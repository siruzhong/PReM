#!/usr/bin/env python
"""Dispatch authorised rebuttal experiment classes on a multi-GPU node.

Paths, GPU count, and concurrency are read from the environment so the same
entry point works outside the original B-node layout:

  PREM_REBUTTAL_RUN_DIR   root for train/eval artifacts (required to launch)
  PREM_REBUTTAL_DATA      decontaminated train JSONL (sha256 must match)
  PREM_REBUTTAL_BASE_DIR  completed Base eval root (for adoption / verify)
  PREM_REBUTTAL_NPROC     visible GPUs / workers (default: 8)
  PREM_REBUTTAL_MAX_JOBS  concurrent jobs (default: 2)
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(os.environ.get("PREM_REBUTTAL_RUN_DIR", ROOT / "outputs/rebuttal_prem/clean"))
DATA = Path(
    os.environ.get(
        "PREM_REBUTTAL_DATA",
        ROOT / "data/llava-video-178k/trainset_9k.jsonl",
    )
)
BASE_DIR = Path(
    os.environ.get(
        "PREM_REBUTTAL_BASE_DIR",
        ROOT / "outputs/rebuttal_prem/table2_k1/eval/base",
    )
)
DATA_HASH = "e630ec50753fcf152263415950c7ba59a97989f92f452375549237c90916d40f"
DATASETS = "videomme,lvbench,longvideobench,mlvu,egoschema,mvbench"
NPROC = int(os.environ.get("PREM_REBUTTAL_NPROC", "8"))
MAX_JOBS = int(os.environ.get("PREM_REBUTTAL_MAX_JOBS", "2"))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_start(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def has_launch_capacity(task, tasks, max_jobs, max_training_jobs):
    running = [item for item in tasks if item["status"] == "running"]
    if len(running) >= max_jobs:
        return False
    return task["kind"] != "train" or sum(
        item["kind"] == "train" for item in running
    ) < max_training_jobs


def gpu_memory_ready(task, free_mib, requirements, stable_seconds, ready_since, now):
    if len(free_mib) != NPROC:
        raise RuntimeError(f"GPU admission requires {NPROC} device readings")
    required = requirements[task["kind"]]
    if min(free_mib) < required:
        ready_since.pop(task["name"], None)
        return False
    since = ready_since.setdefault(task["name"], now)
    return now - since >= stable_seconds


def plan():
    devices = ",".join(str(index) for index in range(NPROC))
    tasks = []
    train_prefix = [
        sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
        "--nnodes", "1", "--nproc_per_node", str(NPROC), "--max_restarts", "0",
    ]
    for variant in ("token_readout", "temporal_mean", "ema", "prem"):
        checkpoint = RUN / "train" / variant / "prem.pt"
        if variant == "token_readout":
            command = train_prefix + [
                "scripts/train/train_same_backbone.py", "--method", variant,
                "--llava_data_file", str(DATA), "--data_protocol", "rebuttal_decontaminated_v1",
                "--model_path", "ckpt/Qwen2.5-VL-3B-Instruct",
                "--llava_video_root", "data/llava-video-178k/frames",
                "--fps", "1.0", "--max_frames", "240", "--visual_buffer_frames", "16",
                "--max_pixels", "200704", "--max_memory_tokens", "128",
                "--num_slots", "1", "--mem_dim", "128", "--layer_groups", "4",
                "--alpha", "0.75", "--pred_weight", "0.2", "--pred_tokens", "4",
                "--lr", "0.0002", "--warmup_ratio", "0.03",
                "--global_batch_size", str(NPROC), "--epochs", "1", "--seed", "13",
                "--save_every", "100",
                "--offload_answer_activations", "--offload_frozen_decoder_for_vision",
                "--distributed_timeout_seconds", "3600",
                "--out_ckpt", str(checkpoint),
            ]
        else:
            command = train_prefix + [
                "scripts/train/train.py",
                "--model_type", "qwen2_5vl",
                "--model_path", "ckpt/Qwen2.5-VL-3B-Instruct",
                "--llava_data_file", str(DATA),
                "--data_protocol", "rebuttal_decontaminated_v1",
                "--llava_video_root", "data/llava-video-178k/frames",
                "--out_ckpt", str(checkpoint),
                "--dev_ratio", "0",
                "--num_slots", "1", "--mem_dim", "128", "--prem_layer_groups", "4",
                "--alpha", "0.75", "--prem_modulation", "attention_kv",
                "--max_frames", "240", "--visual_buffer_frames", "16",
                "--max_pixels", "200704", "--max_memory_tokens", "128",
                "--pred_weight", "0.2", "--pred_tokens", "4",
                "--lr", "0.0002", "--warmup_ratio", "0.03", "--epochs", "1", "--seed", "13",
                "--writer_variant", variant if variant != "prem" else "prem",
                "--offload_answer_activations", "--offload_frozen_decoder_for_vision",
                "--distributed_timeout_seconds", "3600", "--save_every", "100",
            ]
            if variant == "ema":
                command += ["--ema_decay", "0.98"]
        tasks.append({
            "name": variant + "_train", "kind": "train", "variant": variant,
            "priority": 0 if variant in {"prem", "token_readout"} else 3,
            "dependencies": [], "status": "pending", "checkpoint": str(checkpoint),
            "command": command,
        })
    for variant in ("prem", "token_readout", "temporal_mean", "ema"):
        checkpoint = RUN / "train" / variant / "prem.pt"
        output = RUN / "eval" / variant
        command = [
            sys.executable, "-u", "scripts/eval/run_rebuttal_memory.py",
            "--checkpoint", str(checkpoint), "--output_dir", str(output),
            "--devices", devices,
            "--reference_base_dir", str(BASE_DIR),
        ]
        tasks.append({
            "name": variant + "_eval", "kind": "eval", "variant": variant,
            "priority": 1 if variant in {"prem", "token_readout"} else 4,
            "dependencies": [variant + "_train"], "status": "pending",
            "output": str(output), "datasets": DATASETS.split(","), "command": command,
        })
        if variant == "token_readout":
            continue
        modes = ("natural", "needle") if variant == "prem" else ("needle",)
        for mode in modes:
            output = RUN / ("long_stream" if mode == "natural" else "needle") / variant
            command = [
                sys.executable, "-u", "scripts/eval/run_rebuttal_memory.py",
                "--checkpoint", str(checkpoint), "--output_dir", str(output),
                "--mode", mode, "--datasets", "lvbench",
                "--frame_caps", "240,480,960", "--devices", devices,
            ]
            tasks.append({
                "name": variant + "_" + mode, "kind": "eval", "variant": variant,
                "priority": 2 if variant == "prem" else 5,
                "dependencies": [variant + "_eval"], "status": "pending",
                "output": str(output), "datasets": ["lvbench"], "command": command,
            })
    return {
        "created_utc": utc(), "node": os.environ.get("PREM_REBUTTAL_NODE", "local"),
        "max_jobs": MAX_JOBS, "max_training_jobs": 1,
        "data_sha256": DATA_HASH, "tasks": tasks,
    }


def verify(task):
    if task["kind"] == "train":
        import torch
        checkpoint = torch.load(task["checkpoint"], map_location="cpu", weights_only=False)
        expected = {
            "complete": True, "train_steps": 2232, "source_train_qa": 17849,
            "train_manifest_sha256": DATA_HASH, "global_batch_size": NPROC, "skipped": 0,
        }
        if any(checkpoint.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Completed checkpoint contract mismatch: {task['name']}")
        task["checkpoint_sha256"] = hashlib.sha256(Path(task["checkpoint"]).read_bytes()).hexdigest()
    elif task["kind"] == "base":
        expected = dict(
            videomme=2700, lvbench=1549, longvideobench=1337,
            mlvu=2175, egoschema=500, mvbench=4000,
        )
        for dataset, count in expected.items():
            if len(json.loads((BASE_DIR / dataset / "result.json").read_text())) != count:
                raise ValueError(f"Base result coverage mismatch: {dataset}")
    else:
        for dataset in task["datasets"]:
            summary = json.loads((Path(task["output"]) / dataset / "summary.json").read_text())
            if not summary["all_decoder_hashes_match"]:
                raise ValueError("Result alignment failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "campaign.json"
    state = json.loads(path.read_text()) if path.exists() else plan()
    state.setdefault("max_training_jobs", 1)
    if not 1 <= state["max_training_jobs"] <= state["max_jobs"]:
        raise ValueError("Training concurrency must be between one and max_jobs")
    requirements = {"train": 20000, "eval": 10000}
    requirements.update(state.get("min_free_gpu_mib", {}))
    requirements = {kind: float(value) for kind, value in requirements.items()}
    stable_seconds = float(state.get("memory_stable_seconds", 0))
    if not 0 <= stable_seconds < float("inf") or any(
        not 0 < float(value) < float("inf") for value in requirements.values()
    ):
        raise ValueError("Invalid GPU admission thresholds")
    if args.dry_run:
        print(json.dumps(state, indent=2))
        return
    if not DATA.is_file():
        raise FileNotFoundError(f"Missing training data: {DATA}")
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != DATA_HASH:
        raise ValueError("Training data changed")
    import torch
    if torch.cuda.device_count() != NPROC:
        raise RuntimeError(f"This campaign expects {NPROC} visible GPUs")
    import fcntl
    lock = (RUN / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(NPROC)),
        OMP_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        LOAD_STAGGER_SECONDS="1",
    )
    children = {}
    ready_since = {}

    def save():
        state["updated_utc"] = utc()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        os.replace(tmp, path)

    save()
    while True:
        for task in state["tasks"]:
            if task["status"] != "running":
                continue
            process = children.get(task["name"])
            if process is not None:
                code = process.poll()
                if code is None:
                    continue
                task["returncode"] = code
            elif task.get("process_start") and process_start(task["pid"]) == task["process_start"]:
                continue
            try:
                if task.get("returncode", 0) != 0:
                    raise RuntimeError(f"Process returned {task['returncode']}")
                verify(task)
                task["status"] = "verified"
            except Exception as exc:
                task.update(status="failed", error=str(exc))
            task["finished_utc"] = utc()
            save()
            print(json.dumps({k: task[k] for k in ("name", "status", "finished_utc")}), flush=True)
        if any(t["status"] == "failed" for t in state["tasks"]):
            print("Dispatch stopped after a failure; existing jobs remain running.", flush=True)
            raise SystemExit(2)
        if all(t["status"] == "verified" for t in state["tasks"]):
            print("All campaign artifacts verified.", flush=True)
            return
        running = sum(t["status"] == "running" for t in state["tasks"])
        completed = {t["name"] for t in state["tasks"] if t["status"] == "verified"}
        for task in sorted(state["tasks"], key=lambda t: t["priority"]):
            if running >= state["max_jobs"]:
                break
            if task["status"] != "pending" or not set(task["dependencies"]) <= completed:
                continue
            if not has_launch_capacity(task, state["tasks"], state["max_jobs"], state["max_training_jobs"]):
                ready_since.pop(task["name"], None)
                continue
            free = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
            )
            free_mib = [int(x) for x in free.split()]
            if not gpu_memory_ready(
                task, free_mib, requirements, stable_seconds, ready_since, time.monotonic()
            ):
                continue
            log = Path(task.get("launch_log") or (
                ROOT / "logs/rebuttal_prem" / (RUN.name + "_" + task["name"] + ".log")
            ))
            log.parent.mkdir(parents=True, exist_ok=True)
            if task["kind"] == "train":
                Path(task["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as handle:
                process = subprocess.Popen(
                    task["command"], cwd=ROOT, env=env,
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                )
            children[task["name"]] = process
            task.update(status="running", pid=process.pid, started_utc=utc(), log=str(log))
            task["process_start"] = process_start(process.pid)
            task["gpu_admission"] = {
                "free_mib": free_mib,
                "required_free_mib": requirements[task["kind"]],
                "stable_seconds": stable_seconds,
            }
            ready_since.clear()
            running += 1
            save()
            print(json.dumps({"launched": task["name"], "pid": process.pid}), flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
