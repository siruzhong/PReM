#!/usr/bin/env python
"""Unified streaming evaluation for PREM-Online, Flash-VStream, and Qwen2-VL online.

Replaces the five per-model/per-scenario shell scripts with a single entry point.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.run_eval_offline import pack_cuda_devices

PUBLIC_DATASETS = {
    "longvideobench": {
        "video_dir": "data/eval_video/LongVideoBench/videos",
        "gt_file": "data/eval_video/LongVideoBench/raw/lvb_val.json",
    },
    "mlvu": {
        "video_dir": "data/eval_video/mlvu/videos",
        "gt_file": "data/eval_video/mlvu/test_qa.json",
    },
    "videomme": {
        "video_dir": "data/eval_video/videomme/Video-MME/data",
        "gt_file": "data/eval_video/videomme/test_qa.json",
    },
    "egoschema": {
        "video_dir": "data/eval_video/EgoSchema/videos",
        "gt_file": "data/eval_video/EgoSchema/test_qa.json",
    },
    "storm": {
        "video_dir": "data/eval_video/storm_real/video",
        "gt_file": "data/eval_video/storm_real/qa_results/questions.jsonl",
    },
    "mvbench": {
        "video_dir": "data/eval_video/mvbench/videos",
        "gt_file": "data/eval_video/mvbench/test_qa.json",
    },
    "lvbench": {
        "video_dir": "data/eval_video/lvbench/videos",
        "gt_file": "data/eval_video/lvbench/test_qa.json",
    },
}

EVAL_ENGINES = {
    "prem": "scripts/eval/eval_prem_online.py",
    "flash": "scripts/eval/eval_flash_vstream_online.py",
    "qwen2vl": "scripts/eval/eval_qwen2vl_online.py",
    "qwen25vl": "scripts/eval/eval_qwen2vl_online.py",
    "qwen3vl": "scripts/eval/eval_qwen2vl_online.py",
}


def detect_cuda_devices():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        return visible
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return ",".join(line.strip() for line in result.stdout.splitlines() if line.strip())


def run_cmd(cmd, dry_run=False):
    print("[exec] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Unified streaming evaluation")
    ap.add_argument("--mode", choices=["prem", "flash", "qwen2vl", "qwen25vl", "qwen3vl"], required=True,
                    help="Model to evaluate")
    ap.add_argument("--scenario", choices=["public"], required=True,
                    help="Public datasets")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prem_ckpt", default=None,
                    help="PREM-Online checkpoint (required for prem mode)")
    ap.add_argument("--qwen_model", default="ckpt/Qwen2-VL-7B-Instruct",
                    help="Qwen2-VL base model (flash mode)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--evaluation_name", required=True)
    ap.add_argument("--datasets", default="longvideobench,mlvu,videomme,egoschema,mvbench,lvbench")
    ap.add_argument("--cuda_devices", default=None)
    ap.add_argument("--workers_per_gpu", type=int, default=None)
    ap.add_argument("--num_chunks", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=240)
    ap.add_argument("--max_pixels", type=int, default=200704)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--mcq_max_new_tokens", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    # PREM-Online specific
    ap.add_argument("--prem_alpha", type=float, default=1.0, help="Multiplier on the learned PREM steer scale")
    ap.add_argument("--prem_override_alpha", action="store_true", help="Override alpha stored in the PREM checkpoint")
    ap.add_argument(
        "--prem_modulation",
        choices=["attention", "attention_kv", "attention_k", "attention_v"],
        default="attention",
    )
    ap.add_argument("--prem_disable_anti_distractor", action="store_true")
    ap.add_argument("--prem_disable_novelty", action="store_true")
    ap.add_argument("--prem_disable_stability", action="store_true")
    ap.add_argument("--prem_disable_evidence_gate_write", action="store_true")
    ap.add_argument("--prem_uniform_write_route", action="store_true")
    ap.add_argument("--chunk_frames", type=int, default=2)
    ap.add_argument("--initial_chunk_frames", type=int, default=0)
    ap.add_argument("--stream_chunk_tokens", type=int, default=32)
    # Flash-VStream specific
    ap.add_argument("--flash_initial_chunk_frames", type=int, default=120)
    ap.add_argument("--flash_chunk_frames", type=int, default=1)
    # Qwen2-VL online specific
    ap.add_argument("--buffer_frames", type=int, default=16)
    ap.add_argument("--max_video_tokens", type=int, default=11520)
    ap.add_argument("--text_only", action="store_true")
    ap.add_argument("--profile_efficiency", action="store_true")
    # Misc
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.mode == "prem" and not args.prem_ckpt:
        ap.error("--prem_ckpt is required for prem mode")
    if args.profile_efficiency and args.mode == "flash":
        ap.error("--profile_efficiency is currently supported only for PREM and Qwen-VL")

    engine_script = EVAL_ENGINES[args.mode]
    cuda_devices = pack_cuda_devices(args.cuda_devices or detect_cuda_devices(), args.workers_per_gpu)
    if not cuda_devices:
        ap.error("No visible CUDA devices detected")
    devices = [device.strip() for device in cuda_devices.split(",") if device.strip()]
    if not devices:
        ap.error("--cuda_devices must contain at least one device id")
    num_chunks = args.num_chunks if args.num_chunks and args.num_chunks > 0 else len(devices)
    if num_chunks > len(devices):
        ap.error(f"--num_chunks={num_chunks} exceeds --cuda_devices={cuda_devices}")

    common = [
        sys.executable,
        str(REPO_ROOT / engine_script),
        "--model_path", args.model_path,
        "--output_dir", args.output_dir,
        "--evaluation_name", args.evaluation_name,
        "--cuda_devices", cuda_devices,
        "--num_chunks", str(num_chunks),
        "--max_frames", str(args.max_frames),
        "--fps", str(args.fps),
        "--max_pixels", str(args.max_pixels),
        "--mcq_max_new_tokens", str(args.mcq_max_new_tokens),
    ]
    if args.overwrite:
        common.append("--overwrite")
    if args.profile_efficiency:
        common.append("--profile_efficiency")

    if args.mode == "prem":
        common += [
            "--prem_ckpt", args.prem_ckpt,
            "--prem_alpha", str(args.prem_alpha),
            "--prem_modulation", args.prem_modulation,
            "--chunk_frames", str(args.chunk_frames),
            "--initial_chunk_frames", str(args.initial_chunk_frames),
            "--stream_chunk_tokens", str(args.stream_chunk_tokens),
            "--max_new_tokens", str(args.max_new_tokens),
        ]
        if args.prem_override_alpha:
            common.append("--prem_override_alpha")
        for enabled, flag in (
            (args.prem_disable_anti_distractor, "--prem_disable_anti_distractor"),
            (args.prem_disable_novelty, "--prem_disable_novelty"),
            (args.prem_disable_stability, "--prem_disable_stability"),
            (args.prem_disable_evidence_gate_write, "--prem_disable_evidence_gate_write"),
            (args.prem_uniform_write_route, "--prem_uniform_write_route"),
        ):
            if enabled:
                common.append(flag)
    elif args.mode == "flash":
        common += [
            "--qwen_model", args.qwen_model,
            "--initial_chunk_frames", str(args.flash_initial_chunk_frames),
            "--chunk_frames", str(args.flash_chunk_frames),
        ]
    elif args.mode in ("qwen2vl", "qwen25vl", "qwen3vl"):
        common += [
            "--buffer_frames", str(args.buffer_frames),
            "--max_video_tokens", str(args.max_video_tokens),
        ]
        if args.text_only:
            common.append("--text_only")

    selected_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    unknown = [name for name in selected_names if name not in PUBLIC_DATASETS]
    if unknown:
        ap.error(
            f"Unknown datasets: {','.join(unknown)}; "
            f"available: {','.join(PUBLIC_DATASETS)}"
        )
    for name in selected_names:
        ds = PUBLIC_DATASETS[name]
        cmd = common + [
            "--dataset", name,
            "--video_dir", ds["video_dir"],
            "--gt_file", ds["gt_file"],
        ]
        if name == "storm":
            cmd.append("--time_constrained")
        run_cmd(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
