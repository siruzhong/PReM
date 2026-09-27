#!/usr/bin/env python
"""Evaluate PReM on the official LLaVA-Video-7B-Qwen2 backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.prem_llava_video_model import load_prem_llava_video
from scripts.eval.data_io import get_sample_media_name, load_records, normalize_mcq_sample
from scripts.eval.run_eval_offline import DATASETS, dataset_availability, make_available_subset, pack_cuda_devices
from scripts.eval.llava_video_contract import (
    LLAVA_VIDEO_ADAPTER_VERSION,
    LLAVA_VIDEO_DIRECTORY_FPS,
    LLAVA_VIDEO_POOL_MODE,
    LLAVA_VIDEO_TIME_INSTRUCTION_VERSION,
)
from scripts.llava_video_utils import (
    build_answer_inputs,
    load_llava_base_online_frames,
    load_llava_writer_and_decoder_frames,
    numeric_duration_hint,
    preprocess_frames,
)



import re

def _labeled_options(question: str, option_count: int) -> dict[str, str]:
    matches = re.findall(
        r"(?ms)(?:^|\n)\s*\(([A-Z])\)\s*(.*?)(?=(?:\n\s*\([A-Z]\)\s)|\Z)",
        question,
    )
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:option_count])
    return {label: value.strip() for label, value in matches if label in allowed}


def _normalize_choice_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.casefold()).split())


def extract_choice(text: str, option_count: int, question: str | None = None) -> str | None:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:option_count]
    patterns = (
        rf"(?is)<answer>.*?\b([{allowed}])\b.*?</answer>",
        rf"(?i)(?:answer|option|choice)\s*[:：]?\s*\(?([{allowed}])\)?",
        rf"(?i)^\s*\(?([{allowed}])\)?(?:\s|$|[。,.，])",
        rf"(?i)\b([{allowed}])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).upper()

    if question:
        normalized = _normalize_choice_text(text)
        options = {
            label: _normalize_choice_text(value)
            for label, value in _labeled_options(question, option_count).items()
        }
        exact = [label for label, value in options.items() if normalized and normalized == value]
        if len(exact) == 1:
            return exact[0]

        # Upstream models sometimes answer with option text and hit their
        # generation limit mid-phrase. Only accept an unambiguous prefix.
        if len(normalized) >= 4:
            prefixes = [
                label
                for label, value in options.items()
                if value.startswith(normalized) or normalized.startswith(value)
            ]
            if len(prefixes) == 1:
                return prefixes[0]
    return None


def option_count(row: dict) -> int:
    for key in ("options", "candidates"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return len(value)
    labels = {match.upper() for match in re.findall(r"(?m)(?:^|\n)\s*\(?([A-Z])\)[ .]", row["question"])}
    if labels:
        return max(ord(label) - ord("A") + 1 for label in labels)
    return max(4, int(row["answer"]) + 1)


def artifact_identity(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def load_checkpoint(model, checkpoint_path: str, args) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_type") not in (None, "llava_video"):
        raise ValueError(
            f"Checkpoint model_type={checkpoint.get('model_type')!r} is not LLaVA-Video"
        )
    checkpoint_pooling = checkpoint.get("llava_mm_spatial_pool_mode")
    if checkpoint_pooling != "average":
        raise ValueError(
            "LLaVA-Video PReM checkpoint must be trained with official average pooling; "
            f"got {checkpoint_pooling!r}. Retrain this checkpoint with the corrected loader."
        )
    if checkpoint.get("llava_force_sample") is not True:
        raise ValueError(
            "LLaVA-Video PReM checkpoint must use force_sample=true for the "
            "decoder-visible frame budget. Retrain this checkpoint."
        )
    writer_mode = checkpoint.get("writer_mode")
    if writer_mode != "temporal_mean_per_step":
        raise ValueError(
            "LLaVA-Video evaluation requires a spatial pyramid checkpoint, "
            f"got writer_mode={writer_mode!r}"
        )
    checkpoint_modulation = checkpoint.get("prem_modulation", args.prem_modulation)
    if checkpoint_modulation != args.prem_modulation:
        raise ValueError(
            f"Checkpoint modulation={checkpoint_modulation} differs from requested={args.prem_modulation}"
        )
    checkpoint_budget = int(checkpoint["visual_buffer_frames"])
    if args.buffer_frames is not None and int(args.buffer_frames) != checkpoint_budget:
        raise ValueError(
            f"Evaluation buffer must match training: checkpoint={checkpoint_budget}, requested={args.buffer_frames}"
        )
    memory = model.build_prem_memory(
        num_slots=int(checkpoint.get("num_slots", 4)),
        alpha=float(checkpoint.get("alpha", 1.0)),
        mem_dim=int(checkpoint.get("mem_dim", 128)),
        num_layer_groups=int(checkpoint.get("prem_layer_groups", 1)),
        disable_anti_distractor=bool(checkpoint.get("disable_anti_distractor", False)),
        disable_novelty=bool(checkpoint.get("disable_novelty", False)),
        disable_stability=bool(checkpoint.get("disable_stability", False)),
        disable_evidence_gate_write=bool(checkpoint.get("disable_evidence_gate_write", False)),
        uniform_write_route=bool(checkpoint.get("uniform_write_route", False)),
    )
    state_dict = checkpoint.get("prem_state_dict", checkpoint)
    missing, unexpected = memory.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"PReM checkpoint mismatch: missing={list(missing)} unexpected={list(unexpected)}")
    memory.to(device=model.device, dtype=torch.float32).eval()
    return {
        "num_slots": memory.num_slots,
        "mem_dim": memory.key_dim,
        "layer_groups": memory.num_layer_groups,
        "alpha": float(args.prem_alpha),
        "modulation": checkpoint_modulation,
        "buffer_frames": checkpoint_budget,
        "max_memory_tokens": int(checkpoint.get("max_memory_tokens", 128)),
        "writer_mode": writer_mode,
        "disable_anti_distractor": bool(checkpoint.get("disable_anti_distractor", False)),
        "disable_novelty": bool(checkpoint.get("disable_novelty", False)),
        "disable_stability": bool(checkpoint.get("disable_stability", False)),
        "disable_evidence_gate_write": bool(checkpoint.get("disable_evidence_gate_write", False)),
        "uniform_write_route": bool(checkpoint.get("uniform_write_route", False)),
    }


def expected_rows(args, dataset_name: str) -> tuple[list[dict], Path]:
    spec = DATASETS[dataset_name]
    availability = dataset_availability(spec)
    if not availability["ok"] and not args.allow_partial:
        raise RuntimeError(f"Incomplete local data for {dataset_name}: {availability['reason']}")
    qa_path = Path(spec["qa"])
    if not availability["ok"]:
        qa_path = make_available_subset(
            dataset_name,
            spec,
            Path(availability["root"]),
            Path(args.output_dir) / "_subsets",
        )
    rows = [normalize_mcq_sample(row) for row in load_records(qa_path)]
    if args.sample_limit is not None:
        rows = rows[: args.sample_limit]
    return rows, Path(availability["root"])


def prediction_path(args, dataset: str) -> Path:
    return Path(args.output_dir) / dataset / f"pred_{args.num_chunks}_{args.chunk_idx}.jsonl"


def build_state(model, pixels, config: dict, mode: str, chunk_frames: int):
    common = {
        "prem_num_slots": config["num_slots"],
        "prem_mem_dim": config["mem_dim"],
        "prem_layer_groups": config["layer_groups"],
        "prem_alpha": config["alpha"],
        "prem_disable_anti_distractor": config["disable_anti_distractor"],
        "prem_disable_novelty": config["disable_novelty"],
        "prem_disable_stability": config["disable_stability"],
        "prem_disable_evidence_gate_write": config["disable_evidence_gate_write"],
        "prem_uniform_write_route": config["uniform_write_route"],
    }
    if mode == "offline":
        state, _, stats = model.build_prem_state_from_video(
            images=pixels,
            prem_max_memory_tokens=config["max_memory_tokens"],
            encode_chunk_frames=max(1, chunk_frames),
            **common,
        )
        return {"memory": state, "num_updates": len(pixels), "stats": stats}
    stream = model.stream_reset(batch_size=1, **common)
    for start in range(0, len(pixels), max(1, chunk_frames)):
        stream = model.stream_update_from_images(
            stream,
            pixels[start : start + chunk_frames],
            encode_chunk_frames=max(1, chunk_frames),
        )
    return stream


def worker(args) -> None:
    resources = load_prem_llava_video(
        args.model_path,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
    )
    model = resources.model.eval()
    config = load_checkpoint(model, args.prem_ckpt, args)
    signature = json.dumps(
        {
            "protocol": f"llava_video_prem_{args.mode}",
            "model": artifact_identity(Path(args.model_path) / "config.json"),
            "checkpoint": artifact_identity(args.prem_ckpt),
            "writer_max_frames": args.max_frames,
            "buffer_frames": config["buffer_frames"],
            "stream_max_frames": args.stream_max_frames,
            "decoder_sampling": (
                "fps_capped_reservoir"
                if args.mode == "online"
                else "official_force_sample"
            ),
            "answer_decoding": "official_generate_v1",
            "fps": args.fps,
            "chunk_frames": args.chunk_frames,
            "llava_adapter_version": LLAVA_VIDEO_ADAPTER_VERSION,
            "mm_spatial_pool_mode": LLAVA_VIDEO_POOL_MODE,
            "add_time_instruction": bool(
                getattr(model.config, "add_time_instruction", True)
            ),
            "directory_fps": LLAVA_VIDEO_DIRECTORY_FPS,
            "time_instruction_version": LLAVA_VIDEO_TIME_INSTRUCTION_VERSION,
            "alpha": config["alpha"],
            "modulation": config["modulation"],
        },
        sort_keys=True,
    )
    for dataset_name in args.datasets:
        rows, media_root = expected_rows(args, dataset_name)
        rows = grouped_chunk(rows, args.num_chunks, args.chunk_idx)
        pred_path = prediction_path(args, dataset_name)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if pred_path.exists() and not args.overwrite:
            existing = [json.loads(line) for line in pred_path.read_text().splitlines() if line.strip()]
            if any(row.get("eval_signature") != signature for row in existing):
                raise RuntimeError(f"Existing predictions use a different signature: {pred_path}")
        done = {str(row["id"]) for row in existing}
        current_video = None
        current_state = None
        current_answer_pixels = None
        current_answer_timestamps = None
        current_duration = None
        mode = "w" if args.overwrite else "a"
        with pred_path.open(mode, encoding="utf-8") as handle:
            for row in rows:
                sample_id = str(row["id"])
                if sample_id in done:
                    continue
                video_key = get_sample_media_name(row)
                state_reused = video_key == current_video and current_state is not None
                if not state_reused:
                    media_path = resolve_media(media_root, row)
                    duration_hint = numeric_duration_hint(row.get("duration"))
                    (
                        frames,
                        writer_timestamps,
                        answer_frames,
                        current_answer_timestamps,
                        current_duration,
                    ) = load_llava_writer_and_decoder_frames(
                        media_path,
                        writer_max_frames=args.max_frames,
                        writer_fps=args.fps,
                        decoder_budget=config["buffer_frames"],
                        duration_hint=duration_hint,
                        directory_fps=LLAVA_VIDEO_DIRECTORY_FPS,
                        force_sample=True,
                    )
                    if args.mode == "online":
                        (
                            answer_frames,
                            current_answer_timestamps,
                            current_duration,
                        ) = load_llava_base_online_frames(
                            media_path,
                            stream_max_frames=args.stream_max_frames,
                            fps=args.fps,
                            decoder_budget=config["buffer_frames"],
                            video_key=video_key,
                            duration_hint=duration_hint,
                        )
                    writer_pixels = preprocess_frames(
                        resources.image_processor, frames, model.device
                    )
                    current_state = build_state(
                        model, writer_pixels, config, args.mode, args.chunk_frames
                    )
                    current_answer_pixels = preprocess_frames(
                        resources.image_processor, answer_frames, model.device
                    )
                    current_video = video_key
                    del writer_pixels
                choices = option_count(row)
                answer_inputs = build_answer_inputs(
                    row["question"],
                    resources,
                    current_answer_pixels,
                    choices,
                    model.device,
                    timestamps=current_answer_timestamps,
                    duration=current_duration,
                )
                with torch.inference_mode():
                    raw_prediction = model.generate_answer(
                        **answer_inputs,
                        prem_stream_state=current_state["memory"],
                        tokenizer=resources.tokenizer,
                        prem_modulation=config["modulation"],
                        prem_alpha=config["alpha"],
                        prem_router_gamma=0.0,
                    )
                choice = extract_choice(raw_prediction, choices, row["question"])
                pred_index = None if choice is None else ord(choice) - ord("A")
                output = {
                    "id": sample_id,
                    "video": video_key,
                    "question": row["question"],
                    "answer": int(row["answer"]),
                    "pred": raw_prediction,
                    "pred_choice": choice,
                    "pred_index": pred_index,
                    "correct": pred_index == int(row["answer"]),
                    "dataset": dataset_name,
                    "protocol": args.mode,
                    "state_reused": state_reused,
                    "eval_signature": signature,
                    **current_state.get("stats", {}),
                    **(model.prem_last_stats or {}),
                }
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{args.mode}/{dataset_name}] {sample_id}: {choice}", flush=True)


def merge_dataset(args, dataset_name: str) -> dict:
    rows = []
    for chunk_idx in range(args.num_chunks):
        path = Path(args.output_dir) / dataset_name / f"pred_{args.num_chunks}_{chunk_idx}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    expected, _ = expected_rows(args, dataset_name)
    expected_ids = [str(row["id"]) for row in expected]
    actual_ids = [str(row["id"]) for row in rows]
    duplicates = [key for key, count in Counter(actual_ids).items() if count > 1]
    if duplicates or set(expected_ids) != set(actual_ids):
        raise RuntimeError(
            f"Prediction coverage mismatch for {dataset_name}: duplicates={len(duplicates)} "
            f"missing={len(set(expected_ids) - set(actual_ids))} extra={len(set(actual_ids) - set(expected_ids))}"
        )
    order = {sample_id: index for index, sample_id in enumerate(expected_ids)}
    rows.sort(key=lambda row: order[str(row["id"])])
    output_dir = Path(args.output_dir) / dataset_name
    (output_dir / "pred.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    correct = sum(bool(row["correct"]) for row in rows)
    valid = sum(row.get("pred_index") is not None for row in rows)
    signature = rows[0]["eval_signature"] if rows else None
    result = {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "complete_manifest": True,
        "manifest_id_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest(),
        "valid_predictions": valid,
        "invalid_predictions": len(rows) - valid,
        "eval_signature": signature,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def controller(args) -> None:
    cuda_devices = pack_cuda_devices(args.cuda_devices, args.workers_per_gpu)
    devices = [value.strip() for value in cuda_devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("At least one CUDA device is required")
    args.num_chunks = len(devices)
    processes = []
    for chunk_idx, device in enumerate(devices):
        command = [
            sys.executable,
            __file__,
            "--worker",
            "--model-path", args.model_path,
            "--prem-ckpt", args.prem_ckpt,
            "--output-dir", args.output_dir,
            "--datasets", ",".join(args.datasets),
            "--mode", args.mode,
            "--max-frames", str(args.max_frames),
            "--stream-max-frames", str(args.stream_max_frames),
            "--fps", str(args.fps),
            "--chunk-frames", str(args.chunk_frames),
            "--num-chunks", str(args.num_chunks),
            "--chunk-idx", str(chunk_idx),
            "--prem-alpha", str(args.prem_alpha),
            "--prem-modulation", args.prem_modulation,
        ]
        if args.buffer_frames is not None:
            command += ["--buffer-frames", str(args.buffer_frames)]
        if args.sample_limit is not None:
            command += ["--sample-limit", str(args.sample_limit)]
        if args.allow_partial:
            command.append("--allow-partial")
        if args.overwrite:
            command.append("--overwrite")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        print("[exec] CUDA_VISIBLE_DEVICES=" + device + " " + " ".join(command), flush=True)
        processes.append(subprocess.Popen(command, cwd=REPO_ROOT, env=env))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"LLaVA-Video PReM workers failed: {codes}")
    for dataset_name in args.datasets:
        result = merge_dataset(args, dataset_name)
        print(
            f"[result] {args.mode}/{dataset_name}: {result['correct']}/{result['total']} "
            f"({result['accuracy']:.2%})",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="ckpt/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--prem-ckpt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="longvideobench,mlvu,videomme,egoschema,mvbench,lvbench")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--stream-max-frames", type=int, default=240)
    parser.add_argument("--buffer-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--chunk-frames", type=int, default=2)
    parser.add_argument("--prem-alpha", type=float, default=1.0)
    parser.add_argument(
        "--prem-modulation",
        choices=("attention", "attention_kv", "attention_k", "attention_v"),
        default="attention_kv",
    )
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.stream_max_frames <= 0:
        parser.error("--stream-max-frames must be positive")
    args.datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in args.datasets if name not in DATASETS]
    if unknown:
        parser.error(f"Unknown datasets: {unknown}")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        worker(parsed)
    else:
        controller(parsed)
