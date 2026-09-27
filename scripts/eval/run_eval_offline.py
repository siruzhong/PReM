#!/usr/bin/env python
"""Run offline (full-video) benchmark comparisons.

Methods:
  1. Qwen2-VL-7B-Instruct baseline
  2. PREM (unified B+M checkpoint with one-chunk batch ingestion)
  3. Flash-VStream-Qwen-7b

Datasets: LongVideoBench, MLVU, Video-MME, EgoSchema, MVBench, LVBench.
The script checks local video/frame availability before launching each run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

OFFICIAL_MAX_FRAMES = 240
OFFICIAL_MAX_PIXELS = 4 * 224 * 224
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import get_sample_media_name, load_records, normalize_mcq_sample

DATASETS = {
    "storm": {
        "dataset_arg": "storm",
        "qa": "data/eval_video/storm_real/qa_results/questions.jsonl",
        "root": "data/eval_video/storm_real/video",
        "video_exts": [".mp4"],
    },
    "longvideobench": {
        "dataset_arg": "longvideobench",
        "qa": "data/eval_video/LongVideoBench/raw/lvb_val.json",
        "root": "data/eval_video/LongVideoBench/frames",
        "video_root": "data/eval_video/LongVideoBench/videos",
        "video_exts": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
        "archive_hint": "data.tos-link/LongVideoBench_full/videos.tar.part.aa",
    },
    "mlvu": {
        "dataset_arg": "mlvu",
        "qa": "data/eval_video/mlvu/test_qa.json",
        "root": "data/eval_video/mlvu/videos",
        "video_exts": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
    },
    "videomme": {
        "dataset_arg": "videommewo",
        "qa": "data/eval_video/videomme/test_qa.json",
        "root": "data/eval_video/videomme/Video-MME/data",
        "video_exts": [".mp4"],
        "archive_hint": "data.tos-link/Video-MME/videos/videos_chunked_01.zip",
    },
    "egoschema": {
        "dataset_arg": "egoschema",
        "qa": "data/eval_video/EgoSchema/test_qa.json",
        "root": "data/eval_video/EgoSchema/videos",
        "video_exts": [".mp4"],
    },
    "mvbench": {
        "dataset_arg": "mvbench",
        "qa": "data/eval_video/mvbench/test_qa.json",
        "root": "data/eval_video/mvbench/videos",
        "video_exts": [".mp4", ".avi", ".webm"],
        "archive_hint": "data.tos-link/MVBench/raw/video",
    },
    "lvbench": {
        "dataset_arg": "lvbench",
        "qa": "data/eval_video/lvbench/test_qa.json",
        "root": "data/eval_video/lvbench/videos",
        "video_exts": [".mp4"],
        "archive_hint": "data.tos-link/LVBench/raw/_video_zips/all_videos.zip",
    },
}

FRAME_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def media_name(row: dict) -> str:
    return get_sample_media_name(normalize_mcq_sample(row))


def has_media(root: Path, video_id: str, exts: list[str]):
    def nonempty_frame_dir(path: Path):
        return path.is_dir() and any(
            child.is_file() and child.suffix.lower() in FRAME_EXTS
            for child in path.iterdir()
        )

    base = root / video_id
    if base.is_file() and (not exts or base.suffix.lower() in VIDEO_EXTS):
        return True
    if nonempty_frame_dir(base):
        return True
    stem = video_id
    if Path(stem).suffix:
        stem = str(Path(stem).with_suffix(""))
    stem_base = root / stem
    if stem_base.is_file() and (not exts or stem_base.suffix.lower() in VIDEO_EXTS):
        return True
    if nonempty_frame_dir(stem_base):
        return True
    for ext in exts:
        if (root / f"{stem}{ext}").exists():
            return True
    return False


def dataset_availability(spec):
    qa_path = Path(spec["qa"])
    root = Path(spec["root"])
    if not qa_path.exists():
        return {"ok": False, "reason": f"missing qa file: {qa_path}", "rows": 0, "available": 0, "unique": 0}
    rows = load_records(qa_path)
    ids = sorted({media_name(row) for row in rows})
    roots = [root]
    if spec.get("video_root"):
        roots.append(Path(spec["video_root"]))
    available_by_root = {
        str(candidate): sum(1 for video_id in ids if has_media(candidate, video_id, spec["video_exts"]))
        for candidate in roots
    }
    best_root = max(roots, key=lambda candidate: available_by_root[str(candidate)])
    available = available_by_root[str(best_root)]
    ok = best_root.exists() and available == len(ids)
    if ok:
        reason = "ok"
    else:
        archive_hint = spec.get("archive_hint")
        archive_msg = ""
        if archive_hint and Path(archive_hint).exists():
            archive_root = spec.get("video_root", root)
            archive_msg = f"; archive present at {archive_hint} and must be extracted into {archive_root}"
        root_counts = ", ".join(f"{path}: {count}/{len(ids)}" for path, count in available_by_root.items())
        reason = f"direct media available {root_counts}{archive_msg}"
    return {
        "ok": ok,
        "reason": reason,
        "rows": len(rows),
        "available": available,
        "unique": len(ids),
        "root": str(best_root),
        "archive_present": bool(spec.get("archive_hint") and Path(spec["archive_hint"]).exists()),
    }


def make_available_subset(dataset_name: str, spec, media_root: Path, out_dir: Path):
    qa_path = Path(spec["qa"])
    rows = [normalize_mcq_sample(row) for row in load_records(qa_path)]
    subset = [
        row for row in rows
        if has_media(media_root, media_name(row), spec["video_exts"])
    ]
    if not subset:
        raise RuntimeError(f"No locally available media for {dataset_name}")
    subset_dir = out_dir / "_subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    subset_path = subset_dir / f"{dataset_name}_available_{len(subset)}.json"
    with open(subset_path, "w") as f:
        json.dump(subset, f, indent=2)
    return subset_path


def run_cmd(cmd, dry_run=False):
    print("[exec] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def visible_cuda_devices():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if value:
        return value
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False, capture_output=True, text=True,
        )
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return ",".join(ids) or None
    except OSError:
        return None


def default_workers_per_gpu() -> int:
    raw = os.environ.get("WORKERS_PER_GPU", "1").strip()
    try:
        return max(int(raw), 1)
    except ValueError:
        return 1


def pack_cuda_devices(cuda_devices: str | None = None, workers_per_gpu: int | None = None) -> str:
    workers = max(int(workers_per_gpu), 1) if workers_per_gpu is not None else default_workers_per_gpu()
    raw = (cuda_devices or visible_cuda_devices() or "0").strip()
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for device in devices:
        if device not in seen:
            unique.append(device)
            seen.add(device)
    packed = unique * workers
    packed_str = ",".join(packed) if packed else "0"
    print(f"[gpu] {len(unique) or 1} devices x {workers} workers/gpu -> {packed_str}", flush=True)
    return packed_str


def qwen_evaluation_name(model_path: str) -> str:
    """Keep result directory names aligned with the selected Qwen backbone."""
    model_name = Path(model_path).name.lower().replace(".", "_")
    if "qwen3" in model_name:
        family = "qwen3vl"
    elif "qwen2_5" in model_name:
        family = "qwen2_5vl"
    else:
        family = "qwen2vl"
    size = next((part for part in model_name.split("-") if part.endswith("b") and part[:-1].isdigit()), None)
    return f"{family}_{size}" if size else family


def build_methods(args):
    qwen_name = qwen_evaluation_name(args.qwen_model)
    methods = [
        (qwen_name, args.qwen_model, []),
        ("prem_attention", args.qwen_model, []),
        ("flash_vstream", args.flash_model, []),
    ]
    if args.only:
        keep = {"qwen": qwen_name, "prem": "prem_attention", "flash": "flash_vstream"}[args.only]
        methods = [method for method in methods if method[0] == keep]
    return methods



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen_model", default="ckpt/Qwen2-VL-7B-Instruct")
    ap.add_argument("--flash_model", default="ckpt/Flash-VStream-Qwen-7b")
    ap.add_argument("--prem_ckpt", required=True)
    ap.add_argument("--output_dir", default="outputs/prem/public_eval")
    ap.add_argument("--datasets", default="longvideobench,mlvu,videomme,egoschema")
    ap.add_argument("--num_chunks", type=int, default=None)
    ap.add_argument("--cuda_devices", default=None)
    ap.add_argument("--workers_per_gpu", type=int, default=None)
    ap.add_argument("--max_frames", type=int, default=OFFICIAL_MAX_FRAMES)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_pixels", type=int, default=OFFICIAL_MAX_PIXELS)
    ap.add_argument("--mcq_max_new_tokens", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=512)
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
    ap.add_argument("--text_only", action="store_true")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--only", choices=["qwen", "prem", "flash"], default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--profile_efficiency", action="store_true")
    args = ap.parse_args()
    args.cuda_devices = pack_cuda_devices(args.cuda_devices, args.workers_per_gpu)
    if args.num_chunks is None or args.num_chunks <= 0:
        args.num_chunks = len(args.cuda_devices.split(",")) if args.cuda_devices else 1

    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in selected if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}; supported={sorted(DATASETS)}")

    methods = build_methods(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    failed_runs = []
    availability = {}
    for name in selected:
        spec = DATASETS[name]
        status = dataset_availability(spec)
        availability[name] = status
        print(f"[data] {name}: rows={status['rows']} unique={status['unique']} {status['reason']}", flush=True)
        if not status["ok"] and not args.allow_partial:
            print(f"[error] {name}: incomplete local media; official evaluation will not skip this dataset.", flush=True)
            if status.get("archive_present"):
                print(
                    f"[prepare] extract {spec['archive_hint']} into {spec.get('video_root', spec['root'])}",
                    flush=True,
                )
            raise RuntimeError(f"{name} has incomplete local media: {status['reason']}. Use --allow_partial only for diagnostics.")
        stress_args = []
        dataset_arg = spec["dataset_arg"]
        if not status["ok"] and args.allow_partial:
            media_root = Path(status["root"])
            subset_path = make_available_subset(name, spec, media_root, Path(args.output_dir))
            dataset_arg = "overwrite"
            stress_args = [
                "--stress_variant",
                f"{name}_available",
                "--stress_frame_dir",
                str(media_root),
                "--stress_data_file",
                str(subset_path),
            ]
            print(f"[partial] {name}: using subset qa={subset_path}", flush=True)
        for method_name, model_path, extra in methods:
            if method_name == "prem_attention":
                gt_file = str(subset_path) if not status["ok"] and args.allow_partial else spec["qa"]
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/eval/eval_prem_online.py"),
                    "--dataset", dataset_arg,
                    "--model_path", model_path,
                    "--prem_ckpt", args.prem_ckpt,
                    "--video_dir", status["root"],
                    "--gt_file", gt_file,
                    "--output_dir", args.output_dir,
                    "--evaluation_name", method_name,
                    "--num_chunks", str(args.num_chunks),
                    "--max_frames", str(args.max_frames),
                    "--fps", str(args.fps),
                    "--chunk_frames", str(args.max_frames),
                    "--max_pixels", str(args.max_pixels),
                    "--mcq_max_new_tokens", str(args.mcq_max_new_tokens),
                    "--max_new_tokens", str(args.max_new_tokens),
                    "--prem_alpha", str(args.prem_alpha),
                    "--prem_modulation", args.prem_modulation,
                ]
                if args.prem_override_alpha:
                    cmd.append("--prem_override_alpha")
                for enabled, flag in (
                    (args.prem_disable_anti_distractor, "--prem_disable_anti_distractor"),
                    (args.prem_disable_novelty, "--prem_disable_novelty"),
                    (args.prem_disable_stability, "--prem_disable_stability"),
                    (args.prem_disable_evidence_gate_write, "--prem_disable_evidence_gate_write"),
                    (args.prem_uniform_write_route, "--prem_uniform_write_route"),
                ):
                    if enabled:
                        cmd.append(flag)
            else:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/eval/eval_any_dataset.py"),
                    "--dataset", dataset_arg,
                    "--model-path", model_path,
                    "--output_dir", args.output_dir,
                    "--evaluation_name", method_name,
                    "--num_chunks", str(args.num_chunks),
                    "--max_frames", str(args.max_frames),
                    "--fps", str(args.fps),
                    "--max_pixels", str(args.max_pixels),
                    "--mcq_max_new_tokens", str(args.mcq_max_new_tokens),
                    "--max_new_tokens", str(args.max_new_tokens),
                ]
                if args.text_only:
                    if method_name != qwen_evaluation_name(args.qwen_model):
                        raise ValueError("--text_only is supported only with --only qwen")
                    cmd.append("--text_only")
            if args.cuda_devices:
                cmd += ["--cuda_devices", args.cuda_devices]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.profile_efficiency:
                cmd.append("--profile_efficiency")
            if method_name != "prem_attention":
                cmd += stress_args
                cmd += extra
            try:
                run_cmd(cmd, dry_run=args.dry_run)
            except subprocess.CalledProcessError as exc:
                failed_runs.append((name, method_name, exc.returncode))
                print(
                    f"[error] {name}/{method_name} failed with exit code {exc.returncode}; continuing.",
                    flush=True,
                )

    with open(Path(args.output_dir) / "availability.json", "w") as f:
        json.dump(availability, f, indent=2)

    if failed_runs:
        summary = ", ".join(
            f"{dataset}/{method} (exit {returncode})"
            for dataset, method, returncode in failed_runs
        )
        raise RuntimeError(f"Some offline evaluations failed: {summary}")


if __name__ == "__main__":
    main()
