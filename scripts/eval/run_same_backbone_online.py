#!/usr/bin/env python
"""Launch one same-backbone method over the registered public datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.run_eval_offline import pack_cuda_devices
from scripts.eval.run_eval_online import PUBLIC_DATASETS, detect_cuda_devices


DEFAULT_DATASETS = "longvideobench,mlvu,videomme,egoschema,mvbench,lvbench"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("base", "lora", "token_readout", "infinipot_v", "h2o", "prem"),
        required=True,
    )
    parser.add_argument("--model_path", default="ckpt/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--baseline_ckpt", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evaluation_name", default=None)
    parser.add_argument("--datasets", default=DEFAULT_DATASETS)
    parser.add_argument("--cuda_devices", default=None)
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=240)
    parser.add_argument("--buffer_frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--max_video_tokens", type=int, default=11520)
    parser.add_argument("--chunk_frames", type=int, default=2)
    parser.add_argument("--stream_chunk_tokens", type=int, default=32)
    parser.add_argument("--h2o_heavy_tokens", type=int, default=1024)
    parser.add_argument("--h2o_recent_tokens", type=int, default=1024)
    parser.add_argument("--infinipot_block_units", type=int, default=32)
    parser.add_argument("--infinipot_keep_units", type=int, default=24)
    parser.add_argument("--infinipot_tar_ratio", type=float, default=0.5)
    parser.add_argument("--infinipot_query_ratio", type=float, default=0.25)
    parser.add_argument("--prem_modulation", default="attention_kv")
    parser.add_argument("--profile_efficiency", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.method in {"lora", "token_readout", "prem"} and not args.baseline_ckpt:
        parser.error(f"--baseline_ckpt is required for {args.method}")
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = [name for name in selected if name not in PUBLIC_DATASETS]
    if unknown:
        parser.error(f"Unknown datasets: {','.join(unknown)}")
    if "storm" in selected:
        parser.error(
            "STORM is query-time causal and is outside the registered end-of-stream comparison"
        )

    packed = pack_cuda_devices(
        args.cuda_devices or detect_cuda_devices(), args.workers_per_gpu
    )
    devices = [value.strip() for value in packed.split(",") if value.strip()]
    if not devices:
        parser.error("No CUDA devices were detected; pass --cuda_devices")
    num_chunks = args.num_chunks or len(devices)
    if num_chunks > len(devices):
        parser.error(f"num_chunks={num_chunks} exceeds visible worker devices={devices}")

    evaluation_name = args.evaluation_name or args.method
    common = [
        sys.executable,
        str(REPO_ROOT / "scripts/eval/eval_same_backbone_online.py"),
        "--method",
        args.method,
        "--model_path",
        args.model_path,
        "--output_dir",
        args.output_dir,
        "--evaluation_name",
        evaluation_name,
        "--cuda_devices",
        packed,
        "--num_chunks",
        str(num_chunks),
        "--max_frames",
        str(args.max_frames),
        "--buffer_frames",
        str(args.buffer_frames),
        "--fps",
        str(args.fps),
        "--max_pixels",
        str(args.max_pixels),
        "--max_video_tokens",
        str(args.max_video_tokens),
        "--chunk_frames",
        str(args.chunk_frames),
        "--stream_chunk_tokens",
        str(args.stream_chunk_tokens),
        "--h2o_heavy_tokens",
        str(args.h2o_heavy_tokens),
        "--h2o_recent_tokens",
        str(args.h2o_recent_tokens),
        "--infinipot_block_units",
        str(args.infinipot_block_units),
        "--infinipot_keep_units",
        str(args.infinipot_keep_units),
        "--infinipot_tar_ratio",
        str(args.infinipot_tar_ratio),
        "--infinipot_query_ratio",
        str(args.infinipot_query_ratio),
        "--prem_modulation",
        args.prem_modulation,
    ]
    if args.baseline_ckpt:
        common += ["--baseline_ckpt", args.baseline_ckpt]
    if args.profile_efficiency:
        common.append("--profile_efficiency")
    if args.overwrite:
        common.append("--overwrite")

    for name in selected:
        dataset = PUBLIC_DATASETS[name]
        command = common + [
            "--dataset",
            name,
            "--video_dir",
            dataset["video_dir"],
            "--gt_file",
            dataset["gt_file"],
        ]
        print("[exec] " + " ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
