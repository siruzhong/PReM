#!/usr/bin/env python
"""Run one verified rebuttal checkpoint on eight B workers and validate coverage."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.run_eval_online import PUBLIC_DATASETS
from scripts.eval.summarize_rebuttal_memory import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("natural", "needle"), default="natural")
    parser.add_argument("--datasets", default="videomme,lvbench,longvideobench,mlvu,egoschema,mvbench")
    parser.add_argument("--frame_caps", default="240")
    parser.add_argument(
        "--devices",
        default=os.environ.get("PREM_REBUTTAL_DEVICES", "0,1,2,3,4,5,6,7"),
    )
    parser.add_argument(
        "--reference_base_dir",
        default=os.environ.get(
            "PREM_REBUTTAL_BASE_DIR",
            "outputs/rebuttal_prem/table2_k1/eval/base",
        ),
    )
    args = parser.parse_args()
    devices = args.devices.split(",")
    datasets = ["lvbench"] if args.mode == "needle" else args.datasets.split(",")
    for dataset in datasets:
        config = PUBLIC_DATASETS[dataset]
        output = Path(args.output_dir) / dataset
        output.mkdir(parents=True, exist_ok=True)
        jobs = []
        for index, device in enumerate(devices):
            command = [sys.executable, "-u", str(ROOT / "scripts/eval/eval_rebuttal_memory.py"),
                       "--checkpoint", args.checkpoint, "--output_dir", str(output),
                       "--mode", args.mode, "--frame_caps", args.frame_caps,
                       "--gt_file", config["gt_file"], "--video_dir", config["video_dir"],
                       "--num_chunks", str(len(devices)), "--chunk_idx", str(index)]
            env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = device
            with (output / f"worker_{index}.log").open("a") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((process, command))
        (output / "worker_commands.json").write_text(json.dumps(
            [{"pid": p.pid, "command": command} for p, command in jobs], indent=2))
        codes = [process.wait() for process, _ in jobs]
        if any(codes):
            raise RuntimeError(f"Workers failed: {dataset} {codes}; inspect worker logs")
        reference = ROOT / args.reference_base_dir / dataset / "result.json" if args.mode == "natural" else None
        result = summarize(output, reference)
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"dataset": dataset, "questions": result["questions"],
                          "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
