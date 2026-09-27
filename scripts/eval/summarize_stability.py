#!/usr/bin/env python
"""Validate and summarize Table 5 random-seed stability experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = (
    ("longvideobench", 1337),
    ("mlvu", 2175),
    ("videomme", 2700),
    ("egoschema", 500),
    ("mvbench", 4000),
    ("lvbench", 1549),
)

BACKBONES = {
    "qwen25": {
        "label": "Qwen2.5-VL-3B",
        "family": "qwen25",
        "model_type": "qwen2_5vl",
        "model_kind": "qwen2_5_vl",
        "model_path": "ckpt/Qwen2.5-VL-3B-Instruct",
        "data_file": "data/llava-video-178k/trainset_9k.jsonl",
        "lr": 2e-4,
        "eval_alpha": 1.0,
        "base_offline_evaluation": "qwen2_5vl_3b",
        "base_online_evaluation": "qwen25vl_qwen25_public",
        "online_protocol": "qwen2_5_vl_online_reservoir_endtime",
        "original_seed13_checkpoint": "outputs/prem_attention_kv/qwen25_3b_b16/prem.pt",
    },
    "qwen3": {
        "label": "Qwen3-VL-8B",
        "family": "qwen3",
        "model_type": "qwen3vl",
        "model_kind": "qwen3_vl",
        "model_path": "ckpt/Qwen3-VL-8B-Instruct",
        "data_file": "data/llava-video-178k/trainset_9k.jsonl",
        "lr": 1e-4,
        "eval_alpha": 0.5,
        "base_offline_evaluation": "qwen3vl_8b",
        "base_online_evaluation": "qwen3vl_qwen3_public",
        "online_protocol": "qwen3_vl_online_reservoir_endtime",
        "original_seed13_checkpoint": "outputs/prem_attention_kv/qwen3_8b_b16/prem.pt",
    },
}


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in value.replace(",", " ").split()]
    if len(seeds) < 2:
        raise ValueError("Table 5 requires at least two random seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate stability seeds: {seeds}")
    return seeds


def parse_checkpoint_overrides(values: list[str]) -> dict[tuple[str, int], str]:
    checkpoints: dict[tuple[str, int], str] = {}
    for value in values:
        try:
            backbone, seed_text, path = value.split("=", 2)
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --checkpoint {value!r}; expected BACKBONE=SEED=PATH"
            ) from exc
        if backbone not in BACKBONES or not path:
            raise ValueError(f"Invalid checkpoint override: {value!r}")
        checkpoints[(backbone, seed)] = path
    return checkpoints


def default_checkpoint(output_root: Path, backbone: str, seed: int) -> str:
    if seed == 13:
        return BACKBONES[backbone]["original_seed13_checkpoint"]
    return str(output_root / "checkpoints" / backbone / f"seed_{seed}" / "prem.pt")


def checkpoint_path(
    output_root: Path,
    backbone: str,
    seed: int,
    overrides: dict[tuple[str, int], str],
) -> str:
    return overrides.get((backbone, seed), default_checkpoint(output_root, backbone, seed))


def validate_checkpoint(path_text: str, backbone: str, seed: int) -> dict[str, Any]:
    import torch

    config = BACKBONES[backbone]
    path = Path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {backbone} seed {seed} checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    recorded_seed = checkpoint.get("seed")
    if recorded_seed is None and seed == 13:
        recorded_seed = 13
    expected = {
        "complete": True,
        "seed": seed,
        "model_type": config["model_type"],
        "model_path": config["model_path"],
        "llava_data_file": config["data_file"],
        "visual_buffer_frames": 16,
        "max_frames": 64,
        "max_pixels": 200704,
        "alpha": 1.0,
        "router_gamma": 0.05,
        "pred_weight": 0.1,
        "pred_tokens": 8,
        "num_slots": 4,
        "mem_dim": 128,
        "max_memory_tokens": 128,
        "prem_layer_groups": 1,
        "prem_modulation": "attention_kv",
        "writer_mode": "temporal_mean_per_step",
        "warmup_ratio": 0.03,
    }
    actual = {key: checkpoint.get(key) for key in expected}
    actual["seed"] = recorded_seed
    mismatches = {
        key: (actual[key], value)
        for key, value in expected.items()
        if actual[key] != value
    }
    parameter_groups = checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
    lr = parameter_groups[0].get("lr") if parameter_groups else None
    if lr != config["lr"]:
        mismatches["lr"] = (lr, config["lr"])
    if mismatches:
        raise ValueError(f"{backbone} seed {seed} checkpoint mismatch at {path}: {mismatches}")
    return {
        "path": str(path),
        "seed": seed,
        "trainable_params": sum(
            int(tensor.numel()) for tensor in checkpoint["prem_state_dict"].values()
        ),
        "train_steps": int(checkpoint.get("train_steps", 0)),
        "final_train_loss": checkpoint.get("final_train_loss"),
    }


def result_dataset_name(dataset: str, mode: str) -> str:
    return "videommewo" if mode == "offline" and dataset == "videomme" else dataset


def evaluation_name(backbone: str, method: str, mode: str) -> str:
    config = BACKBONES[backbone]
    if method == "prem":
        return "prem_attention" if mode == "offline" else f"prem_{config['family']}_public"
    return config["base_offline_evaluation"] if mode == "offline" else config["base_online_evaluation"]


def row_root(output_root: Path, mode: str, backbone: str, method: str, seed: int | None) -> Path:
    name = "base" if method == "base" else f"prem_seed_{seed}"
    return output_root / mode / backbone / name


def parse_signature(record: dict[str, Any]) -> dict[str, Any]:
    signature = record.get("eval_signature")
    if not isinstance(signature, str):
        raise ValueError("Result record has no serialized eval_signature")
    parsed = json.loads(signature)
    if not isinstance(parsed, dict):
        raise ValueError("eval_signature must decode to an object")
    return parsed


def validate_signature(
    signature: dict[str, Any],
    backbone: str,
    method: str,
    mode: str,
    checkpoint: str | None,
) -> None:
    config = BACKBONES[backbone]
    expected: dict[str, Any] = {"fps": 1.0, "max_pixels": 200704}
    if method == "base" and mode == "offline":
        expected.update({
            "protocol": "full_video",
            "model_kind": config["model_kind"],
            "max_frames": 16,
        })
        model_path = signature.get("model", {}).get("path", "")
    elif method == "base":
        expected.update({
            "protocol": config["online_protocol"],
            "buffer_frames": 16,
            "max_frames": 240,
            "time_constrained": False,
        })
        model_path = signature.get("model_path", "")
    else:
        expected.update({
            "protocol": "unified_bounded_visual_memory",
            "visual_buffer_frames": 16,
            "max_frames": 240,
            "prem_alpha": config["eval_alpha"],
            "prem_modulation": "attention_kv",
            "chunk_frames": 240 if mode == "offline" else 2,
            "time_constrained": False,
        })
        model_path = signature.get("model", {}).get("path", "")
    mismatches = {
        key: (signature.get(key), value)
        for key, value in expected.items()
        if signature.get(key) != value
    }
    if not model_path.endswith(config["model_path"]):
        mismatches["model_path"] = (model_path, config["model_path"])
    if checkpoint is not None:
        actual_checkpoint = signature.get("prem_checkpoint", {}).get("path", "")
        if not actual_checkpoint.endswith(checkpoint):
            mismatches["prem_checkpoint.path"] = (actual_checkpoint, checkpoint)
    if mismatches:
        raise ValueError(
            f"{backbone}/{method}/{mode} evaluation signature mismatch: {mismatches}"
        )


def read_result(
    path: Path,
    expected_count: int,
    backbone: str,
    method: str,
    mode: str,
    checkpoint: str | None,
) -> tuple[int, int]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or len(payload) != expected_count:
        count = len(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"{path}: got {count}, expected {expected_count} records")
    records = list(payload.values())
    signatures = {record.get("eval_signature") for record in records}
    if len(signatures) != 1:
        raise ValueError(f"{path}: found {len(signatures)} evaluation signatures")
    validate_signature(parse_signature(records[0]), backbone, method, mode, checkpoint)
    correct = sum(record.get("acc") == "yes" for record in records)
    csv_path = path.with_name("result.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing Table 5 result summary: {csv_path}")
    with csv_path.open(newline="") as handle:
        summary = next(csv.DictReader(handle))
    counts = (int(summary["count"]), int(summary["expected"]))
    if counts != (expected_count, expected_count):
        raise ValueError(f"{csv_path}: count/expected={counts}, required={expected_count}")
    invalid = int(summary["invalid"])
    if invalid:
        raise ValueError(f"{csv_path}: invalid={invalid}")
    return correct, len(records)


def summarize_method(
    output_root: Path,
    mode: str,
    backbone: str,
    method: str,
    seed: int | None,
    checkpoint: str | None,
) -> dict[str, Any]:
    root = row_root(output_root, mode, backbone, method, seed)
    evaluation = evaluation_name(backbone, method, mode)
    dataset_scores: dict[str, float] = {}
    total_count = 0
    for dataset, expected_count in DATASETS:
        output_dataset = result_dataset_name(dataset, mode)
        path = root / evaluation / output_dataset / "result.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Table 5 result: {path}")
        correct, count = read_result(
            path, expected_count, backbone, method, mode, checkpoint
        )
        dataset_scores[dataset] = correct * 100.0 / count
        total_count += count
    return {
        "dataset_scores": dataset_scores,
        "avg6": sum(dataset_scores.values()) / len(dataset_scores),
        "sample_count": total_count,
    }


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_backbone(
    output_root: Path,
    mode: str,
    backbone: str,
    seeds: list[int],
    overrides: dict[tuple[str, int], str],
) -> dict[str, Any]:
    base = summarize_method(output_root, mode, backbone, "base", None, None)
    checkpoints = {
        seed: checkpoint_path(output_root, backbone, seed, overrides)
        for seed in seeds
    }
    checkpoint_metadata = {
        str(seed): validate_checkpoint(path, backbone, seed)
        for seed, path in checkpoints.items()
    }
    prem_by_seed = {
        seed: summarize_method(
            output_root, mode, backbone, "prem", seed, checkpoints[seed]
        )
        for seed in seeds
    }
    base_values = [base["avg6"] for _ in seeds]
    prem_values = [prem_by_seed[seed]["avg6"] for seed in seeds]
    delta_values = [prem - base_value for prem, base_value in zip(prem_values, base_values)]
    return {
        "backbone": backbone,
        "label": BACKBONES[backbone]["label"],
        "base": base,
        "prem_by_seed": {str(seed): prem_by_seed[seed] for seed in seeds},
        "checkpoint_by_seed": checkpoint_metadata,
        "base_avg6_by_seed": dict(zip(map(str, seeds), base_values)),
        "prem_avg6_by_seed": dict(zip(map(str, seeds), prem_values)),
        "delta_avg6_by_seed": dict(zip(map(str, seeds), delta_values)),
        "base_mean": statistics.mean(base_values),
        "base_std": sample_std(base_values),
        "prem_mean": statistics.mean(prem_values),
        "prem_std": sample_std(prem_values),
        "delta_mean": statistics.mean(delta_values),
        "delta_std": sample_std(delta_values),
    }


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def format_seed_values(
    values: dict[str, float], seeds: list[int], suffix: str = "%"
) -> list[str]:
    return [f"{values[str(seed)]:.2f}{suffix}" for seed in seeds]


def write_outputs(
    output_root: Path,
    mode: str,
    seeds: list[int],
    summaries: list[dict[str, Any]],
) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "mode": mode,
            "visual_budget": 16,
            "memory": "[4,128,128]",
            "fps": 1.0,
            "max_pixels": 200704,
            "seeds": seeds,
            "standard_deviation": "sample standard deviation (n-1)",
            "base_reuse": "Base is deterministic and is evaluated once per backbone, then paired with each PReM seed",
            "fixed_eval_alpha": {
                config["label"]: config["eval_alpha"] for config in BACKBONES.values()
            },
            "datasets": [name for name, _ in DATASETS],
        },
        "backbones": summaries,
    }
    result_root = output_root / mode
    result_root.mkdir(parents=True, exist_ok=True)
    atomic_write(result_root / "summary.json", json.dumps(manifest, indent=2) + "\n")

    csv_path = result_root / "summary.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    fields = ["backbone", "method", *[f"seed_{seed}" for seed in seeds], "mean", "sample_std"]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            for method, values, mean, std in (
                ("Base", summary["base_avg6_by_seed"], summary["base_mean"], summary["base_std"]),
                ("PReM", summary["prem_avg6_by_seed"], summary["prem_mean"], summary["prem_std"]),
                ("PReM-Base delta", summary["delta_avg6_by_seed"], summary["delta_mean"], summary["delta_std"]),
            ):
                writer.writerow({
                    "backbone": summary["label"],
                    "method": method,
                    **{f"seed_{seed}": values[str(seed)] for seed in seeds},
                    "mean": mean,
                    "sample_std": std,
                })
    os.replace(temporary, csv_path)

    seed_headers = " | ".join(f"Seed {seed}" for seed in seeds)
    alignment = " | ".join("---:" for _ in seeds)
    lines = [
        f"Protocol: `{mode}`, B=16, Avg-6. Std. dev. uses n-1.",
        "",
        f"| Backbone | Method | Visual budget | Memory | {seed_headers} | Mean | Std. dev. |",
        f"|---|---|---|---:|{alignment}|---:|---:|",
    ]
    for summary in summaries:
        for method, memory, values, mean, std in (
            ("Base", "M=0", summary["base_avg6_by_seed"], summary["base_mean"], summary["base_std"]),
            ("PReM", "M", summary["prem_avg6_by_seed"], summary["prem_mean"], summary["prem_std"]),
        ):
            seed_cells = " | ".join(format_seed_values(values, seeds))
            lines.append(
                f"| {summary['label']} | {method} | B=16 | {memory} | {seed_cells} | "
                f"{mean:.2f}% | {std:.2f} |"
            )
    lines.extend([
        "",
        f"| Backbone | Metric | {seed_headers} | Mean | Std. dev. |",
        f"|---|---|{alignment}|---:|---:|",
    ])
    for summary in summaries:
        seed_cells = " | ".join(
            format_seed_values(summary["delta_avg6_by_seed"], seeds, "")
        )
        lines.append(
            f"| {summary['label']} | PReM - Base (pp) | {seed_cells} | "
            f"{summary['delta_mean']:.2f} | {summary['delta_std']:.2f} |"
        )
    atomic_write(result_root / "table5.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/table5_stability")
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--seeds", default="13 17 23")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="BACKBONE=SEED=PATH")
    parser.add_argument("--validate-checkpoint", nargs=3, metavar=("PATH", "BACKBONE", "SEED"))
    args = parser.parse_args()

    if args.validate_checkpoint:
        path, backbone, seed_text = args.validate_checkpoint
        metadata = validate_checkpoint(path, backbone, int(seed_text))
        print(json.dumps(metadata, indent=2))
        return

    seeds = parse_seeds(args.seeds)
    overrides = parse_checkpoint_overrides(args.checkpoint)
    output_root = Path(args.output_root)
    summaries = [
        summarize_backbone(output_root, args.mode, backbone, seeds, overrides)
        for backbone in BACKBONES
    ]
    write_outputs(output_root, args.mode, seeds, summaries)
    print(f"[summary] mode={args.mode} backbones={len(summaries)} seeds={seeds} output={output_root / args.mode}")


if __name__ == "__main__":
    main()
