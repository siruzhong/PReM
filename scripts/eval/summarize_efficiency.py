#!/usr/bin/env python
"""Validate and summarize controlled Table 4 efficiency runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

ROWS = (
    {"key": "base_B16", "method": "Qwen2.5 base", "budget": 16, "evaluation": "qwen2_5vl_3b"},
    {"key": "prem_B16", "method": "Qwen2.5 + PReM", "budget": 16, "evaluation": "prem_attention"},
    {"key": "base_B32", "method": "Qwen2.5 base", "budget": 32, "evaluation": "qwen2_5vl_3b"},
    {"key": "prem_B32", "method": "Qwen2.5 + PReM", "budget": 32, "evaluation": "prem_attention"},
    {"key": "base_B64", "method": "Qwen2.5 base", "budget": 64, "evaluation": "qwen2_5vl_3b"},
    {"key": "prem_B64", "method": "Qwen2.5 + PReM", "budget": 64, "evaluation": "prem_attention"},
    {"key": "base_B90", "method": "Qwen2.5 base", "budget": 90, "evaluation": "qwen2_5vl_3b"},
    {"key": "prem_B90", "method": "Qwen2.5 + PReM", "budget": 90, "evaluation": "prem_attention"},
)

ONLINE_ROWS = tuple(
    {**row, "evaluation": "qwen25vl_qwen25_public" if row["key"].startswith("base_") else "prem_qwen25_public"}
    for row in ROWS
)

SUPPORTED_BUDGETS = (16, 32, 64, 90)

DEFAULT_CHECKPOINTS = {
    budget: (
        f"outputs/table4_efficiency/train/"
        f"qwen25_3b_b{budget}_ns1_rg0p10_lg4_pw0p2_pt4_ta0p75/prem.pt"
    )
    for budget in SUPPORTED_BUDGETS
}

EXPECTED_RECIPE = {
    "complete": True,
    "num_slots": 1,
    "mem_dim": 128,
    "max_memory_tokens": 128,
    "prem_modulation": "attention_kv",
    "router_gamma": 0.10,
    "pred_weight": 0.2,
    "pred_tokens": 4,
    "prem_layer_groups": 4,
    "alpha": 0.75,
    "max_frames": 240,
    "writer_mode": "temporal_mean_per_step",
}


def parse_budget_path_overrides(values: list[str], flag: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for value in values:
        try:
            budget_text, path = value.split("=", 1)
            budget = int(budget_text)
        except ValueError as exc:
            raise ValueError(f"Invalid {flag} {value!r}; expected BUDGET=PATH") from exc
        if budget not in SUPPORTED_BUDGETS or not path:
            raise ValueError(f"Unsupported {flag} budget: {budget}")
        mapping[budget] = path
    return mapping


def parse_checkpoint_overrides(values: list[str]) -> dict[int, str]:
    if values:
        return parse_budget_path_overrides(values, "--prem-checkpoint")
    return dict(DEFAULT_CHECKPOINTS)


def metadata_mismatch(checkpoint: dict[str, Any], expected: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    mismatches = {}
    for key, value in expected.items():
        actual = checkpoint.get(key)
        if isinstance(value, float) and isinstance(actual, (int, float)):
            if abs(float(actual) - float(value)) > 1e-8:
                mismatches[key] = (actual, value)
            continue
        if actual != value:
            mismatches[key] = (actual, value)
    return mismatches


def validate_checkpoints(checkpoints: dict[int, str]) -> dict[int, dict[str, Any]]:
    import torch

    stats: dict[int, dict[str, Any]] = {}
    parameter_counts: dict[int, int] = {}
    for budget, path_text in checkpoints.items():
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"PReM B={budget} checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        expected = dict(EXPECTED_RECIPE)
        expected["visual_buffer_frames"] = budget
        mismatches = metadata_mismatch(checkpoint, expected)
        if mismatches:
            raise ValueError(f"PReM B={budget} checkpoint metadata mismatch: {mismatches}")
        state = checkpoint.get("prem_state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"PReM B={budget} checkpoint has no prem_state_dict")
        parameter_counts[budget] = sum(int(tensor.numel()) for tensor in state.values())
        stats[budget] = {
            "trainable_params": parameter_counts[budget],
            "train_wall_seconds": checkpoint.get("train_wall_seconds"),
            "train_peak_gpu_memory_allocated_bytes": checkpoint.get(
                "train_peak_gpu_memory_allocated_bytes"
            ),
        }
    if len(parameter_counts) > 1 and len(set(parameter_counts.values())) != 1:
        raise ValueError(f"PReM trainable parameter counts differ by budget: {parameter_counts}")
    return stats


def load_train_metrics(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def read_signature(record: dict[str, Any]) -> dict[str, Any]:
    signature = record.get("eval_signature")
    if not isinstance(signature, str):
        raise ValueError("result row has no serialized eval_signature")
    parsed = json.loads(signature)
    if not isinstance(parsed, dict):
        raise ValueError("eval_signature must decode to an object")
    return parsed


def dataset_dir_name(name: str) -> str:
    return "videommewo" if name == "videomme" else name


def validate_signature(signature: dict[str, Any], row: dict[str, Any], checkpoint: str | None, protocol: str) -> None:
    expected = {
        "max_frames": row["budget"] if protocol == "offline" and row["key"].startswith("base_") else 240,
        "fps": 1.0,
        "max_pixels": 200704,
        "profile_efficiency": True,
    }
    if protocol == "online":
        expected.update({"protocol": "qwen2_5_vl_online_reservoir_endtime" if row["key"].startswith("base_") else "unified_bounded_visual_memory"})
        if not row["key"].startswith("base_"):
            expected.update({"chunk_frames": 2, "initial_chunk_frames": 0})
    else:
        if row["key"].startswith("base_"):
            expected.update({"protocol": "full_video"})
        else:
            expected.update({"protocol": "unified_bounded_visual_memory", "chunk_frames": 240, "visual_buffer_frames": row["budget"], "prem_modulation": "attention_kv"})
    if not row["key"].startswith("base_"):
        expected.update({"visual_buffer_frames": row["budget"], "prem_modulation": "attention_kv"})
    mismatches = {
        key: (signature.get(key), value)
        for key, value in expected.items()
        if signature.get(key) != value
    }
    if checkpoint is not None:
        actual_path = signature.get("prem_checkpoint", {}).get("path", "")
        if not actual_path.endswith(checkpoint):
            mismatches["prem_checkpoint.path"] = (actual_path, checkpoint)
    if mismatches:
        raise ValueError(f"{row['key']} evaluation signature mismatch: {mismatches}")


def read_dataset(result_dir: Path, row: dict[str, Any], expected: int, checkpoint: str | None, protocol: str) -> dict[str, Any]:
    result_path = result_dir / "result.json"
    csv_path = result_dir / "result.csv"
    with result_path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{result_path} must contain an object keyed by sample id")
    records = list(payload.values())
    if len(records) != expected:
        raise ValueError(f"{result_path}: got {len(records)} rows, expected {expected}")
    signatures = {record.get("eval_signature") for record in records}
    if len(signatures) != 1:
        raise ValueError(f"{result_path}: found {len(signatures)} evaluation signatures")
    validate_signature(read_signature(records[0]), row, checkpoint, protocol)

    with csv_path.open(newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    invalid = int(csv_row["invalid"])
    if int(csv_row["count"]) != expected or int(csv_row["expected"]) != expected:
        raise ValueError(f"{csv_path}: incomplete result counts")

    update = [float(record["stream_update_seconds"]) for record in records]
    answer = [float(record["answer_seconds"]) for record in records]
    allocated = [int(record["peak_gpu_memory_allocated_bytes"]) for record in records]
    reserved = [int(record["peak_gpu_memory_reserved_bytes"]) for record in records]
    correct = sum(record.get("acc") == "yes" for record in records)
    return {
        "accuracy": correct * 100.0 / expected,
        "correct": correct,
        "count": expected,
        "invalid": invalid,
        "update_seconds": update,
        "answer_seconds": answer,
        "peak_allocated_bytes": max(allocated),
        "peak_reserved_bytes": max(reserved),
        "state_reused": sum(bool(record.get("state_reused")) for record in records),
    }


def read_hardware(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-row hardware manifest: {path}")
    hardware = []
    with path.open(newline="") as handle:
        for fields in csv.reader(handle):
            if len(fields) != 5:
                raise ValueError(f"Invalid hardware row in {path}: {fields}")
            index, name, uuid, driver, memory_total = (field.strip() for field in fields)
            hardware.append({
                "index": index,
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "memory_total": memory_total,
            })
    if not hardware:
        raise ValueError(f"Empty hardware manifest: {path}")
    return hardware


def comparable_hardware(hardware: list[dict[str, str]]) -> list[tuple[str, str, str]]:
    return sorted(
        (gpu["name"], gpu["driver_version"], gpu["memory_total"])
        for gpu in hardware
    )


def summarize_row(
    output_root: Path,
    row: dict[str, Any],
    checkpoints: dict[int, str],
    checkpoint_stats: dict[int, dict[str, Any]],
    train_metrics: dict[int, dict[str, Any]],
    protocol: str,
) -> dict[str, Any]:
    checkpoint = None if row["key"].startswith("base_") else checkpoints.get(row["budget"])
    base = output_root / row["key"] / row["evaluation"]
    missing = [
        name
        for name, _ in DATASETS
        if not (base / (dataset_dir_name(name) if protocol == "offline" else name) / "result.json").is_file()
    ]
    stats = checkpoint_stats.get(row["budget"], {}) if checkpoint else {}
    sidecar = train_metrics.get(row["budget"], {}) if checkpoint else {}
    memory_state = "M=0" if checkpoint is None else "[1,128,128]"
    trainable_params = 0 if checkpoint is None else int(stats.get("trainable_params") or 0)
    train_wall = None
    train_peak = None
    if checkpoint is not None:
        sidecar_wall = sidecar.get("train_wall_seconds")
        ckpt_wall = stats.get("train_wall_seconds")
        train_wall = sidecar_wall if sidecar_wall is not None else ckpt_wall
        train_peak = stats.get("train_peak_gpu_memory_allocated_bytes")
    summary: dict[str, Any] = {
        **row,
        "memory_state": memory_state,
        "trainable_params": trainable_params,
        "train_wall_seconds": train_wall,
        "train_peak_gpu_memory_allocated_bytes": train_peak,
        "checkpoint": checkpoint,
        "status": "incomplete" if missing else "complete",
        "missing_datasets": missing,
    }
    if missing:
        return summary

    summary["hardware"] = read_hardware(output_root / row["key"] / "hardware.csv")
    datasets = {
        name: read_dataset(base / (dataset_dir_name(name) if protocol == "offline" else name), row, expected, checkpoint, protocol)
        for name, expected in DATASETS
    }
    updates = [value for dataset in datasets.values() for value in dataset["update_seconds"]]
    answers = [value for dataset in datasets.values() for value in dataset["answer_seconds"]]
    latencies = [update + answer for update, answer in zip(updates, answers)]
    summary.update({
        "accuracy_by_dataset": {name: value["accuracy"] for name, value in datasets.items()},
        "avg6": sum(value["accuracy"] for value in datasets.values()) / len(datasets),
        "sample_count": sum(value["count"] for value in datasets.values()),
        "invalid": sum(value["invalid"] for value in datasets.values()),
        "latency_seconds_total": sum(latencies),
        "latency_seconds_mean": sum(latencies) / len(latencies),
        "latency_seconds_p50": percentile(latencies, 0.50),
        "latency_seconds_p95": percentile(latencies, 0.95),
        "update_seconds_mean": sum(updates) / len(updates),
        "answer_seconds_mean": sum(answers) / len(answers),
        "peak_gpu_memory_allocated_bytes": max(value["peak_allocated_bytes"] for value in datasets.values()),
        "peak_gpu_memory_reserved_bytes": max(value["peak_reserved_bytes"] for value in datasets.values()),
        "state_reused_count": sum(value["state_reused"] for value in datasets.values()),
    })
    return summary


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def write_outputs(output_root: Path, summaries: list[dict[str, Any]], protocol: str) -> None:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "backbone": "Qwen2.5-VL-3B",
            "precision": "bfloat16",
            "fps": 1.0,
            "max_frames": 240,
            "max_pixels": 200704,
            "batch_size": 1,
            "protocol": protocol,
            "latency": "mean per QA of memory update plus answer",
            "ingest_latency": "stream_update_seconds; online rows update incrementally and may reuse state for repeated questions on the same video",
            "peak_gpu_memory": "inference: max torch CUDA allocator bytes across workers and samples; train: checkpoint train_peak_gpu_memory_allocated_bytes",
            "train_time": "1-epoch wall-clock on the efficiency-owned PReM checkpoint; Base has no trained adapter",
            "accuracy": "Avg-6 from the same profiled run, not copied from Table 2",
            "recipe": "ns1_rg0p10_lg4_pw0p2_pt4_ta0p75, T=240, independent of Table 2 checkpoints",
            "datasets": [name for name, _ in DATASETS],
        },
        "rows": summaries,
    }
    atomic_write(output_root / "summary.json", json.dumps(manifest, indent=2) + "\n")

    fieldnames = [
        "key", "method", "budget", "memory_state", "trainable_params", "status",
        "train_wall_seconds", "train_peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_allocated_bytes", "peak_gpu_memory_reserved_bytes",
        "latency_seconds_mean", "latency_seconds_p50", "latency_seconds_p95",
        "update_seconds_mean", "answer_seconds_mean", "avg6", "sample_count", "invalid",
        "state_reused_count",
    ]
    csv_path = output_root / "summary.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    os.replace(temporary, csv_path)

    lines = [
        "| Method | B | Trainable | Train time | Train peak GPU | Infer peak GPU | Ingest / answer | Latency mean / p95 | Avg-6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        if row["status"] != "complete":
            metrics = ("-", "-", "-", "-", "-", "-")
        else:
            infer_gib = row["peak_gpu_memory_allocated_bytes"] / (1024 ** 3)
            train_peak = row.get("train_peak_gpu_memory_allocated_bytes")
            train_gib = (
                f"{train_peak / (1024 ** 3):.2f} GiB"
                if isinstance(train_peak, (int, float)) and train_peak
                else "-"
            )
            train_wall = row.get("train_wall_seconds")
            train_time = (
                f"{train_wall / 3600:.2f} h"
                if isinstance(train_wall, (int, float))
                else "-"
            )
            metrics = (
                train_time,
                train_gib,
                f"{infer_gib:.2f} GiB",
                f"{row['update_seconds_mean']:.2f} / {row['answer_seconds_mean']:.2f} s",
                f"{row['latency_seconds_mean']:.2f} / {row['latency_seconds_p95']:.2f} s",
                f"{row['avg6']:.2f}%",
            )
        lines.append(
            f"| {row['method']} | {row['budget']} | {row['trainable_params']:,} | "
            f"{metrics[0]} | {metrics[1]} | {metrics[2]} | {metrics[3]} | {metrics[4]} | {metrics[5]} |"
        )
    atomic_write(output_root / "table4.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/table4_efficiency")
    parser.add_argument("--prem-checkpoint", action="append", default=[], metavar="BUDGET=PATH")
    parser.add_argument("--train-metrics", action="append", default=[], metavar="BUDGET=PATH")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--protocol", choices=("offline", "online"), default="offline")
    args = parser.parse_args()

    checkpoints = parse_checkpoint_overrides(args.prem_checkpoint) or dict(DEFAULT_CHECKPOINTS)
    present = {budget: path for budget, path in checkpoints.items() if Path(path).is_file()}
    if args.validate_only or args.require_all:
        checkpoint_stats = validate_checkpoints(checkpoints)
    else:
        checkpoint_stats = validate_checkpoints(present) if present else {}
    train_metrics = {
        budget: load_train_metrics(path)
        for budget, path in parse_budget_path_overrides(args.train_metrics, "--train-metrics").items()
    }
    print(
        "[checkpoints] "
        + ", ".join(
            f"B={budget}:{checkpoint_stats[budget]['trainable_params']:,}"
            for budget in sorted(checkpoint_stats)
        )
    )
    if args.validate_only:
        return

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = ONLINE_ROWS if args.protocol == "online" else ROWS
    summaries = [summarize_row(output_root, row, checkpoints, checkpoint_stats, train_metrics, args.protocol) for row in rows]
    incomplete = [row["key"] for row in summaries if row["status"] != "complete"]
    complete_rows = [row for row in summaries if row["status"] == "complete"]
    hardware_sets = {tuple(comparable_hardware(row["hardware"])) for row in complete_rows}
    if len(hardware_sets) > 1:
        raise RuntimeError("Completed Table 4 rows were not measured on identical GPU hardware")
    if args.require_all and incomplete:
        raise RuntimeError(f"Incomplete Table 4 rows: {', '.join(incomplete)}")
    write_outputs(output_root, summaries, args.protocol)
    complete = len(summaries) - len(incomplete)
    print(f"[summary] complete={complete}/{len(summaries)} output={output_root}")


if __name__ == "__main__":
    main()
