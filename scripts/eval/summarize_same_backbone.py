#!/usr/bin/env python
"""Validate and summarize the Qwen2.5-VL same-backbone comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

METHODS = (
    ("base", "Base (Frozen, B=16)"),
    ("lora", "Base + LoRA (r=20)"),
    ("token_readout", "Base + Token-Readout Memory"),
    ("infinipot_v", "Base + InfiniPot-V-style (TaR + VaN)"),
    ("prem", "Base + PReM"),
)
INFINIPOT_V_PUBLIC_REVISION = "81a4dbe2e74660a8148cc2a6cee1148a9a268479"


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
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def single_value(records: list[dict], key: str, context: str):
    values = {record.get(key) for record in records}
    if len(values) != 1:
        raise ValueError(f"{context}: inconsistent {key}: {values}")
    return next(iter(values))


def validate_signature(record: dict, method: str, dataset: str) -> dict:
    try:
        signature = json.loads(record["eval_signature"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{method}/{dataset}: invalid eval_signature") from exc
    expected = {
        "protocol": "same_backbone_qwen25_online_v3",
        "method": method,
        "dataset": dataset,
        "max_frames": 240,
        "buffer_frames": 16,
        "fps": 1.0,
        "max_pixels": 200704,
        "chunk_frames": 2,
        "profile_efficiency": True,
        "single_step_constrained_decoding": True,
        "precision": "bfloat16",
        "attention_backend": "eager",
    }
    mismatch = {
        key: (signature.get(key), value)
        for key, value in expected.items()
        if signature.get(key) != value
    }
    if mismatch:
        raise ValueError(f"{method}/{dataset}: protocol mismatch: {mismatch}")
    if method == "infinipot_v" and (
        signature.get("infinipot_block_units") != 32
        or signature.get("infinipot_keep_units") != 24
        or signature.get("infinipot_tar_ratio") != 0.5
        or signature.get("infinipot_query_ratio") != 0.25
        or signature.get("infinipot_public_revision") != INFINIPOT_V_PUBLIC_REVISION
    ):
        raise ValueError(
            f"{method}/{dataset}: unexpected InfiniPot-V recipe or source revision"
        )
    return signature


def read_dataset(root: Path, method: str, dataset: str, expected: int) -> dict[str, Any]:
    path = root / "eval" / method / dataset / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing result: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or len(payload) != expected:
        actual = len(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"{path}: expected {expected} rows, got {actual}")
    records = list(payload.values())
    if len({str(record["id"]) for record in records}) != expected:
        raise ValueError(f"{path}: duplicate sample ids")
    signatures = {record.get("eval_signature") for record in records}
    if len(signatures) != 1:
        raise ValueError(f"{path}: expected one evaluator signature, got {len(signatures)}")
    signature = validate_signature(records[0], method, dataset)
    if any(record.get("comparison_method") != method for record in records):
        raise ValueError(f"{path}: mixed comparison methods")

    required = (
        "decoder_prompt_tokens",
        "auxiliary_prompt_tokens",
        "stream_update_seconds",
        "answer_seconds",
        "memory_state_bytes",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
    )
    missing = [key for key in required if any(key not in record for record in records)]
    if missing:
        raise ValueError(f"{path}: missing profiled fields {missing}")
    ingestion = [
        float(record["stream_update_seconds"])
        for record in records
        if not bool(record.get("state_reused"))
    ]
    if not ingestion:
        raise ValueError(f"{path}: no first-question ingestion measurements")
    correct = sum(record.get("acc") == "yes" for record in records)
    return {
        "dataset": dataset,
        "count": expected,
        "correct": correct,
        "accuracy": correct * 100.0 / expected,
        "trainable_parameters": int(single_value(records, "trainable_parameters", str(path))),
        "decoder_prompt_tokens": [float(record["decoder_prompt_tokens"]) for record in records],
        "auxiliary_prompt_tokens": [float(record["auxiliary_prompt_tokens"]) for record in records],
        "ingestion_seconds": ingestion,
        "answer_seconds": [float(record["answer_seconds"]) for record in records],
        "peak_allocated_bytes": max(
            int(record["peak_gpu_memory_allocated_bytes"]) for record in records
        ),
        "peak_reserved_bytes": max(
            int(record["peak_gpu_memory_reserved_bytes"]) for record in records
        ),
        "state_bytes": max(int(record.get("state_bytes", 0)) for record in records),
        "memory_state_bytes": max(
            int(record.get("memory_state_bytes", 0)) for record in records
        ),
        "ingested_visual_tokens": [
            float(record.get("ingested_visual_tokens", 0))
            for record in records
            if not bool(record.get("state_reused"))
        ],
        "signature": signature,
    }


def summarize_method(root: Path, method: str, label: str) -> dict[str, Any]:
    datasets = [
        read_dataset(root, method, dataset, expected)
        for dataset, expected in DATASETS
    ]
    parameter_counts = {row["trainable_parameters"] for row in datasets}
    if len(parameter_counts) != 1:
        raise ValueError(f"{method}: trainable parameter count differs across datasets")
    prompts = [value for row in datasets for value in row["decoder_prompt_tokens"]]
    auxiliary = [value for row in datasets for value in row["auxiliary_prompt_tokens"]]
    ingestion = [value for row in datasets for value in row["ingestion_seconds"]]
    answer = [value for row in datasets for value in row["answer_seconds"]]
    ingested_visual = [value for row in datasets for value in row["ingested_visual_tokens"]]
    total_count = sum(row["count"] for row in datasets)
    total_correct = sum(row["correct"] for row in datasets)
    dataset_summaries = [
        {
            "dataset": row["dataset"],
            "count": row["count"],
            "correct": row["correct"],
            "accuracy": row["accuracy"],
            "mean_decoder_prompt_tokens": statistics.fmean(row["decoder_prompt_tokens"]),
            "mean_auxiliary_prompt_tokens": statistics.fmean(row["auxiliary_prompt_tokens"]),
            "mean_ingestion_seconds_per_video": statistics.fmean(row["ingestion_seconds"]),
            "mean_answer_seconds_per_query": statistics.fmean(row["answer_seconds"]),
            "peak_allocated_bytes": row["peak_allocated_bytes"],
            "peak_reserved_bytes": row["peak_reserved_bytes"],
            "signature": row["signature"],
        }
        for row in datasets
    ]
    return {
        "key": method,
        "method": label,
        "trainable_parameters": next(iter(parameter_counts)),
        "macro_average_accuracy": statistics.fmean(row["accuracy"] for row in datasets),
        "micro_average_accuracy": total_correct * 100.0 / total_count,
        "correct": total_correct,
        "count": total_count,
        "mean_decoder_prompt_tokens": statistics.fmean(prompts),
        "p95_decoder_prompt_tokens": percentile(prompts, 0.95),
        "mean_auxiliary_prompt_tokens": statistics.fmean(auxiliary),
        "peak_gpu_memory_allocated_bytes": max(row["peak_allocated_bytes"] for row in datasets),
        "peak_gpu_memory_reserved_bytes": max(row["peak_reserved_bytes"] for row in datasets),
        "mean_ingestion_seconds_per_video": statistics.fmean(ingestion),
        "p95_ingestion_seconds_per_video": percentile(ingestion, 0.95),
        "mean_answer_seconds_per_query": statistics.fmean(answer),
        "p95_answer_seconds_per_query": percentile(answer, 0.95),
        "max_state_bytes": max(row["state_bytes"] for row in datasets),
        "max_memory_state_bytes": max(row["memory_state_bytes"] for row in datasets),
        "mean_ingested_visual_tokens": statistics.fmean(ingested_visual),
        "runtime_comparable": all(
            dataset["signature"].get("runtime_comparable", True)
            for dataset in dataset_summaries
        ),
        "runtime_provenance": sorted(
            {
                dataset["signature"].get("runtime_provenance", "same_backbone_4_worker")
                for dataset in dataset_summaries
            }
        ),
        "datasets": dataset_summaries,
    }


def audit_parameter_matching(rows: list[dict[str, Any]]) -> None:
    indexed = {row["key"]: row for row in rows}
    prem = indexed["prem"]["trainable_parameters"]
    token = indexed["token_readout"]["trainable_parameters"]
    lora = indexed["lora"]["trainable_parameters"]
    if prem <= 0:
        raise ValueError("PReM trainable parameter count must be positive")
    if token != prem:
        raise ValueError(f"Token-Readout ({token:,}) is not parameter-identical to PReM ({prem:,})")
    relative_gap = abs(lora - prem) / prem
    if relative_gap > 0.05:
        raise ValueError(
            f"LoRA ({lora:,}) differs from PReM ({prem:,}) by {relative_gap:.2%}, above 5%"
        )
    if indexed["base"]["trainable_parameters"] != 0:
        raise ValueError("Frozen Base unexpectedly has trainable parameters")
    if indexed["infinipot_v"]["trainable_parameters"] != 0:
        raise ValueError("InfiniPot-V-style unexpectedly has trainable parameters")


def audit_runtime_matching(rows: list[dict[str, Any]]) -> None:
    hardware = set()
    software = set()
    models = set()
    comparable_rows = [row for row in rows if row["runtime_comparable"]]
    noncomparable_rows = [row for row in rows if not row["runtime_comparable"]]
    if {row["key"] for row in noncomparable_rows} - {"prem"}:
        raise ValueError("Only an explicitly imported PReM row may use legacy runtime")
    for row in comparable_rows:
        for dataset in row["datasets"]:
            signature = dataset["signature"]
            hardware.add(json.dumps(signature.get("hardware"), sort_keys=True))
            software.add(json.dumps(signature.get("software"), sort_keys=True))
            models.add(json.dumps(signature.get("model"), sort_keys=True))
    if len(hardware) != 1:
        raise ValueError("Completed rows were not profiled on identical GPU hardware")
    if len(software) != 1:
        raise ValueError("Completed rows used different PyTorch/Transformers/CUDA environments")
    if len(models) != 1:
        raise ValueError("Completed rows used different backbone artifacts")
    for row in noncomparable_rows:
        if row["runtime_provenance"] != ["legacy_table4_efficiency_8_worker"]:
            raise ValueError("Imported PReM row has unrecognized runtime provenance")


def audit_shared_decoder_context(root: Path) -> None:
    shared_methods = ("base", "lora", "token_readout", "prem")
    for dataset, expected in DATASETS:
        by_method = {}
        for method in shared_methods:
            path = root / "eval" / method / dataset / "result.json"
            with path.open() as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict) or len(payload) != expected:
                raise ValueError(f"{path}: cannot audit shared decoder context")
            records = {str(row["id"]): row for row in payload.values()}
            if len(records) != expected:
                raise ValueError(f"{path}: duplicate ids during decoder-context audit")
            policies = {row.get("visual_buffer_policy") for row in records.values()}
            frame_counts = {row.get("visual_buffer_frames") for row in records.values()}
            if policies != {"deterministic_reservoir"}:
                raise ValueError(f"{path}: unexpected visual-buffer policies {policies}")
            if not frame_counts.issubset(set(range(1, 17))) or 16 not in frame_counts:
                raise ValueError(f"{path}: unexpected visual-buffer frame counts {frame_counts}")
            by_method[method] = records

        base_ids = set(by_method["base"])
        for method in shared_methods[1:]:
            if set(by_method[method]) != base_ids:
                raise ValueError(f"{method}/{dataset}: sample ids differ from Base")
        for sample_id in base_ids:
            base = by_method["base"][sample_id]
            base_tokens = int(base["decoder_prompt_tokens"])
            base_frames = int(base["visual_buffer_frames"])
            base_bytes = int(base["visual_buffer_bytes"])
            for method in ("lora", "prem"):
                row = by_method[method][sample_id]
                if (
                    int(row["decoder_prompt_tokens"]) != base_tokens
                    or int(row["visual_buffer_frames"]) != base_frames
                    or int(row["visual_buffer_bytes"]) != base_bytes
                ):
                    raise ValueError(
                        f"{method}/{dataset}/{sample_id}: decoder context differs from Base"
                    )
            token = by_method["token_readout"][sample_id]
            if (
                int(token["decoder_prompt_tokens"]) != base_tokens + 10
                or int(token["auxiliary_prompt_tokens"]) != 10
                or int(token["visual_buffer_frames"]) != base_frames
                or int(token["visual_buffer_bytes"]) != base_bytes
            ):
                raise ValueError(
                    f"token_readout/{dataset}/{sample_id}: expected the Base context plus 10 tokens"
                )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    accuracy_fields = tuple(f"accuracy_{dataset}" for dataset, _ in DATASETS)
    fields = (
        "key",
        "method",
        "trainable_parameters",
        *accuracy_fields,
        "macro_average_accuracy",
        "micro_average_accuracy",
        "mean_decoder_prompt_tokens",
        "mean_auxiliary_prompt_tokens",
        "peak_gpu_memory_allocated_bytes",
        "peak_gpu_memory_reserved_bytes",
        "mean_ingestion_seconds_per_video",
        "mean_answer_seconds_per_query",
        "max_memory_state_bytes",
        "runtime_comparable",
        "runtime_provenance",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in fields if key in row}
            by_dataset = {item["dataset"]: item for item in row["datasets"]}
            record.update(
                {
                    f"accuracy_{dataset}": by_dataset[dataset]["accuracy"]
                    for dataset, _ in DATASETS
                }
            )
            writer.writerow(record)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "<!-- Generated by scripts/eval/summarize_same_backbone.py. Do not hand-edit. -->",
        "### Accuracy (%)",
        "",
        "| Method | Trainable | LVB | MLVU | Video-MME | EgoSchema | MVBench | LVBench | Avg. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        accuracy = {item["dataset"]: item["accuracy"] for item in row["datasets"]}
        lines.append(
            "| {method} | {params:,} | {lvb:.2f} | {mlvu:.2f} | {video:.2f} | "
            "{ego:.2f} | {mv:.2f} | {lv:.2f} | {avg:.2f} |".format(
                method=row["method"],
                params=row["trainable_parameters"],
                lvb=accuracy["longvideobench"],
                mlvu=accuracy["mlvu"],
                video=accuracy["videomme"],
                ego=accuracy["egoschema"],
                mv=accuracy["mvbench"],
                lv=accuracy["lvbench"],
                avg=row["macro_average_accuracy"],
            )
        )
    lines += [
        "",
        "### Resource accounting",
        "",
        "| Method | Prompt Tokens | Added Tokens | Persistent State (MiB) | Peak VRAM (GiB) | Ingestion (s/video) | Answer (s/query) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        runtime_mark = "*" if not row["runtime_comparable"] else ""
        lines.append(
            "| {method} | {prompt:.1f} | {aux:.1f} | "
            "{state:.3f} | {vram:.2f} | {ingest} | {answer} |".format(
                method=row["method"],
                prompt=row["mean_decoder_prompt_tokens"],
                aux=row["mean_auxiliary_prompt_tokens"],
                state=row["max_memory_state_bytes"] / 2**20,
                vram=row["peak_gpu_memory_allocated_bytes"] / 2**30,
                ingest=f'{row["mean_ingestion_seconds_per_video"]:.3f}{runtime_mark}',
                answer=f'{row["mean_answer_seconds_per_query"]:.3f}{runtime_mark}',
            )
        )
    lines += [
        "",
        "Avg. is the unweighted mean over six public benchmarks. Ingestion latency is "
        "averaged once per unique video; answer latency and prompt tokens are averaged per question.",
        "* PReM accuracy and decoder context are validated against the same samples and buffer "
        "policy, but its runtime is reused from the completed legacy 8-worker efficiency run; "
        "the other rows use the registered 4-worker run.",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    display_names = {
        "base": "Base (Frozen, $B{=}16$)",
        "lora": "Base + LoRA ($r{=}20$)",
        "token_readout": "Base + Token-Readout Memory ($K{=}1$)",
        "infinipot_v": "Base + InfiniPot-V-style~\\citep{kim2025infinipotv}",
        "prem": "Base + \\ours{} ($K{=}1$)",
    }
    lines = [
        "% Generated by scripts/eval/summarize_same_backbone.py. Do not hand-edit.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Controlled same-backbone mechanism comparison on Qwen2.5-VL-3B: "
        "uniform visual sampling, parameter-matched tuning, token-based memory, "
        "KV-cache compression, and \\ours{}. "
        "Panel (a) reports all six benchmark accuracies and their macro average. "
        "Panel (b) reports resource use; method-added state excludes "
        "the frozen backbone and shared raw frame buffer; latency is split into "
        "one-time ingestion per video and answer time per query. The parameter-matched "
        "Token-Readout and \\ours{} controls use $K{=}1$. The dagger marks PReM runtime "
        "reused from the prior complete 8-worker efficiency run; all other runtime rows use "
        "the registered 4-worker layout.}",
        "\\label{tab:same-backbone}",
        "\\textbf{(a) Accuracy (\\%)}\\\\[-2pt]",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & Trainable & LVB & MLVU & Video-MME & EgoSchema & MVBench & LVBench & Avg. \\\\",
        "\\midrule",
    ]
    for row in rows:
        parameters = (
            "0"
            if row["trainable_parameters"] == 0
            else f'{row["trainable_parameters"] / 1e6:.3f}M'
        )
        accuracy = {item["dataset"]: item["accuracy"] for item in row["datasets"]}
        lines.append(
            "{name} & {params} & {lvb:.2f} & {mlvu:.2f} & {video:.2f} & "
            "{ego:.2f} & {mv:.2f} & {lv:.2f} & {avg:.2f} \\\\".format(
                name=display_names[row["key"]],
                params=parameters,
                lvb=accuracy["longvideobench"],
                mlvu=accuracy["mlvu"],
                video=accuracy["videomme"],
                ego=accuracy["egoschema"],
                mv=accuracy["mvbench"],
                lv=accuracy["lvbench"],
                avg=row["macro_average_accuracy"],
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\\\[3pt]",
        "\\textbf{(b) Resource accounting}\\\\[-2pt]",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Prompt tok. & Added tok. & State & Peak VRAM & Ingestion & Answer \\\\",
        "\\midrule",
    ]
    for row in rows:
        runtime_mark = "\\textsuperscript{\\dag}" if not row["runtime_comparable"] else ""
        lines.append(
            "{name} & {prompt:.1f} & {aux:.1f} & {state:.3f} MiB & "
            "{vram:.2f} GiB{mark} & {ingest:.3f} s{mark} & {answer:.3f} s{mark} \\\\".format(
                name=display_names[row["key"]],
                prompt=row["mean_decoder_prompt_tokens"],
                aux=row["mean_auxiliary_prompt_tokens"],
                state=row["max_memory_state_bytes"] / 2**20,
                vram=row["peak_gpu_memory_allocated_bytes"] / 2**30,
                ingest=row["mean_ingestion_seconds_per_video"],
                answer=row["mean_answer_seconds_per_query"],
                mark=runtime_mark,
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    rows = [summarize_method(root, key, label) for key, label in METHODS]
    audit_parameter_matching(rows)
    audit_runtime_matching(rows)
    audit_shared_decoder_context(root)
    summary = {
        "schema": "same_backbone_qwen25_summary_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "backbone": "Qwen2.5-VL-3B-Instruct",
            "writer_frames_T": 240,
            "decoder_buffer_frames_B": 16,
            "fps": 1.0,
            "max_pixels": 200704,
            "datasets": [name for name, _ in DATASETS],
            "accuracy_average": "macro_across_datasets",
            "ingestion_latency_unit": "first_question_per_unique_video",
            "answer_latency_unit": "per_question",
        },
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    write_csv(root / "summary.csv", rows)
    write_markdown(root / "summary.md", rows)
    write_latex(root / "summary.tex", rows)
    print(f"[done] {root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
