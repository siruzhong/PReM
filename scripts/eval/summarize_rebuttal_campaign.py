#!/usr/bin/env python
"""Collect the four rebuttal experiment classes without averaging partial runs."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import socket
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.data_io import load_records
from scripts.eval.inference_mcq_vqa import get_sample_media_name, normalize_mcq_sample
from scripts.eval.run_eval_online import PUBLIC_DATASETS
from scripts.eval.prem_checkpoint import normalize_prem_checkpoint
from scripts.eval.summarize_rebuttal_memory import paired_interval, summarize

DATASETS = ("videomme", "lvbench", "longvideobench", "mlvu", "egoschema", "mvbench")
VARIANTS = ("prem", "token_readout", "temporal_mean", "ema")
LABELS = {"base": "Base", "prem": "PReM", "token_readout": "Token-Readout",
          "temporal_mean": "Temporal mean", "ema": "EMA (0.98)"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_rows(directory):
    return [json.loads(line) for path in sorted(directory.glob("[0-9]*_[0-9]*.jsonl"))
            for line in path.open()]


def ground_truth(dataset):
    rows = [normalize_mcq_sample(row) for row in load_records(PUBLIC_DATASETS[dataset]["gt_file"])]
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate ground-truth IDs: {dataset}")
    return result


def verify_identity(rows, variant, checkpoint):
    expected_writer = "prem" if variant == "token_readout" else variant
    expected_interface = "in_sequence_tokens" if variant == "token_readout" else "prefix_kv"
    checkpoint_hash = digest(checkpoint)
    if not rows or any(row["writer_variant"] != expected_writer or
                       row["read_interface"] != expected_interface or
                       row["signature"]["checkpoint_sha256"] != checkpoint_hash for row in rows):
        raise ValueError(f"Writer/readout/checkpoint identity mismatch: {variant}")


def comparable_signature(signature):
    return {key: signature[key] for key in (
        "protocol", "source_sha256", "script_sha256", "software", "model_identity",
        "fps", "max_pixels", "frame_caps", "gt_file", "video_dir", "decoder_buffer",
        "distractor_frames", "needle_samples", "raw_metadata", "gt_sha256", "raw_metadata_sha256",
    )}


def checkpoint_metrics(path, variant):
    import torch
    checkpoint = normalize_prem_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
    expected = {"complete": True, "data_protocol": "rebuttal_decontaminated_v1",
                "source_train_samples": 8953, "source_train_qa": 17849,
                "train_steps": 2232, "global_batch_size": 8, "skipped": 0,
                "num_slots": 1, "mem_dim": 128, "prem_layer_groups": 4,
                "max_frames": 240, "max_memory_tokens": 128,
                "train_manifest_sha256": "e630ec50753fcf152263415950c7ba59a97989f92f452375549237c90916d40f"}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Training contract mismatch: {variant}")
    state = checkpoint["prem_state_dict"]
    result = {**expected, "path": str(path), "sha256": digest(path),
              "writer_variant": checkpoint.get("writer_variant", "prem"),
              "baseline_method": checkpoint.get("baseline_method"),
              "alpha_init": checkpoint["alpha"], "alpha_run": 1.0,
              "alpha_learned": state["log_alpha"].exp().item(),
              "instantiated_parameters": sum(t.numel() for t in state.values()),
              "allocated_state_bytes": 4 * (128 * 128 + 1)}
    result["logical_state_bytes"] = (
        4 * (state["query_norm.weight"].numel() + 1)
        if variant in {"temporal_mean", "ema"} else result["allocated_state_bytes"])
    if variant not in {"temporal_mean", "ema"}:
        result["learned_retention_multiplier"] = state["forget_logit"].sigmoid().item()
    return result


def base_metrics(path, dataset):
    data = json.loads(path.read_text())
    gt = ground_truth(dataset)
    if set(data) != set(gt):
        raise ValueError(f"Incomplete independent Base: {dataset}")
    for sid, row in data.items():
        if int(row["answer"]) != int(gt[sid]["answer"]):
            raise ValueError(f"Independent Base annotation mismatch: {dataset}/{sid}")
        correct = row["pred"].strip().upper() == chr(ord("A") + int(gt[sid]["answer"]))
        if correct != (row["acc"] == "yes") or len(row["decoder_input_sha256"]) != 64:
            raise ValueError(f"Independent Base scoring or hash mismatch: {dataset}/{sid}")
    correct = sum(row["acc"] == "yes" for row in data.values())
    return {"questions": len(data), "correct": correct,
            "accuracy": 100 * correct / len(data), "path": str(path), "sha256": digest(path)}


def duration_metrics(dataset, base_path, prem_directory=None):
    base = json.loads(base_path.read_text())
    gt = ground_truth(dataset)
    duration = {}
    if dataset == "lvbench":
        for line in (ROOT / "data/eval_video/lvbench/raw/video_info.meta.jsonl").open():
            video = json.loads(line)
            minutes = video["video_info"]["duration_minutes"]
            if minutes < 30:
                raise ValueError("LVBench duration outside registered 30-minute lower bound")
            for qa in video["qa"]:
                duration[str(qa["uid"])] = "30–60 min" if minutes < 60 else "≥60 min"
    prem = {str(row["id"]): row for row in prediction_rows(prem_directory)} if prem_directory else {}
    if prem and (set(prem) != set(gt) or any(row["condition"] != 240 for row in prem.values())):
        raise ValueError("Duration table requires the complete 240-frame controlled run")
    groups = defaultdict(list)
    for sid, truth in gt.items():
        group = truth["duration"] if dataset == "videomme" else duration[sid]
        row = {"video_id": get_sample_media_name(truth),
               "base": {"correct": base[sid]["acc"] == "yes"}}
        if prem:
            row["prediction"] = prem[sid]["prediction"]
        groups[group].append(row)
    order = ("short", "medium", "long") if dataset == "videomme" else ("30–60 min", "≥60 min")
    metrics = []
    for group in order:
        rows = groups[group]
        item = {"duration": group, "questions": len(rows),
                "videos": len({row["video_id"] for row in rows}),
                "base_accuracy": 100 * sum(row["base"]["correct"] for row in rows) / len(rows)}
        if prem:
            item["prem_accuracy"] = 100 * sum(row["prediction"]["correct"] for row in rows) / len(rows)
            item["gain_pp"] = item["prem_accuracy"] - item["base_accuracy"]
            item["gain_ci95_pp"] = paired_interval(rows, "prediction")
        metrics.append(item)
    return metrics


def writer_input_view(rows):
    return {(str(row["id"]), row["condition"]): {
        key: row[key] for key in (
            "video_id", "answer", "sampled_frames", "actual_writes", "writer_input_sha256",
            "writer_frame_manifest_sha256", "decoder_frame_manifest_sha256",
        )
    } for row in rows}


def verify_cross_variant_natural(directories):
    reference = None
    for directory in directories:
        view = writer_input_view(prediction_rows(directory))
        if reference is not None and view != reference:
            raise ValueError(f"Writer variants used different natural stream inputs: {directory}")
        reference = view


def verify_cross_variant_needles(directories):
    reference = None
    for directory in directories:
        rows = prediction_rows(directory)
        view = {(str(row["id"]), row["condition"]): {
            "video_id": row["video_id"], "answer": row["answer"],
            "evidence_indices": row["evidence_indices"],
            "distractor_video_id": row["distractor_video_id"],
            "base_pred": row["base"]["pred"],
            "base_hash": row["base"]["decoder_input_sha256"],
            "evidence_visible_pred": row["evidence_visible_base"]["pred"],
            "evidence_visible_hash": row["evidence_visible_base"]["decoder_input_sha256"],
            "actual_writes": row["actual_writes"],
            "writer_input_sha256": row["writer_input_sha256"],
            "target_replaced_writer_sha256": row["target_replaced_writer_sha256"],
            "writer_frame_manifest_sha256": row["writer_frame_manifest_sha256"],
        } for row in rows}
        if reference is not None and view != reference:
            raise ValueError(f"Writer variants used different needle inputs or frozen Base predictions: {directory}")
        reference = view


def collect(run, base_root):
    missing = []
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "source_sha256": digest(__file__), "hostname": socket.gethostname(),
              "datasets": list(DATASETS), "table2": {}, "duration_control": {},
              "training": {}, "long_stream": None, "needle": {}, "missing": missing}
    for variant in VARIANTS:
        path = run / "train" / variant / "prem.pt"
        if path.exists():
            result["training"][variant] = checkpoint_metrics(path, variant)
        else:
            missing.append(f"train/{variant}")
    signatures, natural_directories = {}, defaultdict(list)
    for variant in ("base",) + VARIANTS:
        row = {"datasets": {}, "average_accuracy": None}
        for dataset in DATASETS:
            base_path = base_root / dataset / "result.json"
            if variant == "base":
                if base_path.exists():
                    row["datasets"][dataset] = base_metrics(base_path, dataset)
                else:
                    missing.append(f"base/{dataset}")
                continue
            directory = run / "eval" / variant / dataset
            if not (directory / "summary.json").exists() or not base_path.exists():
                missing.append(f"eval/{variant}/{dataset}")
                continue
            summary = summarize(directory, base_path)
            rows = prediction_rows(directory)
            verify_identity(rows, variant, run / "train" / variant / "prem.pt")
            signature = comparable_signature(summary["signature"])
            if dataset in signatures and signatures[dataset] != signature:
                raise ValueError(f"Cross-method evaluation protocol mismatch: {dataset}/{variant}")
            signatures[dataset] = signature
            natural_directories[dataset].append(directory)
            if summary["signature"]["frame_caps"] != [240]:
                raise ValueError("Six-benchmark table requires cap 240 only")
            metric = summary["metrics"][0]
            row["datasets"][dataset] = {
                "questions": metric["questions"], "accuracy": metric["prediction_accuracy"],
                "gain_ci95_pp": metric["prediction_gain_ci95_pp"],
                "checkpoint_sha256": summary["signature"]["checkpoint_sha256"],
                "summary_path": str(directory / "summary.json"),
                "independent_base_audit": summary["independent_base_audit"],
            }
        if set(row["datasets"]) == set(DATASETS):
            row["average_accuracy"] = sum(x["accuracy"] for x in row["datasets"].values()) / len(DATASETS)
        result["table2"][variant] = row
    for directories in natural_directories.values():
        verify_cross_variant_natural(directories)
    for dataset in ("videomme", "lvbench"):
        if dataset in result["table2"]["base"]["datasets"]:
            prem = run / "eval/prem" / dataset if dataset in result["table2"]["prem"]["datasets"] else None
            result["duration_control"][dataset] = duration_metrics(dataset, base_root / dataset / "result.json", prem)
    directory = run / "long_stream/prem/lvbench"
    if (directory / "summary.json").exists():
        summary = summarize(directory, base_root / "lvbench/result.json")
        rows = prediction_rows(directory)
        verify_identity(rows, "prem", run / "train/prem/prem.pt")
        if summary["signature"]["frame_caps"] != [240, 480, 960]:
            raise ValueError("Unexpected natural long-stream caps")
        reference_rows = prediction_rows(run / "eval/prem/lvbench")
        current_rows = [row for row in rows if row["condition"] == 240]
        if writer_input_view(reference_rows) != writer_input_view(current_rows):
            raise ValueError("Long-stream cap-240 writer inputs differ from the primary PReM evaluation")
        reference = {str(row["id"]): row["prediction"] for row in reference_rows}
        current = {str(row["id"]): row["prediction"] for row in current_rows}
        if set(reference) != set(current) or any(
            current[sid][key] != reference[sid][key]
            for sid in current for key in ("pred", "decoder_input_sha256", "correct")
        ):
            raise ValueError("Long-stream cap-240 anchor differs from the primary PReM evaluation")
        result["long_stream"] = summary
    else:
        missing.append("long_stream/prem/lvbench")
    needle_directories, needle_signature = [], None
    for variant in ("prem", "temporal_mean", "ema"):
        directory = run / "needle" / variant / "lvbench"
        if (directory / "summary.json").exists():
            summary = summarize(directory)
            verify_identity(prediction_rows(directory), variant, run / "train" / variant / "prem.pt")
            signature = summary["signature"]
            if signature["distractor_frames"] != [0, 30, 60, 120, 240, 480] or summary["questions"] != 128:
                raise ValueError("Unexpected registered needle conditions or sample count")
            current_signature = comparable_signature(signature)
            if needle_signature is not None and needle_signature != current_signature:
                raise ValueError("Needle writer variants used different evaluation protocols")
            needle_signature = current_signature
            result["needle"][variant] = summary
            needle_directories.append(directory)
        else:
            missing.append(f"needle/{variant}/lvbench")
    verify_cross_variant_needles(needle_directories)
    result["complete"] = not missing
    result["all_available_inputs_verified"] = True
    return result


def markdown(result):
    lines = ["# PReM 补充实验数值", "",
             "状态：完整。" if result["complete"] else "状态：未完成；— 表示待完成，不计算部分数据集均值。", "",
             "| Method | Video-MME | LVBench | LongVideoBench | MLVU | EgoSchema | MVBench | Avg. |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in result["table2"].items():
        values = [row["datasets"].get(ds, {}).get("accuracy") for ds in DATASETS] + [row["average_accuracy"]]
        cells = ["—" if value is None else f"{value:.2f}" for value in values]
        lines.append("| " + LABELS[name] + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## 严格对齐的时长分段控制", "",
                  "| Dataset / duration | Questions / videos | Base (%) | PReM (%) | Gain, pp [95% CI] |",
                  "|---|---|---:|---:|---|"])
    for dataset, rows in result["duration_control"].items():
        for row in rows:
            prem = f"{row['prem_accuracy']:.2f}" if "prem_accuracy" in row else "—"
            interval = row.get("gain_ci95_pp")
            gain = f"{row['gain_pp']:+.2f} [{interval[0]:.2f}, {interval[1]:.2f}]" if interval else "—"
            lines.append(f"| {dataset} / {row['duration']} | {row['questions']} / {row['videos']} | "
                         f"{row['base_accuracy']:.2f} | {prem} | {gain} |")
    lines.extend(["", "新控制独立于原 Figure 5；不覆盖原图或原 CSV。", ""])
    lines.extend(["## 更长摄入流", "",
                  "| Frame cap | Actual writes | Fixed-buffer Base (%) | PReM (%) | Gain, pp [95% CI] |",
                  "|---:|---|---:|---:|---|"])
    natural = {m["condition"]: m for m in result["long_stream"]["metrics"]} if result["long_stream"] else {}
    for cap in (240, 480, 960):
        if cap not in natural:
            lines.append(f"| {cap} | — | — | — | — |"); continue
        m = natural[cap]; lo, hi = m["prediction_gain_ci95_pp"]
        gain = m["prediction_accuracy"] - m["base_accuracy"]
        lines.append(f"| {cap} | {m['actual_writes_min']}–{m['actual_writes_max']} | "
                     f"{m['base_accuracy']:.2f} | {m['prediction_accuracy']:.2f} | {gain:+.2f} [{lo:.2f}, {hi:.2f}] |")
    lines.extend(["", "## 目标证据留存", "",
                  "| Distractor frames | Base (%) | PReM (%) | Target replaced (%) | Evidence effect, pp [95% CI] |",
                  "|---:|---:|---:|---:|---:|"])
    needles = {m["condition"]: m for m in result["needle"]["prem"]["metrics"]} if "prem" in result["needle"] else {}
    for count in (0, 30, 60, 120, 240, 480):
        if count not in needles:
            lines.append(f"| {count} | — | — | — | — |"); continue
        m = needles[count]; lo, hi = m["evidence_effect_ci95_pp"]
        lines.append(f"| {count} | {m['base_accuracy']:.2f} | {m['prediction_accuracy']:.2f} | "
                     f"{m['target_removed_accuracy']:.2f} | {m['evidence_effect_pp']:+.2f} [{lo:.2f}, {hi:.2f}] |")
    lines.extend(["", "## 简单 Writer 对照", "",
                  "| Writer | Six-benchmark Avg. (%) | Accuracy at D=480 (%) | Logical / allocated state (bytes) | Instantiated parameters |",
                  "|---|---:|---:|---|---:|"])
    for variant in ("temporal_mean", "ema", "prem"):
        avg = result["table2"][variant]["average_accuracy"]
        avg = "—" if avg is None else f"{avg:.2f}"
        probe = result["needle"].get(variant)
        aged = next(m["prediction_accuracy"] for m in probe["metrics"] if m["condition"] == 480) if probe else None
        aged = "—" if aged is None else f"{aged:.2f}"
        train = result["training"].get(variant)
        state = f"{train['logical_state_bytes']} / {train['allocated_state_bytes']}" if train else "—"
        params = f"{train['instantiated_parameters']:,}" if train else "—"
        lines.append(f"| {LABELS[variant]} | {avg} | {aged} | {state} | {params} |")
    lines.extend(["", "D=480 对应证据之后 248 次更新。实例化参数包括未参与该变体计算的模块，不作为活跃参数数量。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/rebuttal_prem/clean_20261106")
    parser.add_argument("--base_root", type=Path, default=ROOT / "outputs/rebuttal_prem/table2_k1_20261106/eval/base")
    parser.add_argument("--allow_partial", action="store_true", help="Write an explicitly incomplete progress artifact")
    args = parser.parse_args()
    result = collect(args.run, args.base_root)
    if not result["complete"] and not args.allow_partial:
        raise RuntimeError("Incomplete campaign: " + ", ".join(result["missing"]))
    stem = "campaign_results" if result["complete"] else "campaign_partial"
    (args.run / (stem + ".json")).write_text(json.dumps(result, indent=2))
    (args.run / (stem + ".md")).write_text(markdown(result))
    print(json.dumps({"complete": result["complete"], "missing": result["missing"],
                      "output": str(args.run / (stem + ".json"))}, indent=2))


if __name__ == "__main__":
    main()
