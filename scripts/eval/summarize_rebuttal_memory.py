#!/usr/bin/env python
"""Validate complete paired rebuttal runs and compute video-clustered intervals."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.data_io import load_records
from scripts.eval.inference_mcq_vqa import normalize_mcq_sample


def paired_interval(rows, key, repetitions=10000, reference="base"):
    groups = defaultdict(list)
    for row in rows:
        groups[row["video_id"]].append(row)
    counts = np.array([len(group) for group in groups.values()])
    differences = np.array([sum(int(row[key]["correct"]) -
                                (int(row[reference]["correct"]) if reference else 0)
                                for row in group) for group in groups.values()])
    rng = np.random.default_rng(13)
    indices = rng.integers(0, len(groups), (repetitions, len(groups)))
    gains = 100 * differences[indices].sum(1) / counts[indices].sum(1)
    return np.percentile(gains, [2.5, 97.5]).tolist()


def verify_writer_inputs(directory, rows, mode):
    verified = set()

    def verify_manifest(digest):
        if digest in verified:
            return
        path = directory / "frame_audit" / (digest + ".json")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Missing or modified writer frame manifest")
        manifest = json.loads(path.read_text())
        verified.add(digest)
        if "foreign_manifest_sha256" in manifest:
            verify_manifest(manifest["foreign_manifest_sha256"])

    fields = ["writer_input_sha256", "writer_frame_manifest_sha256"]
    fields.append("decoder_frame_manifest_sha256" if mode == "natural" else "target_replaced_writer_sha256")
    decoder_manifests = {}
    for row in rows:
        for key in fields:
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Missing or invalid writer input audit: {key}")
        verify_manifest(row["writer_frame_manifest_sha256"])
        if mode == "natural":
            value = row["decoder_frame_manifest_sha256"]
            verify_manifest(value)
            sid = str(row["id"])
            if sid in decoder_manifests and decoder_manifests[sid] != value:
                raise ValueError("Decoder frame manifest changes across conditions")
            decoder_manifests[sid] = value
    return {"all_frame_manifests_verified": True, "frame_manifests": len(verified)}


def summarize(directory, reference_base=None):
    rows = []
    for path in sorted(directory.glob("[0-9]*_[0-9]*.jsonl")):
        rows.extend(json.loads(line) for line in path.open())
    if not rows:
        raise ValueError("No prediction rows")
    signature = rows[0]["signature"]
    if signature["limit_videos"]:
        raise ValueError("Diagnostic runs cannot be summarised as formal results")
    if any(row["signature"] != signature for row in rows):
        raise ValueError("Mixed run signatures")
    conditions = signature["frame_caps"] if signature["mode"] == "natural" else signature["distractor_frames"]
    if signature["mode"] == "natural":
        expected_ids = {str(normalize_mcq_sample(row)["id"]) for row in load_records(signature["gt_file"])}
    else:
        expected_ids = {str(row["id"]) for row in json.loads((directory / "needle_manifest.json").read_text())}
    expected = {(sid, condition) for sid in expected_ids for condition in conditions}
    observed = [(str(row["id"]), row["condition"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"Incomplete/duplicate coverage: observed={len(observed)} expected={len(expected)}")
    writer_audit = verify_writer_inputs(directory, rows, signature["mode"])
    hashes = {}
    for row in rows:
        sid = str(row["id"])
        digest = row["prediction"]["decoder_input_sha256"]
        if row["base"]["decoder_input_sha256"] != digest:
            raise ValueError("Base/prediction input mismatch")
        if "target_removed" in row and row["target_removed"]["decoder_input_sha256"] != digest:
            raise ValueError("Evidence-control input mismatch")
        if sid in hashes and hashes[sid] != digest:
            raise ValueError("Decoder input changes across conditions")
        hashes[sid] = digest
    reference_audit = None
    if reference_base is not None:
        if signature["mode"] != "natural":
            raise ValueError("Native Base predictions cannot validate composite needle streams")
        reference_base = Path(reference_base)
        reference = json.loads(reference_base.read_text())
        if set(reference) != expected_ids:
            raise ValueError("Independent Base coverage mismatch")
        for row in rows:
            frozen = reference[str(row["id"])]
            if frozen["decoder_input_sha256"] != row["base"]["decoder_input_sha256"]:
                raise ValueError("Independent Base input mismatch")
            if frozen["pred"].strip().upper() != row["base"]["pred"].strip().upper():
                raise ValueError("Zero steering differs from independent frozen Base")
        reference_audit = {"path": str(reference_base),
                           "sha256": hashlib.sha256(reference_base.read_bytes()).hexdigest(),
                           "all_inputs_and_predictions_match": True}
    metrics = []
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        metric = {"condition": condition, "questions": len(subset),
                  "videos": len({row['video_id'] for row in subset}),
                  "actual_writes_min": min(row["actual_writes"] for row in subset),
                  "actual_writes_max": max(row["actual_writes"] for row in subset)}
        for key in ("base", "prediction", "target_removed", "evidence_visible_base"):
            if key in subset[0]:
                metric[key + "_accuracy"] = 100 * sum(row[key]["correct"] for row in subset) / len(subset)
                metric[key + "_accuracy_ci95"] = paired_interval(subset, key, reference=None)
                if key in {"prediction", "target_removed"}:
                    metric[key + "_gain_ci95_pp"] = paired_interval(subset, key)
        metric["base_wrong_prediction_right"] = sum(
            not row["base"]["correct"] and row["prediction"]["correct"] for row in subset)
        metric["base_right_prediction_wrong"] = sum(
            row["base"]["correct"] and not row["prediction"]["correct"] for row in subset)
        if "target_removed" in subset[0]:
            metric["evidence_effect_pp"] = metric["prediction_accuracy"] - metric["target_removed_accuracy"]
            metric["evidence_effect_ci95_pp"] = paired_interval(subset, "prediction", reference="target_removed")
        metrics.append(metric)
    return {"signature": signature, "all_decoder_hashes_match": True,
            "writer_input_audit": writer_audit,
            "independent_base_audit": reference_audit,
            "questions": len(expected_ids), "prediction_rows": len(rows), "metrics": metrics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference_base", type=Path)
    args = parser.parse_args()
    summary = summarize(args.directory, args.reference_base)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "signature"}, indent=2))
