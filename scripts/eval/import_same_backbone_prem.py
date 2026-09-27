#!/usr/bin/env python
"""Import a completed legacy PReM run into the same-backbone result schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = (
    ("longvideobench", 1337),
    ("mlvu", 2175),
    ("videomme", 2700),
    ("egoschema", 500),
    ("mvbench", 4000),
    ("lvbench", 1549),
)
TRAINABLE_PARAMETERS = 9_070_982
RUNTIME_PROVENANCE = "legacy_table4_efficiency_8_worker"


def load_records(path: Path, expected: int) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or len(payload) != expected:
        actual = len(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"{path}: expected {expected} records, got {actual}")
    records = {str(row["id"]): row for row in payload.values()}
    if len(records) != expected:
        raise ValueError(f"{path}: duplicate sample ids")
    return records


def import_dataset(
    source_root: Path, output_root: Path, dataset: str, expected: int
) -> None:
    source_path = source_root / dataset / "result.json"
    base_path = output_root / "eval" / "base" / dataset / "result.json"
    source = load_records(source_path, expected)
    base = load_records(base_path, expected)
    if set(source) != set(base):
        missing = sorted(set(base) - set(source))[:5]
        extra = sorted(set(source) - set(base))[:5]
        raise ValueError(
            f"{dataset}: legacy PReM ids differ from Base; missing={missing}, extra={extra}"
        )

    imported: dict[str, dict] = {}
    for sample_id, source_row in source.items():
        base_row = base[sample_id]
        source_signature = json.loads(source_row["eval_signature"])
        if source_signature.get("dataset") != dataset:
            raise ValueError(f"{dataset}/{sample_id}: invalid legacy dataset signature")
        expected_legacy = {
            "protocol": "unified_bounded_visual_memory",
            "max_frames": 240,
            "visual_buffer_frames": 16,
            "fps": 1,
            "max_pixels": 200704,
            "chunk_frames": 2,
            "mcq_max_new_tokens": 1,
            "profile_efficiency": True,
            "prem_modulation": "attention_kv",
        }
        mismatch = {
            key: (source_signature.get(key), value)
            for key, value in expected_legacy.items()
            if source_signature.get(key) != value
        }
        if mismatch:
            raise ValueError(f"{dataset}/{sample_id}: legacy protocol mismatch {mismatch}")
        if int(source_row.get("visual_buffer_frames", 0)) != int(
            base_row["visual_buffer_frames"]
        ):
            raise ValueError(f"{dataset}/{sample_id}: visual frame count differs from Base")
        if int(source_row.get("memory_state_bytes", 0)) != 65_540:
            raise ValueError(f"{dataset}/{sample_id}: unexpected PReM memory state size")

        base_signature = json.loads(base_row["eval_signature"])
        signature = dict(base_signature)
        signature.update(
            {
                "method": "prem",
                "method_label": "Base + PReM",
                "checkpoint": source_signature.get("prem_checkpoint"),
                "trainable_parameters": TRAINABLE_PARAMETERS,
                "runtime_comparable": False,
                "runtime_provenance": RUNTIME_PROVENANCE,
                "runtime_worker_count": 8,
                "imported_from": str(source_path.resolve()),
            }
        )
        row = dict(source_row)
        row.update(
            {
                "eval_signature": json.dumps(signature, sort_keys=True, separators=(",", ":")),
                "comparison_method": "prem",
                "comparison_method_label": "Base + PReM",
                "trainable_parameters": TRAINABLE_PARAMETERS,
                "max_frames": 240,
                "buffer_frames_limit": 16,
                "visual_buffer_frames": int(base_row["visual_buffer_frames"]),
                "visual_buffer_capacity": 16,
                "visual_buffer_policy": "deterministic_reservoir",
                "visual_buffer_bytes": int(base_row["visual_buffer_bytes"]),
                "buffered_frames": int(base_row.get("buffered_frames", 0)),
                "decoder_prompt_tokens": int(base_row["decoder_prompt_tokens"]),
                "auxiliary_prompt_tokens": 0,
                "ingested_visual_tokens": int(base_row.get("ingested_visual_tokens", 0)),
                "runtime_comparable": False,
                "runtime_provenance": RUNTIME_PROVENANCE,
            }
        )
        imported[sample_id] = row

    output_dir = output_root / "eval" / "prem" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "result.json").open("w") as handle:
        json.dump(imported, handle, indent=2)
    with (output_dir / "pred.json").open("w") as handle:
        json.dump(
            {sample_id: row.get("pred", "") for sample_id, row in imported.items()},
            handle,
            indent=2,
        )
    correct = sum(row.get("acc") == "yes" for row in imported.values())
    print(f"[import] {dataset}: {correct}/{expected}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    for dataset, expected in DATASETS:
        import_dataset(source_root, output_root, dataset, expected)
    print(f"[done] imported PReM results into {output_root / 'eval' / 'prem'}", flush=True)


if __name__ == "__main__":
    main()
