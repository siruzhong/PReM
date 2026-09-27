import csv
import json
import subprocess
from pathlib import Path

import pytest

from scripts.eval import summarize_efficiency


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_dataset(root: Path, name: str, signature: dict, records: list[dict]) -> None:
    output = root / name
    output.mkdir(parents=True)
    payload = {}
    for index, record in enumerate(records):
        payload[str(index)] = {
            "eval_signature": json.dumps(signature, sort_keys=True),
            "peak_gpu_memory_allocated_bytes": 100 + index * 50,
            "peak_gpu_memory_reserved_bytes": 200 + index * 50,
            **record,
        }
    (output / "result.json").write_text(json.dumps(payload))
    with (output / "result.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["acc", "score", "count", "expected", "invalid"])
        writer.writeheader()
        writer.writerow({"acc": 0, "score": 0, "count": len(records), "expected": len(records), "invalid": 0})


def test_summarize_row_uses_profiled_latency_memory_and_macro_accuracy(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize_efficiency, "DATASETS", (("one", 2), ("two", 1)))
    row = {
        "key": "base_B16",
        "method": "Qwen2.5 base",
        "budget": 16,
        "evaluation": "qwen2_5vl_3b",
    }
    signature = {
        "protocol": "full_video",
        "max_frames": 16,
        "fps": 1.0,
        "max_pixels": 200704,
        "profile_efficiency": True,
    }
    result_root = tmp_path / "base_B16" / row["evaluation"]
    hardware_path = tmp_path / "base_B16" / "hardware.csv"
    hardware_path.parent.mkdir(parents=True, exist_ok=True)
    hardware_path.write_text("0,NVIDIA H20,GPU-test,555.1,97871 MiB\n")
    write_dataset(result_root, "one", signature, [
        {"acc": "yes", "stream_update_seconds": 1.0, "answer_seconds": 2.0},
        {"acc": "no", "stream_update_seconds": 2.0, "answer_seconds": 3.0},
    ])
    write_dataset(result_root, "two", signature, [
        {"acc": "yes", "stream_update_seconds": 3.0, "answer_seconds": 4.0},
    ])

    summary = summarize_efficiency.summarize_row(tmp_path, row, {}, {}, {}, "offline")

    assert summary["status"] == "complete"
    assert summary["avg6"] == pytest.approx(75.0)
    assert summary["latency_seconds_mean"] == pytest.approx(5.0)
    assert summary["latency_seconds_p50"] == pytest.approx(5.0)
    assert summary["peak_gpu_memory_allocated_bytes"] == 150
    assert summary["peak_gpu_memory_reserved_bytes"] == 250
    assert summary["sample_count"] == 3
    assert summary["invalid"] == 0


def test_summarize_row_maps_offline_videomme_to_videommewo(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize_efficiency, "DATASETS", (("videomme", 1),))
    row = {
        "key": "prem_B16",
        "method": "Qwen2.5 + PReM",
        "budget": 16,
        "evaluation": "prem_attention",
    }
    signature = {
        "protocol": "unified_bounded_visual_memory",
        "max_frames": 240,
        "chunk_frames": 240,
        "visual_buffer_frames": 16,
        "prem_modulation": "attention_kv",
        "fps": 1.0,
        "max_pixels": 200704,
        "profile_efficiency": True,
        "prem_checkpoint": {"path": "outputs/table4_efficiency/train/qwen25_3b_b16_ns1_rg0p10_lg4_pw0p2_pt4_ta0p75/prem.pt"},
    }
    result_root = tmp_path / "prem_B16" / row["evaluation"]
    hardware_path = tmp_path / "prem_B16" / "hardware.csv"
    hardware_path.parent.mkdir(parents=True, exist_ok=True)
    hardware_path.write_text("0,NVIDIA H20,GPU-test,555.1,97871 MiB\n")
    write_dataset(result_root, "videommewo", signature, [
        {"acc": "yes", "stream_update_seconds": 1.0, "answer_seconds": 2.0},
    ])

    summary = summarize_efficiency.summarize_row(
        tmp_path,
        row,
        {16: "outputs/table4_efficiency/train/qwen25_3b_b16_ns1_rg0p10_lg4_pw0p2_pt4_ta0p75/prem.pt"},
        {},
        {},
        "offline",
    )

    assert summary["status"] == "complete"
    assert summary["avg6"] == pytest.approx(100.0)


