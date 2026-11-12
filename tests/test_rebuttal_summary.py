import hashlib
import json

import pytest

from scripts.eval.summarize_rebuttal_memory import summarize


def write_probe(tmp_path):
    manifest = '{"source":"fixture"}'
    manifest_hash = hashlib.sha256(manifest.encode()).hexdigest()
    (tmp_path / "frame_audit").mkdir(exist_ok=True)
    (tmp_path / "frame_audit" / (manifest_hash + ".json")).write_text(manifest)
    signature = {"mode": "needle", "limit_videos": 0, "distractor_frames": [0, 30]}
    (tmp_path / "needle_manifest.json").write_text(json.dumps([{"id": "a"}, {"id": "b"}]))
    rows = []
    for sid in ("a", "b"):
        for count in (0, 30):
            base = {"decoder_input_sha256": sid, "correct": False}
            rows.append({"id": sid, "video_id": sid, "condition": count, "signature": signature,
                         "actual_writes": 16 + count // 2, "base": base,
                         "writer_input_sha256": "a" * 64, "target_replaced_writer_sha256": "b" * 64,
                         "writer_frame_manifest_sha256": manifest_hash,
                         "decoder_frame_manifest_sha256": manifest_hash,
                         "prediction": {**base, "correct": True}, "target_removed": base})
    (tmp_path / "1_0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_complete_paired_probe_reports_clustered_gain(tmp_path):
    write_probe(tmp_path)
    result = summarize(tmp_path)
    assert result["questions"] == 2
    assert result["metrics"][0]["prediction_accuracy"] == 100
    assert result["metrics"][0]["prediction_gain_ci95_pp"] == [100, 100]
    assert result["metrics"][0]["evidence_effect_pp"] == 100
    assert result["metrics"][0]["evidence_effect_ci95_pp"] == [100, 100]
    assert result["metrics"][0]["base_wrong_prediction_right"] == 2
    assert result["writer_input_audit"]["all_frame_manifests_verified"]


def test_modified_frame_manifest_cannot_be_reported(tmp_path):
    rows = write_probe(tmp_path)
    path = tmp_path / "frame_audit" / (rows[0]["writer_frame_manifest_sha256"] + ".json")
    path.write_text('{"source":"changed"}')
    with pytest.raises(ValueError, match="modified writer frame manifest"):
        summarize(tmp_path)


@pytest.mark.parametrize("failure", ["missing", "different_input", "duplicate"])
def test_partial_or_unmatched_probe_cannot_be_reported(tmp_path, failure):
    rows = write_probe(tmp_path)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[0])
    else:
        rows[1]["prediction"]["decoder_input_sha256"] = "changed"
    (tmp_path / "1_0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        summarize(tmp_path)


@pytest.mark.parametrize("failure", [None, "input", "prediction"])
def test_natural_run_must_match_independent_frozen_base(tmp_path, failure):
    rows = write_probe(tmp_path)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps([{"id": sid, "video_id": sid, "question": "Question?", "answer": 0}
                              for sid in ("a", "b")]))
    reference = {}
    for row in rows:
        row["signature"] = {"mode": "natural", "limit_videos": 0,
                            "frame_caps": [0, 30], "gt_file": str(gt)}
        row["base"]["pred"] = "B"
        reference[row["id"]] = {"pred": "B", "decoder_input_sha256": row["id"]}
    if failure == "input":
        reference["a"]["decoder_input_sha256"] = "different"
    elif failure == "prediction":
        reference["a"]["pred"] = "A"
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    (tmp_path / "1_0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if failure:
        with pytest.raises(ValueError):
            summarize(tmp_path, path)
    else:
        assert summarize(tmp_path, path)["independent_base_audit"]["all_inputs_and_predictions_match"]
