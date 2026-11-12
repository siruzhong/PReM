import hashlib
import json

import pytest

from scripts.eval import summarize_rebuttal_campaign as campaign


def test_partial_benchmark_coverage_never_produces_an_average(tmp_path, monkeypatch):
    run, base = tmp_path / "run", tmp_path / "base"
    path = base / "mlvu/result.json"
    path.parent.mkdir(parents=True)
    gt = {"a": {"answer": 0}, "b": {"answer": 0}}
    monkeypatch.setattr(campaign, "ground_truth", lambda dataset: gt)
    path.write_text(json.dumps({
        "a": {"answer": 0, "pred": "A", "acc": "yes", "decoder_input_sha256": "a" * 64},
        "b": {"answer": 0, "pred": "B", "acc": "no", "decoder_input_sha256": "b" * 64},
    }))
    result = campaign.collect(run, base)
    assert not result["complete"]
    assert result["table2"]["base"]["datasets"]["mlvu"]["accuracy"] == 50
    assert result["table2"]["base"]["average_accuracy"] is None


def test_same_needle_accuracy_cannot_hide_different_evidence_frames(tmp_path):
    row = {"id": "a", "condition": 0, "video_id": "video", "answer": 0,
           "evidence_indices": [1, 2], "distractor_video_id": "foreign",
           "actual_writes": 16, "writer_input_sha256": "a" * 64,
           "target_replaced_writer_sha256": "b" * 64, "writer_frame_manifest_sha256": "c" * 64,
           "base": {"pred": "B", "decoder_input_sha256": "same"},
           "evidence_visible_base": {"pred": "A", "decoder_input_sha256": "visible"}}
    directories = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        if index:
            row["evidence_indices"] = [3, 4]
        (directory / "1_0.jsonl").write_text(json.dumps(row) + "\n")
        directories.append(directory)
    with pytest.raises(ValueError, match="different needle inputs"):
        campaign.verify_cross_variant_needles(directories)


@pytest.mark.parametrize("field", ["writer_input_sha256", "writer_frame_manifest_sha256"])
def test_same_decoder_cannot_hide_different_writer_inputs(tmp_path, field):
    row = {"id": "a", "condition": 240, "video_id": "video", "answer": 0,
           "sampled_frames": 240, "actual_writes": 120, "writer_input_sha256": "a" * 64,
           "writer_frame_manifest_sha256": "b" * 64, "decoder_frame_manifest_sha256": "c" * 64}
    directories = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        if index:
            row[field] = "d" * 64
        (directory / "1_0.jsonl").write_text(json.dumps(row) + "\n")
        directories.append(directory)
    with pytest.raises(ValueError, match="different natural stream inputs"):
        campaign.verify_cross_variant_natural(directories)


def test_writer_label_is_verified_from_predictions_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"fixture checkpoint")
    row = {"writer_variant": "prem", "read_interface": "prefix_kv",
           "signature": {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}
    campaign.verify_identity([row], "prem", checkpoint)
    with pytest.raises(ValueError, match="identity mismatch"):
        campaign.verify_identity([row], "temporal_mean", checkpoint)
