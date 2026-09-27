from __future__ import annotations

from PIL import Image

from scripts.eval.eval_prem_online import (
    CausalVideoSource,
    _is_decord_eof,
    get_video_grouped_chunk,
    update_visual_buffer,
)


def test_query_time_rows_are_sorted_within_video():
    rows = [
        {"id": "late", "video_path": "video.mp4", "query_time": 20.0},
        {"id": "early", "video_path": "video.mp4", "query_time": 5.0},
    ]

    grouped = get_video_grouped_chunk(rows, 1, 0)

    assert [row["id"] for row in grouped] == ["early", "late"]


def test_causal_video_source_only_releases_new_prefix_frames(tmp_path):
    video_dir = tmp_path / "frames"
    video_dir.mkdir()
    for index in range(6):
        (video_dir / f"frame_{index:03d}.jpg").write_bytes(b"")

    source = CausalVideoSource(str(tmp_path), {"video_path": "frames"}, 0, 1.0)

    assert len(source.take_until(1.5)) == 2
    assert source.visible_count == 2
    assert len(source.take_until(3.5)) == 2
    assert source.visible_count == 4
    assert source.take_until(3.5) == []


def test_causal_subsampling_preserves_original_timestamps(tmp_path):
    video_dir = tmp_path / "frames"
    video_dir.mkdir()
    for index in range(6):
        (video_dir / f"frame_{index:03d}.jpg").write_bytes(b"")

    source = CausalVideoSource(str(tmp_path), {"video_path": "frames"}, 4, 1.0)

    assert len(source.take_until(1.5)) == 1
    assert source.visible_count == 1
    assert len(source.take_until(3.5)) == 2
    assert source.visible_count == 3


def test_visual_buffer_is_causal_deterministic_and_bounded():
    frames = [Image.new("RGB", (4, 3), color=(index, 0, 0)) for index in range(10)]
    first = {"stats": {}}
    second = {"stats": {}}
    update_visual_buffer(first, frames[:4], capacity=3, max_pixels=6)
    update_visual_buffer(first, frames[4:], capacity=3, max_pixels=6)
    update_visual_buffer(second, frames, capacity=3, max_pixels=6)

    assert len(first["visual_buffer"]) == 3
    assert [index for index, _ in first["visual_buffer"]] == [
        index for index, _ in second["visual_buffer"]
    ]
    assert first["visual_buffer_seen"] == 10
    assert first["stats"]["visual_buffer_bytes"] <= 3 * 6 * 3


def test_decord_eof_errors_are_detected():
    assert _is_decord_eof(RuntimeError("Unable to handle EOF because it takes too long"))
    assert _is_decord_eof(RuntimeError("DECORD_EOF_RETRY_MAX=10240"))
    assert not _is_decord_eof(RuntimeError("CUDA out of memory"))
