import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.eval import rebuttal_input_audit as audit
from scripts.eval import eval_rebuttal_memory as evaluator
from qwen_vl_utils.vision_process import fetch_video


def test_writer_fingerprint_includes_dtype_shape_and_order():
    value = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    expected = audit.tensor_sha256(value)
    assert audit.tensor_sha256(value.clone()) == expected
    assert audit.tensor_sha256(value.float()) != expected
    assert audit.tensor_sha256(value.reshape(4, 2)) != expected
    assert audit.tensor_sha256(value.flip(0)) != expected


@pytest.mark.parametrize("fail", [False, True])
def test_sampler_observer_preserves_original_call_and_restores_it(tmp_path, monkeypatch, fail):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(audit.online, "resolve_video_source", lambda *args: str(source))
    original = audit.online.sample_frame_indices
    sampled = []

    def reader(*args):
        sampled.extend(audit.online.sample_frame_indices(240, 24.0, 1.0, 8))
        if fail:
            raise RuntimeError("fixture decode failure")
        return sampled

    monkeypatch.setattr(audit.online, "resolve_stream_frames", reader)
    if fail:
        with pytest.raises(RuntimeError, match="decode failure"):
            audit.load_frames_with_manifest(tmp_path, {"video_id": "video"}, 8)
    else:
        frames, manifest = audit.load_frames_with_manifest(tmp_path, {"video_id": "video"}, 8)
        assert frames == original(240, 24.0, 1.0, 8) == manifest["requested_indices"]
        assert manifest["all_requested_indices_returned"]
    assert audit.online.sample_frame_indices is original


def test_frame_directory_manifest_records_selected_files(tmp_path, monkeypatch):
    source = tmp_path / "frames"
    source.mkdir()
    for index in range(8):
        (source / f"{index:03d}.jpg").write_bytes(bytes([index]))
    monkeypatch.setattr(audit.online, "resolve_video_source", lambda *args: str(source))
    frames, manifest = audit.load_frames_with_manifest(tmp_path, {"video_id": "frames"}, 4)
    assert [entry["path"] for entry in manifest["selected_files"]] == [str(path) for path in frames]
    assert len(frames) == 4
    digest = audit.save_frame_manifest(tmp_path, manifest)
    saved = tmp_path / "frame_audit" / (digest + ".json")
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
    assert json.loads(saved.read_text()) == manifest


@pytest.mark.parametrize("count", [1, 3, 4])
def test_writer_cache_keeps_singleton_tail_with_original_padding(tmp_path, monkeypatch, count):
    from PIL import Image

    frames = [Image.new("RGB", (56, 56), (10 * (index + 1),) * 3) for index in range(count)]
    observed = []

    def encode(model, processor, chunk, args):
        observed.append(len(chunk))
        images = fetch_video({"video": chunk, "max_frames": len(chunk), "max_pixels": args.max_pixels})
        assert len(images) == 2
        value = sum(image.getpixel((0, 0))[0] for image in images) / 2
        return torch.full((1, 2, 4), value, dtype=torch.bfloat16)

    monkeypatch.setattr(evaluator, "chunk_to_frame_embeds", encode)
    args = SimpleNamespace(feature_cache=str(tmp_path), fps=1.0, max_pixels=200704,
                           model_identity={"fixture": True}, software={"fixture": True})
    features = evaluator.frame_features(None, None, frames, args, {"fixture_frames": count})
    assert len(features) == (count + 1) // 2
    assert observed == [2] * (count // 2) + ([1] if count % 2 else [])
    if count % 2:
        assert torch.equal(features[-1], torch.full((4,), count * 10, dtype=torch.bfloat16))
    observed.clear()
    cached = evaluator.frame_features(None, None, frames, args, {"fixture_frames": count})
    assert torch.equal(cached, features) and not observed
