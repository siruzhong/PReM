"""Observe the existing sampler and fingerprint the actual frozen writer inputs."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import torch

from scripts.eval import eval_prem_online as online
from scripts.eval.inference_mcq_vqa import get_sample_media_name


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_frames_with_manifest(video_dir, sample, max_frames, fps=1.0):
    """Call the original sampling path, recording its chosen indices unchanged.

    Each evaluator is a single-threaded worker. Restore the observed function
    even if decoding fails; this wrapper neither retries nor changes the reader.
    """
    source = Path(online.resolve_video_source(video_dir, get_sample_media_name(sample)))
    stat = source.stat()
    manifest = {"source": str(source.resolve()), "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns, "frame_cap": max_frames, "fps": fps}
    original = online.sample_frame_indices
    observed = []

    def record(total_frames, video_fps, target_fps, max_frames):
        indices = original(total_frames, video_fps, target_fps, max_frames)
        caller = inspect.currentframe().f_back.f_code.co_name
        observed.append({"reader": caller, "source_frames": int(total_frames),
                         "source_fps": float(video_fps), "requested_indices": list(indices)})
        return indices

    online.sample_frame_indices = record
    try:
        frames = online.resolve_stream_frames(video_dir, sample, max_frames, fps)
    finally:
        online.sample_frame_indices = original
    manifest["returned_frames"] = len(frames)
    if source.is_dir():
        manifest["reader"] = "frame_directory"
        manifest["selected_files"] = [{"path": str(Path(frame).resolve()),
                                        "size": Path(frame).stat().st_size,
                                        "mtime_ns": Path(frame).stat().st_mtime_ns} for frame in frames]
    else:
        if not observed:
            raise ValueError("The raw-video reader did not expose its sampling indices")
        manifest.update(observed[-1])
        manifest["all_requested_indices_returned"] = len(observed[-1]["requested_indices"]) == len(frames)
    return frames, manifest


def save_frame_manifest(output_dir, manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    directory = Path(output_dir) / "frame_audit"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + ".json")
    if not path.exists():
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(payload)
        os.replace(temporary, path)
    return digest
