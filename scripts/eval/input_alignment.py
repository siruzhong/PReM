"""Record the decoder inputs before applying a memory readout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoder_input_record(inputs, state: dict) -> dict:
    """Hash names, shapes, dtypes and bytes, including visual pixels and text."""
    digest = hashlib.sha256()
    for key in sorted(inputs):
        value = inputs[key]
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            header = [key, str(tensor.dtype), list(tensor.shape)]
            digest.update(json.dumps(header, separators=(",", ":")).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps([key, value], sort_keys=True).encode())
    return {
        "decoder_input_sha256": digest.hexdigest(),
        "decoder_buffer_stream_indices": [int(index) for index, _ in state.get("visual_buffer", [])],
        "decoder_input_digest_version": 1,
    }
