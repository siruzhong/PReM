from __future__ import annotations

import json
from pathlib import Path


EXPECTED_DATA_SEMANTICS = "unified_bounded_visual_memory_qa_v3"
EXPECTED_WRITER_MODE = "temporal_mean_per_step"
VALID_MODULATION_MODES = ("attention", "attention_kv", "attention_k", "attention_v")


def validate_prem_checkpoint(checkpoint: dict) -> None:
    if not checkpoint.get("complete", False):
        raise ValueError("PREM checkpoint is incomplete and cannot be used for formal evaluation")
    semantics = checkpoint.get("data_semantics")
    if semantics != EXPECTED_DATA_SEMANTICS:
        raise ValueError(
            f"Incompatible PREM data_semantics={semantics!r}; expected {EXPECTED_DATA_SEMANTICS!r}. "
            "Retrain from scratch with the current training entry point."
        )
    writer_mode = checkpoint.get("writer_mode")
    if writer_mode != EXPECTED_WRITER_MODE:
        raise ValueError(
            f"Incompatible PREM writer_mode={writer_mode!r}; expected {EXPECTED_WRITER_MODE!r}"
        )
    if checkpoint.get("qa_objective") != "single_ce":
        raise ValueError("PREM checkpoint must use the unified single-CE QA objective")
    if checkpoint.get("prem_modulation") not in VALID_MODULATION_MODES:
        raise ValueError("PREM checkpoint is missing a valid prem_modulation")
    if checkpoint.get("answer_context") != "bounded_visual_buffer_plus_side_memory":
        raise ValueError("PREM checkpoint must use the unified B+M answer context")
    buffer_frames = int(checkpoint.get("visual_buffer_frames", -1))
    if buffer_frames < 0:
        raise ValueError("PREM checkpoint is missing a valid visual_buffer_frames budget")
    if buffer_frames > 0 and (buffer_frames < 2 or buffer_frames % 2):
        raise ValueError("PREM visual_buffer_frames must be 0 or an even value >= 2")


def artifact_identity(path: str | None) -> dict | None:
    if not path:
        return None
    artifact = Path(path).resolve()
    if not artifact.exists():
        return {"path": path}
    stat = artifact.stat()
    return {
        "path": str(artifact),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def make_eval_signature(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
