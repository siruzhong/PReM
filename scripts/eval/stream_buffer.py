"""Question-agnostic streaming visual buffers shared by online baselines."""

from __future__ import annotations

import hashlib


def deterministic_reservoir_indices(num_items: int, capacity: int) -> list[int]:
    """Causal reservoir indices matching ``update_visual_buffer`` in eval_prem_online."""
    if capacity <= 0:
        raise ValueError("buffer capacity must be positive")
    if num_items <= 0:
        raise ValueError("Cannot sample an empty stream")
    if num_items <= capacity:
        return list(range(num_items))
    reservoir = list(range(capacity))
    for seen in range(capacity, num_items):
        digest = hashlib.sha256(f"prem-buffer:{seen}".encode("ascii")).digest()
        replacement = int.from_bytes(digest[:8], "big") % (seen + 1)
        if replacement < capacity:
            reservoir[replacement] = seen
    return sorted(reservoir)


def stable_reservoir_indices(num_items: int, capacity: int, key: str) -> list[int]:
    """Select a causal, deterministic reservoir without inspecting a question.

    ``key`` is accepted for call-site compatibility; indices match PReM's
    ``update_visual_buffer`` and do not depend on ``key``.
    """
    del key
    return deterministic_reservoir_indices(num_items, capacity)


def uniform_frame_indices(num_items: int, budget: int) -> list[int]:
    """Uniformly sample ``budget`` indices from a length-``num_items`` stream."""
    if budget <= 0:
        raise ValueError("buffer budget must be positive")
    if num_items <= 0:
        raise ValueError("Cannot sample an empty stream")
    if num_items <= budget:
        return list(range(num_items))
    # Local import keeps this module free of a hard torch dependency at import time
    # for lightweight unit tests that only need the reservoir helpers.
    import torch

    return torch.linspace(0, num_items - 1, budget).round().long().tolist()
