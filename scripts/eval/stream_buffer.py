"""Question-agnostic streaming visual buffers shared by online baselines."""

from __future__ import annotations

import hashlib
import random


def stable_reservoir_indices(num_items: int, capacity: int, key: str) -> list[int]:
    """Select a causal, deterministic reservoir without inspecting a question."""
    if capacity <= 0:
        raise ValueError("buffer capacity must be positive")
    if num_items <= 0:
        raise ValueError("Cannot sample an empty stream")
    if num_items <= capacity:
        return list(range(num_items))
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    reservoir = list(range(capacity))
    for index in range(capacity, num_items):
        replacement = rng.randint(0, index)
        if replacement < capacity:
            reservoir[replacement] = index
    return sorted(reservoir)
