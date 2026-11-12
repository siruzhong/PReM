"""Temporal pooling controls with PReM's existing prefix readout.

The recurrent payload stores a pooled, normalized visual feature. At read time
the existing key/value projections bind this feature into an associative matrix;
all question-conditioned reads and prefix steering heads remain unchanged.
The padded tensor preserves the wrapper interface; report logical and allocated
state sizes separately. These controls require K=1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .prem_memory import PReMAttentionMemory


class PooledWriterMemory(PReMAttentionMemory):
    def __init__(self, *args, writer_variant: str, ema_decay: float = 0.98, **kwargs):
        super().__init__(*args, **kwargs)
        if writer_variant not in {"temporal_mean", "ema"}:
            raise ValueError(f"Unknown pooled writer: {writer_variant}")
        if self.num_slots != 1 or self.val_dim * self.key_dim < self.hidden_size:
            raise ValueError("Pooling controls require K=1 and enough payload space for one feature")
        if not 0 <= ema_decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.writer_variant = writer_variant
        self.ema_decay = float(ema_decay)

    def stream_write_step(
        self, state, confidence, token, salience, write_blocked=None,
        apply_decay=True, return_details=False,
    ):
        if not apply_decay:
            raise ValueError("Pooling controls accept one pooled feature per temporal step")
        pooled = state.flatten(1)[:, :self.hidden_size]
        observed = torch.ones_like(confidence)
        if write_blocked is not None:
            observed = observed.masked_fill(write_blocked[:, None], 0)
        count = confidence + observed
        if self.writer_variant == "temporal_mean":
            weight = observed / count.clamp_min(1)
        else:
            # Initialize from the first observed feature, without zero-start bias.
            weight = observed * torch.where(
                confidence == 0, torch.ones_like(confidence), 1 - self.ema_decay,
            )
        updated = pooled + weight * (token - pooled)
        payload = F.pad(updated, (0, self.val_dim * self.key_dim - self.hidden_size))
        new_state = payload.reshape_as(state)
        if return_details:
            return new_state, count, weight, {
                "gate": weight, "observed_steps": count,
                "state_norm": new_state.float().norm(dim=(-2, -1)),
                "write_update_norm": (new_state - state).float().norm(dim=(-2, -1)),
            }
        return new_state, count, weight

    def pooled_binding(self, state):
        pooled = state.flatten(1)[:, :self.hidden_size]
        key = self._slot_keys(self.key_proj(pooled))
        value = self.value_proj(pooled).unsqueeze(1)
        return torch.einsum("bkv,bki->bkvi", value, key)

    def read_sequence(self, state, query_seq):
        return super().read_sequence(self.pooled_binding(state), query_seq)


def install_pooled_writer(model, writer_variant: str, ema_decay: float = 0.98):
    """Replace only the writer, preserving initialization and all read heads."""
    if writer_variant == "prem":
        return model.prem_memory
    source = model.prem_memory
    if source is None:
        raise ValueError("Build the PReM memory before selecting a pooling control")
    # Restore RNG state: replacing a module must not change data order or seeds.
    with torch.random.fork_rng(devices=[]):
        replacement = PooledWriterMemory(
            hidden_size=source.hidden_size, num_slots=source.num_slots,
            key_dim=source.key_dim, val_dim=source.val_dim,
            num_layer_groups=source.num_layer_groups,
            writer_variant=writer_variant, ema_decay=ema_decay,
        )
    replacement.load_state_dict(source.state_dict(), strict=True)
    ref = source.query_proj.weight
    replacement.to(device=ref.device, dtype=ref.dtype)
    replacement.train(source.training)
    model.prem_memory = replacement
    return replacement
