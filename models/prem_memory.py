"""PREM - Associative attention modulation memory.

Unified video-only writer + question-conditioned reader. Uses multi-slot associative
memory with learned write routing, relevance gating, forgetting, and
anti-overwrite protection.  The reader routes question-conditioned hidden
states through per-slot steer heads to produce query/output corrections
for every transformer layer.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PReMAttentionMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 4,
        key_dim: Optional[int] = None,
        val_dim: Optional[int] = None,
        alpha: float = 1.0,
        router_temperature: float = 1.0,
        write_lr: float = 1.0,
        disable_anti_distractor: bool = False,
        disable_novelty: bool = False,
        disable_stability: bool = False,
        disable_evidence_gate_write: bool = False,
        uniform_write_route: bool = False,
        num_layer_groups: int = 1,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_slots = int(num_slots)
        self.key_dim = int(key_dim or hidden_size)
        self.val_dim = int(val_dim or hidden_size)
        self.router_temperature = float(router_temperature)
        self.write_lr = float(write_lr)
        self.disable_anti_distractor = bool(disable_anti_distractor)
        self.disable_novelty = bool(disable_novelty)
        self.disable_stability = bool(disable_stability)
        self.disable_evidence_gate_write = bool(disable_evidence_gate_write)
        self.uniform_write_route = bool(uniform_write_route)
        self.num_layer_groups = int(num_layer_groups)
        if self.num_layer_groups < 1:
            raise ValueError(f"num_layer_groups must be positive, got {num_layer_groups}")

        self.log_alpha = nn.Parameter(torch.tensor(float(__import__("math").log(max(alpha, 1e-4)))))

        self.video_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.query_to_video = nn.Linear(hidden_size, hidden_size, bias=False)
        self.stream_salience_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

        self.query_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.key_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.value_proj = nn.Linear(hidden_size, self.val_dim, bias=False)
        self._init_near_identity(self.query_proj)
        self._init_near_identity(self.key_proj)
        self._init_near_identity(self.value_proj)
        with torch.no_grad():
            self.query_proj.weight.copy_(self.key_proj.weight)

        self.slot_key = nn.Parameter(torch.randn(num_slots, self.key_dim, self.key_dim) * 0.02)
        with torch.no_grad():
            eye = torch.eye(self.key_dim)
            for idx in range(num_slots):
                self.slot_key[idx].add_(eye)

        self.write_router = nn.Linear(self.key_dim, num_slots)
        self.relevance_head = nn.Linear(self.key_dim, num_slots)
        self.forget_logit = nn.Parameter(torch.full((num_slots,), 4.0))

        self.read_router = nn.Sequential(
            nn.Linear(self.val_dim * 3, self.val_dim),
            nn.GELU(),
            nn.Linear(self.val_dim, 1),
        )
        self.q_to_val = nn.Linear(self.key_dim, self.val_dim, bias=False)
        self.query_steer_heads = nn.ModuleList(
            [nn.Linear(self.val_dim, hidden_size, bias=False) for _ in range(num_slots)]
        )
        self.output_steer_heads = nn.ModuleList(
            [nn.Linear(self.val_dim, hidden_size, bias=False) for _ in range(num_slots)]
        )
        self.group_query_steer_heads = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(self.val_dim, hidden_size, bias=False) for _ in range(num_slots)])
                for _ in range(self.num_layer_groups)
            ]
            if self.num_layer_groups > 1 else []
        )
        self.group_output_steer_heads = nn.ModuleList(
            [
                nn.ModuleList([nn.Linear(self.val_dim, hidden_size, bias=False) for _ in range(num_slots)])
                for _ in range(self.num_layer_groups)
            ]
            if self.num_layer_groups > 1 else []
        )
        self._init_steer_heads()

        # optional evidence prediction head (video-only, uses salience gate)
        self.pred_query_proj = nn.Linear(hidden_size, self.val_dim, bias=False)
        self.pred_pos_proj = nn.Linear(1, self.val_dim)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(self.val_dim),
            nn.Linear(self.val_dim, self.val_dim * 2),
            nn.GELU(),
            nn.Linear(self.val_dim * 2, self.val_dim),
        )
        self._init_near_identity(self.pred_query_proj)


    @staticmethod
    def _init_near_identity(linear: nn.Linear):
        out_f, in_f = linear.weight.shape
        with torch.no_grad():
            linear.weight.zero_()
            d = min(out_f, in_f)
            linear.weight[:d, :d] += torch.eye(d)
            linear.weight += 0.01 * torch.randn_like(linear.weight)

    def _init_steer_heads(self):
        heads = list(self.query_steer_heads) + list(self.output_steer_heads)
        for group_heads in self.group_query_steer_heads:
            heads.extend(group_heads)
        for group_heads in self.group_output_steer_heads:
            heads.extend(group_heads)
        for head in heads:
            nn.init.normal_(head.weight, mean=0.0, std=1e-4)


    def reset_memory(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.num_slots,
            self.val_dim,
            self.key_dim,
            device=device,
            dtype=dtype,
        )

    def reset_confidence(self, batch_size: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_slots, device=device, dtype=dtype)

    def condition_video(self, video_seq: torch.Tensor, query: torch.Tensor):
        """Question-conditioned video representation for the READ path."""
        video = self.video_norm(video_seq)
        query = self.query_norm(query)
        query_video = self.query_to_video(query).unsqueeze(1)
        conditioned = video + query_video
        return conditioned

    def condition_stream_chunk(self, chunk_seq: torch.Tensor):
        """Video-only chunk conditioning for streaming writes."""
        conditioned = self.video_norm(chunk_seq)
        salience = torch.sigmoid(self.stream_salience_gate(conditioned)).squeeze(-1)
        return conditioned, salience

    def _slot_keys(self, k_vec: torch.Tensor) -> torch.Tensor:
        slot = torch.einsum("bj,kij->bki", k_vec, self.slot_key)
        return F.normalize(slot, dim=-1)


    def stream_write_step(
        self,
        state: torch.Tensor,
        confidence: torch.Tensor,
        token: torch.Tensor,
        salience: torch.Tensor,
        write_blocked: Optional[torch.Tensor] = None,
        apply_decay: bool = True,
        return_details: bool = False,
    ):
        """Question-free protected delta write for one visual token."""
        k_vec = self.key_proj(token)
        u_vec = self.value_proj(token)
        slot_k = self._slot_keys(k_vec)

        pred = torch.einsum("bkvi,bki->bkv", state, slot_k)
        delta = u_vec.unsqueeze(1) - pred

        if self.uniform_write_route:
            route = torch.full(
                (k_vec.shape[0], self.num_slots),
                1.0 / self.num_slots,
                device=k_vec.device,
                dtype=k_vec.dtype,
            )
        else:
            route = torch.softmax(self.write_router(k_vec), dim=-1)
        stability = torch.sigmoid(self.relevance_head(k_vec))
        if self.disable_stability:
            stability = torch.ones_like(stability)
        if self.disable_novelty:
            novelty = torch.ones_like(stability)
        else:
            novelty = (delta.norm(dim=-1) / u_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(0.0, 1.0)
        # Without salience gating, QA CE can minimize loss by collapsing the
        # video-only salience gate to zero.
        salience_write = (
            torch.ones_like(salience.unsqueeze(-1))
            if self.disable_evidence_gate_write
            else salience.unsqueeze(-1)
        )
        if write_blocked is not None:
            salience_write = salience_write.masked_fill(write_blocked.unsqueeze(-1), 0.0)
        if self.disable_anti_distractor:
            anti_overwrite = torch.ones_like(stability)
        else:
            anti_overwrite = 1.0 - confidence * (1.0 - salience.unsqueeze(-1))
        gate_scalar = route * stability * novelty * salience_write * anti_overwrite.clamp_min(0.0)
        gate = gate_scalar.unsqueeze(-1).unsqueeze(-1)

        outer = torch.einsum("bkv,bki->bkvi", delta, slot_k)
        forget = torch.sigmoid(self.forget_logit).view(1, -1, 1, 1)
        state_base = forget * state if apply_decay else state
        confidence_base = 0.95 * confidence if apply_decay else confidence
        write_update = self.write_lr * gate * outer
        new_state = state_base + write_update
        new_confidence = torch.maximum(confidence_base, (gate_scalar * stability).detach())
        if return_details:
            details = {
                "route": route,
                "stability": stability,
                "novelty": novelty,
                "salience": salience,
                "anti_overwrite": anti_overwrite.clamp_min(0.0),
                "gate": gate_scalar,
                "residual_norm": delta.norm(dim=-1),
                "write_update": write_update,
                "write_update_norm": write_update.float().norm(dim=(-2, -1)),
                "state_norm": new_state.float().norm(dim=(-2, -1)),
                "confidence": new_confidence,
            }
            return new_state, new_confidence, gate_scalar, details
        return new_state, new_confidence, gate_scalar


    def reset_stream(self, batch_size: int, device, dtype):
        state = self.reset_memory(batch_size, device, dtype)
        confidence = self.reset_confidence(batch_size, device, dtype)
        return state, confidence

    def stream_write_sequence(
        self,
        chunk_seq: torch.Tensor,
        salience: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        write_block_mask: Optional[torch.Tensor] = None,
        chunk_tokens: int = 32,
    ):
        if chunk_seq.ndim != 3:
            raise ValueError(f"chunk_seq must have shape [batch, time, hidden], got {tuple(chunk_seq.shape)}")
        if salience.shape != chunk_seq.shape[:2]:
            raise ValueError(
                f"salience must have shape {tuple(chunk_seq.shape[:2])}, got {tuple(salience.shape)}"
            )
        batch_size = chunk_seq.shape[0]
        if state is None or confidence is None:
            state, confidence = self.reset_stream(batch_size, chunk_seq.device, chunk_seq.dtype)

        gates = []
        step = max(1, int(chunk_tokens or 1))
        for start in range(0, chunk_seq.shape[1], step):
            end = min(start + step, chunk_seq.shape[1])
            for idx in range(start, end):
                state, confidence, gate_scalar = self.stream_write_step(
                    state,
                    confidence,
                    chunk_seq[:, idx, :],
                    salience[:, idx],
                    None if write_block_mask is None else write_block_mask[:, idx],
                )
                gates.append(gate_scalar)
        if gates:
            gate_tensor = torch.stack(gates, dim=1)
            self._last_gate_per_slot = gate_tensor.mean(dim=1).detach()
            mean_write_gate = gate_tensor.mean()
        else:
            self._last_gate_per_slot = chunk_seq.new_zeros((batch_size, self.num_slots))
            mean_write_gate = chunk_seq.new_zeros(())
        return state, confidence, mean_write_gate

    def stream_write_grouped_sequence(
        self,
        grouped_seq: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        chunk_tokens: int = 32,
        salience: Optional[torch.Tensor] = None,
    ):
        if grouped_seq.ndim != 4:
            raise ValueError(
                "grouped_seq must have shape [batch, time, spatial_tokens, hidden], "
                f"got {tuple(grouped_seq.shape)}"
            )
        batch_size = grouped_seq.shape[0]
        if state is None or confidence is None:
            state, confidence = self.reset_stream(batch_size, grouped_seq.device, grouped_seq.dtype)
        if salience is None:
            salience = torch.ones(grouped_seq.shape[:3], device=grouped_seq.device, dtype=grouped_seq.dtype)
        elif salience.shape != grouped_seq.shape[:3]:
            raise ValueError(
                f"salience must have shape {tuple(grouped_seq.shape[:3])}, got {tuple(salience.shape)}"
            )

        gates = []
        step = max(1, int(chunk_tokens or 1))
        for start in range(0, grouped_seq.shape[1], step):
            end = min(start + step, grouped_seq.shape[1])
            for idx in range(start, end):
                spatial_tokens = grouped_seq[:, idx, :, :]
                step_salience = salience[:, idx, :]
                # Decay belongs to the temporal step, not its spatial token count.
                for n in range(spatial_tokens.shape[1]):
                    state, confidence, gate_scalar = self.stream_write_step(
                        state,
                        confidence,
                        spatial_tokens[:, n, :],
                        step_salience[:, n],
                        apply_decay=n == 0,
                    )
                    gates.append(gate_scalar)
        if gates:
            gate_tensor = torch.stack(gates, dim=1)
            self._last_gate_per_slot = gate_tensor.mean(dim=1).detach()
            mean_write_gate = gate_tensor.mean()
        else:
            self._last_gate_per_slot = grouped_seq.new_zeros((batch_size, self.num_slots))
            mean_write_gate = grouped_seq.new_zeros(())
        return state, confidence, mean_write_gate




    def _alpha_scale(self, alpha_override: Optional[float] = None):
        scale = torch.exp(self.log_alpha)
        if alpha_override is not None:
            scale = scale * float(alpha_override)
        return scale / self.num_slots


    def read_sequence(self, state: torch.Tensor, query_seq: torch.Tensor):
        q_vec = self.query_proj(self.query_norm(query_seq))
        slot_q = torch.einsum("btj,kij->btki", q_vec, self.slot_key)
        slot_q = F.normalize(slot_q, dim=-1)
        per_slot = torch.einsum("bkvi,btki->btkv", state, slot_q)
        q_exp = q_vec.unsqueeze(2).expand(-1, -1, self.num_slots, -1)
        q_val = self.q_to_val(q_exp)
        router_in = torch.cat([q_val, per_slot, q_val * per_slot], dim=-1)
        logits = self.read_router(router_in).squeeze(-1)
        rho = torch.softmax(logits / max(self.router_temperature, 1e-6), dim=-1)
        readout = (rho.unsqueeze(-1) * per_slot).sum(dim=2)
        return readout, rho, per_slot

    def steer_from_sequence_read(
        self,
        rho: torch.Tensor,
        per_slot: torch.Tensor,
        alpha_override: Optional[float] = None,
    ):
        query_dirs = self._apply_slot_heads(per_slot, self.query_steer_heads)
        output_dirs = self._apply_slot_heads(per_slot, self.output_steer_heads)
        scale = self._alpha_scale(alpha_override)
        query_steer = scale * (rho.unsqueeze(-1) * query_dirs).sum(dim=2)
        output_steer = scale * (rho.unsqueeze(-1) * output_dirs).sum(dim=2)
        return query_steer, output_steer, query_dirs, output_dirs

    @staticmethod
    def _apply_slot_heads(per_slot: torch.Tensor, heads: nn.ModuleList) -> torch.Tensor:
        """Apply all bias-free per-slot heads in one batched contraction."""
        weights = torch.stack([head.weight for head in heads], dim=0)
        return torch.einsum("btkv,khv->btkh", per_slot, weights)

    def contextual_read_per_position(
        self,
        state: torch.Tensor,
        hidden_seq: torch.Tensor,
        query: torch.Tensor,
        layer_group: int,
        alpha_override: Optional[float] = None,
    ):
        """Read shared state from a layer group's current hidden sequence."""
        if not 0 <= int(layer_group) < self.num_layer_groups:
            raise ValueError(f"Invalid layer_group={layer_group} for {self.num_layer_groups} groups")
        conditioned = self.condition_video(hidden_seq, query)
        readout, rho, per_slot = self.read_sequence(state, conditioned)
        query_heads = self.group_query_steer_heads[int(layer_group)]
        output_heads = self.group_output_steer_heads[int(layer_group)]
        query_dirs = self._apply_slot_heads(per_slot, query_heads)
        output_dirs = self._apply_slot_heads(per_slot, output_heads)
        scale = self._alpha_scale(alpha_override)
        query_steer = scale * (rho.unsqueeze(-1) * query_dirs).sum(dim=2)
        output_steer = scale * (rho.unsqueeze(-1) * output_dirs).sum(dim=2)
        return query_steer, output_steer, readout, rho, per_slot, query_dirs, output_dirs



    @staticmethod
    def _top_evidence_span_mask(evidence_gate: torch.Tensor, num_tokens: int) -> torch.Tensor:
        batch_size, seq_len = evidence_gate.shape
        mask = torch.zeros(batch_size, seq_len, device=evidence_gate.device, dtype=torch.bool)
        if num_tokens is None or num_tokens <= 0 or seq_len <= 1:
            return mask
        span = min(int(num_tokens), seq_len - 1)
        scores = evidence_gate.detach().float()
        if span == 1:
            starts = scores.argmax(dim=1)
        else:
            pooled = F.avg_pool1d(scores.unsqueeze(1), kernel_size=span, stride=1).squeeze(1)
            starts = pooled.argmax(dim=1)
        offsets = torch.arange(span, device=evidence_gate.device).unsqueeze(0)
        indices = starts.unsqueeze(1) + offsets
        mask.scatter_(1, indices, True)
        return mask

    def evidence_prediction_loss(
        self,
        state: torch.Tensor,
        conditioned_memory: torch.Tensor,
        query: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        if not bool(target_mask.any()):
            zero = conditioned_memory.new_zeros(())
            return zero, {
                "evidence_prediction_loss": 0.0,
                "evidence_prediction_tokens": 0.0,
            }
        pred_queries = query.unsqueeze(1).expand(-1, conditioned_memory.shape[1], -1)
        readout, _, _ = self.read_sequence(state, pred_queries)
        query_ctx = self.pred_query_proj(self.query_norm(query)).unsqueeze(1)
        positions = torch.linspace(
            0.0,
            1.0,
            conditioned_memory.shape[1],
            device=conditioned_memory.device,
            dtype=self.pred_pos_proj.weight.dtype,
        ).view(1, -1, 1)
        pos_ctx = self.pred_pos_proj(positions).expand(conditioned_memory.shape[0], -1, -1)
        pred = self.pred_head(readout + query_ctx + pos_ctx)
        with torch.no_grad():
            target = self.value_proj(conditioned_memory.detach())
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        token_loss = (pred - target).pow(2).sum(dim=-1)
        mask = target_mask.to(token_loss.dtype)
        denom = mask.sum().clamp_min(1.0)
        loss = (token_loss * mask).sum() / denom
        return loss, {
            "evidence_prediction_loss": float(loss.detach().cpu()),
            "evidence_prediction_tokens": float(denom.detach().cpu()),
        }


    def stream_read_per_position(
        self,
        state: torch.Tensor,
        target_seq: torch.Tensor,
        query: torch.Tensor,
        alpha_override: Optional[float] = None,
    ):
        conditioned_target = self.condition_video(target_seq, query)
        readout, rho, per_slot = self.read_sequence(state, conditioned_target)
        query_steer, output_steer, query_dirs, output_dirs = self.steer_from_sequence_read(
            rho,
            per_slot,
            alpha_override=alpha_override,
        )
        return query_steer, output_steer, readout, rho, per_slot, query_dirs, output_dirs
