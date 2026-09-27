"""Qwen2.5-VL with trainable attention modulation.

Unified PREM write-read module: video-only writer + question-conditioned reader.
The backbone is a standard Qwen2.5-VL. A small side module writes question-agnostic
video evidence into a protected multi-slot associative memory, then reads
question-conditioned hidden states through per-slot steer heads to inject
query/output corrections into frozen self-attention layers.

Based on prem_qwen2vl_model.py, adapted for Qwen2.5-VL API: qwen2_vl -> qwen2_5_vl.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationMixin
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLForConditionalGeneration,
)

from .prem_memory import PReMAttentionMemory


class PReMQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.prem_memory = None
        self.prem_aux_losses = None
        self.prem_last_stats = None
        self.prem_last_stream_state = None

    @property
    def _prem_text_model(self):
        return self.model.language_model

    def _prem_encode_video(self, pixel_values_videos, video_grid_thw):
        features = self.get_video_features(pixel_values_videos, video_grid_thw)
        return torch.cat(features, dim=0)

    def _prem_encode_image(self, pixel_values, image_grid_thw):
        features = self.get_image_features(pixel_values, image_grid_thw)
        return torch.cat(features, dim=0)

    def build_prem_memory(
        self,
        num_slots: int = 4,
        alpha: float = 1.0,
        mem_dim: int = 128,
        disable_anti_distractor: bool = False,
        disable_novelty: bool = False,
        disable_stability: bool = False,
        disable_evidence_gate_write: bool = False,
        uniform_write_route: bool = False,
        num_layer_groups: int = 1,
    ) -> PReMAttentionMemory:
        if self.prem_memory is None:
            mem = PReMAttentionMemory(
                hidden_size=self.config.text_config.hidden_size,
                num_slots=num_slots,
                key_dim=mem_dim,
                val_dim=mem_dim,
                alpha=alpha,
                disable_anti_distractor=disable_anti_distractor,
                disable_novelty=disable_novelty,
                disable_stability=disable_stability,
                disable_evidence_gate_write=disable_evidence_gate_write,
                uniform_write_route=uniform_write_route,
                num_layer_groups=num_layer_groups,
            )
            ref = self.lm_head.weight
            mem = mem.to(device=ref.device, dtype=torch.float32)
            self.prem_memory = mem
        else:
            expected = (int(num_slots), int(mem_dim), int(mem_dim), int(num_layer_groups))
            actual = (
                self.prem_memory.num_slots,
                self.prem_memory.key_dim,
                self.prem_memory.val_dim,
                self.prem_memory.num_layer_groups,
            )
            if actual != expected:
                raise ValueError(f"PREM memory is already initialized as {actual}, requested {expected}")
            self.prem_memory.disable_anti_distractor = bool(disable_anti_distractor)
            self.prem_memory.disable_novelty = bool(disable_novelty)
            self.prem_memory.disable_stability = bool(disable_stability)
            self.prem_memory.disable_evidence_gate_write = bool(disable_evidence_gate_write)
            self.prem_memory.uniform_write_route = bool(uniform_write_route)
        return self.prem_memory

    @staticmethod
    def _prefix_text_mask(text_mask: torch.Tensor, prompt_lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if prompt_lengths is None:
            return text_mask
        prompt_lengths = torch.as_tensor(prompt_lengths, device=text_mask.device).long().view(-1)
        positions = torch.arange(text_mask.shape[1], device=text_mask.device).unsqueeze(0)
        return text_mask & (positions < prompt_lengths.unsqueeze(1))

    @staticmethod
    def _downsample_tokens(seq: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if max_tokens is None or max_tokens <= 0 or seq.shape[0] <= max_tokens:
            return seq
        idx = torch.linspace(0, seq.shape[0] - 1, max_tokens, device=seq.device).round().long()
        return seq[idx]

    def group_video_embeds_by_temporal_step(
        self,
        video_embeds: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Group all Qwen visual tokens by temporal step without pooling them."""
        if video_embeds.ndim != 2:
            raise ValueError(f"video_embeds must have shape [tokens, hidden], got {tuple(video_embeds.shape)}")
        if video_grid_thw is None or video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
            shape = None if video_grid_thw is None else tuple(video_grid_thw.shape)
            raise ValueError(f"video_grid_thw must have shape [videos, 3], got {shape}")

        merge_size = int(self.config.vision_config.spatial_merge_size)
        offset = 0
        grouped = []
        for grid in video_grid_thw.detach().cpu():
            time_steps, height, width = (int(value) for value in grid.tolist())
            if time_steps <= 0 or height <= 0 or width <= 0:
                raise ValueError(f"Invalid video grid: {(time_steps, height, width)}")
            if height % merge_size or width % merge_size:
                raise ValueError(
                    f"Video grid {(time_steps, height, width)} is not divisible by spatial merge size {merge_size}"
                )
            spatial_height = height // merge_size
            spatial_width = width // merge_size
            tokens_per_step = spatial_height * spatial_width
            token_count = time_steps * tokens_per_step
            end = offset + token_count
            if end > video_embeds.shape[0]:
                raise ValueError(
                    f"Video grid requires {end} embeddings, but only {video_embeds.shape[0]} are available"
                )
            grouped.append(video_embeds[offset:end].reshape(time_steps, tokens_per_step, -1))
            offset = end
        if offset != video_embeds.shape[0]:
            raise ValueError(
                f"Unused video embeddings after temporal grouping: consumed={offset}, total={video_embeds.shape[0]}"
            )
        return grouped

    def temporal_pool_video_embeds(
        self,
        video_embeds: torch.Tensor,
        video_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce each visual temporal step to one writer token."""
        grouped = self.group_video_embeds_by_temporal_step(video_embeds, video_grid_thw)
        if len(grouped) != 1:
            raise ValueError(f"Expected one video while building PREM state, got {len(grouped)}")
        return grouped[0].mean(dim=1)

    def stream_reset(
        self,
        batch_size: int = 1,
        *,
        prem_num_slots: int = 4,
        prem_mem_dim: int = 128,
        prem_layer_groups: int = 1,
        prem_alpha: float = 1.0,
        prem_disable_anti_distractor: bool = False,
        prem_disable_novelty: bool = False,
        prem_disable_stability: bool = False,
        prem_disable_evidence_gate_write: bool = False,
        prem_uniform_write_route: bool = False,
    ) -> Dict[str, object]:
        """Create a zero-initialized, video-local PREM stream state."""
        mem = self.build_prem_memory(
            num_slots=prem_num_slots,
            alpha=prem_alpha,
            mem_dim=prem_mem_dim,
            num_layer_groups=prem_layer_groups,
            disable_anti_distractor=prem_disable_anti_distractor,
            disable_novelty=prem_disable_novelty,
            disable_stability=prem_disable_stability,
            disable_evidence_gate_write=prem_disable_evidence_gate_write,
            uniform_write_route=prem_uniform_write_route,
        )
        state, confidence = mem.reset_stream(
            int(batch_size),
            mem.query_proj.weight.device,
            mem.query_proj.weight.dtype,
        )
        return {
            "memory": state,
            "confidence": confidence,
            "num_updates": 0,
            "stats": {"num_stream_updates": 0},
            "config": {
                "prem_num_slots": int(prem_num_slots),
                "prem_mem_dim": int(prem_mem_dim),
                "prem_layer_groups": int(prem_layer_groups),
                "prem_alpha": float(prem_alpha),
                "prem_disable_anti_distractor": bool(prem_disable_anti_distractor),
                "prem_disable_novelty": bool(prem_disable_novelty),
                "prem_disable_stability": bool(prem_disable_stability),
                "prem_disable_evidence_gate_write": bool(prem_disable_evidence_gate_write),
                "prem_uniform_write_route": bool(prem_uniform_write_route),
            },
        }

    def stream_update(
        self,
        stream_state: Dict[str, object],
        frame_embeds: torch.Tensor,
        *,
        update_block_size: int = 32,
        collect_stats: bool = True,
    ) -> Dict[str, object]:
        """Update persistent memory from question-free frame-level embeddings."""
        if not isinstance(stream_state, dict) or "memory" not in stream_state or "confidence" not in stream_state:
            raise TypeError("stream_state must be the dictionary returned by stream_reset()")
        memory = stream_state["memory"]
        confidence = stream_state["confidence"]
        if not torch.is_tensor(memory) or not torch.is_tensor(confidence):
            raise TypeError("stream_state memory and confidence must be tensors")
        config = dict(stream_state.get("config", {}))
        if frame_embeds.ndim == 2:
            frame_embeds = frame_embeds.unsqueeze(0).unsqueeze(2)
        elif frame_embeds.ndim == 3:
            frame_embeds = frame_embeds.unsqueeze(0)
        if frame_embeds.ndim != 4:
            raise ValueError(
                "Stream embeddings must have shape [time, spatial_tokens, hidden] or "
                f"[batch, time, spatial_tokens, hidden], got {tuple(frame_embeds.shape)}"
            )
        if frame_embeds.shape[0] != memory.shape[0]:
            raise ValueError(
                f"Stream batch mismatch: frame_embeds={frame_embeds.shape[0]} state={memory.shape[0]}"
            )

        mem = self.build_prem_memory(
            num_slots=int(config.get("prem_num_slots", memory.shape[1])),
            alpha=float(config.get("prem_alpha", 1.0)),
            mem_dim=int(config.get("prem_mem_dim", memory.shape[-1])),
            num_layer_groups=int(config.get("prem_layer_groups", 1)),
            disable_anti_distractor=bool(config.get("prem_disable_anti_distractor", False)),
            disable_novelty=bool(config.get("prem_disable_novelty", False)),
            disable_stability=bool(config.get("prem_disable_stability", False)),
            disable_evidence_gate_write=bool(config.get("prem_disable_evidence_gate_write", False)),
            uniform_write_route=bool(config.get("prem_uniform_write_route", False)),
        )
        mem_dtype = mem.query_proj.weight.dtype
        raw_spatial_tokens = int(frame_embeds.shape[2])
        temporal_tokens = frame_embeds.mean(dim=2)
        conditioned, salience = mem.condition_stream_chunk(temporal_tokens.to(mem_dtype))
        new_memory, new_confidence, mean_write_gate = mem.stream_write_sequence(
            conditioned,
            state=memory.to(device=conditioned.device, dtype=mem_dtype),
            confidence=confidence.to(device=conditioned.device, dtype=mem_dtype),
            chunk_tokens=update_block_size,
            salience=salience,
        )
        write_gate_per_slot = getattr(mem, "_last_gate_per_slot", None)
        previous_updates = int(stream_state.get("num_updates", 0))
        current_updates = int(frame_embeds.shape[1])
        stats = dict(stream_state.get("stats", {}))
        stats.update({
            "num_stream_updates": previous_updates + current_updates,
            "last_update_steps": current_updates,
            "forget_unit": "temporal_step",
            "stream_writer_mode": "temporal_mean_per_step",
            "stream_visual_tokens_per_step": raw_spatial_tokens,
            "stream_memory_tokens_per_step": 1,
        })
        if collect_stats:
            stats.update({
                "mean_write_gate": float(mean_write_gate.detach().cpu()),
                "mean_memory_norm": float(new_memory.float().norm(dim=(-2, -1)).mean().detach().cpu()),
                "mean_confidence": float(new_confidence.mean().detach().cpu()),
                "forget_multiplier": float(torch.sigmoid(mem.forget_logit).mean().detach().cpu()),
            })
            if torch.is_tensor(write_gate_per_slot):
                stats.update({
                    "mean_write_gate_per_slot": write_gate_per_slot.mean(dim=0).float().cpu().tolist(),
                    "mean_confidence_per_slot": new_confidence.mean(dim=0).float().cpu().tolist(),
                })
        result = {
            **stream_state,
            "memory": new_memory,
            "confidence": new_confidence,
            "num_updates": previous_updates + current_updates,
            "stats": stats,
            **({
                "_last_mean_write_gate": mean_write_gate.detach(),
                "_last_write_gate_per_slot": write_gate_per_slot,
            } if not collect_stats else {}),
        }
        if collect_stats:
            result.pop("_last_mean_write_gate", None)
        return result

    def stream_answer(
        self,
        stream_state: Dict[str, object],
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ):
        """Generate from a bounded visual buffer plus persistent memory state."""
        if generation_kwargs.get("prem_max_memory_tokens") is not None:
            raise ValueError("stream_answer() does not use the full-video prem_max_memory_tokens option")
        forbidden = {"inputs_embeds", "prem_stream_state"}
        present = sorted(key for key in forbidden if generation_kwargs.get(key) is not None)
        if present:
            raise ValueError(f"stream_answer() does not accept an alternate state/input embedding: {present}")
        if input_ids is None:
            raise ValueError("stream_answer() requires text input_ids")
        has_visual_pixels = (
            generation_kwargs.get("pixel_values") is not None
            or generation_kwargs.get("pixel_values_videos") is not None
        )
        has_visual_tokens = bool(
            ((input_ids == self.config.video_token_id) | (input_ids == self.config.image_token_id)).any()
        )
        if has_visual_tokens != has_visual_pixels:
            raise ValueError("stream_answer() visual placeholders and pixel inputs must be provided together")
        if not isinstance(stream_state, dict) or not torch.is_tensor(stream_state.get("memory")):
            raise TypeError("stream_state must be the dictionary returned by stream_reset()/stream_update()")

        for key, value in dict(stream_state.get("config", {})).items():
            generation_kwargs.setdefault(key, value)
        generation_kwargs.setdefault("prem_modulation", "attention")
        generation_kwargs.setdefault("prem_stream_stats", stream_state.get("stats", {}))
        return self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prem_stream_state=stream_state["memory"],
            **generation_kwargs,
        )

    def build_prem_state_from_video(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        prem_prompt_lengths: Optional[torch.Tensor],
        prem_alpha: float = 1.0,
        prem_num_slots: int = 4,
        prem_mem_dim: int = 128,
        prem_layer_groups: int = 1,
        prem_max_memory_tokens: int = 128,
        prem_pred_weight: float = 0.0,
        prem_pred_tokens: int = 8,
        prem_disable_anti_distractor: bool = False,
        prem_disable_novelty: bool = False,
        prem_disable_stability: bool = False,
        prem_disable_evidence_gate_write: bool = False,
        prem_uniform_write_route: bool = False,
    ):
        """Write full-video evidence to M without running the language decoder."""
        inputs_embeds = self.get_input_embeddings()(input_ids)
        video_embeds = self._prem_encode_video(pixel_values_videos, video_grid_thw)
        video_token_mask = input_ids == self.config.video_token_id
        video_mask = video_token_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            video_mask,
            video_embeds.to(inputs_embeds.device, inputs_embeds.dtype),
        )
        self.prem_last_stream_state = None
        self._build_prem_modulation(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            video_token_mask=video_token_mask,
            video_grid_thw=video_grid_thw,
            prem_alpha=prem_alpha,
            prem_num_slots=prem_num_slots,
            prem_mem_dim=prem_mem_dim,
            prem_layer_groups=prem_layer_groups,
            prem_prompt_lengths=prem_prompt_lengths,
            prem_max_memory_tokens=prem_max_memory_tokens,
            prem_router_gamma=0.0,
            prem_disable_anti_distractor=prem_disable_anti_distractor,
            prem_disable_novelty=prem_disable_novelty,
            prem_disable_stability=prem_disable_stability,
            prem_disable_evidence_gate_write=prem_disable_evidence_gate_write,
            prem_uniform_write_route=prem_uniform_write_route,
            prem_pred_weight=prem_pred_weight,
            prem_pred_tokens=prem_pred_tokens,
            writer_only=True,
        )
        if self.prem_last_stream_state is None:
            raise RuntimeError("Full-video writer did not produce an PREM state")
        aux = self.prem_aux_losses or {}
        pred_loss = aux.get("evidence_prediction", inputs_embeds.new_zeros(()))
        stats = dict(self.prem_last_stats or {})
        self.prem_aux_losses = None
        return self.prem_last_stream_state, pred_loss, stats

    def _prem_text_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: Optional[torch.Tensor],
    ):
        text_mask = torch.ones_like(input_ids, dtype=torch.bool)
        text_mask &= input_ids != self.config.video_token_id
        text_mask &= input_ids != self.config.image_token_id
        if attention_mask is not None and attention_mask.ndim == 2:
            row_attention_mask = attention_mask.bool().to(text_mask.device)
            if row_attention_mask.shape[1] != text_mask.shape[1]:
                row_attention_mask = row_attention_mask[:, -text_mask.shape[1]:]
            text_mask &= row_attention_mask
        return self._prefix_text_mask(text_mask, prompt_lengths)
    def _prem_valid_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prompt_lengths: Optional[torch.Tensor],
    ):
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
        valid_mask &= input_ids != self.config.video_token_id
        valid_mask &= input_ids != self.config.image_token_id
        if attention_mask is not None and attention_mask.ndim == 2:
            row_attention_mask = attention_mask.bool().to(valid_mask.device)
            if row_attention_mask.shape[1] != valid_mask.shape[1]:
                row_attention_mask = row_attention_mask[:, -valid_mask.shape[1]:]
            valid_mask &= row_attention_mask
        if prompt_lengths is not None:
            prompt_lengths = torch.as_tensor(prompt_lengths, device=valid_mask.device).long().view(-1)
            positions = torch.arange(valid_mask.shape[1], device=valid_mask.device).unsqueeze(0)
            valid_mask &= positions < prompt_lengths.unsqueeze(1)
        return valid_mask

    def _build_prem_modulation(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        video_token_mask: torch.Tensor,
        video_grid_thw: torch.Tensor,
        prem_alpha: float,
        prem_num_slots: int,
        prem_mem_dim: int,
        prem_layer_groups: int,
        prem_prompt_lengths: Optional[torch.Tensor],
        prem_max_memory_tokens: int,
        prem_router_gamma: float,
        prem_disable_anti_distractor: bool,
        prem_disable_novelty: bool,
        prem_disable_stability: bool,
        prem_disable_evidence_gate_write: bool,
        prem_uniform_write_route: bool,
        prem_pred_weight: float = 0.0,
        prem_pred_tokens: int = 8,
        writer_only: bool = False,
    ):
        """Video-only write + question-conditioned read modulation."""
        mem = self.build_prem_memory(
            num_slots=prem_num_slots,
            alpha=1.0,
            mem_dim=prem_mem_dim,
            num_layer_groups=prem_layer_groups,
            disable_anti_distractor=prem_disable_anti_distractor,
            disable_novelty=prem_disable_novelty,
            disable_stability=prem_disable_stability,
            disable_evidence_gate_write=prem_disable_evidence_gate_write,
            uniform_write_route=prem_uniform_write_route,
        )
        text_mask = self._prem_text_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prem_prompt_lengths,
        )
        valid_mask = self._prem_valid_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prem_prompt_lengths,
        )

        query_correction = None if writer_only else torch.zeros_like(inputs_embeds)
        output_correction = None if writer_only else torch.zeros_like(inputs_embeds)
        router_loss = None
        fired = 0
        pred_loss = None
        last_stats = None
        context_states = []
        context_queries = []

        for batch_idx in range(input_ids.shape[0]):
            row_video = video_token_mask[batch_idx]
            row_text = text_mask[batch_idx]
            row_valid = valid_mask[batch_idx]
            if (
                not bool(row_video.any())
                or not bool(row_text.any())
                or (not writer_only and not bool(row_valid.any()))
            ):
                continue

            video_seq = inputs_embeds[batch_idx, row_video]
            temporal_seq = self.temporal_pool_video_embeds(
                video_seq,
                video_grid_thw[batch_idx: batch_idx + 1],
            )
            temporal_seq = self._downsample_tokens(temporal_seq, prem_max_memory_tokens)
            query = inputs_embeds[batch_idx, row_text].mean(dim=0, keepdim=True)
            mem_dtype = mem.query_proj.weight.dtype
            # video-only write
            conditioned, salience = mem.condition_stream_chunk(temporal_seq.unsqueeze(0).to(mem_dtype))
            state, confidence = mem.reset_stream(1, conditioned.device, mem_dtype)
            state, confidence, mean_write_gate = mem.stream_write_sequence(
                conditioned, salience, state=state, confidence=confidence,
            )

            last_stats = {
                "mean_salience": float(salience.mean().detach().cpu()),
                "mean_confidence": float(confidence.mean().detach().cpu()),
                "mean_write_gate": float(mean_write_gate.detach().cpu()),
                "mean_memory_norm": float(state.norm(dim=(-2, -1)).mean().detach().cpu()),
                "stream_writer_mode": "temporal_mean_per_step",
                "stream_memory_tokens_per_step": 1,
                "stream_num_temporal_steps": int(temporal_seq.shape[0]),
                "disable_anti_distractor": self.prem_memory.disable_anti_distractor,
                "disable_novelty": self.prem_memory.disable_novelty,
                "disable_stability": self.prem_memory.disable_stability,
                "disable_evidence_gate_write": self.prem_memory.disable_evidence_gate_write,
                "uniform_write_route": self.prem_memory.uniform_write_route,
            }
            context_states.append(state)
            fired += 1

            if prem_pred_weight > 0:
                pred_mask = mem._top_evidence_span_mask(salience, prem_pred_tokens)
                pred_state, pred_confidence = mem.reset_stream(1, conditioned.device, mem_dtype)
                pred_state, _, _ = mem.stream_write_sequence(
                    conditioned,
                    salience,
                    state=pred_state,
                    confidence=pred_confidence,
                    write_block_mask=pred_mask,
                )
                pred_loss_val, _ = mem.evidence_prediction_loss(
                    pred_state, conditioned, query.to(mem_dtype), pred_mask,
                )
                pred_loss = pred_loss_val if pred_loss is None else pred_loss + pred_loss_val

            if writer_only:
                continue

            # question-conditioned read
            target_seq = inputs_embeds[batch_idx, row_valid]
            out = mem.stream_read_per_position(
                state,
                target_seq.unsqueeze(0).to(mem_dtype),
                query.to(mem_dtype),
                alpha_override=prem_alpha,
            )
            query_steer, output_steer, _, rho, per_slot, _, _ = out
            query_correction[batch_idx, row_valid] = query_steer[0].to(query_correction.dtype)
            output_correction[batch_idx, row_valid] = output_steer[0].to(output_correction.dtype)

            avg_rho = rho.reshape(-1, mem.num_slots).mean(dim=0)
            mean_read_contribution = per_slot.float().norm(dim=-1).reshape(-1, mem.num_slots).mean(dim=0)
            uniform = torch.full_like(avg_rho, 1.0 / mem.num_slots)
            router_balance = ((avg_rho - uniform) ** 2).sum()
            router_loss = router_balance if router_loss is None else router_loss + router_balance
            last_stats.update({
                "router_entropy": float((-(rho * rho.clamp_min(1e-8).log()).sum(-1)).mean().detach().cpu()),
                "max_router_weight": float(rho.max(dim=-1).values.mean().detach().cpu()),
                "mean_query_steer_norm": float(query_steer.norm(dim=-1).mean().detach().cpu()),
                "mean_output_steer_norm": float(output_steer.norm(dim=-1).mean().detach().cpu()),
                "mean_rho": avg_rho.detach().float().cpu().tolist(),
                "mean_read_contribution_per_slot": mean_read_contribution.detach().cpu().tolist(),
            })
            context_queries.append(query.to(mem_dtype))

        if fired > 0:
            router_balance = inputs_embeds.new_zeros(()) if writer_only else router_loss / fired
            self.prem_aux_losses = {
                "router_balance": router_balance,
                "evidence_prediction": pred_loss / fired if prem_pred_weight > 0 else inputs_embeds.new_zeros(()),
                "router_gamma": float(prem_router_gamma),
                "pred_weight": float(prem_pred_weight),
            }
            self.prem_last_stats = {
                **last_stats,
                "modulation": "attention",
                "alpha": float((mem._alpha_scale(prem_alpha) * mem.num_slots).detach().cpu()),
                "max_memory_tokens": int(prem_max_memory_tokens),
                "placement": "unified_per_position",
                "disable_anti_distractor": bool(prem_disable_anti_distractor),
                "disable_novelty": bool(prem_disable_novelty),
                "disable_stability": bool(prem_disable_stability),
                "disable_evidence_gate_write": bool(prem_disable_evidence_gate_write),
                "uniform_write_route": bool(prem_uniform_write_route),
                "pred_weight": float(prem_pred_weight),
                "pred_tokens": int(prem_pred_tokens),
                "router_balance_loss": float(router_balance.detach().cpu()),
                "evidence_prediction_loss": float((pred_loss / fired).detach().cpu()) if prem_pred_weight > 0 else None,
            }
        else:
            self.prem_aux_losses = None
            self.prem_last_stats = None

        context = None
        if fired == input_ids.shape[0]:
            stream_state = torch.cat(context_states, dim=0)
            self.prem_last_stream_state = stream_state
            if not writer_only:
                context = {
                    "memory": mem,
                    "state": stream_state,
                    "query": torch.cat(context_queries, dim=0),
                    "valid_mask": valid_mask,
                    "alpha": prem_alpha,
                }
        return query_correction, output_correction, fired, context

    def _build_prem_modulation_from_stream_state(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prem_stream_state: torch.Tensor,
        prem_alpha: float,
        prem_num_slots: int,
        prem_mem_dim: int,
        prem_layer_groups: int,
        prem_prompt_lengths: Optional[torch.Tensor],
        prem_router_gamma: float,
        prem_stream_stats: Optional[dict],
        prem_disable_anti_distractor: bool,
        prem_disable_novelty: bool,
        prem_disable_stability: bool,
        prem_disable_evidence_gate_write: bool,
        prem_uniform_write_route: bool,
    ):
        """question-conditioned read from a pre-built stream state."""
        mem = self.build_prem_memory(
            num_slots=prem_num_slots,
            alpha=1.0,
            mem_dim=prem_mem_dim,
            num_layer_groups=prem_layer_groups,
            disable_anti_distractor=prem_disable_anti_distractor,
            disable_novelty=prem_disable_novelty,
            disable_stability=prem_disable_stability,
            disable_evidence_gate_write=prem_disable_evidence_gate_write,
            uniform_write_route=prem_uniform_write_route,
        )
        text_mask = self._prem_text_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prem_prompt_lengths,
        )
        valid_mask = self._prem_valid_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prem_prompt_lengths,
        )
        query_correction = torch.zeros_like(inputs_embeds)
        output_correction = torch.zeros_like(inputs_embeds)
        router_loss = None
        fired = 0
        last_stats = None
        context_states = []
        context_queries = []
        mem_dtype = mem.query_proj.weight.dtype
        state = prem_stream_state.to(device=mem.query_proj.weight.device, dtype=mem_dtype)

        for batch_idx in range(input_ids.shape[0]):
            row_text = text_mask[batch_idx]
            row_valid = valid_mask[batch_idx]
            if not bool(row_valid.any()) or not bool(row_text.any()):
                continue
            query = inputs_embeds[batch_idx, row_text].mean(dim=0, keepdim=True)
            target_seq = inputs_embeds[batch_idx, row_valid]
            state_row = state[batch_idx: batch_idx + 1] if state.shape[0] == input_ids.shape[0] else state[:1]
            out = mem.stream_read_per_position(
                state_row,
                target_seq.unsqueeze(0).to(mem_dtype),
                query.to(mem_dtype),
                alpha_override=prem_alpha,
            )
            query_steer, output_steer, _, rho, per_slot, _, _ = out
            query_correction[batch_idx, row_valid] = query_steer[0].to(query_correction.dtype)
            output_correction[batch_idx, row_valid] = output_steer[0].to(output_correction.dtype)
            avg_rho = rho.reshape(-1, mem.num_slots).mean(dim=0)
            mean_read_contribution = per_slot.float().norm(dim=-1).reshape(-1, mem.num_slots).mean(dim=0)
            uniform = torch.full_like(avg_rho, 1.0 / mem.num_slots)
            router_balance = ((avg_rho - uniform) ** 2).sum()
            router_loss = router_balance if router_loss is None else router_loss + router_balance
            last_stats = {
                "router_entropy": float((-(rho * rho.clamp_min(1e-8).log()).sum(-1)).mean().detach().cpu()),
                "max_router_weight": float(rho.max(dim=-1).values.mean().detach().cpu()),
                "mean_query_steer_norm": float(query_steer.norm(dim=-1).mean().detach().cpu()),
                "mean_output_steer_norm": float(output_steer.norm(dim=-1).mean().detach().cpu()),
                "mean_rho": avg_rho.detach().float().cpu().tolist(),
                "mean_read_contribution_per_slot": mean_read_contribution.detach().cpu().tolist(),
                **(prem_stream_stats or {}),
            }
            context_states.append(state_row)
            context_queries.append(query.to(mem_dtype))
            fired += 1

        if fired > 0:
            self.prem_aux_losses = {
                "router_balance": router_loss / fired,
                "router_gamma": float(prem_router_gamma),
            }
            self.prem_last_stats = {
                **last_stats,
                "modulation": "attention",
                "alpha": float((mem._alpha_scale(prem_alpha) * mem.num_slots).detach().cpu()),
                "placement": "stream_state_per_position",
                "streaming_prem": True,
                "router_balance_loss": float((router_loss / fired).detach().cpu()),
            }
        else:
            self.prem_aux_losses = None
            self.prem_last_stats = None
        context = None
        if fired == input_ids.shape[0]:
            context = {
                "memory": mem,
                "state": torch.cat(context_states, dim=0),
                "query": torch.cat(context_queries, dim=0),
                "valid_mask": valid_mask,
                "alpha": prem_alpha,
            }
        return query_correction, output_correction, fired, context


    
    @staticmethod
    def _parse_modulation_mode(modulation: str) -> str:
        """Map the public modulation name to its attention projection target."""
        modes = {
            "attention": "qo",
            "attention_kv": "kv",
            "attention_k": "k",
            "attention_v": "v",
        }
        try:
            return modes[modulation]
        except KeyError as exc:
            raise ValueError(
                f"Unknown prem_modulation: {modulation}. Expected one of {tuple(modes)}."
            ) from exc
    
    def _prem_hook_stack(self, query_correction: torch.Tensor, output_correction: torch.Tensor, modulation_mode: str = "qo"):
        """Register forward hooks for attention modulation.

        Args:
            query_correction: correction for q_proj (qo mode) or k_proj (kv mode)
            output_correction: correction for o_proj (qo mode) or v_proj (kv mode)
            modulation_mode: "qo" (default, modulates query & output) or "kv" (modulates key & value)
        """
        stack = ExitStack()

        def make_q_hook():
            def hook(module, module_inputs, module_output):
                hidden_states = module_inputs[0]
                correction = query_correction.to(device=hidden_states.device, dtype=hidden_states.dtype)
                if correction.shape[-1] != module_output.shape[-1]:
                    correction = torch.nn.functional.linear(correction, module.weight)
                return module_output + correction

            return hook

        def make_o_hook():
            def hook(module, module_inputs, module_output):
                correction = output_correction.to(device=module_output.device, dtype=module_output.dtype)
                if correction.shape[-1] != module_output.shape[-1]:
                    correction = torch.nn.functional.linear(correction, module.weight)
                return module_output + correction

            return hook

        targets = {
            "qo": ("q_proj", "o_proj"),
            "kv": ("k_proj", "v_proj"),
            "k": ("k_proj", None),
            "v": (None, "v_proj"),
        }
        if modulation_mode not in targets:
            raise ValueError(f"Unknown modulation_mode: {modulation_mode}. Use one of {tuple(targets)}.")
        proj_a, proj_b = targets[modulation_mode]

        for layer in self._prem_text_model.layers:
            if proj_a is not None:
                a_handle = getattr(layer.self_attn, proj_a).register_forward_hook(make_q_hook())
                stack.callback(a_handle.remove)
            if proj_b is not None:
                b_handle = getattr(layer.self_attn, proj_b).register_forward_hook(make_o_hook())
                stack.callback(b_handle.remove)
        return stack

    def _prem_contextual_hook_stack(self, context: dict, modulation_mode: str = "qo"):
        """Build per-layer-group hooks that re-read shared state.

        Args:
            context: dict with memory, state, query, valid_mask, alpha
            modulation_mode: "qo" (default, modulates query & output) or "kv" (modulates key & value)
        """
        mem = context["memory"]
        state = context["state"]
        query = context["query"]
        valid_mask = context["valid_mask"]
        num_groups = mem.num_layer_groups
        num_layers = len(self._prem_text_model.layers)
        stack = ExitStack()

        targets = {
            "qo": ("q_proj", "o_proj"),
            "kv": ("k_proj", "v_proj"),
            "k": ("k_proj", None),
            "v": (None, "v_proj"),
        }
        if modulation_mode not in targets:
            raise ValueError(f"Unknown modulation_mode: {modulation_mode}. Use one of {tuple(targets)}.")
        proj_a, proj_b = targets[modulation_mode]

        def contextual_corrections(module, module_inputs, module_output, layer_group: int):
            hidden_states = module_inputs[0]
            mem_dtype = mem.query_proj.weight.dtype
            query_steer, output_steer, _, _, _, _, _ = mem.contextual_read_per_position(
                state.to(device=hidden_states.device, dtype=mem_dtype),
                hidden_states.to(dtype=mem_dtype),
                query.to(device=hidden_states.device, dtype=mem_dtype),
                layer_group=layer_group,
                alpha_override=context["alpha"],
            )
            if valid_mask.shape == hidden_states.shape[:2]:
                mask = valid_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
                query_steer = query_steer * mask
                output_steer = output_steer * mask
            return query_steer, output_steer

        def make_q_hook(layer_group: int, pending_output: list[torch.Tensor] | None):
            def hook(module, module_inputs, module_output):
                query_steer, output_steer = contextual_corrections(
                    module, module_inputs, module_output, layer_group
                )
                query_steer = query_steer.to(device=module_output.device, dtype=module_output.dtype)
                if query_steer.shape[-1] != module_output.shape[-1]:
                    query_steer = torch.nn.functional.linear(query_steer, module.weight)
                if pending_output is not None:
                    pending_output.append(output_steer.to(device=module_output.device, dtype=module_output.dtype))
                return module_output + query_steer

            return hook

        def make_contextual_o_hook(layer_group: int):
            def hook(module, module_inputs, module_output):
                _, correction = contextual_corrections(module, module_inputs, module_output, layer_group)
                correction = correction.to(device=module_output.device, dtype=module_output.dtype)
                if correction.shape[-1] != module_output.shape[-1]:
                    correction = torch.nn.functional.linear(correction, module.weight)
                return module_output + correction

            return hook

        def make_o_hook(pending_output: list[torch.Tensor]):
            def hook(module, module_inputs, module_output):
                if not pending_output:
                    raise RuntimeError("PREM contextual o_proj hook fired before q_proj hook")
                correction = pending_output.pop()
                if correction.shape[-1] != module_output.shape[-1]:
                    correction = torch.nn.functional.linear(correction, module.weight)
                return module_output + correction.to(device=module_output.device, dtype=module_output.dtype)

            return hook

        for layer_idx, layer in enumerate(self._prem_text_model.layers):
            layer_group = min(num_groups - 1, layer_idx * num_groups // max(1, num_layers))
            pending_output: list[torch.Tensor] = []
            if proj_a is not None:
                queued_output = pending_output if proj_b is not None else None
                a_handle = getattr(layer.self_attn, proj_a).register_forward_hook(
                    make_q_hook(layer_group, queued_output)
                )
                stack.callback(a_handle.remove)
            if proj_b is not None:
                output_hook = (
                    make_o_hook(pending_output)
                    if proj_a is not None
                    else make_contextual_o_hook(layer_group)
                )
                b_handle = getattr(layer.self_attn, proj_b).register_forward_hook(output_hook)
                stack.callback(b_handle.remove)
        return stack

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        prem_modulation: Optional[str] = None,
        prem_alpha: float = 1.0,
        prem_num_slots: int = 4,
        prem_mem_dim: int = 128,
        prem_layer_groups: int = 1,
        prem_prompt_lengths: Optional[torch.Tensor] = None,
        prem_max_memory_tokens: int = 128,
        prem_router_gamma: float = 0.0,
        prem_disable_anti_distractor: bool = False,
        prem_disable_novelty: bool = False,
        prem_disable_stability: bool = False,
        prem_disable_evidence_gate_write: bool = False,
        prem_uniform_write_route: bool = False,
        prem_pred_weight: float = 0.0,
        prem_pred_tokens: int = 8,
        prem_stream_state: Optional[torch.Tensor] = None,
        prem_stream_stats: Optional[dict] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hook_stack = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self._prem_encode_image(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                self.prem_last_stats = None
                self.prem_aux_losses = None
                if prem_stream_state is None:
                    self.prem_last_stream_state = None
                video_embeds = self._prem_encode_video(pixel_values_videos, video_grid_thw)
                video_token_mask = input_ids == self.config.video_token_id
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                _, video_mask = self.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    video_features=video_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                if prem_modulation and prem_modulation != "none":
                    modulation_mode = self._parse_modulation_mode(prem_modulation)
                    if prem_stream_state is not None:
                        query_corr, output_corr, fired, contextual_context = self._build_prem_modulation_from_stream_state(
                            inputs_embeds=inputs_embeds,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            prem_stream_state=prem_stream_state,
                            prem_alpha=prem_alpha,
                            prem_num_slots=prem_num_slots,
                            prem_mem_dim=prem_mem_dim,
                            prem_layer_groups=prem_layer_groups,
                            prem_prompt_lengths=prem_prompt_lengths,
                            prem_router_gamma=prem_router_gamma,
                            prem_stream_stats=prem_stream_stats,
                            prem_disable_anti_distractor=prem_disable_anti_distractor,
                            prem_disable_novelty=prem_disable_novelty,
                            prem_disable_stability=prem_disable_stability,
                            prem_disable_evidence_gate_write=prem_disable_evidence_gate_write,
                            prem_uniform_write_route=prem_uniform_write_route,
                        )
                    else:
                        query_corr, output_corr, fired, contextual_context = self._build_prem_modulation(
                            inputs_embeds=inputs_embeds,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            video_token_mask=video_token_mask,
                            video_grid_thw=video_grid_thw,
                            prem_alpha=prem_alpha,
                            prem_num_slots=prem_num_slots,
                            prem_mem_dim=prem_mem_dim,
                            prem_layer_groups=prem_layer_groups,
                            prem_prompt_lengths=prem_prompt_lengths,
                            prem_max_memory_tokens=prem_max_memory_tokens,
                            prem_router_gamma=prem_router_gamma,
                            prem_disable_anti_distractor=prem_disable_anti_distractor,
                            prem_disable_novelty=prem_disable_novelty,
                            prem_disable_stability=prem_disable_stability,
                            prem_disable_evidence_gate_write=prem_disable_evidence_gate_write,
                            prem_uniform_write_route=prem_uniform_write_route,
                            prem_pred_weight=prem_pred_weight,
                            prem_pred_tokens=prem_pred_tokens,
                        )
                    if fired > 0:
                        hook_stack = (
                            self._prem_contextual_hook_stack(contextual_context, modulation_mode=modulation_mode)
                            if prem_layer_groups > 1 and contextual_context is not None
                            else self._prem_hook_stack(query_corr, output_corr, modulation_mode=modulation_mode)
                        )
            elif prem_stream_state is not None and prem_modulation and prem_modulation != "none":
                self.prem_last_stats = None
                self.prem_aux_losses = None
                modulation_mode = self._parse_modulation_mode(prem_modulation)
                query_corr, output_corr, fired, contextual_context = self._build_prem_modulation_from_stream_state(
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prem_stream_state=prem_stream_state,
                    prem_alpha=prem_alpha,
                    prem_num_slots=prem_num_slots,
                    prem_mem_dim=prem_mem_dim,
                    prem_layer_groups=prem_layer_groups,
                    prem_prompt_lengths=prem_prompt_lengths,
                    prem_router_gamma=prem_router_gamma,
                    prem_stream_stats=prem_stream_stats,
                    prem_disable_anti_distractor=prem_disable_anti_distractor,
                    prem_disable_novelty=prem_disable_novelty,
                    prem_disable_stability=prem_disable_stability,
                    prem_disable_evidence_gate_write=prem_disable_evidence_gate_write,
                    prem_uniform_write_route=prem_uniform_write_route,
                )
                if fired > 0:
                    hook_stack = (
                        self._prem_contextual_hook_stack(contextual_context, modulation_mode=modulation_mode)
                        if prem_layer_groups > 1 and contextual_context is not None
                        else self._prem_hook_stack(query_corr, output_corr, modulation_mode=modulation_mode)
                    )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        def run_backbone():
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                second_per_grid_ts=second_per_grid_ts,
            )

        outputs = run_backbone() if hook_stack is None else None
        if hook_stack is not None:
            with hook_stack:
                outputs = run_backbone()

        hidden_states = outputs[0]
        loss = None
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            valid_labels = shift_labels.ne(-100)
            loss_fct = CrossEntropyLoss()
            if bool(valid_labels.any()):
                selected_hidden = hidden_states[..., :-1, :][valid_labels].contiguous()
                selected_logits = self.lm_head(selected_hidden).float()
                selected_labels = shift_labels[valid_labels].to(selected_logits.device)
                loss = loss_fct(selected_logits, selected_labels)
            else:
                loss = hidden_states.sum() * 0.0
            aux = self.prem_aux_losses
            if loss is not None and aux is not None:
                loss = loss + aux["router_gamma"] * aux["router_balance"].to(loss.device)
                if aux.get("pred_weight", 0) > 0:
                    loss = loss + aux["pred_weight"] * aux["evidence_prediction"].to(loss.device)
            self.prem_aux_losses = None
            logits = None
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        prem_modulation=None,
        prem_alpha=1.0,
        prem_num_slots=4,
        prem_mem_dim=128,
        prem_layer_groups=1,
        prem_prompt_lengths=None,
        prem_max_memory_tokens=128,
        prem_router_gamma=0.0,
        prem_pred_weight=0.0,
        prem_pred_tokens=8,
        prem_disable_anti_distractor=False,
        prem_disable_novelty=False,
        prem_disable_stability=False,
        prem_disable_evidence_gate_write=False,
        prem_uniform_write_route=False,
        prem_stream_state=None,
        prem_stream_stats=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
        if cache_position is not None and cache_position[0] != 0:
            prem_modulation = None
            prem_stream_state = None
            prem_stream_stats = None

        model_inputs.update(
            {
                "prem_modulation": prem_modulation,
                "prem_alpha": prem_alpha,
                "prem_num_slots": prem_num_slots,
                "prem_mem_dim": prem_mem_dim,
                "prem_layer_groups": prem_layer_groups,
                "prem_prompt_lengths": prem_prompt_lengths,
                "prem_max_memory_tokens": prem_max_memory_tokens,
                "prem_router_gamma": prem_router_gamma,
                "prem_pred_weight": prem_pred_weight,
                "prem_pred_tokens": prem_pred_tokens,
                "prem_disable_anti_distractor": prem_disable_anti_distractor,
                "prem_disable_novelty": prem_disable_novelty,
                "prem_disable_stability": prem_disable_stability,
                "prem_disable_evidence_gate_write": prem_disable_evidence_gate_write,
                "prem_uniform_write_route": prem_uniform_write_route,
                "prem_stream_state": prem_stream_state,
                "prem_stream_stats": prem_stream_stats,
            })
        return model_inputs
