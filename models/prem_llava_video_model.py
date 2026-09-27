"""Associative Side Memory adapter for the official LLaVA-Video Qwen2 backbone."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .prem_memory import PReMAttentionMemory
from scripts.eval.llava_video_contract import LLAVA_VIDEO_POOL_MODE
from scripts.llava_video_utils import (
    materialized_llava_qwen_loader,
    validate_materialized_vision_tower,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LLAVA_SOURCE = REPO_ROOT / "external/baselines/llava_next"
DEFAULT_VISION_TOWER = REPO_ROOT / "ckpt/siglip-so400m-patch14-384"
IGNORE_INDEX = -100


@dataclass
class LlavaVideoResources:
    model: "PReMLlavaVideoForCausalLM"
    tokenizer: object
    image_processor: object
    context_len: int
    default_image_token: str
    image_token_index: int
    conversation_templates: dict
    tokenizer_image_token: object


def load_prem_llava_video(
    model_path: str | Path,
    *,
    vision_tower_path: str | Path = DEFAULT_VISION_TOWER,
    device_map="auto",
    torch_dtype="bfloat16",
    attn_implementation: Optional[str] = None,
) -> LlavaVideoResources:
    """Load the pinned official LLaVA-Video implementation and attach PReM."""
    source = str(LLAVA_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model import llava_arch
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

    vision_tower = Path(vision_tower_path)
    if not (vision_tower / "config.json").exists():
        raise FileNotFoundError(f"Missing LLaVA-Video vision tower: {vision_tower}")
    upstream_builder = llava_arch.build_vision_tower

    def build_local_vision_tower(vision_tower_cfg, **kwargs):
        configured = getattr(
            vision_tower_cfg,
            "mm_vision_tower",
            getattr(vision_tower_cfg, "vision_tower", None),
        )
        if configured == str(vision_tower):
            return SigLipVisionTower(configured, vision_tower_cfg=vision_tower_cfg, **kwargs)
        return upstream_builder(vision_tower_cfg, **kwargs)

    llava_arch.build_vision_tower = build_local_vision_tower
    attn = attn_implementation or (
        "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
    )
    try:
        target_device = (
            device_map.get("", "cuda:0")
            if isinstance(device_map, dict)
            else "cuda:0" if device_map == "auto" else device_map
        )
        with materialized_llava_qwen_loader(LlavaQwenForCausalLM):
            tokenizer, backbone, image_processor, context_len = load_pretrained_model(
                str(model_path),
                None,
                "llava_qwen",
                torch_dtype=(
                    "bfloat16" if torch_dtype == torch.bfloat16 else
                    "float16" if torch_dtype == torch.float16 else
                    torch_dtype
                ),
                device_map="auto",
                attn_implementation=attn,
                overwrite_config={
                    "delay_load": True,
                    "mm_vision_tower": str(vision_tower),
                    "mm_spatial_pool_mode": LLAVA_VIDEO_POOL_MODE,
                },
            )
    finally:
        llava_arch.build_vision_tower = upstream_builder
    backbone.to(device=target_device)
    validate_materialized_vision_tower(backbone)

    if str(backbone.config.mm_spatial_pool_mode) != LLAVA_VIDEO_POOL_MODE:
        raise RuntimeError(
            "Official LLaVA-Video evaluation requires mm_spatial_pool_mode=average"
        )

    return LlavaVideoResources(
        model=PReMLlavaVideoForCausalLM(backbone),
        tokenizer=tokenizer,
        image_processor=image_processor,
        context_len=int(context_len),
        default_image_token=DEFAULT_IMAGE_TOKEN,
        image_token_index=int(IMAGE_TOKEN_INDEX),
        conversation_templates=conv_templates,
        tokenizer_image_token=tokenizer_image_token,
    )


class PReMLlavaVideoForCausalLM(nn.Module):
    """Thin PReM wrapper around the official ``LlavaQwenForCausalLM`` model.

    The vision tower and language model remain frozen. The wrapper exposes the
    same writer/read contract used by the Qwen-VL PReM implementations while
    respecting LLaVA-Video's negative image placeholder and expanded visual
    embedding sequence.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.prem_memory: Optional[PReMAttentionMemory] = None
        self.prem_aux_losses = None
        self.prem_last_stats = None

    @property
    def config(self):
        return self.backbone.config

    @property
    def device(self):
        return self.backbone.device

    @property
    def _prem_text_model(self):
        return self.backbone.model

    @property
    def lm_head(self):
        return self.backbone.lm_head

    def train(self, mode: bool = True):
        # The frozen backbone stays deterministic; only PReM enters train mode.
        self.backbone.eval()
        if self.prem_memory is not None:
            self.prem_memory.train(mode)
        return self

    def eval(self):
        return self.train(False)

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
            self.prem_memory = PReMAttentionMemory(
                hidden_size=int(self.config.hidden_size),
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
            ).to(device=self.lm_head.weight.device, dtype=torch.float32)
        else:
            expected = (int(num_slots), int(mem_dim), int(mem_dim), int(num_layer_groups))
            actual = (
                self.prem_memory.num_slots,
                self.prem_memory.key_dim,
                self.prem_memory.val_dim,
                self.prem_memory.num_layer_groups,
            )
            if actual != expected:
                raise ValueError(f"PReM memory is already initialized as {actual}, requested {expected}")
            self.prem_memory.disable_anti_distractor = bool(disable_anti_distractor)
            self.prem_memory.disable_novelty = bool(disable_novelty)
            self.prem_memory.disable_stability = bool(disable_stability)
            self.prem_memory.disable_evidence_gate_write = bool(disable_evidence_gate_write)
            self.prem_memory.uniform_write_route = bool(uniform_write_route)
        return self.prem_memory

    @staticmethod
    def _downsample_tokens(seq: torch.Tensor, max_tokens: int) -> torch.Tensor:
        if max_tokens is None or max_tokens <= 0 or seq.shape[0] <= max_tokens:
            return seq
        indices = torch.linspace(0, seq.shape[0] - 1, max_tokens, device=seq.device)
        return seq[indices.round().long()]

    def encode_video_frames(self, images: torch.Tensor, *, chunk_frames: int = 0) -> torch.Tensor:
        """Return projected per-frame spatial tokens as ``[time, spatial, hidden]``."""
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError(f"Expected one video tensor, got {len(images)}")
            images = images[0]
        if images.ndim != 4:
            raise ValueError(f"images must have shape [frames, channels, height, width], got {images.shape}")
        step = int(chunk_frames or images.shape[0])
        chunks = []
        for start in range(0, images.shape[0], max(1, step)):
            chunks.append(self.backbone.encode_images(images[start : start + step]))
        return torch.cat(chunks, dim=0)

    def _write_features(
        self,
        frame_features: torch.Tensor,
        *,
        state: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        max_memory_tokens: int = 128,
        update_block_size: int = 32,
    ):
        mem = self.prem_memory
        if mem is None:
            raise RuntimeError("build_prem_memory() must be called before writing")
        mem_dtype = mem.query_proj.weight.dtype
        frame_features = frame_features.to(device=mem.query_proj.weight.device, dtype=mem_dtype)
        temporal_tokens = frame_features.mean(dim=1)
        conditioned, salience = mem.condition_stream_chunk(temporal_tokens.unsqueeze(0))
        conditioned = self._downsample_tokens(
            conditioned[0], max_memory_tokens
        ).unsqueeze(0)
        salience = self._downsample_tokens(salience[0], max_memory_tokens).unsqueeze(0)
        if state is None or confidence is None:
            state, confidence = mem.reset_stream(1, conditioned.device, mem_dtype)
        state, confidence, mean_write_gate = mem.stream_write_sequence(
            conditioned,
            salience=salience,
            state=state,
            confidence=confidence,
            chunk_tokens=update_block_size,
        )
        stats = {
            "num_stream_updates": int(conditioned.shape[1]),
            "mean_salience": float(salience.mean().detach().cpu()),
            "mean_confidence": float(confidence.mean().detach().cpu()),
            "mean_write_gate": float(mean_write_gate.detach().cpu()),
            "mean_memory_norm": float(state.float().norm(dim=(-2, -1)).mean().detach().cpu()),
            "stream_writer_mode": "temporal_mean_per_step",
            "stream_memory_tokens_per_step": 1,
        }
        return state, confidence, conditioned, salience, stats

    def build_prem_state_from_video(
        self,
        *,
        images: torch.Tensor,
        query_embeddings: Optional[torch.Tensor] = None,
        prem_num_slots: int = 4,
        prem_mem_dim: int = 128,
        prem_layer_groups: int = 1,
        prem_alpha: float = 1.0,
        prem_max_memory_tokens: int = 128,
        prem_pred_weight: float = 0.0,
        prem_pred_tokens: int = 8,
        prem_disable_anti_distractor: bool = False,
        prem_disable_novelty: bool = False,
        prem_disable_stability: bool = False,
        prem_disable_evidence_gate_write: bool = False,
        prem_uniform_write_route: bool = False,
        encode_chunk_frames: int = 0,
        **_unused,
    ):
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
        if isinstance(images, list):
            images = images[0]
        encode_step = max(1, int(encode_chunk_frames or 16))
        conditioned_chunks = []
        salience_chunks = []
        mem_dtype = mem.query_proj.weight.dtype
        for start in range(0, images.shape[0], encode_step):
            features = self.backbone.encode_images(images[start : start + encode_step])
            temporal_tokens = features.to(
                device=mem.query_proj.weight.device, dtype=mem_dtype
            ).mean(dim=1)
            chunk_conditioned, chunk_salience = mem.condition_stream_chunk(
                temporal_tokens.unsqueeze(0)
            )
            conditioned_chunks.append(chunk_conditioned[0])
            salience_chunks.append(chunk_salience[0])
        conditioned = torch.cat(conditioned_chunks, dim=0)
        salience = torch.cat(salience_chunks, dim=0)
        conditioned = self._downsample_tokens(
            conditioned, prem_max_memory_tokens
        ).unsqueeze(0)
        salience = self._downsample_tokens(salience, prem_max_memory_tokens).unsqueeze(0)
        state, confidence = mem.reset_stream(1, conditioned.device, conditioned.dtype)
        state, confidence, mean_write_gate = mem.stream_write_sequence(
            conditioned, salience=salience, state=state, confidence=confidence
        )
        stats = {
            "num_stream_updates": int(conditioned.shape[1]),
            "mean_salience": float(salience.mean().detach().cpu()),
            "mean_confidence": float(confidence.mean().detach().cpu()),
            "mean_write_gate": float(mean_write_gate.detach().cpu()),
            "mean_memory_norm": float(state.float().norm(dim=(-2, -1)).mean().detach().cpu()),
            "stream_writer_mode": "temporal_mean_per_step",
            "stream_memory_tokens_per_step": 1,
        }
        pred_loss = conditioned.new_zeros(())
        if prem_pred_weight > 0:
            if query_embeddings is None:
                raise ValueError("query_embeddings are required when evidence prediction is enabled")
            global_conditioned = conditioned
            global_salience = salience
            pred_mask = mem._top_evidence_span_mask(global_salience, prem_pred_tokens)
            pred_state, pred_confidence = mem.reset_stream(1, conditioned.device, conditioned.dtype)
            pred_state, _, _ = mem.stream_write_sequence(
                conditioned,
                salience=salience,
                state=pred_state,
                confidence=pred_confidence,
                write_block_mask=pred_mask,
            )
            pred_loss, _ = mem.evidence_prediction_loss(
                pred_state,
                global_conditioned,
                query_embeddings.to(device=conditioned.device, dtype=conditioned.dtype),
                pred_mask,
            )
        stats.update(
            {
                "modulation": "attention",
                "alpha": float((mem._alpha_scale(prem_alpha) * mem.num_slots).detach().cpu()),
                "max_memory_tokens": int(prem_max_memory_tokens),
                "evidence_prediction_loss": float(pred_loss.detach().cpu()),
            }
        )
        self.prem_last_stats = stats
        return state, pred_loss, stats

    def stream_reset(self, batch_size: int = 1, **config):
        mem = self.build_prem_memory(
            num_slots=int(config.get("prem_num_slots", 4)),
            alpha=1.0,
            mem_dim=int(config.get("prem_mem_dim", 128)),
            num_layer_groups=int(config.get("prem_layer_groups", 1)),
            disable_anti_distractor=bool(config.get("prem_disable_anti_distractor", False)),
            disable_novelty=bool(config.get("prem_disable_novelty", False)),
            disable_stability=bool(config.get("prem_disable_stability", False)),
            disable_evidence_gate_write=bool(config.get("prem_disable_evidence_gate_write", False)),
            uniform_write_route=bool(config.get("prem_uniform_write_route", False)),
        )
        state, confidence = mem.reset_stream(
            int(batch_size), mem.query_proj.weight.device, mem.query_proj.weight.dtype
        )
        return {"memory": state, "confidence": confidence, "num_updates": 0, "stats": {}, "config": config}

    def stream_update_from_images(
        self,
        stream_state: dict,
        images: torch.Tensor,
        *,
        encode_chunk_frames: int = 0,
        update_block_size: int = 32,
    ):
        features = self.encode_video_frames(images, chunk_frames=encode_chunk_frames)
        state, confidence, _conditioned, _salience, stats = self._write_features(
            features,
            state=stream_state["memory"],
            confidence=stream_state["confidence"],
            max_memory_tokens=0,
            update_block_size=update_block_size,
        )
        previous = int(stream_state.get("num_updates", 0))
        current = int(features.shape[0])
        stats["num_stream_updates"] = previous + current
        return {
            **stream_state,
            "memory": state,
            "confidence": confidence,
            "num_updates": previous + current,
            "stats": {**stream_state.get("stats", {}), **stats},
        }

    @staticmethod
    def _expanded_visual_mask(
        raw_input_ids: torch.Tensor,
        expanded_length: int,
        visual_tokens: int,
        image_token_index: int,
    ) -> torch.Tensor:
        masks = []
        for row in raw_input_ids:
            positions = torch.nonzero(row == image_token_index, as_tuple=False).flatten()
            if len(positions) != 1:
                raise ValueError(f"Expected one LLaVA image placeholder, found {len(positions)}")
            start = int(positions[0])
            mask = torch.zeros(expanded_length, dtype=torch.bool, device=row.device)
            mask[start : min(expanded_length, start + visual_tokens)] = True
            masks.append(mask)
        return torch.stack(masks)

    def _prepare_multimodal(
        self,
        input_ids,
        attention_mask,
        labels,
        images,
        modalities,
        image_token_index: int,
    ):
        encoded = self.encode_video_frames(images, chunk_frames=16)
        original_encode = self.backbone.encode_images
        self.backbone.encode_images = lambda _images: encoded
        try:
            _, position_ids, expanded_attention, _, inputs_embeds, expanded_labels = (
                self.backbone.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    None,
                    attention_mask,
                    None,
                    labels,
                    [images] if not isinstance(images, list) else images,
                    modalities,
                    image_sizes=None,
                )
            )
        finally:
            self.backbone.encode_images = original_encode
        # LLaVA pools each frame before insertion, so infer the inserted span
        # from the final sequence length rather than the unpooled feature count.
        raw_valid = int(attention_mask[0].sum()) if attention_mask is not None else input_ids.shape[1]
        visual_tokens = inputs_embeds.shape[1] - raw_valid + 1
        visual_mask = self._expanded_visual_mask(
            input_ids, inputs_embeds.shape[1], visual_tokens, image_token_index
        )
        return position_ids, expanded_attention, inputs_embeds, expanded_labels, visual_mask

    def _corrections_from_state(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        visual_mask: torch.Tensor,
        state: torch.Tensor,
        *,
        prompt_lengths: Optional[torch.Tensor],
        alpha: float,
        router_gamma: float,
    ):
        mem = self.prem_memory
        if mem is None:
            raise RuntimeError("PReM memory has not been initialized")
        valid = ~visual_mask
        if attention_mask is not None:
            valid &= attention_mask.bool().to(valid.device)
        if prompt_lengths is not None:
            lengths = torch.as_tensor(prompt_lengths, device=valid.device).long().view(-1)
            positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
            valid &= positions < lengths.unsqueeze(1)
        query_correction = torch.zeros_like(inputs_embeds)
        output_correction = torch.zeros_like(inputs_embeds)
        router_losses = []
        stats = {}
        mem_dtype = mem.query_proj.weight.dtype
        for batch_idx in range(inputs_embeds.shape[0]):
            row_valid = valid[batch_idx]
            if not bool(row_valid.any()):
                continue
            text = inputs_embeds[batch_idx, row_valid]
            query = text.mean(dim=0, keepdim=True).to(mem_dtype)
            out = mem.stream_read_per_position(
                state[batch_idx : batch_idx + 1].to(mem_dtype),
                text.unsqueeze(0).to(mem_dtype),
                query,
                alpha_override=alpha,
            )
            query_steer, output_steer, _, rho, _, _, _ = out
            query_correction[batch_idx, row_valid] = query_steer[0].to(query_correction.dtype)
            output_correction[batch_idx, row_valid] = output_steer[0].to(output_correction.dtype)
            avg_rho = rho.reshape(-1, mem.num_slots).mean(dim=0)
            uniform = torch.full_like(avg_rho, 1.0 / mem.num_slots)
            router_losses.append(((avg_rho - uniform) ** 2).sum())
            stats = {
                "router_entropy": float((-(rho * rho.clamp_min(1e-8).log()).sum(-1)).mean().detach().cpu()),
                "max_router_weight": float(rho.max(dim=-1).values.mean().detach().cpu()),
                "mean_query_steer_norm": float(query_steer.norm(dim=-1).mean().detach().cpu()),
                "mean_output_steer_norm": float(output_steer.norm(dim=-1).mean().detach().cpu()),
            }
        router_loss = (
            torch.stack(router_losses).mean() if router_losses else inputs_embeds.new_zeros(())
        )
        self.prem_last_stats = {
            **(self.prem_last_stats or {}),
            **stats,
            "router_balance_loss": float(router_loss.detach().cpu()),
        }
        self.prem_aux_losses = {"router_balance": router_loss, "router_gamma": float(router_gamma)}
        return query_correction, output_correction

    def _hook_stack(self, query_correction, output_correction, modulation_mode: str):
        targets = {
            "qo": ("q_proj", "o_proj"),
            "kv": ("k_proj", "v_proj"),
            "k": ("k_proj", None),
            "v": (None, "v_proj"),
        }
        if modulation_mode not in targets:
            raise ValueError(f"Unknown modulation mode {modulation_mode!r}")
        first, second = targets[modulation_mode]
        stack = ExitStack()

        def make_hook(correction):
            def hook(module, module_inputs, module_output):
                value = correction.to(device=module_output.device, dtype=module_output.dtype)
                if value.shape[-1] != module_output.shape[-1]:
                    value = torch.nn.functional.linear(value, module.weight)
                # Prefill sees the expanded prompt; decode steps are [B, 1, D].
                if value.shape[:2] != module_output.shape[:2]:
                    return module_output
                return module_output + value

            return hook

        for layer in self._prem_text_model.layers:
            if first is not None:
                handle = getattr(layer.self_attn, first).register_forward_hook(
                    make_hook(query_correction)
                )
                stack.callback(handle.remove)
            if second is not None:
                handle = getattr(layer.self_attn, second).register_forward_hook(
                    make_hook(output_correction)
                )
                stack.callback(handle.remove)
        return stack

    @staticmethod
    def _parse_modulation(value: str) -> str:
        return {
            "attention": "qo",
            "attention_kv": "kv",
            "attention_k": "k",
            "attention_v": "v",
        }.get(value, value)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: torch.Tensor,
        modalities=("video",),
        image_token_index: int = -200,
        prem_stream_state: torch.Tensor,
        prem_prompt_lengths: Optional[torch.Tensor] = None,
        prem_modulation: str = "attention_kv",
        prem_alpha: float = 1.0,
        prem_router_gamma: float = 0.05,
        use_cache: bool = False,
        **_kwargs,
    ):
        position_ids, expanded_attention, embeds, expanded_labels, visual_mask = self._prepare_multimodal(
            input_ids, attention_mask, labels, images, list(modalities), image_token_index
        )
        if prem_prompt_lengths is not None:
            raw_visual = (input_ids == image_token_index).sum(dim=1)
            expanded_prompt_lengths = (
                prem_prompt_lengths.to(input_ids.device)
                + visual_mask.sum(dim=1)
                - raw_visual
            )
        else:
            expanded_prompt_lengths = None
        corrections = self._corrections_from_state(
            embeds,
            expanded_attention,
            visual_mask,
            prem_stream_state,
            prompt_lengths=expanded_prompt_lengths,
            alpha=prem_alpha,
            router_gamma=prem_router_gamma,
        )
        hooks = self._hook_stack(*corrections, self._parse_modulation(prem_modulation))
        with hooks:
            outputs = self.backbone.model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=expanded_attention,
                inputs_embeds=embeds,
                use_cache=use_cache,
                return_dict=True,
            )
        hidden = outputs.last_hidden_state
        loss = None
        logits = None
        if expanded_labels is not None:
            shifted = expanded_labels[..., 1:].contiguous()
            valid_labels = shifted.ne(IGNORE_INDEX)
            if bool(valid_labels.any()):
                selected_hidden = hidden[..., :-1, :][valid_labels]
                selected_logits = self.lm_head(selected_hidden).float()
                loss = CrossEntropyLoss()(selected_logits, shifted[valid_labels])
            else:
                loss = hidden.sum() * 0.0
            loss = loss + prem_router_gamma * self.prem_aux_losses["router_balance"].to(loss.device)
        else:
            logits = self.lm_head(hidden[:, -1:, :]).float()
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    def predict_choice(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        image_token_index: int,
        prem_stream_state: torch.Tensor,
        tokenizer,
        option_labels: list[str],
        **prem_kwargs,
    ) -> str:
        output = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            images=images,
            image_token_index=image_token_index,
            prem_stream_state=prem_stream_state,
            prem_prompt_lengths=torch.tensor([input_ids.shape[1]], device=input_ids.device),
            **prem_kwargs,
        )
        logits = output.logits[0, -1]
        candidates = []
        for label in option_labels:
            token_ids = tokenizer(label, add_special_tokens=False).input_ids
            if not token_ids:
                raise ValueError(f"Tokenizer produced no id for option {label!r}")
            candidates.append(int(token_ids[0]))
        scores = logits[torch.tensor(candidates, device=logits.device)]
        return option_labels[int(scores.argmax())]

    def generate_answer(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        image_token_index: int,
        prem_stream_state: torch.Tensor,
        tokenizer,
        prem_modulation: str = "attention_kv",
        prem_alpha: float = 1.0,
        prem_router_gamma: float = 0.0,
        max_new_tokens: int = 8,
        **_kwargs,
    ) -> str:
        """Match official LLaVA-Video Base: greedy generate, then parse the letter."""
        position_ids, expanded_attention, embeds, expanded_labels, visual_mask = self._prepare_multimodal(
            input_ids, attention_mask, None, images, ["video"], image_token_index
        )
        del position_ids, expanded_labels
        query_correction, output_correction = self._corrections_from_state(
            embeds,
            expanded_attention,
            visual_mask,
            prem_stream_state,
            prompt_lengths=torch.tensor([input_ids.shape[1]], device=input_ids.device),
            alpha=prem_alpha,
            router_gamma=prem_router_gamma,
        )
        image_list = images if isinstance(images, list) else [images]
        hooks = self._hook_stack(
            query_correction, output_correction, self._parse_modulation(prem_modulation)
        )
        with hooks:
            generated = self.backbone.generate(
                input_ids,
                images=image_list,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
