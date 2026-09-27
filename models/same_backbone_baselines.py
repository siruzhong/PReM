"""Mechanism controls for the Qwen2.5-VL same-backbone comparison.

The controls in this module are opt-in and deliberately separate from the
PReM model wrappers.  They provide a parameter-audited LoRA configuration, a
token-injection readout over the same recurrent writer used by PReM,
InfiniPot-V-style TaR + VaN KV compression, and an optional H2O diagnostic.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


QWEN25_LORA_RANK = 20
QWEN25_LORA_ALPHA = 40
INFINIPOT_V_REFERENCE = "https://github.com/aiha-lab/InfiniPot-V"
INFINIPOT_V_PUBLIC_REVISION = "81a4dbe2e74660a8148cc2a6cee1148a9a268479"
QWEN25_LORA_TARGET_PATTERN = (
    r"^model\.language_model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)$"
)


def qwen25_lora_parameter_count(config, rank: int = QWEN25_LORA_RANK) -> int:
    """Return the exact q/k/v/o LoRA parameter count for a Qwen2.5 text stack."""
    text = config.text_config if hasattr(config, "text_config") else config
    hidden = int(text.hidden_size)
    head_dim = hidden // int(text.num_attention_heads)
    kv_hidden = int(text.num_key_value_heads) * head_dim
    per_layer = rank * (
        (hidden + hidden)  # q_proj
        + (hidden + kv_hidden)  # k_proj
        + (hidden + kv_hidden)  # v_proj
        + (hidden + hidden)  # o_proj
    )
    return int(text.num_hidden_layers) * per_layer


def make_qwen25_lora_config(
    rank: int = QWEN25_LORA_RANK,
    alpha: int = QWEN25_LORA_ALPHA,
    dropout: float = 0.0,
):
    """Build a language-attention-only PEFT configuration."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        target_modules=QWEN25_LORA_TARGET_PATTERN,
    )


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters() if parameter.requires_grad)


def token_readout_count(memory) -> int:
    """Number of injected tokens when every PReM read head remains active."""
    group_count = len(memory.group_query_steer_heads)
    return 2 * int(memory.num_slots) * (1 + group_count)


def recurrent_state_bytes(stream_state: dict[str, Any] | torch.Tensor) -> int:
    if torch.is_tensor(stream_state):
        return int(stream_state.numel() * stream_state.element_size())
    total = 0
    for key in ("memory", "confidence"):
        value = stream_state.get(key)
        if torch.is_tensor(value):
            total += int(value.numel() * value.element_size())
    return total


def token_readout_embeddings(
    memory,
    state: torch.Tensor,
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Read recurrent state into prompt tokens without adding parameters.

    All query/output heads used by the matched PReM module become token
    projectors.  Consequently, the writer and trainable parameter count are
    identical; only the read interface changes from prefix K/V steering to
    in-sequence token injection.
    """
    if query_embeddings.ndim != 3 or query_embeddings.shape[1] == 0:
        raise ValueError("query_embeddings must have shape [batch, text, hidden] with text > 0")
    if reference_embeddings.ndim != 3:
        raise ValueError("reference_embeddings must have shape [batch, seq, hidden]")

    mem_dtype = memory.query_proj.weight.dtype
    query = query_embeddings.to(mem_dtype).mean(dim=1, keepdim=True)
    conditioned_query = memory.condition_video(query, query[:, 0])
    _, rho, per_slot = memory.read_sequence(state.to(mem_dtype), conditioned_query)

    head_sets = [memory.query_steer_heads, memory.output_steer_heads]
    for group_idx in range(len(memory.group_query_steer_heads)):
        head_sets.extend(
            [
                memory.group_query_steer_heads[group_idx],
                memory.group_output_steer_heads[group_idx],
            ]
        )

    projected = []
    for heads in head_sets:
        directions = memory._apply_slot_heads(per_slot, heads).squeeze(1)
        projected.append(rho.squeeze(1).unsqueeze(-1) * directions)
    tokens = torch.cat(projected, dim=1)

    # Qwen applies RMSNorm per token.  Matching the average backbone embedding
    # norm avoids an arbitrary scale advantage while introducing no parameters.
    target_norm = reference_embeddings.detach().float().norm(dim=-1).mean(dim=1, keepdim=True)
    tokens = F.normalize(tokens.float(), dim=-1) * target_norm.unsqueeze(-1)
    tokens = tokens * memory._alpha_scale().float()
    return tokens.to(device=reference_embeddings.device, dtype=reference_embeddings.dtype)


def _video_features(model, pixel_values_videos, video_grid_thw) -> torch.Tensor:
    if pixel_values_videos is None:
        raise ValueError("Token-readout inputs require pixel_values_videos")
    if hasattr(model, "_prem_encode_video"):
        features = model._prem_encode_video(pixel_values_videos, video_grid_thw)
    else:
        features = model.get_video_features(pixel_values_videos, video_grid_thw)
    if isinstance(features, (tuple, list)):
        features = torch.cat(list(features), dim=0)
    return features


def materialize_qwen25_embeddings(model, inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace Qwen video placeholders and return embeddings plus 3D positions."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    embeddings = model.get_input_embeddings()(input_ids)
    video_features = _video_features(
        model,
        inputs.get("pixel_values_videos"),
        inputs.get("video_grid_thw"),
    ).to(device=embeddings.device, dtype=embeddings.dtype)
    _, video_mask = model.model.get_placeholder_mask(
        input_ids,
        inputs_embeds=embeddings,
        video_features=video_features,
    )
    embeddings = embeddings.masked_scatter(video_mask, video_features)
    position_ids, _ = model.model.get_rope_index(
        input_ids,
        image_grid_thw=inputs.get("image_grid_thw"),
        video_grid_thw=inputs.get("video_grid_thw"),
        second_per_grid_ts=inputs.get("second_per_grid_ts"),
        attention_mask=attention_mask,
    )
    return embeddings, position_ids


def insert_prompt_embeddings(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    inserted: torch.Tensor,
    insert_at: int,
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Insert continuous prompt tokens into a batch-size-one Qwen sequence."""
    if embeddings.shape[0] != 1 or inserted.shape[0] != 1:
        raise ValueError("Token-readout comparison currently requires batch size one")
    seq_len = int(embeddings.shape[1])
    insert_at = int(insert_at)
    if not 0 <= insert_at <= seq_len:
        raise ValueError(f"insert_at={insert_at} is outside sequence length {seq_len}")
    count = int(inserted.shape[1])
    if count <= 0:
        raise ValueError("At least one memory token must be inserted")

    expanded_embeddings = torch.cat(
        [embeddings[:, :insert_at], inserted, embeddings[:, insert_at:]], dim=1
    )
    inserted_mask = torch.ones(
        (1, count), device=attention_mask.device, dtype=attention_mask.dtype
    )
    expanded_mask = torch.cat(
        [attention_mask[:, :insert_at], inserted_mask, attention_mask[:, insert_at:]], dim=1
    )

    if insert_at < seq_len:
        start = position_ids[:, :, insert_at : insert_at + 1]
    elif insert_at:
        start = position_ids[:, :, insert_at - 1 : insert_at] + 1
    else:
        start = torch.zeros(
            (position_ids.shape[0], 1, 1),
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
    offsets = torch.arange(count, device=position_ids.device, dtype=position_ids.dtype).view(1, 1, -1)
    inserted_positions = start + offsets
    suffix_positions = position_ids[:, :, insert_at:] + count
    expanded_positions = torch.cat(
        [position_ids[:, :, :insert_at], inserted_positions, suffix_positions], dim=2
    )

    result = {
        "inputs_embeds": expanded_embeddings,
        "attention_mask": expanded_mask,
        "position_ids": expanded_positions,
    }
    if labels is not None:
        ignored = torch.full(
            (1, count), -100, device=labels.device, dtype=labels.dtype
        )
        result["labels"] = torch.cat(
            [labels[:, :insert_at], ignored, labels[:, insert_at:]], dim=1
        )
    return result


def build_token_readout_inputs(
    model,
    inputs: dict[str, Any],
    stream_state: dict[str, Any] | torch.Tensor,
    prompt_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Materialize a Qwen prompt and insert recurrent-memory readout tokens."""
    embeddings, position_ids = materialize_qwen25_embeddings(model, inputs)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    text_mask = attention_mask.bool() & positions.lt(int(prompt_length))
    for token_id in (model.config.video_token_id, model.config.image_token_id):
        text_mask &= input_ids.ne(int(token_id))
    query_embeddings = embeddings[text_mask].view(1, -1, embeddings.shape[-1])
    state_tensor = stream_state["memory"] if isinstance(stream_state, dict) else stream_state
    memory_tokens = token_readout_embeddings(
        model.prem_memory,
        state_tensor,
        query_embeddings,
        embeddings[:, : int(prompt_length)],
    )
    expanded = insert_prompt_embeddings(
        embeddings,
        attention_mask,
        position_ids,
        memory_tokens,
        int(prompt_length),
        inputs.get("labels"),
    )
    stats = {
        "auxiliary_prompt_tokens": int(memory_tokens.shape[1]),
        "decoder_prompt_tokens": int(attention_mask.sum().item() + memory_tokens.shape[1]),
        "memory_state_bytes": recurrent_state_bytes(stream_state),
    }
    return expanded, stats


def dynamic_cache_state_bytes(cache: DynamicCache) -> int:
    """Return bytes occupied by initialized K/V tensors in a dynamic cache."""
    total = 0
    for layer in cache.layers:
        if layer.is_initialized:
            total += int(layer.keys.numel() * layer.keys.element_size())
            total += int(layer.values.numel() * layer.values.element_size())
    return total


def clone_dynamic_cache(cache: DynamicCache, config) -> DynamicCache:
    """Clone a question-independent K/V state for one answer request."""
    copied = DynamicCache(config=config)
    for source, target in zip(cache.layers, copied.layers):
        if not source.is_initialized:
            continue
        target.lazy_initialization(source.keys)
        target.keys = source.keys.clone()
        target.values = source.values.clone()
    return copied


def infinipot_v_indices(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    token_per_frame: int,
    frames_to_keep: int,
    tar_ratio: float = 0.5,
    query_ratio: float = 0.25,
) -> torch.Tensor:
    """Select visual KV indices with the public InfiniPot-V TaR + VaN rule.

    The public implementation treats the final ``query_ratio`` of cached
    visual units as temporal anchors (TaR), adds the most dissimilar earlier
    patch keys, then spends the remaining budget on high value-norm (VaN)
    tokens. Selection is independent for every KV head.
    """
    if key_states.shape != value_states.shape or key_states.ndim != 4:
        raise ValueError("key_states and value_states must share [batch, heads, seq, dim]")
    if int(key_states.shape[0]) != 1:
        raise ValueError("InfiniPot-V-style selection currently requires batch size one")
    token_per_frame = int(token_per_frame)
    frames_to_keep = int(frames_to_keep)
    if token_per_frame <= 0 or frames_to_keep <= 0:
        raise ValueError("token_per_frame and frames_to_keep must be positive")
    if not 0.0 <= float(tar_ratio) <= 1.0:
        raise ValueError("tar_ratio must be in [0, 1]")
    if not 0.0 < float(query_ratio) < 1.0:
        raise ValueError("query_ratio must be in (0, 1)")

    sequence_length = int(key_states.shape[2])
    if sequence_length % token_per_frame:
        raise ValueError(
            f"Visual sequence ({sequence_length}) is not divisible by "
            f"token_per_frame ({token_per_frame})"
        )
    frame_count = sequence_length // token_per_frame
    total_to_keep = frames_to_keep * token_per_frame
    if frame_count <= frames_to_keep:
        return torch.arange(sequence_length, device=key_states.device).view(1, -1).expand(
            int(key_states.shape[1]), -1
        )

    van_budget = round((1.0 - float(tar_ratio)) * total_to_keep)
    tar_budget = total_to_keep - van_budget
    query_frames = int(frame_count * float(query_ratio))
    # The recent-anchor span is part of the TaR budget.  When the retained
    # budget is reduced (e.g. 8 units for a 16-frame-aligned comparison), the
    # original ratio can otherwise request more recent units than TaR owns.
    query_frames = max(1, min(query_frames, frame_count - 1, tar_budget))
    query_length = query_frames * token_per_frame
    historical_budget = tar_budget - query_length
    historical_length = sequence_length - query_length
    if historical_budget < 0:
        raise ValueError(
            "TaR budget is smaller than its recent-anchor span; increase tar_ratio "
            "or decrease query_ratio"
        )
    if historical_budget > historical_length:
        raise ValueError("TaR historical budget exceeds available visual tokens")

    keys = key_states.float()
    query_keys = F.normalize(keys[:, :, -query_length:, :], dim=-1)
    historical_keys = F.normalize(keys[:, :, :-query_length, :], dim=-1)
    query_keys = query_keys.reshape(
        1,
        key_states.shape[1],
        query_frames,
        token_per_frame,
        key_states.shape[-1],
    )
    historical_keys = historical_keys.reshape(
        1,
        key_states.shape[1],
        frame_count - query_frames,
        token_per_frame,
        key_states.shape[-1],
    )
    # Match corresponding spatial patches, average over recent anchor frames,
    # and negate cosine similarity so redundant history receives a low score.
    dissimilarity = -(
        query_keys.unsqueeze(3) * historical_keys.unsqueeze(2)
    ).sum(dim=-1).mean(dim=2).reshape(1, key_states.shape[1], -1)
    if historical_budget:
        historical = torch.topk(
            dissimilarity, historical_budget, dim=-1
        ).indices.squeeze(0)
    else:
        historical = torch.empty(
            (key_states.shape[1], 0), device=key_states.device, dtype=torch.long
        )
    recent = torch.arange(
        historical_length,
        sequence_length,
        device=key_states.device,
        dtype=torch.long,
    ).view(1, -1).expand(int(key_states.shape[1]), -1)
    tar_indices = torch.cat([historical, recent], dim=-1)

    value_scores = value_states.float().norm(dim=-1).squeeze(0)
    if tar_indices.numel():
        head_indices = torch.arange(
            key_states.shape[1], device=key_states.device
        ).unsqueeze(1).expand_as(tar_indices)
        value_scores[head_indices, tar_indices] = value_scores.max() + 1.0
    selected = torch.topk(value_scores, total_to_keep, dim=-1).indices
    return selected.sort(dim=-1).values


def compress_infinipot_v_cache(
    cache: DynamicCache,
    system_tokens: int,
    token_per_frame: int,
    frames_to_keep: int,
    tar_ratio: float = 0.5,
    query_ratio: float = 0.25,
) -> list[torch.Tensor]:
    """Apply InfiniPot-V-style per-layer, per-head visual KV selection."""
    system_tokens = int(system_tokens)
    selected_by_layer = []
    for layer in cache.layers:
        if not layer.is_initialized:
            continue
        if system_tokens < 0 or system_tokens > int(layer.keys.shape[2]):
            raise ValueError("system_tokens is outside the initialized KV sequence")
        visual_keys = layer.keys[:, :, system_tokens:, :]
        visual_values = layer.values[:, :, system_tokens:, :]
        selected = infinipot_v_indices(
            visual_keys,
            visual_values,
            token_per_frame,
            frames_to_keep,
            tar_ratio,
            query_ratio,
        )
        gather = selected.unsqueeze(0).unsqueeze(-1).expand(
            1, selected.shape[0], selected.shape[1], layer.keys.shape[-1]
        )
        kept_keys = torch.gather(visual_keys, 2, gather)
        kept_values = torch.gather(visual_values, 2, gather)
        layer.keys = torch.cat([layer.keys[:, :, :system_tokens, :], kept_keys], dim=2)
        layer.values = torch.cat(
            [layer.values[:, :, :system_tokens, :], kept_values], dim=2
        )
        selected_by_layer.append(selected)
    return selected_by_layer


class H2ODynamicCache(DynamicCache):
    """Dynamic cache with the H2O heavy-hitter + recent policy.

    Policy reference: https://github.com/FMInference/H2O
    """

    def __init__(self, config, heavy_size: int, recent_size: int):
        if int(heavy_size) <= 0 or int(recent_size) <= 0:
            raise ValueError("H2O heavy_size and recent_size must both be positive")
        super().__init__(config=config)
        self.config = config
        self.heavy_size = int(heavy_size)
        self.recent_size = int(recent_size)
        self.cache_size = self.heavy_size + self.recent_size
        self.hh_scores: list[torch.Tensor | None] = [None] * len(self.layers)

    @staticmethod
    def _kv_head_scores(attention_weights: torch.Tensor, kv_heads: int) -> torch.Tensor:
        batch, query_heads, query_len, seq_len = attention_weights.shape
        if query_heads % kv_heads:
            raise ValueError(f"query_heads={query_heads} is not divisible by kv_heads={kv_heads}")
        groups = query_heads // kv_heads
        return (
            attention_weights.detach()
            .float()
            .view(batch, kv_heads, groups, query_len, seq_len)
            .sum(dim=(2, 3))
        )

    def _gather_layer(self, layer_idx: int, keep: torch.Tensor) -> None:
        """Keep one ordered token set in a layer and its score accumulator."""
        layer = self.layers[int(layer_idx)]
        key_gather = keep.unsqueeze(-1).expand(-1, -1, -1, layer.keys.shape[-1])
        value_gather = keep.unsqueeze(-1).expand(-1, -1, -1, layer.values.shape[-1])
        layer.keys = torch.gather(layer.keys, dim=2, index=key_gather)
        layer.values = torch.gather(layer.values, dim=2, index=value_gather)
        scores = self.hh_scores[int(layer_idx)]
        if scores is not None:
            self.hh_scores[int(layer_idx)] = torch.gather(scores, dim=2, index=keep)

    def prepare_for_tokens(self, num_coming: int) -> None:
        """Evict before a query block is appended to the cache.

        H2O reserves the recent portion for incoming tokens.  Doing this
        before attention is important: otherwise a prefill block can attend to
        an over-budget cache and only be evicted after the expensive attention
        matrix has already been materialized.
        """
        num_coming = int(num_coming)
        if num_coming <= 0:
            return
        if num_coming > self.recent_size:
            raise ValueError(
                f"H2O query block ({num_coming}) exceeds recent budget ({self.recent_size})"
            )

        for layer_idx, layer in enumerate(self.layers):
            if not layer.is_initialized:
                continue
            seq_len = int(layer.keys.shape[-2])
            if seq_len + num_coming <= self.cache_size:
                continue
            scores = self.hh_scores[layer_idx]
            if scores is None or int(scores.shape[-1]) != seq_len:
                raise RuntimeError(
                    "H2O score state is unavailable or misaligned before eviction "
                    f"(layer={layer_idx}, cache={seq_len})"
                )

            keep_count = self.cache_size - num_coming
            # This is H2O's evict_for_space rule: preserve all heavy-hitter
            # slots, and replace `num_coming` entries of the recent window.
            heavy_keep = self.heavy_size
            recent_keep = self.recent_size - num_coming
            candidate_end = seq_len - recent_keep
            candidate_scores = scores[..., :candidate_end]
            if heavy_keep > 0:
                heavy = torch.topk(candidate_scores, heavy_keep, dim=-1).indices
                heavy = heavy.sort(dim=-1).values
            else:
                heavy = torch.empty(
                    (*scores.shape[:2], 0), device=scores.device, dtype=torch.long
                )
            if recent_keep > 0:
                recent = torch.arange(
                    seq_len - recent_keep,
                    seq_len,
                    device=heavy.device,
                    dtype=heavy.dtype,
                ).view(1, 1, -1).expand(heavy.shape[0], heavy.shape[1], -1)
            else:
                recent = torch.empty(
                    (*scores.shape[:2], 0), device=heavy.device, dtype=torch.long
                )
            keep = torch.cat([heavy, recent], dim=-1)
            if int(keep.shape[-1]) != keep_count:
                raise RuntimeError(
                    f"H2O eviction selected {keep.shape[-1]} tokens, expected {keep_count}"
                )
            self._gather_layer(layer_idx, keep)

    def update_scores_and_evict(self, layer_idx: int, attention_weights: torch.Tensor) -> None:
        layer = self.layers[int(layer_idx)]
        if not layer.is_initialized:
            return
        seq_len = int(layer.keys.shape[-2])
        scores = self._kv_head_scores(attention_weights, int(layer.keys.shape[1]))
        previous = self.hh_scores[int(layer_idx)]
        if previous is not None:
            old_len = min(int(previous.shape[-1]), int(scores.shape[-1]))
            scores[..., :old_len] += previous[..., :old_len]

        if seq_len <= self.cache_size:
            self.hh_scores[int(layer_idx)] = scores
            return

        heavy_end = seq_len - self.recent_size
        heavy_scores = scores[..., :heavy_end]
        keep_heavy = torch.topk(heavy_scores, self.heavy_size, dim=-1).indices.sort(dim=-1).values
        recent = torch.arange(
            heavy_end, seq_len, device=keep_heavy.device, dtype=keep_heavy.dtype
        ).view(1, 1, -1).expand(keep_heavy.shape[0], keep_heavy.shape[1], -1)
        keep = torch.cat([keep_heavy, recent], dim=-1)
        self._gather_layer(layer_idx, keep)

    def clone(self) -> "H2ODynamicCache":
        copied = H2ODynamicCache(self.config, self.heavy_size, self.recent_size)
        for source, target in zip(self.layers, copied.layers):
            if not source.is_initialized:
                continue
            target.lazy_initialization(source.keys)
            target.keys = source.keys.clone()
            target.values = source.values.clone()
        copied.hh_scores = [None if score is None else score.clone() for score in self.hh_scores]
        return copied

    def state_bytes(self) -> int:
        total = dynamic_cache_state_bytes(self)
        for score in self.hh_scores:
            if score is not None:
                total += int(score.numel() * score.element_size())
        return total


def install_h2o_attention(model) -> None:
    """Attach post-attention H2O eviction to a Qwen2.5-VL model once."""
    if model.config.model_type not in {"qwen2_5_vl", "qwen2_5_vl_text"}:
        raise ValueError("The same-backbone H2O control supports Qwen2.5-VL only")
    if model.config._attn_implementation != "eager":
        raise ValueError("H2O requires eager attention scores")

    for layer_idx, layer in enumerate(model.model.language_model.layers):
        attention = layer.self_attn
        if getattr(attention, "_same_backbone_h2o_installed", False):
            continue
        original = attention.forward

        @functools.wraps(original)
        def wrapped(*args, __original=original, __layer_idx=layer_idx, **kwargs):
            output = __original(*args, **kwargs)
            cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
            if isinstance(cache, H2ODynamicCache):
                weights = output[1]
                if weights is None:
                    raise RuntimeError("H2O did not receive attention probabilities")
                cache.update_scores_and_evict(__layer_idx, weights)
            return output

        attention.forward = wrapped
        attention._same_backbone_h2o_installed = True
