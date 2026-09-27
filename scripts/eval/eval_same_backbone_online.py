#!/usr/bin/env python
"""Evaluate same-backbone mechanism controls under one streaming protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    __version__ as transformers_version,
)
from transformers.cache_utils import DynamicCache

from models.prem_qwen2_5_vl_model import PReMQwen2_5_VLForConditionalGeneration
from models.same_backbone_baselines import (
    H2ODynamicCache,
    INFINIPOT_V_PUBLIC_REVISION,
    INFINIPOT_V_REFERENCE,
    build_token_readout_inputs,
    clone_dynamic_cache,
    compress_infinipot_v_cache,
    count_trainable_parameters,
    dynamic_cache_state_bytes,
    install_h2o_attention,
    make_qwen25_lora_config,
    qwen25_lora_parameter_count,
    recurrent_state_bytes,
)
from qwen_vl_utils import process_vision_info
from scripts.eval.data_io import load_records
from scripts.eval.eval_prem_online import (
    build_buffered_answer_inputs,
    build_stream_state,
    efficiency_stats,
    get_video_grouped_chunk,
    load_prem_memory,
    mcq_option_token_ids,
    resolve_stream_frames,
    synchronize_cuda,
    update_visual_buffer,
    write_result_files,
)
from scripts.eval.inference_mcq_vqa import get_sample_media_name, normalize_mcq_sample
from scripts.eval.prem_checkpoint import artifact_identity, make_eval_signature


METHOD_LABELS = {
    "base": "Base (Frozen, B=16)",
    "lora": "Base + LoRA (r=20)",
    "token_readout": "Base + Token-Readout Memory",
    "infinipot_v": "Base + InfiniPot-V-style (TaR + VaN)",
    "h2o": "Base + H2O-style (Qwen2.5 integration)",
    "prem": "Base + PReM",
}


def choose_option(logits: torch.Tensor, processor, question: str) -> str:
    allowed = mcq_option_token_ids(processor, question)
    allowed_tensor = torch.tensor(allowed, device=logits.device, dtype=torch.long)
    scores = logits[0, -1].index_select(0, allowed_tensor)
    token_id = allowed[int(scores.argmax().item())]
    return processor.tokenizer.decode([token_id], skip_special_tokens=True).strip()


def load_base_or_lora(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    trainable_parameters = 0
    checkpoint = None
    if args.method == "lora":
        checkpoint = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
        validate_control_checkpoint(checkpoint, args, "lora")
        from peft import get_peft_model, set_peft_model_state_dict

        rank = int(checkpoint["lora_rank"])
        model = get_peft_model(
            model,
            make_qwen25_lora_config(
                rank,
                int(checkpoint["lora_alpha"]),
                float(checkpoint["lora_dropout"]),
            ),
        )
        result = set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
        missing_adapters = [
            key for key in getattr(result, "missing_keys", []) if "lora_" in key
        ]
        if missing_adapters or getattr(result, "unexpected_keys", None):
            raise ValueError(f"LoRA state mismatch: {result}")
        trainable_parameters = count_trainable_parameters(model)
        expected = qwen25_lora_parameter_count(model.config, rank)
        if trainable_parameters != expected or trainable_parameters != int(
            checkpoint["trainable_parameters"]
        ):
            raise ValueError("LoRA parameter count does not match its audited checkpoint")
    return model.eval(), trainable_parameters, checkpoint


def load_prem_wrapper(args):
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    if config.model_type != "qwen2_5_vl":
        raise ValueError(f"Expected Qwen2.5-VL, got {config.model_type!r}")
    config.architectures = [PReMQwen2_5_VLForConditionalGeneration.__name__]
    model = PReMQwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        device_map="cuda",
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    return model


def validate_control_checkpoint(checkpoint: dict, args, method: str) -> None:
    expected = {
        "complete": True,
        "baseline_method": method,
        "model_type": "qwen2_5_vl",
        "max_frames": args.max_frames,
        "visual_buffer_frames": args.buffer_frames,
        "max_pixels": args.max_pixels,
        "data_semantics": "same_backbone_controls_v1",
        "source_train_samples": 9000,
        "source_train_qa": 18038,
        "train_samples": 18038,
        "train_steps": 2255,
        "skipped": 0,
        "skip_oom": False,
        "skip_bad_samples": False,
        "qa_pairs_flattened": True,
        "global_batch_size": 8,
        "epochs": 1,
        "lr": 2e-4,
        "warmup_ratio": 0.03,
        "seed": 13,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Control checkpoint metadata mismatch: {mismatches}")
    recipe = (
        {
            "lora_rank": 20,
            "lora_alpha": 40,
            "lora_dropout": 0.0,
            "trainable_parameters": 9_216_000,
        }
        if method == "lora"
        else {
            "num_slots": 1,
            "mem_dim": 128,
            "prem_layer_groups": 4,
            "alpha": 0.75,
            "max_memory_tokens": 128,
            "pred_weight": 0.2,
            "pred_tokens": 4,
            "memory_prompt_tokens": 10,
            "trainable_parameters": 9_070_982,
        }
    )
    recipe_mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in recipe.items()
        if checkpoint.get(key) != value
    }
    if recipe_mismatches:
        raise ValueError(f"Control checkpoint recipe mismatch: {recipe_mismatches}")


def load_token_readout(args):
    checkpoint = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    validate_control_checkpoint(checkpoint, args, "token_readout")
    if checkpoint.get("read_interface") != "in_sequence_tokens":
        raise ValueError("Token-readout checkpoint has the wrong read interface")
    model = load_prem_wrapper(args)
    memory = model.build_prem_memory(
        num_slots=int(checkpoint["num_slots"]),
        alpha=float(checkpoint["alpha"]),
        mem_dim=int(checkpoint["mem_dim"]),
        num_layer_groups=int(checkpoint["prem_layer_groups"]),
    )
    memory.load_state_dict(checkpoint["prem_state_dict"], strict=True)
    memory.to(device=model.device, dtype=torch.float32).eval()
    trainable_parameters = sum(
        int(value.numel()) for value in checkpoint["prem_state_dict"].values()
    )
    if trainable_parameters != int(checkpoint["trainable_parameters"]):
        raise ValueError("Token-readout parameter count does not match its checkpoint")
    cfg = {
        "num_slots": int(checkpoint["num_slots"]),
        "mem_dim": int(checkpoint["mem_dim"]),
        "layer_groups": int(checkpoint["prem_layer_groups"]),
        "alpha": 1.0,
        "stream_chunk_tokens": args.stream_chunk_tokens,
        "visual_buffer_frames": args.buffer_frames,
        "disable_anti_distractor": False,
        "disable_novelty": False,
        "disable_stability": False,
        "disable_evidence_gate_write": False,
        "uniform_write_route": False,
    }
    return model, cfg, trainable_parameters, checkpoint


def load_prem(args):
    model = load_prem_wrapper(args)
    args.visual_buffer_frames = args.buffer_frames
    cfg = load_prem_memory(model, args.baseline_ckpt, args)
    checkpoint = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    trainable_parameters = sum(
        int(value.numel()) for value in checkpoint["prem_state_dict"].values()
    )
    return model, cfg, trainable_parameters, checkpoint


def load_kv_compression(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval()
    if args.method == "h2o":
        install_h2o_attention(model)
    return model


def load_method(args):
    if args.method in {"base", "lora"}:
        model, count, checkpoint = load_base_or_lora(args)
        return model, None, count, checkpoint
    if args.method == "token_readout":
        return load_token_readout(args)
    if args.method == "prem":
        return load_prem(args)
    if args.method in {"infinipot_v", "h2o"}:
        return load_kv_compression(args), None, 0, None
    raise ValueError(f"Unsupported same-backbone method: {args.method}")


def make_reservoir_state(model, processor, frames: list, video_key: str, args) -> dict:
    del model, processor, video_key
    state: dict = {}
    update_visual_buffer(
        state,
        frames,
        args.buffer_frames,
        args.max_pixels,
    )
    stats = dict(state["stats"])
    stats.update(
        {
            "num_stream_frames": len(frames),
            "visible_frame_count": len(frames),
            "buffered_frames": int(stats["visual_buffer_frames"]),
            "visual_tokens": 0,
            "ingested_visual_tokens": 0,
            "memory_state_bytes": 0,
            "state_bytes": 0,
        }
    )
    state["stats"] = stats
    return state


def answer_reservoir(model, processor, question: str, state: dict, args):
    inputs, _ = build_buffered_answer_inputs(
        model,
        processor,
        question,
        state,
        args,
        is_mcq=True,
    )
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    attention_mask = model_inputs["attention_mask"]
    with torch.inference_mode():
        output = model(
            **model_inputs,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    grid = model_inputs.get("video_grid_thw")
    visual_tokens = 0
    if grid is not None:
        merge_size = int(model.config.vision_config.spatial_merge_size)
        visual_tokens = int(
            (grid.to(dtype=torch.long).prod(dim=1) // (merge_size**2)).sum().item()
        )
    return choose_option(output.logits, processor, question), {
        "decoder_prompt_tokens": int(attention_mask.sum().item()),
        "auxiliary_prompt_tokens": 0,
        "visual_tokens": visual_tokens,
        "ingested_visual_tokens": visual_tokens,
    }


def answer_token_readout(model, processor, question: str, state: dict, args):
    inputs, prompt_length = build_buffered_answer_inputs(
        model,
        processor,
        question,
        state,
        args,
        is_mcq=True,
    )
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        # The stream state is materialized under inference_mode during
        # ingestion; keep readout construction in the same context so frozen
        # memory heads cannot trigger autograd's inference-tensor guard.
        expanded, stats = build_token_readout_inputs(
            model,
            model_inputs,
            state,
            prompt_length,
        )
        output = model(**expanded, use_cache=False, logits_to_keep=1, return_dict=True)
    return choose_option(output.logits, processor, question), stats


def answer_prem(model, processor, question: str, state: dict, cfg: dict, args):
    inputs, prompt_length = build_buffered_answer_inputs(
        model,
        processor,
        question,
        state,
        args,
        is_mcq=True,
    )
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
            prem_modulation=args.prem_modulation,
            prem_alpha=cfg["alpha"],
            prem_num_slots=cfg["num_slots"],
            prem_mem_dim=cfg["mem_dim"],
            prem_layer_groups=cfg["layer_groups"],
            prem_prompt_lengths=torch.tensor([prompt_length], device=model.device),
            prem_stream_state=state["memory"],
            prem_stream_stats=state.get("stats", {}),
            prem_disable_anti_distractor=cfg["disable_anti_distractor"],
            prem_disable_novelty=cfg["disable_novelty"],
            prem_disable_stability=cfg["disable_stability"],
            prem_disable_evidence_gate_write=cfg["disable_evidence_gate_write"],
            prem_uniform_write_route=cfg["uniform_write_route"],
            **model_inputs,
        )
    return choose_option(output.logits, processor, question), {
        "decoder_prompt_tokens": int(attention_mask.sum().item()),
        "auxiliary_prompt_tokens": 0,
        "memory_state_bytes": recurrent_state_bytes(state),
    }


def h2o_template_parts(
    processor, question: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the token prefix/suffix around one video placeholder.

    The prefix is deliberately independent of the question.  H2O's state is
    built once per video and then cloned for each question, so putting a
    question before the visual stream would make the KV state question
    dependent and invalidate state reuse.
    """
    instruction = (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the option letter. "
    )
    text = instruction + question if question is not None else None
    content = [{"type": "video"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "Best option: ("
    ids = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    video_positions = torch.nonzero(
        ids[0].eq(int(processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"))),
        as_tuple=False,
    ).flatten()
    if int(video_positions.numel()) != 1:
        raise ValueError(f"Expected one video placeholder in H2O template, got {video_positions.tolist()}")
    split = int(video_positions[0].item())
    return ids[:, :split], ids[:, split + 1 :]


def encode_h2o_chunk(model, processor, frames: list, args):
    video_info = {
        "type": "video",
        "video": frames,
        "fps": args.fps,
        "max_frames": len(frames),
        "max_pixels": args.max_pixels,
    }
    messages = [{"role": "user", "content": [video_info]}]
    _, video_inputs = process_vision_info(messages)
    processed = processor(
        text=["<|vision_start|><|video_pad|><|vision_end|>"],
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        fps=[args.fps],
    )
    pixels = processed.pixel_values_videos.to(model.device)
    grid = processed.video_grid_thw.to(model.device)
    with torch.inference_mode():
        features = torch.cat(model.get_video_features(pixels, grid), dim=0)
    seconds = processed.get("second_per_grid_ts")
    second_per_grid = float(seconds[0].item()) if seconds is not None else 1.0
    return features.unsqueeze(0), grid[0], second_per_grid


def h2o_language_step(model, cache, embeddings, position_ids):
    if int(embeddings.shape[1]) <= 0:
        raise ValueError("H2O language step requires at least one token")
    output = None
    # H2O's real-drop eviction reserves at most `recent_size` entries for one
    # incoming block.  Split unusually long questions/chunks without changing
    # their absolute multimodal positions.
    for start in range(0, int(embeddings.shape[1]), cache.recent_size):
        end = min(start + cache.recent_size, int(embeddings.shape[1]))
        block = embeddings[:, start:end]
        block_positions = position_ids[..., start:end]
        cache.prepare_for_tokens(int(block.shape[1]))
        with torch.inference_mode():
            output = model.model.language_model(
                inputs_embeds=block,
                position_ids=block_positions,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
    return output


def kv_language_step(model, cache, embeddings, position_ids):
    """Append one causal block to a standard Qwen language-model KV cache."""
    if int(embeddings.shape[1]) <= 0:
        raise ValueError("KV language step requires at least one token")
    with torch.inference_mode():
        return model.model.language_model(
            inputs_embeds=embeddings,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )


def make_infinipot_state(model, processor, frames: list, args) -> dict:
    """Build a reusable video state with public TaR + VaN KV compression.

    Qwen video RoPE advances once per temporal vision unit (two sampled RGB
    frames at the registered setting). The public InfiniPot-V 32/24 block
    recipe is therefore applied in those units, while encoding remains in
    two-frame causal chunks to preserve the online protocol.
    """
    prefix, _ = h2o_template_parts(processor)
    prefix = prefix.to(model.device)
    cache = DynamicCache(config=model.config.text_config)
    prefix_embeddings = model.get_input_embeddings()(prefix)
    prefix_positions = torch.arange(prefix.shape[1], device=model.device).view(1, 1, -1)
    prefix_positions = prefix_positions.expand(3, 1, -1)
    kv_language_step(model, cache, prefix_embeddings, prefix_positions)

    temporal_patch = int(model.config.vision_config.temporal_patch_size)
    usable_frames = len(frames) - (len(frames) % temporal_patch)
    prefix_length = int(prefix.shape[1])
    time_offset = 0.0
    max_position = prefix_length - 1
    visual_tokens = 0
    cached_temporal_units = 0
    token_per_temporal_unit = None
    chunks = 0
    compression_events = 0

    for start in range(0, usable_frames, args.chunk_frames):
        chunk = frames[start : min(start + args.chunk_frames, usable_frames)]
        if len(chunk) % temporal_patch:
            raise RuntimeError("InfiniPot-V chunk is not aligned to Qwen temporal patches")
        embeddings, grid, seconds = encode_h2o_chunk(model, processor, chunk, args)
        temporal = int(grid[0].item())
        height = int(grid[1].item()) // int(model.config.vision_config.spatial_merge_size)
        width = int(grid[2].item()) // int(model.config.vision_config.spatial_merge_size)
        per_unit = height * width
        if int(embeddings.shape[1]) != temporal * per_unit:
            raise ValueError("InfiniPot-V visual features do not form fixed spatial units")
        if token_per_temporal_unit is None:
            token_per_temporal_unit = per_unit
        elif token_per_temporal_unit != per_unit:
            raise ValueError(
                "InfiniPot-V requires a constant visual token grid within each video"
            )

        interval = seconds * float(model.config.vision_config.tokens_per_second)
        temporal_ids = (
            torch.arange(temporal, device=model.device, dtype=torch.float32) * interval
            + time_offset
        ).long()
        temporal_ids = temporal_ids.view(-1, 1).expand(-1, per_unit).flatten()
        height_ids = (
            torch.arange(height, device=model.device)
            .view(1, -1, 1)
            .expand(temporal, -1, width)
            .flatten()
        )
        width_ids = (
            torch.arange(width, device=model.device)
            .view(1, 1, -1)
            .expand(temporal, height, -1)
            .flatten()
        )
        positions = torch.stack([temporal_ids, height_ids, width_ids]) + prefix_length
        positions = positions.unsqueeze(1)
        kv_language_step(model, cache, embeddings, positions)

        time_offset += temporal * interval
        max_position = max(max_position, int(positions.max().item()))
        visual_tokens += int(embeddings.shape[1])
        cached_temporal_units += temporal
        chunks += 1

        consumed = start + len(chunk)
        remaining_units = (usable_frames - consumed) // temporal_patch
        # Also compress the final block. Without this finalization step a
        # stream ending at a block boundary leaves the full block in cache.
        at_stream_end = consumed >= usable_frames
        if cached_temporal_units >= args.infinipot_block_units and (
            remaining_units or at_stream_end
        ):
            keep_units = args.infinipot_keep_units
            # When more stream remains, preserve the pending tail while
            # compressing the completed block. At end-of-stream there is no
            # tail to preserve: keep the registered 24-unit budget exactly.
            if remaining_units and keep_units + remaining_units < args.infinipot_block_units:
                keep_units = args.infinipot_block_units - remaining_units
            compress_infinipot_v_cache(
                cache,
                prefix_length,
                token_per_temporal_unit,
                keep_units,
                args.infinipot_tar_ratio,
                args.infinipot_query_ratio,
            )
            cached_temporal_units = keep_units
            compression_events += 1

    state_bytes = dynamic_cache_state_bytes(cache)
    return {
        "cache": cache,
        "prefix_ids": prefix.detach().cpu(),
        "max_position": max_position,
        "stats": {
            "num_stream_frames": len(frames),
            "visible_frame_count": usable_frames,
            "buffered_frames": 0,
            "visual_tokens": int(cache.get_seq_length()),
            "ingested_visual_tokens": visual_tokens,
            "num_stream_chunks": chunks,
            "memory_state_bytes": state_bytes,
            "state_bytes": state_bytes,
            "infinipot_block_units": args.infinipot_block_units,
            "infinipot_keep_units": args.infinipot_keep_units,
            "infinipot_tar_ratio": args.infinipot_tar_ratio,
            "infinipot_query_ratio": args.infinipot_query_ratio,
            "infinipot_token_per_unit": token_per_temporal_unit or 0,
            "infinipot_compression_events": compression_events,
            "infinipot_cache_tokens": int(cache.get_seq_length()),
        },
    }


def answer_infinipot(model, processor, question: str, state: dict, args):
    prefix, suffix = h2o_template_parts(processor, question)
    if not torch.equal(prefix.cpu(), state["prefix_ids"]):
        raise RuntimeError("InfiniPot-V template prefix changed across questions")
    suffix = suffix.to(model.device)
    cache = clone_dynamic_cache(state["cache"], model.config.text_config)
    embeddings = model.get_input_embeddings()(suffix)
    start = int(state["max_position"]) + 1
    positions = torch.arange(start, start + suffix.shape[1], device=model.device)
    positions = positions.view(1, 1, -1).expand(3, 1, -1)
    output = kv_language_step(model, cache, embeddings, positions)
    with torch.inference_mode():
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])
    return choose_option(logits, processor, question), {
        "decoder_prompt_tokens": int(cache.get_seq_length()),
        "auxiliary_prompt_tokens": 0,
        "memory_state_bytes": dynamic_cache_state_bytes(state["cache"]),
        "infinipot_cache_tokens": int(state["cache"].get_seq_length()),
    }


def make_h2o_state(model, processor, frames: list, args) -> dict:
    # Build a question-independent state so all questions for one video share
    # the same causal visual stream and only differ in the answer suffix.
    prefix, _ = h2o_template_parts(processor)
    prefix = prefix.to(model.device)
    cache = H2ODynamicCache(
        model.config.text_config,
        args.h2o_heavy_tokens,
        args.h2o_recent_tokens,
    )
    prefix_embeddings = model.get_input_embeddings()(prefix)
    prefix_positions = torch.arange(prefix.shape[1], device=model.device).view(1, 1, -1)
    prefix_positions = prefix_positions.expand(3, 1, -1)
    h2o_language_step(model, cache, prefix_embeddings, prefix_positions)

    prefix_length = int(prefix.shape[1])
    time_offset = 0.0
    max_position = prefix_length - 1
    visual_tokens = 0
    chunks = 0
    for start in range(0, len(frames), args.chunk_frames):
        chunk = frames[start : start + args.chunk_frames]
        if len(chunk) % 2:
            chunk = chunk[:-1]
        if not chunk:
            continue
        embeddings, grid, seconds = encode_h2o_chunk(model, processor, chunk, args)
        temporal = int(grid[0].item())
        height = int(grid[1].item()) // int(model.config.vision_config.spatial_merge_size)
        width = int(grid[2].item()) // int(model.config.vision_config.spatial_merge_size)
        interval = seconds * float(model.config.vision_config.tokens_per_second)
        temporal_ids = (
            torch.arange(temporal, device=model.device, dtype=torch.float32) * interval
            + time_offset
        ).long()
        temporal_ids = temporal_ids.view(-1, 1).expand(-1, height * width).flatten()
        height_ids = (
            torch.arange(height, device=model.device)
            .view(1, -1, 1)
            .expand(temporal, -1, width)
            .flatten()
        )
        width_ids = (
            torch.arange(width, device=model.device)
            .view(1, 1, -1)
            .expand(temporal, height, -1)
            .flatten()
        )
        positions = torch.stack([temporal_ids, height_ids, width_ids]) + prefix_length
        positions = positions.unsqueeze(1)
        if int(embeddings.shape[1]) != int(positions.shape[-1]):
            raise ValueError(
                f"H2O visual token/position mismatch: {embeddings.shape[1]} vs {positions.shape[-1]}"
            )
        h2o_language_step(model, cache, embeddings, positions)
        time_offset += temporal * interval
        max_position = max(max_position, int(positions.max().item()))
        visual_tokens += int(embeddings.shape[1])
        chunks += 1

    return {
        "cache": cache,
        "prefix_ids": prefix.detach().cpu(),
        "max_position": max_position,
        "stats": {
            "num_stream_frames": len(frames),
            "visible_frame_count": len(frames),
            "buffered_frames": 0,
            "visual_tokens": int(cache.get_seq_length()),
            "ingested_visual_tokens": visual_tokens,
            "num_stream_chunks": chunks,
            "memory_state_bytes": cache.state_bytes(),
            "state_bytes": cache.state_bytes(),
            "h2o_heavy_tokens": args.h2o_heavy_tokens,
            "h2o_recent_tokens": args.h2o_recent_tokens,
            "h2o_cache_tokens": int(cache.get_seq_length()),
        },
    }


def answer_h2o(model, processor, question: str, state: dict, args):
    prefix, suffix = h2o_template_parts(processor, question)
    if not torch.equal(prefix.cpu(), state["prefix_ids"]):
        raise RuntimeError("H2O template prefix changed across questions")
    suffix = suffix.to(model.device)
    cache = state["cache"].clone()
    embeddings = model.get_input_embeddings()(suffix)
    start = int(state["max_position"]) + 1
    positions = torch.arange(start, start + suffix.shape[1], device=model.device)
    positions = positions.view(1, 1, -1).expand(3, 1, -1)
    output = h2o_language_step(model, cache, embeddings, positions)
    with torch.inference_mode():
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])
    resident_before_answer = int(state["cache"].get_seq_length())
    stats = {
        "decoder_prompt_tokens": int(cache.get_seq_length()),
        "auxiliary_prompt_tokens": 0,
        "memory_state_bytes": state["cache"].state_bytes(),
        "h2o_cache_tokens": resident_before_answer,
    }
    return choose_option(logits, processor, question), stats


def make_signature(args, checkpoint, trainable_parameters: int) -> str:
    payload = {
        "protocol": "same_backbone_qwen25_online_v3",
        "method": args.method,
        "method_label": METHOD_LABELS[args.method],
        "dataset": args.dataset,
        "model": artifact_identity(args.model_path),
        "checkpoint": artifact_identity(args.baseline_ckpt),
        "max_frames": args.max_frames,
        "buffer_frames": args.buffer_frames,
        "fps": args.fps,
        "max_pixels": args.max_pixels,
        "max_video_tokens": args.max_video_tokens,
        "chunk_frames": args.chunk_frames,
        "stream_chunk_tokens": args.stream_chunk_tokens,
        "h2o_heavy_tokens": args.h2o_heavy_tokens if args.method == "h2o" else None,
        "h2o_recent_tokens": args.h2o_recent_tokens if args.method == "h2o" else None,
        "infinipot_block_units": (
            args.infinipot_block_units if args.method == "infinipot_v" else None
        ),
        "infinipot_keep_units": (
            args.infinipot_keep_units if args.method == "infinipot_v" else None
        ),
        "infinipot_tar_ratio": (
            args.infinipot_tar_ratio if args.method == "infinipot_v" else None
        ),
        "infinipot_query_ratio": (
            args.infinipot_query_ratio if args.method == "infinipot_v" else None
        ),
        "infinipot_reference": (
            INFINIPOT_V_REFERENCE if args.method == "infinipot_v" else None
        ),
        "infinipot_public_revision": (
            INFINIPOT_V_PUBLIC_REVISION if args.method == "infinipot_v" else None
        ),
        "profile_efficiency": bool(args.profile_efficiency),
        "single_step_constrained_decoding": True,
        "trainable_parameters": int(trainable_parameters),
        "precision": "bfloat16",
        "attention_backend": "eager",
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "compute_capability": list(torch.cuda.get_device_capability()),
        },
        "software": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda": torch.version.cuda,
        },
    }
    return make_eval_signature(payload)


def run_worker(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Same-backbone evaluation requires a visible CUDA device")
    if args.profile_efficiency:
        torch.cuda.reset_peak_memory_stats()
    model, method_cfg, trainable_parameters, checkpoint = load_method(args)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    rows = get_video_grouped_chunk(rows, args.num_chunks, args.chunk_idx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / (
        f"{args.num_chunks}_{args.chunk_idx}.json" if args.num_chunks > 1 else "pred.json"
    )
    signature = make_signature(args, checkpoint, trainable_parameters)
    done = set()
    if pred_file.exists() and not args.overwrite:
        existing = [json.loads(line) for line in pred_file.open()]
        if any(row.get("eval_signature") != signature for row in existing):
            raise RuntimeError(f"Existing predictions use a different protocol: {pred_file}")
        done = {str(row["id"]) for row in existing}

    current_video = None
    current_state = None
    with pred_file.open("w" if args.overwrite else "a") as handle:
        for sample in tqdm(rows, desc=f"{args.method}:cuda:{args.chunk_idx}"):
            if str(sample["id"]) in done:
                continue
            question = sample["question"] if "question" in sample else sample["question1"]
            video_key = get_sample_media_name(sample)
            state_reused = video_key == current_video and current_state is not None
            synchronize_cuda(args.profile_efficiency)
            update_started = time.perf_counter()
            if not state_reused:
                frames = resolve_stream_frames(
                    args.video_dir,
                    sample,
                    args.max_frames,
                    args.fps,
                )
                if args.method in {"base", "lora"}:
                    current_state = make_reservoir_state(
                        model, processor, frames, video_key, args
                    )
                elif args.method in {"token_readout", "prem"}:
                    current_state = build_stream_state(
                        model,
                        processor,
                        frames,
                        method_cfg,
                        args,
                    )
                elif args.method == "infinipot_v":
                    current_state = make_infinipot_state(
                        model, processor, frames, args
                    )
                else:
                    current_state = make_h2o_state(model, processor, frames, args)
                current_video = video_key
            synchronize_cuda(args.profile_efficiency)
            update_seconds = time.perf_counter() - update_started

            synchronize_cuda(args.profile_efficiency)
            answer_started = time.perf_counter()
            if args.method in {"base", "lora"}:
                prediction, answer_stats = answer_reservoir(
                    model, processor, question, current_state, args
                )
            elif args.method == "token_readout":
                prediction, answer_stats = answer_token_readout(
                    model, processor, question, current_state, args
                )
            elif args.method == "prem":
                prediction, answer_stats = answer_prem(
                    model, processor, question, current_state, method_cfg, args
                )
            elif args.method == "infinipot_v":
                prediction, answer_stats = answer_infinipot(
                    model, processor, question, current_state, args
                )
            else:
                prediction, answer_stats = answer_h2o(
                    model, processor, question, current_state, args
                )
            synchronize_cuda(args.profile_efficiency)
            answer_seconds = time.perf_counter() - answer_started
            state_stats = dict(current_state.get("stats", {}))
            row = {
                "id": sample["id"],
                "question": question,
                "answer": sample["answer"],
                "pred": prediction,
                "eval_signature": signature,
                "comparison_method": args.method,
                "comparison_method_label": METHOD_LABELS[args.method],
                "trainable_parameters": trainable_parameters,
                "state_reused": state_reused,
                "stream_update_seconds": update_seconds,
                "answer_seconds": answer_seconds,
                "max_frames": args.max_frames,
                "buffer_frames_limit": args.buffer_frames,
                **state_stats,
                **answer_stats,
                **efficiency_stats(args.profile_efficiency),
                **{
                    key: sample[key]
                    for key in (
                        "variant",
                        "stress_variant",
                        "question_category",
                        "level",
                        "topic_category",
                        "duration_group",
                        "duration",
                        "question_type",
                        "question_subtype",
                        "episode_id",
                    )
                    if key in sample
                },
            }
            handle.write(json.dumps(row) + "\n")
            handle.flush()


def launch(args) -> None:
    output_base = Path(args.output_dir) / args.evaluation_name / args.dataset
    if not args.cuda_devices:
        worker_args = argparse.Namespace(**vars(args))
        worker_args.output_dir = str(output_base)
        run_worker(worker_args)
        write_result_files(output_base, output_base / "pred.json", args.gt_file)
        return

    devices = [value.strip() for value in args.cuda_devices.split(",") if value.strip()]
    if args.num_chunks > len(devices):
        raise ValueError(f"num_chunks={args.num_chunks} exceeds cuda_devices={devices}")
    common = [
        sys.executable,
        __file__,
        "--worker",
        "--method",
        args.method,
        "--dataset",
        args.dataset,
        "--model_path",
        args.model_path,
        "--video_dir",
        args.video_dir,
        "--gt_file",
        args.gt_file,
        "--output_dir",
        str(output_base),
        "--num_chunks",
        str(args.num_chunks),
        "--max_frames",
        str(args.max_frames),
        "--buffer_frames",
        str(args.buffer_frames),
        "--fps",
        str(args.fps),
        "--max_pixels",
        str(args.max_pixels),
        "--max_video_tokens",
        str(args.max_video_tokens),
        "--chunk_frames",
        str(args.chunk_frames),
        "--initial_chunk_frames",
        str(args.initial_chunk_frames),
        "--stream_chunk_tokens",
        str(args.stream_chunk_tokens),
        "--h2o_heavy_tokens",
        str(args.h2o_heavy_tokens),
        "--h2o_recent_tokens",
        str(args.h2o_recent_tokens),
        "--infinipot_block_units",
        str(args.infinipot_block_units),
        "--infinipot_keep_units",
        str(args.infinipot_keep_units),
        "--infinipot_tar_ratio",
        str(args.infinipot_tar_ratio),
        "--infinipot_query_ratio",
        str(args.infinipot_query_ratio),
        "--prem_modulation",
        args.prem_modulation,
    ]
    if args.baseline_ckpt:
        common += ["--baseline_ckpt", args.baseline_ckpt]
    if args.profile_efficiency:
        common.append("--profile_efficiency")
    if args.overwrite:
        common.append("--overwrite")

    processes = []
    for index in range(args.num_chunks):
        command = common + ["--chunk_idx", str(index)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index])
        print(
            f"[exec] CUDA_VISIBLE_DEVICES={devices[index]} " + " ".join(command),
            flush=True,
        )
        processes.append(subprocess.Popen(command, env=env))
    failures = []
    for index, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append((index, code))
    if failures:
        raise RuntimeError(f"Same-backbone workers failed: {failures}")

    pred_file = output_base / "pred.json"
    if args.num_chunks > 1:
        with pred_file.open("w") as merged:
            for index in range(args.num_chunks):
                chunk = output_base / f"{args.num_chunks}_{index}.json"
                if not chunk.exists():
                    raise FileNotFoundError(chunk)
                merged.write(chunk.read_text())
    write_result_files(output_base, pred_file, args.gt_file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=tuple(METHOD_LABELS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_path", default="ckpt/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--baseline_ckpt", default=None)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evaluation_name", default="same_backbone")
    parser.add_argument("--cuda_devices", default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=240)
    parser.add_argument("--buffer_frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--max_video_tokens", type=int, default=11520)
    parser.add_argument("--chunk_frames", type=int, default=2)
    parser.add_argument("--initial_chunk_frames", type=int, default=0)
    parser.add_argument("--stream_chunk_tokens", type=int, default=32)
    parser.add_argument("--h2o_heavy_tokens", type=int, default=1024)
    parser.add_argument("--h2o_recent_tokens", type=int, default=1024)
    parser.add_argument("--infinipot_block_units", type=int, default=32)
    parser.add_argument("--infinipot_keep_units", type=int, default=24)
    parser.add_argument("--infinipot_tar_ratio", type=float, default=0.5)
    parser.add_argument("--infinipot_query_ratio", type=float, default=0.25)
    parser.add_argument(
        "--prem_modulation",
        choices=("attention", "attention_kv", "attention_k", "attention_v"),
        default="attention_kv",
    )
    parser.add_argument("--prem_alpha", type=float, default=1.0)
    parser.add_argument("--prem_override_alpha", action="store_true")
    parser.add_argument("--prem_num_slots", type=int, default=1)
    parser.add_argument("--prem_mem_dim", type=int, default=128)
    parser.add_argument("--prem_layer_groups", type=int, default=4)
    parser.add_argument("--prem_disable_anti_distractor", action="store_true")
    parser.add_argument("--prem_disable_novelty", action="store_true")
    parser.add_argument("--prem_disable_stability", action="store_true")
    parser.add_argument("--prem_disable_evidence_gate_write", action="store_true")
    parser.add_argument("--prem_uniform_write_route", action="store_true")
    parser.add_argument("--profile_efficiency", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_frames != 240 or args.buffer_frames != 16:
        parser.error("The registered same-backbone protocol requires T=240 and B=16")
    if args.chunk_frames != 2 or args.initial_chunk_frames != 0:
        parser.error("The registered streaming protocol requires 2-frame causal chunks")
    if args.method in {"lora", "token_readout", "prem"} and not args.baseline_ckpt:
        parser.error(f"--baseline_ckpt is required for {args.method}")
    if args.method == "h2o" and (
        args.h2o_heavy_tokens <= 0 or args.h2o_recent_tokens <= 0
    ):
        parser.error("H2O heavy/recent budgets must both be positive")
    if args.method == "infinipot_v":
        if not 0 < args.infinipot_keep_units < args.infinipot_block_units:
            parser.error("InfiniPot-V requires 0 < keep_units < block_units")
        if not 0.0 <= args.infinipot_tar_ratio <= 1.0:
            parser.error("InfiniPot-V tar_ratio must be in [0, 1]")
        if not 0.0 < args.infinipot_query_ratio < 1.0:
            parser.error("InfiniPot-V query_ratio must be in (0, 1)")
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
