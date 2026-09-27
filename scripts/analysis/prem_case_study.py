#!/usr/bin/env python3
"""Trace PReM writes, question-conditioned reads, and slot interventions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoConfig, AutoProcessor

from models.prem_qwen2_5_vl_model import PReMQwen2_5_VLForConditionalGeneration
from models.prem_qwen2vl_model import PReMQwen2VLForConditionalGeneration
from models.prem_qwen3_vl_model import PReMQwen3VLForConditionalGeneration
from scripts.eval.data_io import load_records, normalize_mcq_sample
from scripts.eval.eval_prem_online import (
    build_buffered_answer_inputs,
    chunk_to_frame_embeds,
    load_prem_memory,
    mcq_option_token_ids,
    sample_frame_indices,
    update_visual_buffer,
)
from scripts.eval.inference_mcq_vqa import get_sample_media_name, resolve_video_source


DEFAULT_SAMPLE_IDS = ["001006", "001118", "001132", "001236"]
DEFAULT_EVIDENCE_TIMES = {
    "000974": [395.0, 398.0],
    "000977": [365.0, 370.0],
    "000979": [450.0, 460.0],
    "000983": [385.0, 388.0],
    "001006": [139.0],
    "001118": [130.0],
    "001132": [176.0],
    "001236": [87.0],
}


def relative_path(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def tensor_values(value: torch.Tensor) -> list[float]:
    return value.detach().float().cpu().reshape(-1).tolist()


def load_model(args):
    config_path = Path(args.model_path) / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        model_type = json.load(handle).get("model_type")
    model_classes = {
        "qwen2_vl": PReMQwen2VLForConditionalGeneration,
        "qwen2_5_vl": PReMQwen2_5_VLForConditionalGeneration,
        "qwen3_vl": PReMQwen3VLForConditionalGeneration,
    }
    model_cls = model_classes.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unsupported Qwen-VL model type: {model_type!r}")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    config.architectures = [model_cls.__name__]
    attention = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "eager"
    model = model_cls.from_pretrained(
        args.model_path,
        config=config,
        device_map="cuda",
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation=attention,
    ).eval()
    processor_kwargs = {"trust_remote_code": True}
    if model_type != "qwen3_vl":
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    prem_config = load_prem_memory(model, args.prem_ckpt, args)
    return model, processor, prem_config


def load_case_rows(args) -> list[dict]:
    selected_ids = args.sample_id or DEFAULT_SAMPLE_IDS
    rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    by_id = {str(row["id"]): row for row in rows}
    missing = [sample_id for sample_id in selected_ids if sample_id not in by_id]
    if missing:
        raise KeyError(f"Sample ids not found in {args.gt_file}: {missing}")
    selected = [by_id[sample_id] for sample_id in selected_ids]
    videos = {get_sample_media_name(row) for row in selected}
    if len(videos) != 1:
        raise ValueError(f"A case study must use one video; got {sorted(videos)}")
    return selected


def sample_video(video_path: str, fps: float, max_frames: int):
    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    source_fps = float(reader.get_avg_fps())
    indices = sample_frame_indices(len(reader), source_fps, fps, max_frames)
    arrays = reader.get_batch(indices).asnumpy()
    frames = [Image.fromarray(frame).convert("RGB") for frame in arrays]
    timestamps = [index / source_fps for index in indices]
    return frames, indices, timestamps, source_fps, len(reader)


def temporal_timestamps(frame_times: list[float], count: int) -> list[float]:
    if count <= 0:
        return []
    if count == len(frame_times):
        return list(frame_times)
    groups = np.array_split(np.asarray(frame_times, dtype=float), count)
    return [float(group.mean()) for group in groups]


def trace_video_writes(model, processor, frames, frame_times, prem_config, args):
    stream_state = model.stream_reset(
        prem_num_slots=prem_config["num_slots"],
        prem_mem_dim=prem_config["mem_dim"],
        prem_layer_groups=prem_config["layer_groups"],
        prem_alpha=prem_config["alpha"],
        prem_disable_anti_distractor=prem_config["disable_anti_distractor"],
        prem_disable_novelty=prem_config["disable_novelty"],
        prem_disable_stability=prem_config["disable_stability"],
        prem_disable_evidence_gate_write=prem_config["disable_evidence_gate_write"],
        prem_uniform_write_route=prem_config["uniform_write_route"],
    )
    update_visual_buffer(
        stream_state,
        frames,
        prem_config["visual_buffer_frames"],
        max_pixels=args.max_pixels,
    )
    memory_module = model.prem_memory
    state = stream_state["memory"]
    confidence = stream_state["confidence"]
    trace = {
        key: []
        for key in (
            "timestamp_s",
            "route",
            "stability",
            "novelty",
            "salience",
            "anti_overwrite",
            "gate",
            "residual_norm",
            "write_update_norm",
            "state_norm",
            "confidence",
        )
    }
    write_updates = []

    for start in range(0, len(frames), args.chunk_frames):
        end = min(start + args.chunk_frames, len(frames))
        frame_embeds = chunk_to_frame_embeds(model, processor, frames[start:end], args)
        temporal = frame_embeds.mean(dim=1).unsqueeze(0)
        conditioned, salience = memory_module.condition_stream_chunk(
            temporal.to(memory_module.query_proj.weight.dtype)
        )
        token_times = temporal_timestamps(frame_times[start:end], conditioned.shape[1])
        for index, timestamp in enumerate(token_times):
            state, confidence, _, details = memory_module.stream_write_step(
                state,
                confidence,
                conditioned[:, index],
                salience[:, index],
                return_details=True,
            )
            trace["timestamp_s"].append(timestamp)
            trace["salience"].append(float(details["salience"].detach().float().cpu()[0]))
            write_updates.append(details["write_update"][0].detach().float().cpu())
            for key in trace:
                if key in {"timestamp_s", "salience"}:
                    continue
                trace[key].append(tensor_values(details[key][0]))

    stream_state["memory"] = state
    stream_state["confidence"] = confidence
    stream_state["num_updates"] = len(trace["timestamp_s"])
    forget = torch.sigmoid(memory_module.forget_logit).detach().float().cpu()
    return stream_state, trace, torch.stack(write_updates), forget


def prepare_question(model, processor, question, stream_state, args):
    inputs, prompt_length = build_buffered_answer_inputs(
        model, processor, question, stream_state, args, is_mcq=True
    )
    prepared = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    return prepared, prompt_length


def temporal_read_attribution(memory_module, write_updates, forget, slot_queries, rho):
    steps = int(write_updates.shape[0])
    exponents = torch.arange(steps - 1, -1, -1, dtype=torch.float32)
    decay = forget.unsqueeze(0).pow(exponents.unsqueeze(1))
    device = memory_module.query_proj.weight.device
    decayed_updates = write_updates.to(device) * decay[:, :, None, None].to(device)
    attribution = torch.zeros(
        steps,
        memory_module.num_slots,
        device=device,
        dtype=torch.float32,
    )
    for start in range(0, slot_queries.shape[1], 64):
        end = min(start + 64, slot_queries.shape[1])
        retrieved = torch.einsum(
            "tkvi,pki->tpkv",
            decayed_updates,
            slot_queries[0, start:end].float(),
        )
        attribution += (
            rho[0, start:end].float().unsqueeze(0) * retrieved.norm(dim=-1)
        ).sum(dim=1)
    attribution /= max(1, slot_queries.shape[1])
    return attribution.detach().cpu()


def temporal_memory_segments(write_updates, forget, timestamps, num_segments):
    """Decompose the final memory into contiguous, time-local write segments."""
    steps = int(write_updates.shape[0])
    exponents = torch.arange(steps - 1, -1, -1, dtype=torch.float32)
    decay = forget.unsqueeze(0).pow(exponents.unsqueeze(1))
    decayed_updates = write_updates * decay[:, :, None, None]
    groups = np.array_split(np.arange(steps), min(int(num_segments), steps))
    contributions = []
    metadata = []
    for index, group in enumerate(groups):
        start = int(group[0])
        end = int(group[-1])
        contributions.append(decayed_updates[group].sum(dim=0))
        metadata.append(
            {
                "segment": index,
                "start_step": start,
                "end_step": end,
                "start_s": float(timestamps[start]),
                "end_s": float(timestamps[end]),
                "center_s": 0.5 * float(timestamps[start] + timestamps[end]),
                "memory_norm": float(contributions[-1].norm()),
            }
        )
    return contributions, metadata


def read_summary(
    model,
    memory_module,
    prepared,
    prompt_length,
    state,
    alpha,
    write_updates,
    forget,
):
    input_ids = prepared["input_ids"]
    attention_mask = prepared.get("attention_mask")
    prompt_lengths = torch.tensor([prompt_length], device=model.device)
    text_mask = model._prem_text_mask(input_ids, attention_mask, prompt_lengths)
    valid_mask = model._prem_valid_mask(input_ids, attention_mask, prompt_lengths)
    embeddings = model.get_input_embeddings()(input_ids)
    query = embeddings[0, text_mask[0]].mean(dim=0, keepdim=True)
    target = embeddings[0, valid_mask[0]].unsqueeze(0)
    with torch.inference_mode():
        conditioned_target = memory_module.condition_video(
            target.to(memory_module.query_proj.weight.dtype),
            query.to(memory_module.query_proj.weight.dtype),
        )
        _, rho, per_slot = memory_module.read_sequence(
            state.to(memory_module.query_proj.weight.dtype), conditioned_target
        )
        _, _, key_dirs, value_dirs = memory_module.steer_from_sequence_read(
            rho, per_slot, alpha_override=alpha
        )
        query_vectors = memory_module.query_proj(memory_module.query_norm(conditioned_target))
        slot_queries = torch.einsum("btj,kij->btki", query_vectors, memory_module.slot_key)
        slot_queries = torch.nn.functional.normalize(slot_queries, dim=-1)
    scale = memory_module._alpha_scale(alpha).detach().float()
    key_contribution = scale * rho.float() * key_dirs.float().norm(dim=-1)
    value_contribution = scale * rho.float() * value_dirs.float().norm(dim=-1)
    contribution = 0.5 * (key_contribution + value_contribution)
    mean_contribution = contribution.mean(dim=1)[0]
    contribution_share = mean_contribution / mean_contribution.sum().clamp_min(1e-8)
    temporal_attribution = temporal_read_attribution(
        memory_module, write_updates, forget, slot_queries, rho
    )
    return {
        "prefix_positions": int(rho.shape[1]),
        "mean_rho": tensor_values(rho.float().mean(dim=1)[0]),
        "read_contribution": tensor_values(mean_contribution),
        "read_contribution_share": tensor_values(contribution_share),
        "temporal_read_attribution": temporal_attribution.tolist(),
    }


def score_options(
    model,
    processor,
    prepared,
    prompt_length,
    question,
    state,
    stream_stats,
    prem_config,
    args,
    alpha,
):
    model_inputs = dict(prepared)
    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask", None)
    option_ids = mcq_option_token_ids(processor, question)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            prem_stream_state=state,
            prem_stream_stats=stream_stats,
            prem_modulation=args.prem_modulation,
            prem_alpha=alpha,
            prem_num_slots=prem_config["num_slots"],
            prem_mem_dim=prem_config["mem_dim"],
            prem_layer_groups=prem_config["layer_groups"],
            prem_prompt_lengths=torch.tensor([prompt_length], device=model.device),
            prem_disable_anti_distractor=prem_config["disable_anti_distractor"],
            prem_disable_novelty=prem_config["disable_novelty"],
            prem_disable_stability=prem_config["disable_stability"],
            prem_disable_evidence_gate_write=prem_config["disable_evidence_gate_write"],
            prem_uniform_write_route=prem_config["uniform_write_route"],
            **model_inputs,
        )
    position = int(attention_mask[0].sum().item()) - 1 if attention_mask is not None else -1
    scores = outputs.logits[0, position, option_ids].detach().float().cpu()
    return scores


def summarize_scores(scores: torch.Tensor, gold_index: int) -> dict:
    if not 0 <= gold_index < scores.numel():
        raise ValueError(f"Gold option {gold_index} is outside {scores.numel()} choices")
    other = torch.cat([scores[:gold_index], scores[gold_index + 1 :]])
    margin = scores[gold_index] - other.max()
    prediction = int(scores.argmax())
    return {
        "option_logits": scores.tolist(),
        "prediction_index": prediction,
        "prediction": chr(65 + prediction),
        "gold_margin": float(margin),
    }


def analyze_question(
    model,
    processor,
    row,
    stream_state,
    prem_config,
    args,
    write_updates,
    forget,
    temporal_contributions,
    temporal_segments,
):
    question = row["question"]
    prepared, prompt_length = prepare_question(model, processor, question, stream_state, args)
    memory_module = model.prem_memory
    full_state = stream_state["memory"]
    read = read_summary(
        model,
        memory_module,
        prepared,
        prompt_length,
        full_state,
        prem_config["alpha"],
        write_updates,
        forget,
    )
    full_scores = score_options(
        model,
        processor,
        prepared,
        prompt_length,
        question,
        full_state,
        stream_state.get("stats", {}),
        prem_config,
        args,
        prem_config["alpha"],
    )
    no_memory_scores = score_options(
        model,
        processor,
        prepared,
        prompt_length,
        question,
        full_state,
        stream_state.get("stats", {}),
        prem_config,
        args,
        0.0,
    )
    full_summary = summarize_scores(full_scores, int(row["answer"]))
    no_memory_summary = summarize_scores(no_memory_scores, int(row["answer"]))

    ablations = []
    if args.include_slot_ablation:
        for slot in range(prem_config["num_slots"]):
            ablated_state = full_state.clone()
            ablated_state[:, slot].zero_()
            scores = score_options(
                model,
                processor,
                prepared,
                prompt_length,
                question,
                ablated_state,
                stream_state.get("stats", {}),
                prem_config,
                args,
                prem_config["alpha"],
            )
            summary = summarize_scores(scores, int(row["answer"]))
            summary["slot"] = slot
            summary["margin_drop"] = full_summary["gold_margin"] - summary["gold_margin"]
            ablations.append(summary)

    temporal_ablation = []
    if not args.skip_temporal_ablation:
        for segment, contribution in zip(temporal_segments, temporal_contributions):
            ablated_state = full_state - contribution.to(
                device=full_state.device, dtype=full_state.dtype
            ).unsqueeze(0)
            scores = score_options(
                model,
                processor,
                prepared,
                prompt_length,
                question,
                ablated_state,
                stream_state.get("stats", {}),
                prem_config,
                args,
                prem_config["alpha"],
            )
            summary = summarize_scores(scores, int(row["answer"]))
            summary.update(segment)
            summary["margin_drop"] = full_summary["gold_margin"] - summary["gold_margin"]
            temporal_ablation.append(summary)

    return {
        "id": str(row["id"]),
        "question": question,
        "gold_index": int(row["answer"]),
        "gold": chr(65 + int(row["answer"])),
        "read": read,
        "prem": full_summary,
        "no_memory": no_memory_summary,
        "slot_ablation": ablations,
        "temporal_ablation": temporal_ablation,
        "evidence_timestamps_s": DEFAULT_EVIDENCE_TIMES.get(str(row["id"]), []),
    }


def build_trace(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("PReM case-study tracing requires a CUDA device")
    rows = load_case_rows(args)
    video_name = get_sample_media_name(rows[0])
    video_path = resolve_video_source(args.video_dir, video_name)
    frames, source_indices, frame_times, source_fps, total_source_frames = sample_video(
        video_path, args.fps, args.max_frames
    )
    model, processor, prem_config = load_model(args)
    stream_state, write_trace, write_updates, forget = trace_video_writes(
        model, processor, frames, frame_times, prem_config, args
    )
    temporal_contributions, temporal_segments = temporal_memory_segments(
        write_updates,
        forget,
        write_trace["timestamp_s"],
        args.temporal_segments,
    )
    questions = [
        analyze_question(
            model,
            processor,
            row,
            stream_state,
            prem_config,
            args,
            write_updates,
            forget,
            temporal_contributions,
            temporal_segments,
        )
        for row in rows
    ]
    buffer_sample_indices = [int(index) for index, _ in stream_state.get("visual_buffer", [])]
    return {
        "schema_version": 2,
        "method": "PReM",
        "model": relative_path(args.model_path),
        "checkpoint": relative_path(args.prem_ckpt),
        "video": {
            "path": relative_path(video_path),
            "media_name": video_name,
            "source_fps": source_fps,
            "total_source_frames": total_source_frames,
            "sampled_source_indices": source_indices,
            "sampled_timestamps_s": frame_times,
            "buffer_sample_indices": buffer_sample_indices,
        },
        "config": {
            "num_slots": prem_config["num_slots"],
            "mem_dim": prem_config["mem_dim"],
            "alpha": prem_config["alpha"],
            "visual_buffer_frames": prem_config["visual_buffer_frames"],
            "max_frames": args.max_frames,
            "fps": args.fps,
            "chunk_frames": args.chunk_frames,
            "modulation": prem_config["modulation"],
        },
        "write": write_trace,
        "temporal_segments": temporal_segments,
        "questions": questions,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="ckpt/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--prem-ckpt", default="outputs/prem_attention_kv/qwen25_3b/prem.pt")
    parser.add_argument("--gt-file", default="data/eval_video/mlvu/test_qa.json")
    parser.add_argument("--video-dir", default="data/eval_video/mlvu/videos")
    parser.add_argument("--sample-id", action="append", help="Repeat for multiple questions on one video")
    parser.add_argument("--trace-output", default="outputs/interpretability/prem_case_ego52_qwen25.json")
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--chunk-frames", type=int, default=240)
    parser.add_argument("--temporal-segments", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--prem-alpha", type=float, default=1.0)
    parser.add_argument("--prem-modulation", default="attention_kv")
    parser.add_argument("--prem-num-slots", type=int, default=4)
    parser.add_argument("--prem-mem-dim", type=int, default=128)
    parser.add_argument("--prem-layer-groups", type=int, default=1)
    parser.add_argument("--stream-chunk-tokens", type=int, default=32)
    parser.add_argument("--visual-buffer-frames", type=int, default=None)
    parser.add_argument("--prem-disable-anti-distractor", action="store_true")
    parser.add_argument("--prem-disable-novelty", action="store_true")
    parser.add_argument("--prem-disable-evidence-gate-write", action="store_true")
    parser.add_argument("--prem-uniform-write-route", action="store_true")
    parser.add_argument("--include-slot-ablation", action="store_true")
    parser.add_argument("--skip-temporal-ablation", action="store_true")
    args = parser.parse_args()
    args.prem_override_alpha = True
    args.initial_chunk_frames = 0
    args.mcq_max_new_tokens = 1
    args.max_new_tokens = 1
    return args


def main():
    args = parse_args()
    trace = build_trace(args)
    trace_path = Path(args.trace_output)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"[trace] {trace_path}", flush=True)


if __name__ == "__main__":
    main()
