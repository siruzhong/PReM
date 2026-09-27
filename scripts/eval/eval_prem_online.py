#!/usr/bin/env python
"""Unified fixed-budget evaluation for PREM.

This script mirrors Flash-VStream's realtime protocol at the evaluation level:
video frames are split into chronological chunks, each chunk updates a
persistent PREM memory state and a bounded visual reservoir without seeing the
question. The answer path always consumes question + B + M.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Must be set before importing decord; shared GPFS + DASH MP4s often exceed the default.
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor

from scripts.eval.inference_mcq_vqa import (
    FRAME_EXTS,
    get_chunk,
    get_sample_media_name,
    list_frame_paths,
    normalize_mcq_sample,
    resolve_video_source,
)
from scripts.eval.data_io import load_records
from scripts.eval.prem_checkpoint import artifact_identity, make_eval_signature, validate_prem_checkpoint
from models.prem_qwen2_5_vl_model import PReMQwen2_5_VLForConditionalGeneration
from models.prem_qwen2vl_model import PReMQwen2VLForConditionalGeneration
from models.prem_qwen3_vl_model import PReMQwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info, qwen3_video_metadata




def extract_answer(text: str) -> int | None:
    answer = re.search(r"(?<![A-Z])[A-E](?![A-Z])", str(text).upper())
    if answer is None:
        return None
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[answer.group(0)]


def mcq_option_token_ids(processor, question: str) -> list[int]:
    labels = []
    for line in str(question).splitlines():
        match = re.match(r"\s*[\(\[]?([A-E])[\)\].:]\s+", line.upper())
        if match and match.group(1) not in labels:
            labels.append(match.group(1))
    if not labels:
        raise ValueError(f"Could not derive MCQ option labels from question: {question!r}")

    token_ids = []
    for label in labels:
        encoded = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(encoded) != 1:
            raise ValueError(f"Option label {label!r} is not a single tokenizer token: {encoded}")
        token_ids.append(int(encoded[0]))
    return token_ids


def synchronize_cuda(enabled: bool) -> None:
    if enabled:
        torch.cuda.synchronize()


def efficiency_stats(enabled: bool) -> dict[str, int]:
    if not enabled:
        return {}
    return {
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }




def load_prem_memory(model, ckpt_path: str, args):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    validate_prem_checkpoint(ckpt)
    checkpoint_modulation = ckpt["prem_modulation"]
    if args.prem_modulation != checkpoint_modulation:
        raise ValueError("Evaluation modulation mode must match training")
    checkpoint_model_type = ckpt.get("model_type")
    aliases = {
        "qwen2vl": "qwen2_vl",
        "qwen2_5vl": "qwen2_5_vl",
        "qwen3vl": "qwen3_vl",
    }
    if checkpoint_model_type and aliases.get(checkpoint_model_type, checkpoint_model_type) != model.config.model_type:
        raise ValueError(
            f"Checkpoint backbone type {checkpoint_model_type!r} does not match "
            f"loaded model type {model.config.model_type!r}"
        )
    state_dict = ckpt.get("prem_state_dict", ckpt)
    num_slots = int(ckpt.get("num_slots", args.prem_num_slots))
    mem_dim = int(ckpt.get("mem_dim", args.prem_mem_dim))
    layer_groups = int(ckpt.get("prem_layer_groups", 1))
    ckpt_alpha = float(args.prem_alpha if args.prem_override_alpha else ckpt.get("alpha", args.prem_alpha))
    stream_chunk_tokens = int(ckpt.get("stream_chunk_tokens", args.stream_chunk_tokens))
    checkpoint_buffer_frames = int(ckpt["visual_buffer_frames"])
    if args.visual_buffer_frames is not None and args.visual_buffer_frames != checkpoint_buffer_frames:
        raise ValueError(
            "Evaluation visual buffer budget must match training: "
            f"checkpoint={checkpoint_buffer_frames}, requested={args.visual_buffer_frames}"
        )
    mem = model.build_prem_memory(
        num_slots=num_slots,
        alpha=ckpt_alpha,
        mem_dim=mem_dim,
        num_layer_groups=layer_groups,
        disable_anti_distractor=args.prem_disable_anti_distractor or bool(ckpt.get("disable_anti_distractor", False)),
        disable_novelty=args.prem_disable_novelty or bool(ckpt.get("disable_novelty", False)),
        disable_stability=args.prem_disable_stability or bool(ckpt.get("disable_stability", False)),
        disable_evidence_gate_write=args.prem_disable_evidence_gate_write or bool(ckpt.get("disable_evidence_gate_write", False)),
        uniform_write_route=args.prem_uniform_write_route or bool(ckpt.get("uniform_write_route", False)),
    )
    missing, unexpected = mem.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(
            "PREM-Online checkpoint does not match the current Attention memory module; "
            f"missing={list(missing)} unexpected={list(unexpected)}"
        )
    mem.to(device=model.device, dtype=torch.float32).eval()
    return {
        "num_slots": num_slots,
        "mem_dim": mem_dim,
        "layer_groups": layer_groups,
        "alpha": float(args.prem_alpha),
        "modulation": checkpoint_modulation,
        "stream_chunk_tokens": stream_chunk_tokens,
        "visual_buffer_frames": checkpoint_buffer_frames,
        "disable_anti_distractor": args.prem_disable_anti_distractor or bool(ckpt.get("disable_anti_distractor", False)),
        "disable_novelty": args.prem_disable_novelty or bool(ckpt.get("disable_novelty", False)),
        "disable_stability": args.prem_disable_stability or bool(ckpt.get("disable_stability", False)),
        "disable_evidence_gate_write": args.prem_disable_evidence_gate_write or bool(ckpt.get("disable_evidence_gate_write", False)),
        "uniform_write_route": args.prem_uniform_write_route or bool(ckpt.get("uniform_write_route", False)),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def sample_frame_indices(total_frames: int, video_fps: float, target_fps: float, max_frames: int) -> list[int]:
    """Build an even-sized FPS-capped manifest matching Qwen video sampling."""
    total_frames = int(total_frames)
    if total_frames <= 0:
        raise ValueError("Video contains no frames")
    if video_fps <= 0 or target_fps <= 0:
        raise ValueError(f"video_fps and target_fps must be positive, got {video_fps}, {target_fps}")
    if total_frames == 1:
        return [0]

    frame_factor = 2
    requested = round((total_frames / float(video_fps)) * float(target_fps) / frame_factor) * frame_factor
    requested = max(4, requested)
    requested = min(requested, int(max_frames) if max_frames else requested)
    requested = min(requested, total_frames - (total_frames % frame_factor))
    requested = max(2, requested)
    return torch.linspace(0, total_frames - 1, requested).round().long().tolist()


def _is_decord_eof(exc: BaseException) -> bool:
    text = str(exc)
    return "DECORD_EOF_RETRY_MAX" in text or "Unable to handle EOF" in text


def _ffprobe_duration_fps(video_path: str) -> tuple[float, float]:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json", video_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(proc.stdout)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    rate = str(stream.get("avg_frame_rate") or "30/1")
    if "/" in rate:
        num, den = rate.split("/", 1)
        video_fps = float(num) / max(float(den), 1e-6)
    else:
        video_fps = float(rate or 30.0)
    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    if video_fps <= 0 or duration <= 0:
        raise RuntimeError(f"ffprobe could not read duration/fps from {video_path}")
    return duration, video_fps


def sample_raw_video_frames_ffmpeg(video_path: str, max_frames: int, fps: float = 1.0) -> list:
    duration, video_fps = _ffprobe_duration_fps(video_path)
    total = max(1, int(round(duration * video_fps)))
    indices = sample_frame_indices(total, video_fps, fps, max_frames)
    expr = "+".join(f"eq(n\\,{index})" for index in indices)
    with tempfile.TemporaryDirectory() as tmp:
        pattern = os.path.join(tmp, "%06d.jpg")
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", video_path,
                "-vf", f"select='{expr}',setpts=N/TB",
                "-vsync", "vfr",
                "-q:v", "2",
                pattern,
            ],
            check=True,
        )
        paths = sorted(Path(tmp).glob("*.jpg"))
        if not paths:
            raise RuntimeError(f"ffmpeg produced no frames from {video_path}")
        return [Image.open(path).convert("RGB").copy() for path in paths]


def sample_raw_video_frames(video_path: str, max_frames: int, fps: float = 1.0) -> list:
    last_exc = None
    for attempt in range(3):
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            total = len(vr)
            indices = sample_frame_indices(total, float(vr.get_avg_fps()), fps, max_frames)
            frames = vr.get_batch(indices).asnumpy()
            return [Image.fromarray(frame).convert("RGB") for frame in frames]
        except Exception as exc:
            last_exc = exc
            if not _is_decord_eof(exc) or attempt == 2:
                break
            time.sleep(0.5 * (attempt + 1))
    if last_exc is not None and _is_decord_eof(last_exc):
        print(f"[warn] decord EOF on {video_path}, falling back to ffmpeg", flush=True)
        return sample_raw_video_frames_ffmpeg(video_path, max_frames, fps)
    raise last_exc


class CausalVideoSource:
    """Lazily expose only frames at or before the requested timestamp."""

    def __init__(self, video_dir: str, sample: dict, max_frames: int, fps: float = 1.0):
        self.video_path = resolve_video_source(video_dir, get_sample_media_name(sample))
        self.fps = float(fps)
        self._cursor = 0
        self._reader = None
        if os.path.isdir(self.video_path):
            self._frames = list_frame_paths(self.video_path)
            if not self._frames:
                raise FileNotFoundError(f"Frame directory {self.video_path} contains no image frames.")
            source_indices = list(range(len(self._frames)))
            if max_frames and len(self._frames) > max_frames:
                idx = torch.linspace(0, len(self._frames) - 1, max_frames).round().long().tolist()
                self._frames = [self._frames[i] for i in idx]
                source_indices = idx
            self._timestamps = [index / self.fps for index in source_indices]
        else:
            try:
                self._reader = VideoReader(self.video_path, ctx=cpu(0), num_threads=1)
                total = len(self._reader)
                source_fps = float(self._reader.get_avg_fps())
                self._indices = sample_frame_indices(total, source_fps, self.fps, max_frames)
                self._timestamps = [index / source_fps for index in self._indices]
                self._frames = None
            except Exception as exc:
                if not _is_decord_eof(exc):
                    raise
                print(f"[warn] decord EOF on {self.video_path}, falling back to ffmpeg", flush=True)
                frames = sample_raw_video_frames_ffmpeg(self.video_path, max_frames, self.fps)
                self._reader = None
                self._frames = frames
                self._timestamps = [index / self.fps for index in range(len(frames))]

    @property
    def visible_count(self) -> int:
        return self._cursor

    def take_until(self, query_time: float) -> list:
        cutoff = bisect.bisect_right(self._timestamps, float(query_time) + 1e-6)
        if cutoff <= self._cursor:
            return []
        start, end = self._cursor, cutoff
        if self._reader is not None:
            frames = self._reader.get_batch(self._indices[start:end]).asnumpy()
            result = [Image.fromarray(frame).convert("RGB") for frame in frames]
        else:
            result = self._frames[start:end]
        self._cursor = end
        return result


def resolve_stream_frames(video_dir: str, sample: dict, max_frames: int, fps: float = 1.0) -> list:
    video_name = get_sample_media_name(sample)
    video_path = resolve_video_source(video_dir, video_name)
    if os.path.isdir(video_path):
        frames = list_frame_paths(video_path)
        if not frames:
            raise FileNotFoundError(f"Frame directory {video_path} contains no image frames.")
        if max_frames and len(frames) > max_frames:
            idx = torch.linspace(0, len(frames) - 1, max_frames).round().long().tolist()
            frames = [frames[i] for i in idx]
        return frames
    return sample_raw_video_frames(video_path, max_frames, fps=fps)


def split_chunks(items: list, chunk_frames: int, initial_chunk_frames: int = 0) -> list[list]:
    chunk_frames = max(1, int(chunk_frames))
    initial_chunk_frames = max(0, int(initial_chunk_frames or 0))
    if initial_chunk_frames <= 0 or initial_chunk_frames >= len(items):
        return [items[start: start + chunk_frames] for start in range(0, len(items), chunk_frames)]
    chunks = [items[:initial_chunk_frames]]
    chunks.extend(
        items[start: start + chunk_frames]
        for start in range(initial_chunk_frames, len(items), chunk_frames)
    )
    return chunks


def _materialize_frame(frame, max_pixels: int | None = None):
    if isinstance(frame, (str, os.PathLike)):
        with Image.open(frame) as image:
            materialized = image.convert("RGB").copy()
    elif isinstance(frame, Image.Image):
        materialized = frame.convert("RGB").copy()
    else:
        materialized = Image.fromarray(frame).convert("RGB")
    if max_pixels and materialized.width * materialized.height > int(max_pixels):
        scale = (int(max_pixels) / (materialized.width * materialized.height)) ** 0.5
        width = max(1, int(materialized.width * scale))
        height = max(1, int(materialized.height * scale))
        materialized = materialized.resize((width, height), Image.Resampling.BICUBIC)
    return materialized


def update_visual_buffer(
    stream_state: dict,
    frames: list,
    capacity: int,
    max_pixels: int | None = None,
) -> None:
    """Causally maintain a deterministic, question-agnostic reservoir of at most B frames."""
    capacity = int(capacity)
    buffer = list(stream_state.get("visual_buffer", []))
    seen = int(stream_state.get("visual_buffer_seen", 0))
    for frame in frames:
        materialized = None
        if capacity > 0 and len(buffer) < capacity:
            materialized = _materialize_frame(frame, max_pixels)
            buffer.append((seen, materialized))
        elif capacity > 0:
            digest = hashlib.sha256(f"prem-buffer:{seen}".encode("ascii")).digest()
            replacement = int.from_bytes(digest[:8], "big") % (seen + 1)
            if replacement < capacity:
                materialized = _materialize_frame(frame, max_pixels)
                buffer[replacement] = (seen, materialized)
        seen += 1
    buffer.sort(key=lambda item: item[0])
    stream_state["visual_buffer"] = buffer
    stream_state["visual_buffer_seen"] = seen
    stats = dict(stream_state.get("stats", {}))
    visual_buffer_bytes = sum(frame.width * frame.height * len(frame.getbands()) for _, frame in buffer)
    stats.update(
        {
            "visual_buffer_frames": len(buffer),
            "visual_buffer_capacity": capacity,
            "visual_buffer_policy": "deterministic_reservoir",
            "visual_buffer_bytes": visual_buffer_bytes,
        }
    )
    stream_state["stats"] = stats


def get_video_grouped_chunk(rows: list[dict], num_chunks: int, chunk_idx: int) -> list[dict]:
    """Keep every question for a video on one worker and in adjacent rows."""
    groups = {}
    for row in rows:
        groups.setdefault(get_sample_media_name(row), []).append(row)
    for group in groups.values():
        if all("query_time" in row for row in group):
            group.sort(key=lambda row: float(row["query_time"]))
    selected_groups = get_chunk(list(groups.values()), num_chunks, chunk_idx)
    return [row for group in selected_groups for row in group]


def chunk_to_frame_embeds(model, processor, frames: list, args) -> torch.Tensor:
    video_info = {
        "type": "video",
        "video": frames,
        "fps": args.fps,
        "max_frames": len(frames),
        "max_pixels": args.max_pixels,
    }
    messages = [
        {
            "role": "user",
            "content": [
                video_info,
                {"type": "text", "text": "stream"},
            ],
        }
    ]
    _, video_inputs = process_vision_info(messages)
    video_kwargs = {}
    if model.config.model_type == "qwen3_vl":
        video_kwargs["video_metadata"] = [qwen3_video_metadata(video_info)]
        video_kwargs["do_sample_frames"] = False
    inputs = processor(
        text=["<|vision_start|><|video_pad|><|vision_end|>"],
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    pixel_values_videos = inputs.pixel_values_videos.to(model.device)
    video_grid_thw = inputs.video_grid_thw.to(model.device)
    with torch.inference_mode():
        visual_dtype = next(model.visual.parameters()).dtype
        pixel_values_videos = pixel_values_videos.to(dtype=visual_dtype)
        if hasattr(model, "_prem_encode_video"):
            encoded = model._prem_encode_video(pixel_values_videos, video_grid_thw)
            video_embeds = encoded[0] if isinstance(encoded, tuple) else encoded
        else:
            video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
    grouped = model.group_video_embeds_by_temporal_step(video_embeds, video_grid_thw)
    if len(grouped) != 1:
        raise ValueError(f"Expected one stream video chunk, got {len(grouped)}")
    return grouped[0].to(device=model.device, dtype=model.get_input_embeddings().weight.dtype)


def update_stream_state(model, processor, stream_state, frames: list, prem_cfg: dict, args):
    """Write a newly arrived frame segment into an existing stream state."""
    if stream_state is None:
        stream_state = model.stream_reset(
            prem_num_slots=prem_cfg["num_slots"],
            prem_mem_dim=prem_cfg["mem_dim"],
            prem_layer_groups=prem_cfg["layer_groups"],
            prem_alpha=prem_cfg["alpha"],
            prem_disable_anti_distractor=prem_cfg["disable_anti_distractor"],
            prem_disable_novelty=prem_cfg["disable_novelty"],
            prem_disable_stability=prem_cfg["disable_stability"],
            prem_disable_evidence_gate_write=prem_cfg["disable_evidence_gate_write"],
            prem_uniform_write_route=prem_cfg["uniform_write_route"],
        )
        initial_chunk_frames = args.initial_chunk_frames
    else:
        initial_chunk_frames = 0

    if not frames:
        if int(stream_state.get("num_updates", 0)) == 0:
            raise RuntimeError("No stream chunks were produced")
        return stream_state

    update_visual_buffer(
        stream_state,
        frames,
        prem_cfg["visual_buffer_frames"],
        max_pixels=args.max_pixels,
    )

    chunks = split_chunks(frames, args.chunk_frames, initial_chunk_frames)
    # Online evaluation is deliberately chunk-causal: the visual encoder only
    # sees the currently arrived chunk, and memory is updated before the next
    # chunk is processed. Full-video visual replay belongs to offline code.
    for chunk in chunks:
        frame_embeds = chunk_to_frame_embeds(model, processor, chunk, args)
        with torch.inference_mode():
            stream_state = model.stream_update(
                stream_state,
                frame_embeds,
                update_block_size=prem_cfg["stream_chunk_tokens"],
                collect_stats=False,
            )
        stats = dict(stream_state.get("stats", {}))
        last_gate_per_slot = stream_state.pop("_last_write_gate_per_slot", None)
        if torch.is_tensor(last_gate_per_slot):
            step_count = int(frame_embeds.shape[1])
            previous_sum = stream_state.get("_write_gate_sum_per_slot")
            if not torch.is_tensor(previous_sum):
                previous_sum = torch.zeros_like(last_gate_per_slot)
            stream_state["_write_gate_sum_per_slot"] = previous_sum + last_gate_per_slot * step_count
            stream_state["_write_gate_count"] = int(stream_state.get("_write_gate_count", 0)) + step_count
        stats["num_stream_chunks"] = int(stats.get("num_stream_chunks", 0)) + 1
        stream_state["stats"] = stats

    stats = dict(stream_state.get("stats", {}))
    last_gate = stream_state.pop("_last_mean_write_gate", None)
    if torch.is_tensor(last_gate):
        stats.update({
            "mean_write_gate": float(last_gate.cpu()),
            "mean_memory_norm": float(stream_state["memory"].float().norm(dim=(-2, -1)).mean().cpu()),
            "mean_confidence": float(stream_state["confidence"].mean().cpu()),
            "forget_multiplier": float(torch.sigmoid(model.prem_memory.forget_logit).mean().cpu()),
        })
    gate_sum_per_slot = stream_state.pop("_write_gate_sum_per_slot", None)
    gate_count = int(stream_state.pop("_write_gate_count", 0))
    if torch.is_tensor(gate_sum_per_slot) and gate_count > 0:
        stats.update({
            "mean_write_gate_per_slot": (gate_sum_per_slot / gate_count).mean(dim=0).detach().float().cpu().tolist(),
            "mean_confidence_per_slot": stream_state["confidence"].mean(dim=0).detach().float().cpu().tolist(),
        })
    stats.update({
        "num_stream_frames": int(stats.get("num_stream_frames", 0)) + len(frames),
        "sampling_fps": float(args.fps),
        "initial_chunk_frames": int(args.initial_chunk_frames),
        "chunk_frames": int(args.chunk_frames),
        "stream_chunk_tokens": int(prem_cfg["stream_chunk_tokens"]),
        "visual_encoding_mode": "chunk_causal",
        "memory_state_bytes": int(
            stream_state["memory"].numel() * stream_state["memory"].element_size()
            + stream_state["confidence"].numel() * stream_state["confidence"].element_size()
        ),
    })
    stats["state_bytes"] = stats["memory_state_bytes"] + int(stats.get("visual_buffer_bytes", 0))
    stream_state["stats"] = stats
    return stream_state


def build_stream_state(model, processor, frames: list, prem_cfg: dict, args):
    """Replay a complete video stream for end-of-stream or batch evaluation.

    The Qwen2-VL visual encoder still needs its own grid metadata internally,
    but PREM memory only receives sequential visual embeddings. It does not use
    Flash-VStream's spa/thw memory geometry.
    """
    return update_stream_state(model, processor, None, frames, prem_cfg, args)


def build_question_text(processor, q_base: str, is_mcq: bool) -> tuple[str, str]:
    if is_mcq:
        prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the option letter. "
    else:
        prompt = "Answer the following open-ended question based on the video. "
    question = prompt + q_base
    return question, "Best option: (" if is_mcq else ""


def build_buffered_answer_inputs(model, processor, q_base: str, stream_state: dict, args, is_mcq: bool):
    question, answer_prefix = build_question_text(processor, q_base, is_mcq)
    buffered_items = stream_state.get("visual_buffer", [])
    buffered_frames = [frame for _, frame in buffered_items]
    buffered_indices = [index for index, _ in buffered_items]
    encoder_frames = buffered_frames if len(buffered_frames) != 1 else buffered_frames * 2
    encoder_indices = buffered_indices if len(buffered_indices) != 1 else buffered_indices * 2
    content = []
    if encoder_frames:
        video_info = {
                "type": "video",
                "video": encoder_frames,
                "fps": args.fps,
                "max_frames": len(encoder_frames),
                "max_pixels": args.max_pixels,
            }
        content.append(video_info)
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + answer_prefix
    if encoder_frames:
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {}
        if model.config.model_type == "qwen3_vl":
            video_kwargs["video_metadata"] = [
                qwen3_video_metadata(
                    video_info,
                    frame_indices=encoder_indices,
                    total_num_frames=max(
                        int(stream_state.get("visual_buffer_seen", 0)),
                        max(encoder_indices) + 1,
                    ),
                )
            ]
            video_kwargs["do_sample_frames"] = False
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
    else:
        inputs = processor.tokenizer([text], padding=True, return_tensors="pt")
    prompt_len = int(inputs.input_ids.shape[1])
    return inputs, prompt_len


def answer_from_state(model, processor, q_base: str, stream_state: dict, prem_cfg: dict, args, is_mcq: bool = True):
    inputs, prompt_len = build_buffered_answer_inputs(
        model, processor, q_base, stream_state, args, is_mcq=is_mcq
    )
    model_inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    input_ids = model_inputs.pop("input_ids")
    attention_mask = model_inputs.pop("attention_mask")
    max_new_tokens = args.mcq_max_new_tokens if is_mcq else args.max_new_tokens
    generation_kwargs = {}
    if is_mcq:
        allowed_token_ids = mcq_option_token_ids(processor, q_base)
        generation_kwargs["prefix_allowed_tokens_fn"] = lambda _batch_id, _input_ids: allowed_token_ids
    with torch.inference_mode():
        generated_ids = model.stream_answer(
            stream_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            prem_modulation=args.prem_modulation,
            prem_alpha=prem_cfg["alpha"],
            prem_num_slots=prem_cfg["num_slots"],
            prem_mem_dim=prem_cfg["mem_dim"],
            prem_layer_groups=prem_cfg["layer_groups"],
            prem_prompt_lengths=torch.tensor([prompt_len], device=model.device),
            prem_disable_anti_distractor=prem_cfg["disable_anti_distractor"],
            prem_disable_novelty=prem_cfg["disable_novelty"],
            prem_disable_stability=prem_cfg["disable_stability"],
            prem_disable_evidence_gate_write=prem_cfg["disable_evidence_gate_write"],
            prem_uniform_write_route=prem_cfg["uniform_write_route"],
            **model_inputs,
            **generation_kwargs,
        )
    generated_ids_trimmed = generated_ids[:, input_ids.shape[1]:]
    answer = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    answer_stats = {
        key: value
        for key, value in (model.prem_last_stats or {}).items()
        if key in {
            "placement",
            "router_entropy",
            "max_router_weight",
            "mean_query_steer_norm",
            "mean_output_steer_norm",
            "mean_rho",
            "mean_read_contribution_per_slot",
            "mean_write_gate_per_slot",
            "mean_confidence_per_slot",
        }
    }
    return answer, answer_stats


def write_result_files(output_dir: Path, pred_file: Path, gt_file: str):
    pred_rows = [json.loads(line) for line in open(pred_file)]
    gt_rows = [normalize_mcq_sample(row) for row in load_records(gt_file)]
    expected = len(gt_rows)
    if len(pred_rows) != expected:
        raise RuntimeError(f"Incomplete predictions: got={len(pred_rows)} expected={expected}")
    pred_ids = [str(row["id"]) for row in pred_rows]
    if len(set(pred_ids)) != len(pred_ids):
        raise RuntimeError("Prediction file contains duplicate sample ids")
    prediction_set = {}
    correct = 0
    invalid = 0
    for row in pred_rows:
        pred_idx = extract_answer(row["pred"])
        invalid += int(pred_idx is None)
        ok = pred_idx is not None and pred_idx == row["answer"]
        correct += int(ok)
        prediction_set[str(row["id"])] = {
            "acc": "yes" if ok else "no",
            "score": 1.0 if ok else 0.0,
            **row,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "result.json", "w") as f:
        json.dump(prediction_set, f, indent=2)
    acc = correct / max(1, expected) * 100.0
    with open(output_dir / "result.csv", "w") as f:
        f.write("acc,score,count,expected,invalid\n")
        f.write(f"{acc:.6f},{acc:.6f},{len(pred_rows)},{expected},{invalid}\n")
    print(
        f"[result] acc={acc:.6f} correct={correct}/{expected} invalid={invalid}",
        flush=True,
    )


def run_worker(args):
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            "Streaming evaluation requires CUDA, but PyTorch cannot see a CUDA device. "
            f"CUDA_VISIBLE_DEVICES={visible!r}; check the requested GPU ids and the job allocation."
        )
    if args.profile_efficiency:
        torch.cuda.reset_peak_memory_stats()
    use_flash_attn = importlib.util.find_spec("flash_attn") is not None
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    # Auto-detect the Qwen-VL generation API from checkpoint config.
    model_type = None
    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_type = json.load(f).get("model_type")
    model_classes = {
        "qwen2_vl": PReMQwen2VLForConditionalGeneration,
        "qwen2_5_vl": PReMQwen2_5_VLForConditionalGeneration,
        "qwen3_vl": PReMQwen3VLForConditionalGeneration,
    }
    ModelCls = model_classes.get(model_type)
    if ModelCls is None:
        raise RuntimeError(
            f"Unsupported model_type={model_type!r}; Qwen3-VL requires transformers>=4.57.1"
        )
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    config.architectures = [ModelCls.__name__]
    model = ModelCls.from_pretrained(
        args.model_path,
        config=config,
        device_map="cuda",
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).eval()
    processor_kwargs = {"trust_remote_code": True}
    if model_type != "qwen3_vl":
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    prem_cfg = load_prem_memory(model, args.prem_ckpt, args)

    rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    if args.time_constrained and any("query_time" not in row for row in rows):
        raise ValueError("--time_constrained requires query_time in every QA row")
    rows = get_video_grouped_chunk(rows, args.num_chunks, args.chunk_idx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / (f"{args.num_chunks}_{args.chunk_idx}.json" if args.num_chunks > 1 else "pred.json")
    eval_signature = make_eval_signature({
        "protocol": "unified_bounded_visual_memory",
        "dataset": args.dataset,
        "model": artifact_identity(args.model_path),
        "prem_checkpoint": artifact_identity(args.prem_ckpt),
        "max_frames": args.max_frames,
        "fps": args.fps,
        "prem_alpha": args.prem_alpha,
        "prem_modulation": prem_cfg["modulation"],
        "chunk_frames": args.chunk_frames,
        "initial_chunk_frames": args.initial_chunk_frames,
        "max_pixels": args.max_pixels,
        "mcq_max_new_tokens": args.mcq_max_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "stream_chunk_tokens": args.stream_chunk_tokens,
        "visual_buffer_frames": prem_cfg["visual_buffer_frames"],
        "disable_anti_distractor": prem_cfg["disable_anti_distractor"],
        "disable_novelty": prem_cfg["disable_novelty"],
        "disable_stability": prem_cfg["disable_stability"],
        "disable_evidence_gate_write": prem_cfg["disable_evidence_gate_write"],
        "uniform_write_route": prem_cfg["uniform_write_route"],
        "time_constrained": bool(args.time_constrained),
        "profile_efficiency": bool(args.profile_efficiency),
    })
    done = set()
    if pred_file.exists() and not args.overwrite:
        existing = [json.loads(line) for line in open(pred_file)]
        if any(row.get("eval_signature") != eval_signature for row in existing):
            raise RuntimeError(f"Existing predictions use a different evaluator/checkpoint: {pred_file}")
        done = {row["id"] for row in existing}
    mode = "w" if args.overwrite else "a"
    current_video = None
    current_source = None
    current_stream_state = None
    with open(pred_file, mode) as f:
        for sample in tqdm(rows, desc=f"cuda:{args.chunk_idx}"):
            if sample["id"] in done:
                continue
            try:
                video_key = get_sample_media_name(sample)
                same_video = video_key == current_video
                if not same_video:
                    current_source = None
                    current_stream_state = None
                    current_video = video_key
                state_reused = same_video and current_stream_state is not None
                synchronize_cuda(args.profile_efficiency)
                update_start = time.perf_counter()
                if args.time_constrained:
                    if current_source is None:
                        current_source = CausalVideoSource(
                            args.video_dir, sample, args.max_frames, args.fps
                        )
                    new_frames = current_source.take_until(float(sample["query_time"]))
                    current_stream_state = update_stream_state(
                        model,
                        processor,
                        current_stream_state,
                        new_frames,
                        prem_cfg,
                        args,
                    )
                    visible_frame_count = current_source.visible_count
                elif not state_reused:
                    frames = resolve_stream_frames(args.video_dir, sample, args.max_frames, args.fps)
                    current_stream_state = build_stream_state(model, processor, frames, prem_cfg, args)
                    visible_frame_count = len(frames)
                else:
                    visible_frame_count = int(current_stream_state["stats"].get("num_stream_frames", 0))
                stream_state = current_stream_state
                synchronize_cuda(args.profile_efficiency)
                update_seconds = time.perf_counter() - update_start
                q_base_list = [sample["question"]] if "question" in sample else [sample["question1"], sample["question2"]]
                synchronize_cuda(args.profile_efficiency)
                answer_start = time.perf_counter()
                pred, answer_stats = answer_from_state(
                    model,
                    processor,
                    q_base_list[0],
                    stream_state,
                    prem_cfg,
                    args,
                    is_mcq=True,
                )
                synchronize_cuda(args.profile_efficiency)
                answer_seconds = time.perf_counter() - answer_start
                row = {
                    "id": sample["id"],
                    "question": q_base_list[0],
                    "answer": sample["answer"],
                    "pred": pred,
                    "eval_signature": eval_signature,
                    "state_reused": state_reused,
                    "stream_update_seconds": update_seconds,
                    "mean_update_seconds": update_seconds / max(1, int(stream_state["num_updates"])),
                    "answer_seconds": answer_seconds,
                    **{k: sample[k] for k in (
                        "variant",
                        "stress_variant",
                        "question_category",
                        "level",
                        "topic_category",
                        "duration_group",
                        "duration",
                        "num_frames",
                        "query_time",
                        "question_type",
                        "question_subtype",
                        "episode_id",
                    ) if k in sample},
                    "time_constrained": bool(args.time_constrained),
                    "visible_frame_count": visible_frame_count,
                    "causal_cutoff_seconds": float(sample["query_time"]) if args.time_constrained else None,
                    **stream_state["stats"],
                    **answer_stats,
                    **efficiency_stats(args.profile_efficiency),
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
            except Exception as exc:
                raise RuntimeError(
                    f"Streaming evaluation failed for id={sample.get('id')}: {type(exc).__name__}: {exc}"
                ) from exc


def launch(args):
    output_base = Path(args.output_dir) / args.evaluation_name / args.dataset
    if args.cuda_devices:
        devices = [d.strip() for d in args.cuda_devices.split(",") if d.strip()]
        num_chunks = args.num_chunks if args.num_chunks and args.num_chunks > 0 else len(devices)
        if num_chunks > len(devices):
            raise ValueError(f"num_chunks={num_chunks} exceeds cuda_devices={devices}")
        procs = []
        for idx in range(num_chunks):
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--dataset",
                args.dataset,
                "--model_path",
                args.model_path,
                "--prem_ckpt",
                args.prem_ckpt,
                "--video_dir",
                args.video_dir,
                "--gt_file",
                args.gt_file,
                "--output_dir",
                str(output_base),
                "--num_chunks",
                str(num_chunks),
                "--chunk_idx",
                str(idx),
                "--max_frames",
                str(args.max_frames),
                "--fps",
                str(args.fps),
                "--prem_alpha",
                str(args.prem_alpha),
                "--prem_modulation",
                str(args.prem_modulation),
                "--chunk_frames",
                str(args.chunk_frames),
                "--initial_chunk_frames",
                str(args.initial_chunk_frames),
                "--max_pixels",
                str(args.max_pixels),
                "--mcq_max_new_tokens",
                str(args.mcq_max_new_tokens),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--stream_chunk_tokens",
                str(args.stream_chunk_tokens),
            ]
            if args.visual_buffer_frames is not None:
                cmd += ["--visual_buffer_frames", str(args.visual_buffer_frames)]
            if args.time_constrained:
                cmd.append("--time_constrained")
            if args.profile_efficiency:
                cmd.append("--profile_efficiency")
            if args.overwrite:
                cmd.append("--overwrite")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = devices[idx]
            print("[exec] CUDA_VISIBLE_DEVICES=" + devices[idx] + " " + " ".join(cmd), flush=True)
            procs.append(subprocess.Popen(cmd, env=env))
        failed = []
        for idx, proc in enumerate(procs):
            ret = proc.wait()
            if ret != 0:
                failed.append((idx, ret))
        if failed:
            raise RuntimeError(f"Streaming eval workers failed: {failed}")
        if num_chunks > 1:
            pred_file = output_base / "pred.json"
            with open(pred_file, "w") as fout:
                for idx in range(num_chunks):
                    chunk_file = output_base / f"{num_chunks}_{idx}.json"
                    if not chunk_file.exists():
                        raise FileNotFoundError(chunk_file)
                    for line in open(chunk_file):
                        fout.write(line)
        else:
            pred_file = output_base / "pred.json"
        write_result_files(output_base, pred_file, args.gt_file)
    else:
        worker_args = argparse.Namespace(**vars(args))
        worker_args.output_dir = str(output_base)
        run_worker(worker_args)
        pred_file = output_base / ("pred.json" if args.num_chunks <= 1 else f"{args.num_chunks}_{args.chunk_idx}.json")
        write_result_files(output_base, pred_file, args.gt_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_path", default="ckpt/Qwen2-VL-7B-Instruct")
    parser.add_argument("--prem_ckpt", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evaluation_name", default="prem_unified")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--cuda_devices", default=None)
    parser.add_argument("--max_frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--chunk_frames", type=int, default=2)
    parser.add_argument("--initial_chunk_frames", type=int, default=0)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--mcq_max_new_tokens", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prem_alpha", type=float, default=1.0, help="Multiplier on the learned PREM steer scale")
    parser.add_argument("--prem_override_alpha", action="store_true", help="Override alpha stored in the PREM checkpoint")
    parser.add_argument(
        "--prem_modulation",
        choices=["attention", "attention_kv", "attention_k", "attention_v"],
        default="attention",
        help="Attention projection modulation used by the checkpoint",
    )
    parser.add_argument("--prem_num_slots", type=int, default=4)
    parser.add_argument("--prem_mem_dim", type=int, default=128)
    parser.add_argument("--prem_layer_groups", type=int, default=1)
    parser.add_argument("--stream_chunk_tokens", type=int, default=32)
    parser.add_argument(
        "--visual_buffer_frames",
        type=int,
        default=None,
        help="Must match the checkpoint's fixed B budget; defaults to checkpoint metadata",
    )
    parser.add_argument(
        "--time_constrained",
        action="store_true",
        help="Answer each question using only frames at or before its query_time",
    )
    parser.add_argument("--prem_disable_anti_distractor", action="store_true")
    parser.add_argument("--prem_disable_novelty", action="store_true")
    parser.add_argument("--prem_disable_stability", action="store_true")
    parser.add_argument("--prem_disable_evidence_gate_write", action="store_true")
    parser.add_argument("--prem_uniform_write_route", action="store_true")
    parser.add_argument(
        "--profile_efficiency",
        action="store_true",
        help="Synchronize CUDA timing and record per-worker peak allocator memory",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mcq_max_new_tokens != 1:
        parser.error("Streaming MCQ evaluation requires --mcq_max_new_tokens 1 for constrained decoding")
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
