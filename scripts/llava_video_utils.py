"""Shared frame sampling and prompt construction for LLaVA-Video PReM runs."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image

from scripts.eval.llava_video_contract import (
    LLAVA_VIDEO_DIRECTORY_FPS,
    format_time_instruction,
)


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")
FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IGNORE_INDEX = -100


@contextmanager
def materialized_llava_qwen_loader(model_class):
    """Load the full fine-tuned checkpoint without the broken meta path."""
    upstream_from_pretrained = model_class.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs.pop("device_map", None)
        kwargs["low_cpu_mem_usage"] = False
        return upstream_from_pretrained(*args, **kwargs)

    model_class.from_pretrained = from_pretrained
    try:
        yield
    finally:
        model_class.from_pretrained = upstream_from_pretrained


def validate_materialized_vision_tower(model) -> None:
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        raise RuntimeError("LLaVA-Video vision tower was not loaded")
    meta_parameters = [
        name
        for name, parameter in vision_tower.vision_tower.named_parameters()
        if parameter.is_meta
    ]
    if meta_parameters:
        raise RuntimeError(
            "LLaVA-Video vision tower has unloaded meta parameters: "
            + ", ".join(meta_parameters[:5])
        )


def numeric_duration_hint(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def resolve_video_path(video_root: Path, video_name: str) -> Path:
    direct = video_root / video_name
    candidates = [direct, direct.with_suffix("") if direct.suffix else direct]
    stem = candidates[-1]
    candidates.extend(Path(str(stem) + extension) for extension in VIDEO_EXTENSIONS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir() and any(
            child.is_file() and child.suffix.lower() in FRAME_EXTENSIONS
            for child in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError(f"Cannot resolve LLaVA-Video media {video_name!r} under {video_root}")


def _sample_indices(total: int, source_fps: float, target_fps: float, max_frames: int) -> list[int]:
    if total <= 0:
        raise ValueError("Video contains no frames")
    requested = max(1, round(total / max(source_fps, 1e-6) * target_fps))
    requested = min(total, requested, max_frames if max_frames > 0 else requested)
    return torch.linspace(0, total - 1, requested).round().long().tolist()


def llava_official_frame_indices(
    length: int, count: int, *, force_sample: bool = False
) -> np.ndarray:
    """Match official LLaVA-Video Base: linspace over native frames, not a 1fps grid."""
    if length <= 0:
        raise ValueError("Cannot sample an empty video")
    count = max(1, count)
    if not force_sample:
        count = min(length, count)
    return np.linspace(0, length - 1, count).round().astype(int)


def _directory_frame_paths(media_path: Path) -> list[Path]:
    return sorted(
        child
        for child in media_path.iterdir()
        if child.is_file() and child.suffix.lower() in FRAME_EXTENSIONS
    )


def _empty_rgb_clip(source: np.ndarray | None = None) -> np.ndarray:
    if source is not None and source.ndim == 4:
        return source[:0]
    return np.zeros((0, 1, 1, 3), dtype=np.uint8)


def _gather_indexed_frames(
    unique_indices: list[int],
    loaded: dict[int, np.ndarray],
    selected: list[int] | np.ndarray,
) -> np.ndarray:
    if len(selected) == 0:
        sample = loaded[unique_indices[0]] if unique_indices else None
        return _empty_rgb_clip(None if sample is None else sample[None, ...])
    return np.stack([loaded[int(index)] for index in selected])


def load_video_frames(
    media_path: Path,
    *,
    max_frames: int,
    fps: float,
    duration_hint: float | None = None,
    return_metadata: bool = False,
):
    duration_hint = numeric_duration_hint(duration_hint)
    if media_path.is_dir():
        paths = sorted(
            child for child in media_path.iterdir()
            if child.is_file() and child.suffix.lower() in FRAME_EXTENSIONS
        )
        indices = _sample_indices(len(paths), 1.0, fps, max_frames)
        frames = np.stack([
            np.asarray(Image.open(paths[index]).convert("RGB")) for index in indices
        ])
        timestamps = [float(index) for index in indices]
        duration = duration_hint if duration_hint is not None else float(len(paths))
        return (frames, timestamps, duration) if return_metadata else frames
    reader = VideoReader(str(media_path), ctx=cpu(0), num_threads=1)
    source_fps = max(float(reader.get_avg_fps()), 1e-6)
    indices = _sample_indices(len(reader), source_fps, fps, max_frames)
    frames = reader.get_batch(indices).asnumpy()
    timestamps = [float(index) / source_fps for index in indices]
    duration = len(reader) / source_fps
    return (frames, timestamps, duration) if return_metadata else frames


def load_llava_official_frames(
    media_path: Path,
    max_frames: int,
    *,
    force_sample: bool = False,
    directory_fps: float = LLAVA_VIDEO_DIRECTORY_FPS,
    duration_hint: float | None = None,
):
    """Decoder-visible frames for official LLaVA-Video evaluation (Base and PReM)."""
    duration_hint = numeric_duration_hint(duration_hint)
    if media_path.is_dir():
        frame_paths = _directory_frame_paths(media_path)
        if not frame_paths:
            raise ValueError("Video contains no frames")
        indices = llava_official_frame_indices(
            len(frame_paths), max_frames, force_sample=force_sample
        )
        frames = np.stack([
            np.asarray(Image.open(frame_paths[index]).convert("RGB"))
            for index in indices
        ])
        source_fps = max(float(directory_fps), 1e-6)
        timestamps = [float(index) / source_fps for index in indices]
        inferred_duration = len(frame_paths) / source_fps
        duration = duration_hint if duration_hint is not None else inferred_duration
        return frames, timestamps, duration

    reader = VideoReader(str(media_path), ctx=cpu(0), num_threads=1)
    indices = llava_official_frame_indices(
        len(reader), max_frames, force_sample=force_sample
    )
    source_fps = max(float(reader.get_avg_fps()), 1e-6)
    frames = reader.get_batch(indices).asnumpy()
    timestamps = [float(index) / source_fps for index in indices]
    duration = len(reader) / source_fps
    return frames, timestamps, duration


def load_llava_writer_and_decoder_frames(
    media_path: Path,
    *,
    writer_max_frames: int,
    writer_fps: float,
    decoder_budget: int,
    duration_hint: float | None = None,
    directory_fps: float = LLAVA_VIDEO_DIRECTORY_FPS,
    force_sample: bool = True,
):
    """Load the 1fps writer stream and the official decoder-B frames independently.

    Memory still writes a dense 1fps (cap `writer_max_frames`) stream. The decoder
    buffer matches LLaVA-Video Base: `force_sample` linspace over native frames,
    including the official time-instruction timestamps.
    """
    duration_hint = numeric_duration_hint(duration_hint)
    if decoder_budget < 0:
        raise ValueError(f"decoder_budget must be non-negative, got {decoder_budget}")

    if media_path.is_dir():
        frame_paths = _directory_frame_paths(media_path)
        if not frame_paths:
            raise ValueError("Video contains no frames")
        source_fps = max(float(directory_fps), 1e-6)
        writer_indices = _sample_indices(
            len(frame_paths), source_fps, writer_fps, writer_max_frames
        )
        decoder_indices = (
            llava_official_frame_indices(
                len(frame_paths), decoder_budget, force_sample=force_sample
            )
            if decoder_budget > 0
            else np.zeros((0,), dtype=int)
        )
        unique_indices = sorted(
            {int(index) for index in writer_indices}
            | {int(index) for index in decoder_indices}
        )
        loaded = {
            index: np.asarray(Image.open(frame_paths[index]).convert("RGB"))
            for index in unique_indices
        }
        writer_frames = _gather_indexed_frames(unique_indices, loaded, writer_indices)
        decoder_frames = _gather_indexed_frames(unique_indices, loaded, decoder_indices)
        writer_timestamps = [float(index) / source_fps for index in writer_indices]
        decoder_timestamps = [float(index) / source_fps for index in decoder_indices]
        inferred_duration = len(frame_paths) / source_fps
        duration = duration_hint if duration_hint is not None else inferred_duration
        return (
            writer_frames,
            writer_timestamps,
            decoder_frames,
            decoder_timestamps,
            duration,
        )

    reader = VideoReader(str(media_path), ctx=cpu(0), num_threads=1)
    total = len(reader)
    source_fps = max(float(reader.get_avg_fps()), 1e-6)
    writer_indices = _sample_indices(total, source_fps, writer_fps, writer_max_frames)
    decoder_indices = (
        llava_official_frame_indices(total, decoder_budget, force_sample=force_sample)
        if decoder_budget > 0
        else np.zeros((0,), dtype=int)
    )
    unique_indices = sorted(
        {int(index) for index in writer_indices}
        | {int(index) for index in decoder_indices}
    )
    batch = reader.get_batch(unique_indices).asnumpy()
    loaded = {
        index: batch[position] for position, index in enumerate(unique_indices)
    }
    writer_frames = _gather_indexed_frames(unique_indices, loaded, writer_indices)
    decoder_frames = _gather_indexed_frames(unique_indices, loaded, decoder_indices)
    writer_timestamps = [float(index) / source_fps for index in writer_indices]
    decoder_timestamps = [float(index) / source_fps for index in decoder_indices]
    duration = total / source_fps
    return (
        writer_frames,
        writer_timestamps,
        decoder_frames,
        decoder_timestamps,
        duration,
    )


def select_stream_buffer_frames(
    frames: np.ndarray,
    timestamps: list[float],
    budget: int,
    video_key: str,
) -> tuple[np.ndarray, list[float]]:
    """Keep a question-agnostic online buffer of at most `budget` stream frames."""
    from scripts.eval.stream_buffer import stable_reservoir_indices

    if len(frames) != len(timestamps):
        raise ValueError(
            f"Frame/timestamp length mismatch: {len(frames)} != {len(timestamps)}"
        )
    if budget <= 0:
        return frames[:0], []
    indices = stable_reservoir_indices(len(frames), budget, video_key)
    return frames[np.asarray(indices, dtype=int)], [
        float(timestamps[index]) for index in indices
    ]


def load_llava_base_online_frames(
    media_path: Path,
    *,
    stream_max_frames: int,
    fps: float,
    decoder_budget: int,
    video_key: str,
    duration_hint: float | None = None,
):
    """Decoder frames for LLaVA online: 1fps cap `stream_max_frames`, then reservoir to B."""
    stream_frames, stream_timestamps, duration = load_video_frames(
        media_path,
        max_frames=stream_max_frames,
        fps=fps,
        duration_hint=duration_hint,
        return_metadata=True,
    )
    frames, timestamps = select_stream_buffer_frames(
        stream_frames, stream_timestamps, decoder_budget, video_key
    )
    return frames, timestamps, duration


def uniformly_select(
    frames: np.ndarray,
    budget: int,
    *,
    force_sample: bool = False,
) -> np.ndarray:
    if budget <= 0:
        return frames[:0]
    if len(frames) <= budget and not force_sample:
        return frames
    indices = torch.linspace(0, len(frames) - 1, budget).round().long().tolist()
    return frames[indices]


def uniformly_select_with_timestamps(
    frames: np.ndarray,
    timestamps: list[float],
    budget: int,
    *,
    force_sample: bool = False,
) -> tuple[np.ndarray, list[float]]:
    if len(frames) != len(timestamps):
        raise ValueError(
            f"Frame/timestamp length mismatch: {len(frames)} != {len(timestamps)}"
        )
    if budget <= 0:
        return frames[:0], []
    if len(frames) <= budget and not force_sample:
        return frames, list(timestamps)
    indices = torch.linspace(0, len(frames) - 1, budget).round().long().tolist()
    return frames[indices], [float(timestamps[index]) for index in indices]


def preprocess_frames(image_processor, frames: np.ndarray, device, dtype=torch.bfloat16):
    pixels = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    return pixels.to(device=device, dtype=dtype)


def strip_media_marker(text: str) -> str:
    return (
        str(text)
        .replace("<image>\n", "")
        .replace("<video>\n", "")
        .replace("<image>", "")
        .replace("<video>", "")
        .strip()
    )


def _find_last_subsequence(values: list[int], target: list[int]) -> int | None:
    if not target:
        return None
    for start in range(len(values) - len(target), -1, -1):
        if values[start : start + len(target)] == target:
            return start
    return None


def build_training_inputs(row: dict, resources, pixels: torch.Tensor, device):
    conversation = resources.conversation_templates["qwen_1_5"].copy()
    first_user = True
    assistant_texts = []
    for turn in row["conversations"]:
        role = str(turn.get("from", "")).lower()
        text = strip_media_marker(turn.get("value", ""))
        if role in {"human", "user"}:
            if first_user:
                text = resources.default_image_token + "\n" + text
                first_user = False
            conversation.append_message(conversation.roles[0], text)
        elif role in {"gpt", "assistant"}:
            assistant_texts.append(text)
            conversation.append_message(conversation.roles[1], text)
        else:
            raise ValueError(f"Unsupported conversation role {turn.get('from')!r}")
    prompt = conversation.get_prompt()
    input_ids = resources.tokenizer_image_token(
        prompt,
        resources.tokenizer,
        resources.image_token_index,
        return_tensors="pt",
    ).unsqueeze(0)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    values = input_ids[0].tolist()
    first_answer = input_ids.shape[1]
    search_end = len(values)
    for answer in reversed(assistant_texts):
        answer_ids = resources.tokenizer(answer, add_special_tokens=False).input_ids
        start = _find_last_subsequence(values[:search_end], answer_ids)
        if start is None:
            continue
        end = min(start + len(answer_ids), input_ids.shape[1])
        labels[0, start:end] = input_ids[0, start:end]
        first_answer = min(first_answer, start)
        search_end = start
    if not bool(labels.ne(IGNORE_INDEX).any()):
        raise ValueError(f"No assistant labels found for sample {row.get('id')}")
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "images": pixels,
        "modalities": ["video"],
        "image_token_index": resources.image_token_index,
    }, int(first_answer)


def build_answer_inputs(
    question: str,
    resources,
    pixels: torch.Tensor,
    option_count: int,
    device,
    *,
    timestamps: list[float] | None = None,
    duration: float | None = None,
):
    labels = ", ".join(chr(65 + index) for index in range(option_count))
    instruction = (
        "Select the best answer to the following multiple-choice question "
        f"based on the video. Respond with only the letter ({labels})."
    )
    if (
        bool(getattr(resources.model.config, "add_time_instruction", True))
        and timestamps is not None
        and duration is not None
    ):
        video_context = format_time_instruction(duration, timestamps, len(pixels)) + " "
    else:
        video_context = f"{len(pixels)} frames are uniformly sampled from the video. "
    prompt_question = (
        resources.default_image_token + "\n" + video_context + instruction + "\n" + question
    )
    conversation = resources.conversation_templates["qwen_1_5"].copy()
    conversation.append_message(conversation.roles[0], prompt_question)
    conversation.append_message(conversation.roles[1], None)
    input_ids = resources.tokenizer_image_token(
        conversation.get_prompt(),
        resources.tokenizer,
        resources.image_token_index,
        return_tensors="pt",
    ).unsqueeze(0).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "images": pixels,
        "modalities": ["video"],
        "image_token_index": resources.image_token_index,
    }


def prefix_query_embeddings(model, input_ids: torch.Tensor, prompt_length: int, image_token_index: int):
    ids = input_ids[0, :prompt_length]
    ids = ids[ids != image_token_index]
    embeddings = model.backbone.get_model().embed_tokens(ids)
    return embeddings.mean(dim=0, keepdim=True)
