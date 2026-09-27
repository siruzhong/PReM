#!/usr/bin/env python
"""Budget-constrained end-time evaluation for unmodified Qwen-VL backbones.

This is an online baseline, not a recurrent-memory model.  Before the
question arrives it keeps a fixed, question-agnostic reservoir of video
frames. At end time the backbone answers from the question and that bounded raw
video buffer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from scripts.eval.eval_prem_online import (
    CausalVideoSource,
    efficiency_stats,
    extract_answer,
    get_video_grouped_chunk,
    mcq_option_token_ids,
    resolve_stream_frames,
    synchronize_cuda,
    write_result_files,
)
from scripts.eval.data_io import load_records
from scripts.eval.inference_mcq_vqa import get_sample_media_name, normalize_mcq_sample
from qwen_vl_utils import process_vision_info, qwen3_video_metadata
from scripts.eval.stream_buffer import stable_reservoir_indices


def uniform_subsample(items: list, target_count: int) -> list:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(items) <= target_count:
        return list(items)
    indices = torch.linspace(0, len(items) - 1, target_count).round().long().tolist()
    return [items[index] for index in indices]


def make_video_inputs(processor, frames: list, max_pixels: int, model_type: str, fps: float, metadata=None):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "max_frames": len(frames),
                    "max_pixels": max_pixels,
                }
            ],
        }
    ]
    _, video_inputs = process_vision_info(messages)
    video_kwargs = {}
    if model_type == "qwen3_vl":
        video_kwargs = {"video_metadata": [metadata], "do_sample_frames": False}
    else:
        video_kwargs = {"fps": [fps]}
    inputs = processor(
        text=["<|vision_start|><|video_pad|><|vision_end|>"],
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    return video_inputs, inputs.video_grid_thw


def visual_token_count(video_grid_thw: torch.Tensor, spatial_merge_size: int) -> int:
    per_video = video_grid_thw.to(dtype=torch.long).prod(dim=1)
    return int((per_video // (int(spatial_merge_size) ** 2)).sum().item())


def select_online_buffer(processor, frames: list, video_key: str, args, spatial_merge_size: int, model_type: str):
    indices = stable_reservoir_indices(len(frames), args.buffer_frames, video_key)
    buffer_frames = [frames[index] for index in indices]
    if len(buffer_frames) > 1 and len(buffer_frames) % 2:
        buffer_frames = buffer_frames[:-1]
        indices = indices[:-1]

    while True:
        video_info = {"video": buffer_frames, "fps": args.fps}
        metadata = qwen3_video_metadata(
            video_info,
            frame_indices=indices,
            total_num_frames=len(frames),
        )
        video_inputs, grid_thw = make_video_inputs(
            processor,
            buffer_frames,
            args.max_pixels,
            model_type,
            args.fps,
            metadata,
        )
        token_count = visual_token_count(grid_thw, spatial_merge_size)
        if token_count <= args.max_video_tokens:
            return buffer_frames, video_inputs, token_count, metadata
        if len(buffer_frames) <= 2:
            raise ValueError(
                f"Two buffered frames require {token_count} visual tokens, exceeding "
                f"--max_video_tokens={args.max_video_tokens}"
            )
        target = max(2, int(len(buffer_frames) * args.max_video_tokens / token_count))
        target = min(target, len(buffer_frames) - 1)
        if target % 2:
            target -= 1
        selected = torch.linspace(0, len(buffer_frames) - 1, max(2, target)).round().long().tolist()
        buffer_frames = [buffer_frames[index] for index in selected]
        indices = [indices[index] for index in selected]


def answer_from_buffer(model, processor, q_base: str, video_inputs, video_metadata, args) -> str:
    prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the option letter. "
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_inputs},
                {"type": "text", "text": prompt + q_base},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "Best option: ("
    video_kwargs = {}
    if model.config.model_type == "qwen3_vl":
        video_kwargs = {"video_metadata": [video_metadata], "do_sample_frames": False}
    else:
        video_kwargs = {"fps": [args.fps]}
    inputs = processor(
        text=[text],
        images=None,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)
    allowed_token_ids = mcq_option_token_ids(processor, q_base)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.mcq_max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: allowed_token_ids,
        )
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def answer_text_only(model, processor, q_base: str, args) -> str:
    prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the option letter. "
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt + q_base}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "Best option: ("
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    allowed_token_ids = mcq_option_token_ids(processor, q_base)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.mcq_max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: allowed_token_ids,
        )
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def run_worker(args):
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            "Qwen-VL online evaluation requires CUDA, but PyTorch cannot see a CUDA device. "
            f"CUDA_VISIBLE_DEVICES={visible!r}; check the requested GPU ids and the job allocation."
        )
    if args.profile_efficiency:
        torch.cuda.reset_peak_memory_stats()
    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "eager"
    model_type = AutoConfig.from_pretrained(args.model_path).model_type
    if model_type == "qwen3_vl":
        from transformers import Qwen3VLForConditionalGeneration

        model_cls = Qwen3VLForConditionalGeneration
    elif model_type == "qwen2_5_vl":
        model_cls = Qwen2_5_VLForConditionalGeneration
    elif model_type == "qwen2_vl":
        model_cls = Qwen2VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported Qwen-VL model_type: {model_type!r}")
    model = model_cls.from_pretrained(
        args.model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
    ).eval()
    processor_kwargs = {} if model_type == "qwen3_vl" else {"use_fast": False}
    processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)
    merge_size = int(model.config.vision_config.spatial_merge_size)

    rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    if args.time_constrained and any("query_time" not in row for row in rows):
        raise ValueError("--time_constrained requires query_time in every QA row")
    rows = get_video_grouped_chunk(rows, args.num_chunks, args.chunk_idx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / (f"{args.num_chunks}_{args.chunk_idx}.json" if args.num_chunks > 1 else "pred.json")
    signature = json.dumps(
        {
            "protocol": (
                f"{model_type}_text_only"
                if args.text_only
                else f"{model_type}_online_reservoir_endtime"
            ),
            "model_path": str(Path(args.model_path).resolve()),
            "max_frames": args.max_frames,
            "fps": args.fps,
            "buffer_frames": args.buffer_frames,
            "max_video_tokens": args.max_video_tokens,
            "max_pixels": args.max_pixels,
            "time_constrained": bool(args.time_constrained),
            "profile_efficiency": bool(args.profile_efficiency),
        },
        sort_keys=True,
    )
    done = set()
    if pred_file.exists() and not args.overwrite:
        existing = [json.loads(line) for line in open(pred_file)]
        if any(row.get("eval_signature") != signature for row in existing):
            raise RuntimeError(f"Existing predictions use a different evaluator: {pred_file}")
        done = {row["id"] for row in existing}

    current_video = None
    current_source = None
    current_visible_frames = []
    current_buffer = None
    current_stats = None
    with open(pred_file, "w" if args.overwrite else "a") as handle:
        for sample in rows:
            if sample["id"] in done:
                continue
            try:
                q_base = sample["question"] if "question" in sample else sample["question1"]
                if args.text_only:
                    synchronize_cuda(args.profile_efficiency)
                    answer_start = time.perf_counter()
                    prediction = answer_text_only(model, processor, q_base, args)
                    synchronize_cuda(args.profile_efficiency)
                    row = {
                        "id": sample["id"],
                        "question": q_base,
                        "answer": sample["answer"],
                        "pred": prediction,
                        "eval_signature": signature,
                        "state_reused": False,
                        "stream_update_seconds": 0.0,
                        "answer_seconds": time.perf_counter() - answer_start,
                        "time_constrained": bool(args.time_constrained),
                        "query_time": sample.get("query_time"),
                        "causal_cutoff_seconds": None,
                        "protocol": f"{model_type}_text_only",
                        "num_stream_frames": 0,
                        "visible_frame_count": 0,
                        "buffered_frames": 0,
                        "visual_tokens": 0,
                        **efficiency_stats(args.profile_efficiency),
                        **{key: sample[key] for key in ("question_category", "question_type", "question_subtype", "episode_id", "level", "topic_category", "duration_group", "duration") if key in sample},
                    }
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    continue
                video_key = get_sample_media_name(sample)
                same_video = video_key == current_video
                if not same_video:
                    current_source = None
                    current_visible_frames = []
                    current_buffer = None
                    current_video = video_key
                state_reused = same_video and current_buffer is not None
                synchronize_cuda(args.profile_efficiency)
                update_start = time.perf_counter()
                if args.time_constrained:
                    if current_source is None:
                        current_source = CausalVideoSource(
                            args.video_dir, sample, args.max_frames, args.fps
                        )
                    new_frames = current_source.take_until(float(sample["query_time"]))
                    current_visible_frames.extend(new_frames)
                    state_reused = same_video and not new_frames and current_buffer is not None
                    current_buffer, video_inputs, visual_tokens, video_metadata = select_online_buffer(
                        processor, current_visible_frames, video_key, args, merge_size, model_type
                    )
                    current_stats = {
                        "protocol": f"{model_type}_causal_query_time",
                        "num_stream_frames": current_source.visible_count,
                        "visible_frame_count": current_source.visible_count,
                        "sampling_fps": float(args.fps),
                        "buffered_frames": len(current_buffer),
                        "visual_tokens": visual_tokens,
                        "buffer_frames_limit": args.buffer_frames,
                        "max_video_tokens": args.max_video_tokens,
                    }
                elif not state_reused:
                    frames = resolve_stream_frames(args.video_dir, sample, args.max_frames, args.fps)
                    current_buffer, video_inputs, visual_tokens, video_metadata = select_online_buffer(
                        processor, frames, video_key, args, merge_size, model_type
                    )
                    current_stats = {
                        "protocol": f"{model_type}_online_reservoir_endtime",
                        "num_stream_frames": len(frames),
                        "visible_frame_count": len(frames),
                        "sampling_fps": float(args.fps),
                        "buffered_frames": len(current_buffer),
                        "visual_tokens": visual_tokens,
                        "buffer_frames_limit": args.buffer_frames,
                        "max_video_tokens": args.max_video_tokens,
                    }
                    update_seconds = time.perf_counter() - update_start
                    current_video = video_key
                synchronize_cuda(args.profile_efficiency)
                update_seconds = time.perf_counter() - update_start

                synchronize_cuda(args.profile_efficiency)
                answer_start = time.perf_counter()
                prediction = answer_from_buffer(
                    model,
                    processor,
                    q_base,
                    video_inputs,
                    video_metadata,
                    args,
                )
                synchronize_cuda(args.profile_efficiency)
                row = {
                    "id": sample["id"],
                    "question": q_base,
                    "answer": sample["answer"],
                    "pred": prediction,
                    "eval_signature": signature,
                    "state_reused": state_reused,
                    "stream_update_seconds": update_seconds,
                    "answer_seconds": time.perf_counter() - answer_start,
                    "time_constrained": bool(args.time_constrained),
                    "query_time": sample.get("query_time"),
                    "causal_cutoff_seconds": float(sample["query_time"]) if args.time_constrained else None,
                    **{key: sample[key] for key in ("question_category", "question_type", "question_subtype", "episode_id", "level", "topic_category", "duration_group", "duration") if key in sample},
                    **current_stats,
                    **efficiency_stats(args.profile_efficiency),
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            except Exception as exc:
                raise RuntimeError(
                    f"Qwen-VL-online evaluation failed for id={sample.get('id')}: {type(exc).__name__}: {exc}"
                ) from exc


def launch(args):
    output_base = Path(args.output_dir) / args.evaluation_name / args.dataset
    if not args.cuda_devices:
        worker_args = argparse.Namespace(**vars(args))
        worker_args.output_dir = str(output_base)
        run_worker(worker_args)
        write_result_files(output_base, output_base / "pred.json", args.gt_file)
        return

    devices = [device.strip() for device in args.cuda_devices.split(",") if device.strip()]
    if args.num_chunks > len(devices):
        raise ValueError(f"num_chunks={args.num_chunks} exceeds cuda_devices={devices}")
    procs = []
    for index in range(args.num_chunks):
        cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--dataset", args.dataset,
            "--model_path", args.model_path,
            "--video_dir", args.video_dir,
            "--gt_file", args.gt_file,
            "--output_dir", str(output_base),
            "--num_chunks", str(args.num_chunks),
            "--chunk_idx", str(index),
            "--max_frames", str(args.max_frames),
            "--fps", str(args.fps),
            "--buffer_frames", str(args.buffer_frames),
            "--max_video_tokens", str(args.max_video_tokens),
            "--max_pixels", str(args.max_pixels),
            "--mcq_max_new_tokens", str(args.mcq_max_new_tokens),
        ]
        if args.time_constrained:
            cmd.append("--time_constrained")
        if args.text_only:
            cmd.append("--text_only")
        if args.profile_efficiency:
            cmd.append("--profile_efficiency")
        if args.overwrite:
            cmd.append("--overwrite")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index])
        print("[exec] CUDA_VISIBLE_DEVICES=" + devices[index] + " " + " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
    failed = []
    for index, proc in enumerate(procs):
        result = proc.wait()
        if result != 0:
            failed.append((index, result))
    if failed:
        raise RuntimeError(f"Qwen-VL-online workers failed: {failed}")
    if args.num_chunks > 1:
        pred_file = output_base / "pred.json"
        with open(pred_file, "w") as merged:
            for index in range(args.num_chunks):
                chunk = output_base / f"{args.num_chunks}_{index}.json"
                if not chunk.exists():
                    raise FileNotFoundError(chunk)
                merged.write(chunk.read_text())
    else:
        pred_file = output_base / "pred.json"
    write_result_files(output_base, pred_file, args.gt_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_path", default="ckpt/Qwen2-VL-7B-Instruct")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evaluation_name", default="qwen_vl_online_reservoir_endtime")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--cuda_devices", default=None)
    parser.add_argument("--max_frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--buffer_frames", type=int, default=16)
    parser.add_argument("--max_video_tokens", type=int, default=11520)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--mcq_max_new_tokens", type=int, default=1)
    parser.add_argument(
        "--time_constrained",
        action="store_true",
        help="Answer each question using only frames at or before its query_time",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument(
        "--profile_efficiency",
        action="store_true",
        help="Synchronize CUDA timing and record per-worker peak allocator memory",
    )
    args = parser.parse_args()
    if args.mcq_max_new_tokens != 1:
        parser.error("Qwen-VL-online MCQ evaluation requires --mcq_max_new_tokens 1")
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
