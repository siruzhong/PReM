#!/usr/bin/env python
"""Stateful streaming evaluation for Flash-VStream.

This turns the realtime CLI demo protocol into a batch QA evaluator:
chronological frames update Flash Memory first, then the question is answered
from the current memory state.
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
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

from scripts.eval.data_io import load_records

from scripts.eval.eval_prem_online import (
    CausalVideoSource,
    get_video_grouped_chunk,
    mcq_option_token_ids,
    resolve_stream_frames,
    split_chunks,
    write_result_files,
)
from scripts.eval.inference_mcq_vqa import get_sample_media_name, normalize_mcq_sample
from models.flash_memory_constants import DEFAULT_FLASH_MEMORY_CONFIG
from models.vstream_qwen2vl_model import FlashVStreamQwen2VLConfig
from models.vstream_qwen2vl_processor import FlashVStreamQwen2VLProcessor
from models.vstream_qwen2vl_realtime import FlashVStreamQwen2VLModel


def load_model(args):
    if not torch.cuda.is_available():
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        raise RuntimeError(
            "Flash-VStream evaluation requires CUDA, but PyTorch cannot see a CUDA device. "
            f"CUDA_VISIBLE_DEVICES={visible!r}; check the requested GPU ids and the job allocation."
        )
    use_flash_attn = importlib.util.find_spec("flash_attn") is not None
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    config = FlashVStreamQwen2VLConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG.copy()
    model = FlashVStreamQwen2VLModel.from_pretrained(
        args.model_path,
        config=config,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).eval()
    processor = FlashVStreamQwen2VLProcessor.from_pretrained(args.qwen_model)
    model.use_video_streaming_mode = True
    model.video_embedding_memory = []
    return model, processor, config.vision_config.flash_memory_config


def update_flash_memory(model, processor, flash_memory_config: dict, frames: list, args):
    model.video_embedding_memory = []
    frame_cnt = 0
    opened_frames = [
        Image.open(frame).convert("RGB") if isinstance(frame, (str, os.PathLike)) else frame
        for frame in frames
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": opened_frames,
                    "max_frames": len(opened_frames),
                    "max_pixels": args.max_pixels,
                }
            ],
        }
    ]
    _, video_inputs = process_vision_info(messages)
    if video_inputs is None or len(video_inputs) != 1:
        count = 0 if video_inputs is None else len(video_inputs)
        raise ValueError(f"Expected one preprocessed Flash video, got {count}")
    stream_frames = video_inputs[0]
    stats = {
        "num_candidate_frames": len(frames),
        "num_stream_frames": len(stream_frames),
        "num_stream_chunks": 0,
        "sampling_fps": float(args.fps),
        "max_pixels": int(args.max_pixels),
        "initial_chunk_frames": int(args.initial_chunk_frames),
        "chunk_frames": int(args.chunk_frames),
        "protocol": "flash_realtime_endtime_pixels",
    }
    if hasattr(stream_frames, "shape") and len(stream_frames.shape) >= 3:
        stats["processed_frame_height"] = int(stream_frames.shape[-2])
        stats["processed_frame_width"] = int(stream_frames.shape[-1])
    elif len(stream_frames) > 0 and isinstance(stream_frames[0], Image.Image):
        stats["processed_frame_width"] = int(stream_frames[0].width)
        stats["processed_frame_height"] = int(stream_frames[0].height)
    for chunk in split_chunks(stream_frames, args.chunk_frames, args.initial_chunk_frames):
        video_inputs = processor.image_processor(
            images=None,
            videos=chunk,
            return_tensors="pt",
            additional_pool_size=flash_memory_config["flash_memory_temporal_poolsize"],
        )
        with torch.inference_mode():
            model.embed_new_video_clip(**video_inputs, start_idx=frame_cnt)
        frame_cnt += len(chunk)
        stats["num_stream_chunks"] += 1
    stats["state_bytes"] = int(
        sum(
            item.numel() * item.element_size()
            for item in model.video_embedding_memory
            if torch.is_tensor(item)
        )
    )
    return stats


def answer_from_flash_memory(model, processor, flash_memory_config: dict, q_base: str, args):
    prompt = "Select the best answer to the following multiple-choice question based on the video. Respond with only the option letter. "
    question = prompt + q_base
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<|vision_start|><|video_pad|><|vision_end|>" + question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text += "Best option: ("
    cuda_list = model.get_video_embedding_memory_cuda_list()
    if cuda_list is None or len(cuda_list) == 0:
        raise RuntimeError("Flash memory is empty before answer")
    if args.flash_dummy_video_tokens > 0:
        video_embed_size = int(args.flash_dummy_video_tokens)
    else:
        tem_thw = cuda_list[1]
        spa_thw = cuda_list[5]
        video_embed_size = int((tem_thw.prod() // 4 + spa_thw.prod() // 4).item())
    inputs = processor(
        text=[text],
        images=None,
        videos=None,
        padding=True,
        return_tensors="pt",
        flash_memory_config=flash_memory_config,
        dummy_video_tokens=video_embed_size * 4,
    ).to("cuda")
    allowed_token_ids = mcq_option_token_ids(processor, q_base)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.mcq_max_new_tokens,
            do_sample=False,
            use_cache=False,
            prefix_allowed_tokens_fn=lambda _batch_id, _input_ids: allowed_token_ids,
        )
    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def run_worker(args):
    model, processor, flash_memory_config = load_model(args)
    rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    if args.time_constrained and any("query_time" not in row for row in rows):
        raise ValueError("--time_constrained requires query_time in every QA row")
    rows = get_video_grouped_chunk(rows, args.num_chunks, args.chunk_idx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_file = output_dir / (f"{args.num_chunks}_{args.chunk_idx}.json" if args.num_chunks > 1 else "pred.json")
    eval_signature = json.dumps(
        {
            "protocol": "flash_realtime_endtime_pixels",
            "model_path": str(Path(args.model_path).resolve()),
            "qwen_model": str(Path(args.qwen_model).resolve()),
            "max_frames": args.max_frames,
            "max_pixels": args.max_pixels,
            "fps": args.fps,
            "chunk_frames": args.chunk_frames,
            "initial_chunk_frames": args.initial_chunk_frames,
            "flash_dummy_video_tokens": args.flash_dummy_video_tokens,
            "time_constrained": bool(args.time_constrained),
        },
        sort_keys=True,
    )
    done = set()
    if pred_file.exists() and not args.overwrite:
        existing = [json.loads(line) for line in open(pred_file)]
        if any(row.get("eval_signature") != eval_signature for row in existing):
            raise RuntimeError(f"Existing predictions use a different evaluator/checkpoint: {pred_file}")
        done = {row["id"] for row in existing}
    mode = "w" if args.overwrite else "a"
    current_video = None
    current_source = None
    current_cumulative_frames = None
    current_stats = None
    current_update_seconds = None
    with open(pred_file, mode) as f:
        for sample in tqdm(rows, desc=f"cuda:{args.chunk_idx}"):
            if sample["id"] in done:
                continue
            try:
                video_key = get_sample_media_name(sample)
                same_video = video_key == current_video
                if not same_video:
                    current_source = None
                    current_cumulative_frames = None
                    current_stats = None
                    current_video = video_key
                state_reused = same_video and current_stats is not None
                update_start = time.perf_counter()
                if args.time_constrained:
                    if current_source is None:
                        current_source = CausalVideoSource(args.video_dir, sample, args.max_frames, args.fps)
                        current_cumulative_frames = []
                    new_frames = current_source.take_until(float(sample["query_time"]))
                    if new_frames:
                        current_cumulative_frames.extend(new_frames)
                        current_stats = update_flash_memory(model, processor, flash_memory_config, current_cumulative_frames, args)
                    visible_frame_count = current_source.visible_count
                elif not state_reused:
                    frames = resolve_stream_frames(args.video_dir, sample, args.max_frames, args.fps)
                    current_stats = update_flash_memory(model, processor, flash_memory_config, frames, args)
                    visible_frame_count = len(frames)
                else:
                    visible_frame_count = int(current_stats.get("num_stream_frames", 0))
                current_update_seconds = time.perf_counter() - update_start
                stats = current_stats
                update_seconds = 0.0 if state_reused else float(current_update_seconds)
                q_base = sample["question"] if "question" in sample else sample["question1"]
                answer_start = time.perf_counter()
                pred = answer_from_flash_memory(model, processor, flash_memory_config, q_base, args)
                answer_seconds = time.perf_counter() - answer_start
                row = {
                    "id": sample["id"],
                    "question": q_base,
                    "answer": sample["answer"],
                    "pred": pred,
                    "eval_signature": eval_signature,
                    "state_reused": state_reused,
                    "stream_update_seconds": update_seconds,
                    "mean_update_seconds": update_seconds / max(1, stats["num_stream_chunks"]),
                    "answer_seconds": answer_seconds,
                    "visible_frame_count": visible_frame_count,
                    **{k: sample[k] for k in (
                        "variant",
                        "stress_variant",
                        "question_category",
                        "level",
                        "topic_category",
                        "duration_group",
                        "duration",
                        "num_frames",
                    ) if k in sample},
                    **stats,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
            except Exception as exc:
                raise RuntimeError(
                    f"Flash streaming evaluation failed for id={sample.get('id')}: "
                    f"{type(exc).__name__}: {exc}"
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
                "--dataset", args.dataset,
                "--model_path", args.model_path,
                "--qwen_model", args.qwen_model,
                "--video_dir", args.video_dir,
                "--gt_file", args.gt_file,
                "--output_dir", str(output_base),
                "--num_chunks", str(num_chunks),
                "--chunk_idx", str(idx),
                "--max_frames", str(args.max_frames),
                "--max_pixels", str(args.max_pixels),
                "--fps", str(args.fps),
                "--initial_chunk_frames", str(args.initial_chunk_frames),
                "--chunk_frames", str(args.chunk_frames),
                "--mcq_max_new_tokens", str(args.mcq_max_new_tokens),
                "--flash_dummy_video_tokens", str(args.flash_dummy_video_tokens),
            ]
            if args.time_constrained:
                cmd.append("--time_constrained")
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
            raise RuntimeError(f"Flash streaming eval workers failed: {failed}")
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
    parser.add_argument("--model_path", default="ckpt/Flash-VStream-Qwen-7b")
    parser.add_argument("--qwen_model", default="ckpt/Qwen2-VL-7B-Instruct")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--evaluation_name", default="flash_vstream_streaming_pixels200704_constrained")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--cuda_devices", default=None)
    parser.add_argument("--max_frames", type=int, default=240)
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=200704,
        help="Resize Flash realtime input frames before chunking; 200704 matches the released realtime CLI.",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--initial_chunk_frames", type=int, default=120)
    parser.add_argument("--chunk_frames", type=int, default=1)
    parser.add_argument(
        "--mcq_max_new_tokens",
        type=int,
        default=1,
        help="Flash-VStream realtime demo generates one MCQ option token; keep this at 1 for aligned streaming eval.",
    )
    parser.add_argument(
        "--flash_dummy_video_tokens",
        type=int,
        default=0,
        help="0 computes the current Flash Memory visual token count dynamically; set 10800 to mimic the demo hard-code.",
    )
    parser.add_argument("--time_constrained", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.mcq_max_new_tokens != 1:
        parser.error("Flash streaming MCQ evaluation requires --mcq_max_new_tokens 1")
    if args.worker:
        run_worker(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
