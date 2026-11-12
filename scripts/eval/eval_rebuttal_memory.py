#!/usr/bin/env python
"""PReM rebuttal: frame-cap scaling, evidence retention, and pooled writers.

Separate from the registered Table 2 entry point. Natural-video runs keep the
decoder's original 240-frame reservoir fixed. The needle probe uses annotated
LVBench evidence and an unrelated, fixed final buffer. Results are JSONL;
sample selection and cache identities are independent of model correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoProcessor

from models.prem_writer_controls import install_pooled_writer
from scripts.eval.data_io import load_records
from scripts.eval.eval_prem_online import (
    chunk_to_frame_embeds, extract_answer, get_video_grouped_chunk,
    resolve_video_source, update_visual_buffer,
)
from scripts.eval.eval_same_backbone_online import answer_prem, answer_token_readout, load_prem_wrapper
from scripts.eval.inference_mcq_vqa import get_sample_media_name, normalize_mcq_sample
from scripts.eval.input_alignment import file_sha256
from scripts.eval.prem_checkpoint import normalize_prem_checkpoint, validate_prem_checkpoint
from scripts.eval.rebuttal_input_audit import load_frames_with_manifest, save_frame_manifest, tensor_sha256


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_model(args):
    ckpt = normalize_prem_checkpoint(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
    variant = ckpt.get("writer_variant", "prem")
    is_token = ckpt.get("baseline_method") == "token_readout"
    args.read_interface = "in_sequence_tokens" if is_token else "prefix_kv"
    if not args.limit_videos:
        expected = {"data_protocol": "rebuttal_decontaminated_v1", "source_train_samples": 8953,
                    "source_train_qa": 17849, "train_steps": 2232, "global_batch_size": 8,
                    "num_slots": ckpt.get("num_slots") if is_token else 1,
                    "mem_dim": 128, "max_frames": 240, "skipped": 0,
                    "train_manifest_sha256": "e630ec50753fcf152263415950c7ba59a97989f92f452375549237c90916d40f"}
        mismatches = {key: (ckpt.get(key), value) for key, value in expected.items() if ckpt.get(key) != value}
        if mismatches:
            raise ValueError(f"Formal rebuttal checkpoint contract mismatch: {mismatches}")
    if is_token:
        expected = {"complete": True, "read_interface": "in_sequence_tokens",
                    "memory_prompt_tokens": 10 * int(ckpt.get("num_slots", 0)), "prem_modulation": "none",
                    "prem_layer_groups": 4, "max_memory_tokens": 128,
                    "pred_weight": 0.2, "pred_tokens": 4}
        if any(ckpt.get(key) != value for key, value in expected.items()):
            raise ValueError("Token-Readout checkpoint recipe mismatch")
        if ckpt.get("num_slots") not in (1, 4, 8):
            raise ValueError("Token-Readout sweep supports K in {1, 4, 8}")
        args.token_readout_position = ckpt.get("token_readout_position", "answer")
    elif variant == "prem":
        validate_prem_checkpoint(ckpt)
    else:
        if variant not in {"temporal_mean", "ema"} or ckpt["writer_mode"] != "pooled_normalized_feature":
            raise ValueError("Unrecognised writer control checkpoint")
        # Check shared training contract separately from the explicit control identity.
        validate_prem_checkpoint({**ckpt, "writer_mode": "temporal_mean_per_step"})
    if (not is_token and ckpt["prem_modulation"] != "attention_kv") or ckpt["visual_buffer_frames"] != 16:
        raise ValueError("This rebuttal protocol requires KV prefix steering and B=16")
    model = load_prem_wrapper(args)
    mem = model.build_prem_memory(
        num_slots=ckpt["num_slots"], mem_dim=ckpt["mem_dim"],
        num_layer_groups=ckpt["prem_layer_groups"], alpha=ckpt["alpha"],
    )
    mem.load_state_dict(ckpt["prem_state_dict"], strict=True)
    if variant != "prem":
        mem = install_pooled_writer(model, variant, ckpt.get("ema_decay") or 0.98)
    model.eval().requires_grad_(False)
    cfg = {
        "num_slots": ckpt["num_slots"], "mem_dim": ckpt["mem_dim"],
        "layer_groups": ckpt["prem_layer_groups"], "alpha": 1.0,
    }
    for key in ("disable_anti_distractor", "disable_novelty", "disable_stability",
                "disable_evidence_gate_write", "uniform_write_route"):
        cfg[key] = bool(ckpt.get(key, False))
    return model, cfg, ckpt, variant


def reset_state(model, cfg):
    return model.stream_reset(
        prem_num_slots=cfg["num_slots"], prem_mem_dim=cfg["mem_dim"],
        prem_layer_groups=cfg["layer_groups"], prem_alpha=1.0,
        **{"prem_" + key: cfg[key] for key in (
            "disable_anti_distractor", "disable_novelty", "disable_stability",
            "disable_evidence_gate_write", "uniform_write_route")},
    )


def update(model, state, features):
    if not len(features):
        return state
    with torch.inference_mode():
        return model.stream_update(state, features.to(model.device), collect_stats=False)


def frame_features(model, processor, frames, args, identity):
    """Cache exactly the spatial mean consumed by the original stream writer."""
    if not frames:
        raise ValueError("Writer input contains no frames")
    expected_steps = (len(frames) + 1) // 2
    key = stable_hash({"feature_protocol": "qwen25_two_frame_spatial_mean_v1",
                       "source": identity, "fps": args.fps, "max_pixels": args.max_pixels,
                       "model": args.model_identity, "software": args.software,
                       "encoder_source": file_sha256(ROOT / "scripts/eval/eval_prem_online.py")})
    path = Path(args.feature_cache) / (key + ".pt")
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["key"] != key or saved["frames"] != len(frames)
                or len(saved["features"]) != expected_steps):
            raise ValueError("Feature cache identity mismatch")
        return saved["features"]
    pooled = []
    for start in range(0, len(frames), 2):
        # The original Qwen preprocessing repeats a final singleton frame.
        # Pass it through unchanged, as update_stream_state does.
        grouped = chunk_to_frame_embeds(model, processor, frames[start:start + 2], args)
        # stream_update computes this BF16 spatial mean before casting to FP32.
        pooled.append(grouped.mean(dim=1).detach().cpu())
    features = torch.cat(pooled)
    if len(features) != expected_steps:
        raise ValueError("Unexpected number of actual temporal writer steps")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save({"key": key, "frames": len(frames), "features": features}, tmp)
    try:
        # A shared cache entry is immutable: concurrent writers use the first
        # complete tensor instead of replacing one another's inputs.
        os.link(tmp, path)
    except FileExistsError:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["key"] != key or saved["frames"] != len(frames)
                or len(saved["features"]) != expected_steps):
            raise ValueError("Concurrent feature cache identity mismatch")
        features = saved["features"]
    finally:
        tmp.unlink(missing_ok=True)
    return features


def media_identity(video_dir, sample):
    path = Path(resolve_video_source(video_dir, get_sample_media_name(sample)))
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def buffer_state(frames, args):
    buffer = {}
    update_visual_buffer(buffer, frames, 16, args.max_pixels)
    return buffer


def attach_buffer(state, buffer):
    return {**state, "visual_buffer": buffer["visual_buffer"],
            "visual_buffer_seen": buffer["visual_buffer_seen"]}


def answer(model, processor, sample, state, cfg, args):
    if args.read_interface == "in_sequence_tokens" and cfg["alpha"] != 0:
        pred, stats = answer_token_readout(model, processor, sample["question"], state, args)
    else:
        pred, stats = answer_prem(model, processor, sample["question"], state, cfg, args)
    return {"pred": pred, "correct": extract_answer(pred) == int(sample["answer"]), **stats}


def natural_rows(model, processor, cfg, args):
    rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    groups = {}
    for row in get_video_grouped_chunk(rows, args.num_chunks, args.chunk_idx):
        groups.setdefault(get_sample_media_name(row), []).append(row)
    if args.limit_videos:
        groups = dict(list(groups.items())[:args.limit_videos])
    for video, questions in groups.items():
        needed = [(sample, cap) for sample in questions for cap in args.frame_caps
                  if (str(sample["id"]), cap) not in args.done]
        if not needed:
            continue
        sample = questions[0]
        canonical, decoder_manifest = load_frames_with_manifest(args.video_dir, sample, 240, args.fps)
        decoder_manifest_hash = save_frame_manifest(args.output_dir, decoder_manifest)
        buffer = buffer_state(canonical, args)
        baseline = {}
        for cap in args.frame_caps:
            pending = [s for s in questions if (str(s["id"]), cap) not in args.done]
            if not pending:
                continue
            frames, manifest = ((canonical, decoder_manifest) if cap == 240 else
                                load_frames_with_manifest(args.video_dir, sample, cap, args.fps))
            manifest_hash = save_frame_manifest(args.output_dir, manifest)
            identity = {**media_identity(args.video_dir, sample), "frame_cap": cap,
                        "frame_manifest_sha256": manifest_hash}
            features = frame_features(model, processor, frames, args, identity)
            writer_hash = tensor_sha256(features)
            state = update(model, reset_state(model, cfg), features)
            state = attach_buffer(state, buffer)
            for s in pending:
                sid = str(s["id"])
                result = answer(model, processor, s, state, cfg, args)
                if sid not in baseline:
                    baseline[sid] = answer(model, processor, s, state, {**cfg, "alpha": 0.0}, args)
                base = baseline[sid]
                assert base["decoder_input_sha256"] == result["decoder_input_sha256"]
                yield {"id": sid, "video_id": video, "condition": cap,
                       "sampled_frames": len(frames), "actual_writes": state["num_updates"],
                       "answer": s["answer"], "prediction": result, "base": base,
                       "writer_input_sha256": writer_hash,
                       "writer_frame_manifest_sha256": manifest_hash,
                       "decoder_frame_manifest_sha256": decoder_manifest_hash,
                       "buffer_source_frame_cap": 240,
                       "matrix_state_bytes": state["memory"].numel() * state["memory"].element_size(),
                       "confidence_state_bytes": state["confidence"].numel() * state["confidence"].element_size()}


def seconds(value):
    parts = [int(x) for x in value.split(":")]
    result = 0
    for part in parts:
        result = result * 60 + part
    return result


def select_needles(args):
    videos = [json.loads(line) for line in Path(args.raw_metadata).open()]
    candidates = []
    for video in videos:
        for qa in video["qa"]:
            match = re.fullmatch(r"\s*(\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?)\s*", qa.get("time_reference", ""))
            if not match or not set(qa["question_type"]) & {"entity recognition", "key information retrieval"}:
                continue
            start, end = map(seconds, match.groups())
            if not 1 <= end - start <= 30:
                continue
            candidates.append({"id": str(qa["uid"]), "video_id": video["key"],
                               "question": qa["question"], "answer": ord(qa["answer"]) - ord("A"),
                               "evidence_seconds": [start, end], "type": video["type"],
                               "question_type": qa["question_type"]})
    candidates.sort(key=lambda row: stable_hash([13, row["id"]]))
    selected, per_video = [], {}
    for row in candidates:
        if per_video.get(row["video_id"], 0) >= 2:
            continue
        foreign = sorted(v["key"] for v in videos if v["type"] == row["type"] and v["key"] != row["video_id"])
        if not foreign:
            continue
        row["distractor_video_id"] = foreign[int(stable_hash([13, row["id"], "foreign"]), 16) % len(foreign)]
        selected.append(row)
        per_video[row["video_id"]] = per_video.get(row["video_id"], 0) + 1
        if len(selected) == args.needle_samples:
            break
    return selected


def evidence_frames(args, sample):
    path = resolve_video_source(args.video_dir, sample["video_id"])
    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    start, end = sample["evidence_seconds"]
    first, last = int(round(start * fps)), min(len(reader) - 1, int(end * fps) - 1)
    indices = torch.linspace(first, last, 16).round().long().tolist()
    frames = [Image.fromarray(frame).convert("RGB") for frame in reader.get_batch(indices).asnumpy()]
    return frames, indices


def needle_rows(model, processor, cfg, args):
    selected = select_needles(args)
    manifest = Path(args.output_dir) / "needle_manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != selected:
            raise ValueError("Needle selection differs from saved manifest")
    else:
        # All workers deterministically select the same manifest.
        tmp = manifest.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(selected, indent=2)); os.replace(tmp, manifest)
    selected = selected[args.chunk_idx::args.num_chunks]
    if args.limit_videos:
        selected = selected[:args.limit_videos]
    for sample in selected:
        sid = sample["id"]
        if all((sid, n) in args.done for n in args.distractor_frames):
            continue
        evidence, evidence_indices = evidence_frames(args, sample)
        foreign_sample = {"video_id": sample["distractor_video_id"]}
        max_distractors = max(args.distractor_frames)
        foreign, foreign_manifest = load_frames_with_manifest(
            args.video_dir, foreign_sample, max_distractors + 32, args.fps)
        foreign_manifest_hash = save_frame_manifest(args.output_dir, foreign_manifest)
        if len(foreign) != max_distractors + 32:
            raise ValueError("Distractor source is shorter than the registered probe")
        foreign_features = frame_features(model, processor, foreign, args,
            {**media_identity(args.video_dir, foreign_sample), "frame_cap": max_distractors + 32,
             "frame_manifest_sha256": foreign_manifest_hash})
        evidence_features = frame_features(model, processor, evidence, args,
            {**media_identity(args.video_dir, sample), "evidence_indices": evidence_indices})
        tail = buffer_state(foreign[-16:], args)
        evidence_buffer = buffer_state(evidence, args)
        # Replacing the target with neutral frames preserves the exact write count.
        states = {
            "present": update(model, reset_state(model, cfg), evidence_features),
            "removed": update(model, reset_state(model, cfg), foreign_features[:8]),
        }
        base = answer(model, processor, sample, attach_buffer(states["present"], tail),
                      {**cfg, "alpha": 0.0}, args)
        visible = answer(model, processor, sample, attach_buffer(states["present"], evidence_buffer),
                         {**cfg, "alpha": 0.0}, args)
        previous = 0
        for count in args.distractor_frames:
            extra = foreign_features[8 + previous // 2:8 + count // 2]
            states = {name: update(model, state, extra) for name, state in states.items()}
            previous = count
            if (sid, count) in args.done:
                continue
            predictions = {}
            for name, state in states.items():
                answered_state = update(model, state, foreign_features[-8:])
                answered_state = attach_buffer(answered_state, tail)
                if answered_state["num_updates"] != 16 + count // 2:
                    raise ValueError("Unexpected actual recurrent horizon")
                predictions[name] = answer(model, processor, sample, answered_state, cfg, args)
                assert predictions[name]["decoder_input_sha256"] == base["decoder_input_sha256"]
            middle = foreign_features[8:8 + count // 2]
            writer_hash = tensor_sha256(torch.cat([evidence_features, middle, foreign_features[-8:]]))
            replaced_hash = tensor_sha256(torch.cat([foreign_features[:8], middle, foreign_features[-8:]]))
            manifest_hash = save_frame_manifest(args.output_dir, {
                "evidence_source": media_identity(args.video_dir, sample),
                "evidence_indices": evidence_indices, "foreign_manifest_sha256": foreign_manifest_hash,
                "middle_frame_range": [16, 16 + count],
                "tail_frame_range": [len(foreign) - 16, len(foreign)],
                "replacement_frame_range": [0, 16],
            })
            yield {"id": sid, "video_id": sample["video_id"], "condition": count,
                   "distractor_frames": count, "actual_writes": answered_state["num_updates"],
                   "evidence_age_writes": answered_state["num_updates"] - len(evidence_features), "evidence_frames": 16,
                   "evidence_indices": evidence_indices, "evidence_seconds": sample["evidence_seconds"],
                   "distractor_video_id": sample["distractor_video_id"],
                   "target_excluded_from_buffer": True, "answer": sample["answer"],
                   "writer_input_sha256": writer_hash,
                   "target_replaced_writer_sha256": replaced_hash,
                   "writer_frame_manifest_sha256": manifest_hash,
                   "prediction": predictions["present"], "target_removed": predictions["removed"],
                   "base": base, "evidence_visible_base": visible}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("natural", "needle"), required=True)
    parser.add_argument("--model_path", default="ckpt/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_cache", default="outputs/rebuttal_prem/frozen_features")
    parser.add_argument("--gt_file", default="data/eval_video/lvbench/test_qa.json")
    parser.add_argument("--video_dir", default="data/eval_video/lvbench/videos")
    parser.add_argument("--raw_metadata", default="data/eval_video/lvbench/raw/video_info.meta.jsonl")
    parser.add_argument("--frame_caps", default="240,480,960")
    parser.add_argument("--distractor_frames", default="0,30,60,120,240,480")
    parser.add_argument("--needle_samples", type=int, default=128)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--limit_videos", type=int, default=0, help="Implementation diagnostic only")
    args = parser.parse_args()
    args.frame_caps = sorted(set(map(int, args.frame_caps.split(","))))
    args.distractor_frames = sorted(set(map(int, args.distractor_frames.split(","))))
    if any(x <= 0 or x % 2 for x in args.frame_caps) or any(x < 0 or x % 2 for x in args.distractor_frames):
        parser.error("Frame counts must be even, with positive natural-video caps")
    args.fps, args.max_pixels, args.buffer_frames = 1.0, 200704, 16
    args.prem_modulation = "attention_kv"
    args.software = {"torch": torch.__version__, "cuda": torch.version.cuda,
                     "transformers": importlib.metadata.version("transformers"),
                     "decord": importlib.metadata.version("decord")}
    args.model_identity = {"path": str(Path(args.model_path).resolve()),
                           "config_sha256": file_sha256(Path(args.model_path) / "config.json"),
                           "weights": [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                                       for p in sorted(Path(args.model_path).glob("*.safetensors"))],
                           "processor_files": {name: file_sha256(Path(args.model_path) / name)
                                               for name in ("preprocessor_config.json", "processor_config.json",
                                                            "video_preprocessor_config.json", "chat_template.json",
                                                            "chat_template.jinja")
                                               if (Path(args.model_path) / name).exists()},
                           "encoder_dependencies": {name: file_sha256(ROOT / name) for name in (
                               "models/prem_qwen2_5_vl_model.py", "qwen_vl_utils/vision_process.py")}}
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    signature = {**vars(args), "checkpoint_sha256": file_sha256(args.checkpoint),
                 "script_sha256": file_sha256(__file__), "node": "B",
                 "protocol": "rebuttal_memory_v1", "decoder_buffer": "fixed_16_frames"}
    signature["gt_sha256"] = file_sha256(args.gt_file)
    signature["raw_metadata_sha256"] = file_sha256(args.raw_metadata)
    signature["source_sha256"] = {name: file_sha256(ROOT / name) for name in (
        "models/prem_memory.py", "models/prem_writer_controls.py",
        "models/prem_qwen2_5_vl_model.py", "models/same_backbone_baselines.py",
        "scripts/eval/eval_same_backbone_online.py", "scripts/eval/eval_prem_online.py",
        "scripts/eval/prem_checkpoint.py", "scripts/eval/input_alignment.py",
        "scripts/eval/summarize_rebuttal_memory.py", "scripts/eval/rebuttal_input_audit.py",
    )}
    signature["hardware"] = {"hostname": os.uname().nodename,
                             "gpu": torch.cuda.get_device_name(),
                             "visible_gpus_per_worker": torch.cuda.device_count()}
    signature.pop("chunk_idx")
    signature = json.loads(json.dumps(signature))
    args.done = set()
    prediction_path = output / f"{args.num_chunks}_{args.chunk_idx}.jsonl"
    if prediction_path.exists():
        for line in prediction_path.open():
            row = json.loads(line)
            if row["signature"] != signature:
                raise ValueError("Resume signature mismatch")
            args.done.add((str(row["id"]), row["condition"]))
    model, cfg, ckpt, variant = load_model(args)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    generator = natural_rows if args.mode == "natural" else needle_rows
    with prediction_path.open("a") as handle:
        for row in generator(model, processor, cfg, args):
            row.update(signature=signature, writer_variant=variant, K=cfg["num_slots"],
                       read_interface=args.read_interface,
                       learned_retention=float(torch.sigmoid(model.prem_memory.forget_logit).mean()))
            handle.write(json.dumps(row) + "\n"); handle.flush()
            print(f"[{args.mode}:{args.chunk_idx}] id={row['id']} condition={row['condition']} writes={row['actual_writes']}", flush=True)


if __name__ == "__main__":
    main()
