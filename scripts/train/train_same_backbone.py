#!/usr/bin/env python
"""Train parameter-matched controls for the Qwen2.5-VL comparison.

The script intentionally lives outside the main PReM trainer.  It supports
language-attention LoRA and a token-readout memory control while preserving the
same LLaVA-Video split, writer horizon, and decoder frame budget used by PReM.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.prem_qwen2_5_vl_model import (  # noqa: E402
    PReMQwen2_5_VLForConditionalGeneration,
)
from models.same_backbone_baselines import (  # noqa: E402
    QWEN25_LORA_ALPHA,
    QWEN25_LORA_RANK,
    build_token_readout_inputs,
    count_trainable_parameters,
    make_qwen25_lora_config,
    qwen25_lora_parameter_count,
    token_readout_count,
)
from scripts.train.train import (  # noqa: E402
    build_llava_inputs,
    cleanup_cuda,
    count_llava_qa_pairs,
    flatten_llava_qa_rows,
    is_cuda_oom,
    load_llava_split,
    split_id_payload,
    warmup_steps_from_ratio,
)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def trainable_state(model, method: str) -> dict[str, torch.Tensor]:
    if method == "lora":
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(model)
    else:
        state = model.prem_memory.state_dict()
    return {key: value.detach().cpu() for key, value in state.items()}


def normalize_gradients(
    parameters: list[torch.nn.Parameter],
    local_examples: int,
    device: str,
    distributed: bool,
) -> int:
    """Average one fixed-global-batch update without wrapping the frozen backbone."""
    active_count = torch.tensor([int(local_examples)], device=device, dtype=torch.long)
    if distributed:
        dist.all_reduce(active_count, op=dist.ReduceOp.SUM)
    count = int(active_count.item())
    if count == 0:
        return 0
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if distributed:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(count)
    return count


def load_models(args, device: str):
    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "eager"
    if args.method == "lora":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        from peft import get_peft_model

        model = get_peft_model(
            model,
            make_qwen25_lora_config(args.lora_rank, args.lora_alpha, args.lora_dropout),
        )
        expected = qwen25_lora_parameter_count(model.config, args.lora_rank)
        actual = count_trainable_parameters(model)
        if actual != expected:
            raise RuntimeError(f"LoRA parameter audit failed: actual={actual:,} expected={expected:,}")
        memory = None
    else:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
        if config.model_type != "qwen2_5_vl":
            raise ValueError(f"Expected Qwen2.5-VL, got {config.model_type!r}")
        config.architectures = [PReMQwen2_5_VLForConditionalGeneration.__name__]
        model = PReMQwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            config=config,
            device_map={"": device},
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        memory = model.build_prem_memory(
            num_slots=args.num_slots,
            alpha=args.alpha,
            mem_dim=args.mem_dim,
            num_layer_groups=args.layer_groups,
        )
        memory.to(device=model.device, dtype=torch.float32)
        for parameter in memory.parameters():
            parameter.requires_grad_(True)
        actual = count_trainable_parameters(memory)

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    return model, memory, processor, attn, actual


def load_resume(args, model, optimizer, device: str) -> tuple[int, int, int, list[float]]:
    if not args.resume:
        return 0, 0, 0, []
    latest = Path(str(args.out_ckpt) + ".latest")
    path = latest if latest.exists() else Path(args.out_ckpt)
    if not path.exists():
        print(f"[resume] no checkpoint at {path}; starting fresh", flush=True)
        return 0, 0, 0, []
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("baseline_method") != args.method:
        raise ValueError(f"Checkpoint method mismatch: {path}")
    if int(checkpoint.get("visual_buffer_frames", -1)) != args.visual_buffer_frames:
        raise ValueError(f"Checkpoint B mismatch: {path}")
    if int(checkpoint.get("max_frames", -1)) != args.max_frames:
        raise ValueError(f"Checkpoint T mismatch: {path}")
    if int(checkpoint.get("global_batch_size", -1)) != args.global_batch_size:
        raise ValueError(f"Checkpoint global batch mismatch: {path}")
    if int(checkpoint.get("world_size", -1)) != int(os.environ.get("WORLD_SIZE", "1")):
        raise ValueError(
            f"Checkpoint WORLD_SIZE mismatch; resume with the original GPU count: {path}"
        )
    if args.method == "lora":
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, checkpoint["adapter_state_dict"])
    else:
        model.prem_memory.load_state_dict(checkpoint["prem_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    print(
        f"[resume] path={path} step={checkpoint['train_steps']} "
        f"epoch={checkpoint['epoch']} seen={checkpoint['seen_in_epoch']}",
        flush=True,
    )
    return (
        int(checkpoint["train_steps"]),
        int(checkpoint["epoch"]),
        int(checkpoint["seen_in_epoch"]),
        list(checkpoint.get("recent_losses", [])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("lora", "token_readout"), required=True)
    parser.add_argument("--model_path", default="ckpt/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--llava_data_file", default="data/llava-video-178k/trainset_9k.jsonl")
    parser.add_argument("--llava_video_root", default="data/llava-video-178k/frames")
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=240)
    parser.add_argument("--visual_buffer_frames", type=int, default=16)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--max_memory_tokens", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--global_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stream_jsonl", action="store_true", default=True)
    parser.add_argument("--no_stream_jsonl", dest="stream_jsonl", action="store_false")
    parser.add_argument("--filter_missing_media", action="store_true")
    parser.add_argument("--skip_oom", action="store_true")
    parser.add_argument("--skip_bad_samples", action="store_true")
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lora_rank", type=int, default=QWEN25_LORA_RANK)
    parser.add_argument("--lora_alpha", type=int, default=QWEN25_LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--num_slots", type=int, default=1)
    parser.add_argument("--mem_dim", type=int, default=128)
    parser.add_argument("--layer_groups", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--pred_weight", type=float, default=0.2)
    parser.add_argument("--pred_tokens", type=int, default=4)
    args = parser.parse_args()

    if args.max_frames != 240 or args.visual_buffer_frames != 16:
        parser.error("The registered same-backbone protocol requires T=240 and B=16")
    if args.visual_buffer_frames % 2:
        parser.error("--visual_buffer_frames must be even for Qwen video patches")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("--warmup_ratio must be in [0, 1]")
    out_path = Path(args.out_ckpt)
    if out_path.exists():
        existing = torch.load(out_path, map_location="cpu", weights_only=False)
        if existing.get("complete") and not args.overwrite:
            print(f"[skip] complete checkpoint already exists: {out_path}", flush=True)
            return
        if not args.resume and not args.overwrite:
            parser.error(f"Incomplete checkpoint exists; pass --resume or --overwrite: {out_path}")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size < world_size or args.global_batch_size % world_size:
        parser.error(
            "--global_batch_size must be divisible by WORLD_SIZE and at least WORLD_SIZE"
        )
    gradient_accumulation_steps = args.global_batch_size // world_size
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    is_main = rank == 0
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, memory, processor, attn, trainable_count = load_models(args, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    media_root = Path(args.llava_video_root) if args.filter_missing_media else None
    source_rows, _dev_rows = load_llava_split(
        Path(args.llava_data_file),
        0.0,
        args.seed,
        args.limit,
        video_root=media_root,
        stream_jsonl=args.stream_jsonl,
    )
    source_count = len(source_rows)
    qa_count = count_llava_qa_pairs(source_rows)
    rows = flatten_llava_qa_rows(source_rows)
    all_train_ids = split_id_payload(rows, 5000)
    if distributed:
        rows = rows[rank::world_size]
    local_count = len(rows)
    if distributed:
        count_tensor = torch.tensor([local_count], device=device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.MAX)
        epoch_size = int(count_tensor.item())
    else:
        epoch_size = local_count
    optimizer_updates_per_epoch = math.ceil(epoch_size / gradient_accumulation_steps)
    warmup_steps = warmup_steps_from_ratio(
        optimizer_updates_per_epoch, args.epochs, args.warmup_ratio
    )

    if is_main:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path.parent / f"{args.method}_split.json", "w") as handle:
            json.dump(
                {
                    "data_semantics": "same_backbone_controls_v1",
                    "llava_data_file": args.llava_data_file,
                    "llava_video_root": args.llava_video_root,
                    "source_train_samples": source_count,
                    "source_train_qa": qa_count,
                    "train_samples": len(flatten_llava_qa_rows(source_rows)),
                    "train_ids": all_train_ids,
                    "seed": args.seed,
                    "dev_ratio": 0.0,
                    "world_size": world_size,
                    "global_batch_size": args.global_batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                },
                handle,
                indent=2,
            )
        print(
            f"[load] method={args.method} model={args.model_path} attn={attn} "
            f"world_size={world_size} global_batch={args.global_batch_size} "
            f"grad_accum={gradient_accumulation_steps} updates_per_epoch="
            f"{optimizer_updates_per_epoch} trainable={trainable_count:,} train_qa={qa_count}",
            flush=True,
        )

    step, start_epoch, seen_in_epoch, losses = load_resume(
        args, model, optimizer, device
    )
    started = time.time()
    skipped = 0
    latest_path = Path(str(out_path) + ".latest")

    def make_checkpoint(epoch: int, seen: int, complete: bool) -> dict:
        state_key = "adapter_state_dict" if args.method == "lora" else "prem_state_dict"
        checkpoint = {
            state_key: trainable_state(model, args.method),
            "optimizer_state_dict": optimizer.state_dict(),
            "baseline_method": args.method,
            "mechanism_label": (
                "LoRA (language q/k/v/o)" if args.method == "lora" else "Token-Readout Memory"
            ),
            "model_type": "qwen2_5_vl",
            "model_path": args.model_path,
            "trainable_parameters": trainable_count,
            "max_frames": args.max_frames,
            "visual_buffer_frames": args.visual_buffer_frames,
            "max_pixels": args.max_pixels,
            "fps": args.fps,
            "llava_data_file": args.llava_data_file,
            "llava_video_root": args.llava_video_root,
            "source_train_samples": source_count,
            "source_train_qa": qa_count,
            "train_samples": qa_count,
            "data_semantics": "same_backbone_controls_v1",
            "qa_pairs_flattened": True,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "train_steps": step,
            "epoch": epoch,
            "seen_in_epoch": seen,
            "recent_losses": losses[-50:],
            "skipped": skipped,
            "skip_oom": bool(args.skip_oom),
            "skip_bad_samples": bool(args.skip_bad_samples),
            "world_size": world_size,
            "global_batch_size": args.global_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "train_wall_seconds": time.time() - started,
            "train_peak_gpu_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0
            ),
            "complete": complete,
        }
        if args.method == "lora":
            checkpoint.update(
                {
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_dropout,
                    "lora_targets": "language_model.self_attn.{q,k,v,o}_proj",
                }
            )
        else:
            checkpoint.update(
                {
                    "num_slots": args.num_slots,
                    "mem_dim": args.mem_dim,
                    "prem_layer_groups": args.layer_groups,
                    "alpha": args.alpha,
                    "prem_modulation": "none",
                    "read_interface": "in_sequence_tokens",
                    "memory_prompt_tokens": token_readout_count(memory),
                    "max_memory_tokens": args.max_memory_tokens,
                    "pred_weight": args.pred_weight,
                    "pred_tokens": args.pred_tokens,
                }
            )
        return checkpoint

    model.train()
    if memory is not None:
        memory.train()
    for epoch in range(start_epoch, args.epochs):
        row_start = seen_in_epoch if epoch == start_epoch else 0
        if row_start not in {0, epoch_size} and row_start % gradient_accumulation_steps:
            raise ValueError(
                f"Resume cursor {row_start} is not on a gradient-accumulation boundary"
            )
        optimizer.zero_grad(set_to_none=True)
        active_local_window = 0
        for row_index in range(row_start, epoch_size):
            row = rows[row_index] if row_index < local_count else None
            active = False
            loss_value = None
            try:
                if row is not None:
                    if args.method == "lora":
                        answer_inputs, _ = build_llava_inputs(
                            row,
                            Path(args.llava_video_root),
                            processor,
                            args.visual_buffer_frames,
                            args.max_pixels,
                            args.fps,
                            device,
                            "qwen2_5vl",
                        )
                        output = model(**answer_inputs, use_cache=False, return_dict=True)
                        loss = output.loss
                    else:
                        writer_inputs, writer_prompt_length = build_llava_inputs(
                            row,
                            Path(args.llava_video_root),
                            processor,
                            args.max_frames,
                            args.max_pixels,
                            args.fps,
                            device,
                            "qwen2_5vl",
                        )
                        stream_state, pred_loss, _ = model.build_prem_state_from_video(
                            input_ids=writer_inputs["input_ids"],
                            attention_mask=writer_inputs["attention_mask"],
                            pixel_values_videos=writer_inputs["pixel_values_videos"],
                            video_grid_thw=writer_inputs["video_grid_thw"],
                            prem_prompt_lengths=torch.tensor(
                                [writer_prompt_length], device=device
                            ),
                            prem_alpha=1.0,
                            prem_num_slots=args.num_slots,
                            prem_mem_dim=args.mem_dim,
                            prem_layer_groups=args.layer_groups,
                            prem_max_memory_tokens=args.max_memory_tokens,
                            prem_pred_weight=args.pred_weight,
                            prem_pred_tokens=args.pred_tokens,
                        )
                        answer_inputs, answer_prompt_length = build_llava_inputs(
                            row,
                            Path(args.llava_video_root),
                            processor,
                            args.visual_buffer_frames,
                            args.max_pixels,
                            args.fps,
                            device,
                            "qwen2_5vl",
                        )
                        expanded, _ = build_token_readout_inputs(
                            model,
                            answer_inputs,
                            stream_state,
                            answer_prompt_length,
                        )
                        output = model(**expanded, use_cache=False, return_dict=True)
                        loss = output.loss + args.pred_weight * pred_loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss: {float(loss.detach())}")
                    loss.backward()
                    loss_value = float(loss.detach().cpu())
                    active = True
            except Exception as exc:
                can_skip = (
                    gradient_accumulation_steps == 1
                    and ((args.skip_oom and is_cuda_oom(exc)) or args.skip_bad_samples)
                )
                if not can_skip:
                    raise RuntimeError(
                        f"training failed method={args.method} rank={rank} row={row_index}: {exc}"
                    ) from exc
                skipped += 1
                print(
                    f"[skip][rank={rank}] row={row_index} {type(exc).__name__}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
                optimizer.zero_grad(set_to_none=True)
                cleanup_cuda()

            active_local_window += int(active)
            seen = row_index + 1
            window_end = (
                seen % gradient_accumulation_steps == 0 or seen == epoch_size
            )
            if not window_end:
                if loss_value is not None:
                    losses.append(loss_value)
                continue

            active_examples = normalize_gradients(
                trainable,
                active_local_window,
                device,
                distributed,
            )
            if active_examples:
                step += 1
                scale = min(1.0, step / max(1, warmup_steps)) if warmup_steps else 1.0
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * scale
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            active_local_window = 0
            if loss_value is not None:
                losses.append(loss_value)
            if is_main and step and step % args.log_every == 0:
                recent = sum(losses[-args.log_every :]) / max(1, len(losses[-args.log_every :]))
                print(
                    f"[train] method={args.method} epoch={epoch + 1}/{args.epochs} "
                    f"row={seen}/{epoch_size} step={step} loss={recent:.5f} "
                    f"batch_examples={active_examples}",
                    flush=True,
                )
            if is_main and step and args.save_every > 0 and step % args.save_every == 0:
                atomic_torch_save(make_checkpoint(epoch, seen, False), latest_path)
        seen_in_epoch = 0
        if distributed:
            dist.barrier()

    if is_main:
        final = make_checkpoint(args.epochs, 0, True)
        atomic_torch_save(final, out_path)
        if latest_path.exists():
            latest_path.unlink()
        print(
            f"[done] checkpoint={out_path} steps={step} "
            f"trainable={trainable_count:,} skipped={skipped}",
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
