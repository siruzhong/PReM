#!/usr/bin/env python
"""Train PREM attention modulation on supported video-language backbones.
Optional question-conditioned evidence prediction can be enabled with
``--pred_weight``. Only ``prem_memory`` parameters are trainable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_utils import extract_vision_info, process_vision_info, qwen3_video_metadata  # noqa: E402
OFFICIAL_MAX_FRAMES = 240
OFFICIAL_MAX_PIXELS = 4 * 224 * 224

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
IGNORE_INDEX = -100


def strip_media_marker(text: str) -> str:
    return str(text).replace("<image>\n", "").replace("<video>\n", "").replace("<image>", "").replace("<video>", "").strip()


def role_name(value: str) -> str:
    value = str(value).lower()
    if value in {"human", "user"}:
        return "user"
    if value in {"gpt", "assistant"}:
        return "assistant"
    raise ValueError(f"Unsupported conversation role: {value}")


def resolve_llava_video(video_root: Path, video_name: str) -> Path:
    base = video_root / video_name
    candidates = [base]
    if base.suffix:
        candidates.append(base.with_suffix(""))
    else:
        candidates.extend(base.with_suffix(ext) for ext in VIDEO_EXTS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing LLaVA-Video media: {video_root / video_name}")


def resolve_device(requested: str):
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return f"cuda:{int(local_rank)}"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return "cuda"
    best_idx = None
    best_free = -1
    for line in out.splitlines():
        if not line.strip():
            continue
        idx_str, free_str = [part.strip() for part in line.split(",", 1)]
        free_mb = int(free_str)
        if free_mb > best_free:
            best_idx = int(idx_str)
            best_free = free_mb
    return "cuda" if best_idx is None else f"cuda:{best_idx}"


def iter_llava_rows(data_file: Path):
    if data_file.suffix == ".jsonl":
        with open(data_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(data_file) as f:
            yield from json.load(f)


def conversation_qa_pair_indices(row: dict) -> list[tuple[int, int]]:
    """Return strict adjacent user/assistant turn pairs for one source video row."""
    conversations = row.get("conversations")
    if not conversations:
        raise ValueError(f"Missing conversations for sample {row.get('id')}")

    pairs = []
    pending_user_idx = None
    for turn_idx, turn in enumerate(conversations):
        role = role_name(turn.get("from"))
        if role == "user":
            if pending_user_idx is not None:
                raise ValueError(
                    f"Conversation must alternate user and assistant turns for sample {row.get('id')}"
                )
            pending_user_idx = turn_idx
            continue
        if pending_user_idx is None:
            raise ValueError(
                f"Assistant turn without an adjacent user question for sample {row.get('id')}"
            )
        pairs.append((pending_user_idx, turn_idx))
        pending_user_idx = None

    if pending_user_idx is not None:
        raise ValueError(f"Conversation ends without an assistant answer for sample {row.get('id')}")
    if not pairs:
        raise ValueError(f"No user/assistant QA pairs for sample {row.get('id')}")
    return pairs


def make_qa_pair_row(row: dict, qa_index: int) -> dict:
    """Create one QA training row with its own video-only writer."""
    pair_indices = conversation_qa_pair_indices(row)
    try:
        user_idx, assistant_idx = pair_indices[qa_index]
    except IndexError as exc:
        raise IndexError(f"Invalid QA pair index {qa_index} for sample {row.get('id')}") from exc

    source_id = row.get("id")
    sample_id = source_id if source_id is not None else row.get("video", "row")
    qa_row = dict(row)
    qa_row["id"] = f"{sample_id}::qa{qa_index}"
    qa_row["_source_row_id"] = source_id
    qa_row["_qa_index"] = qa_index
    qa_row["conversations"] = [
        dict(row["conversations"][user_idx]),
        dict(row["conversations"][assistant_idx]),
    ]
    return qa_row


class JsonlOffsetRows:
    def __init__(self, data_file: Path, offsets: list[int], qa_pair_counts: list[int] | None = None):
        self.data_file = Path(data_file)
        self.offsets = offsets
        self.qa_pair_counts = qa_pair_counts

    @classmethod
    def build(cls, data_file: Path, seed: int, limit: int | None):
        records = []
        with open(data_file, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    row = json.loads(line)
                    if row.get("video") or row.get("videos"):
                        records.append((offset, len(conversation_qa_pair_indices(row))))
        rng = random.Random(seed)
        rng.shuffle(records)
        if limit is not None:
            records = records[:limit]
        return cls(
            data_file,
            [offset for offset, _ in records],
            [qa_pair_count for _, qa_pair_count in records],
        )

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            pair_counts = self.qa_pair_counts[idx] if self.qa_pair_counts is not None else None
            return JsonlOffsetRows(self.data_file, self.offsets[idx], pair_counts)
        with open(self.data_file, "rb") as f:
            f.seek(self.offsets[idx])
            return json.loads(f.readline().decode("utf-8"))


class QAPairRows:
    """Lazy view that exposes one adjacent QA pair as one training sample."""

    def __init__(self, source_rows, source_qa_indices: list[tuple[int, int]]):
        self.source_rows = source_rows
        self.source_qa_indices = source_qa_indices

    @classmethod
    def build(cls, source_rows):
        source_qa_indices = []
        pair_counts = getattr(source_rows, "qa_pair_counts", None)
        if pair_counts is not None:
            for source_idx, pair_count in enumerate(pair_counts):
                source_qa_indices.extend((source_idx, qa_idx) for qa_idx in range(pair_count))
        else:
            for source_idx in range(len(source_rows)):
                for qa_idx, _ in enumerate(conversation_qa_pair_indices(source_rows[source_idx])):
                    source_qa_indices.append((source_idx, qa_idx))
        return cls(source_rows, source_qa_indices)

    def __len__(self):
        return len(self.source_qa_indices)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return QAPairRows(self.source_rows, self.source_qa_indices[idx])
        source_idx, qa_idx = self.source_qa_indices[idx]
        return make_qa_pair_row(self.source_rows[source_idx], qa_idx)


def flatten_llava_qa_rows(rows):
    return QAPairRows.build(rows)


def count_llava_qa_pairs(rows) -> int:
    pair_counts = getattr(rows, "qa_pair_counts", None)
    if pair_counts is not None:
        return int(sum(pair_counts))
    return sum(len(conversation_qa_pair_indices(rows[idx])) for idx in range(len(rows)))





def checkpoint_resume_cursor(checkpoint: dict) -> tuple[int, int]:
    """Return the epoch and row index of the next unprocessed training sample."""
    epoch = int(checkpoint.get("epoch", 0))
    seen_in_epoch = int(checkpoint.get("seen_in_epoch", 0))
    if epoch < 0 or seen_in_epoch < 0:
        raise ValueError(
            f"Invalid resume cursor: epoch={epoch}, seen_in_epoch={seen_in_epoch}"
        )

    # Older completed checkpoints stored the last completed epoch. Current
    # checkpoints always store the next unprocessed epoch/row explicitly.
    if checkpoint.get("resume_cursor") != "next_unprocessed" and checkpoint.get("complete") and seen_in_epoch == 0:
        epoch += 1
    return epoch, seen_in_epoch





def load_llava_split(
    data_file: Path,
    dev_ratio: float,
    seed: int,
    limit: int | None,
    video_root: Path | None = None,
    stream_jsonl: bool = False,
):
    if stream_jsonl:
        if data_file.suffix != ".jsonl":
            raise ValueError("--stream_jsonl requires a .jsonl --llava_data_file")
        if dev_ratio > 0:
            raise ValueError("--stream_jsonl currently requires --dev_ratio 0")
        if video_root is not None:
            raise ValueError("--stream_jsonl does not support --filter_missing_media")
        return JsonlOffsetRows.build(data_file, seed, limit), []

    rows = list(iter_llava_rows(data_file))
    rows = [row for row in rows if row.get("video") or row.get("videos")]
    if video_root is not None:
        kept = []
        for row in rows:
            video = row.get("video")
            if not video and row.get("videos"):
                video = str(row["videos"][0])
                if not Path(video).suffix:
                    video = video + ".mp4"
            try:
                resolve_llava_video(video_root, str(video))
            except FileNotFoundError:
                continue
            kept.append(row)
        rows = kept
    groups: dict[str, list] = {}
    for row in rows:
        video = row.get("video")
        if not video and row.get("videos"):
            video = str(row["videos"][0])
            if not Path(video).suffix:
                video = video + ".mp4"
            row = dict(row)
            row["video"] = video
        groups.setdefault(str(video), []).append(row)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_dev = 0 if dev_ratio <= 0 else max(1, int(len(keys) * dev_ratio))
    dev_keys = set(keys[:n_dev])
    train_rows = [row for key in keys if key not in dev_keys for row in groups[key]]
    dev_rows = [row for key in keys if key in dev_keys for row in groups[key]]
    rng.shuffle(train_rows)
    rng.shuffle(dev_rows)
    if limit is not None:
        train_rows = train_rows[:limit]
    return train_rows, dev_rows


def split_id_payload(rows, limit: int):
    if limit < 0:
        ids = [row.get("id") for row in rows]
        return ids
    count = len(rows)
    keep = min(count, limit, 100)
    ids = [rows[idx].get("id") for idx in range(keep)]
    if count <= keep:
        return ids
    return {"count": count, "first": ids, "omitted": count - keep}


def build_llava_messages(row, video_root: Path, max_frames, max_pixels, fps):
    conversations = row.get("conversations")
    if not conversations:
        raise ValueError(f"Missing conversations for sample {row.get('id')}")
    video_name = row.get("video")
    if not video_name and row.get("videos"):
        video_name = str(row["videos"][0])
        if not Path(video_name).suffix:
            video_name = video_name + ".mp4"
    video_path = resolve_llava_video(video_root, str(video_name))

    messages = []
    first_user = True
    for turn in conversations:
        role = role_name(turn.get("from"))
        text = strip_media_marker(turn.get("value", ""))
        if role == "user" and first_user:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": str(video_path),
                            "fps": fps,
                            "max_frames": max_frames,
                            "max_pixels": max_pixels,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            )
            first_user = False
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    return messages




def token_ids(processor, text: str):
    return processor.tokenizer(text, add_special_tokens=False).input_ids


def find_last_subsequence(sequence: list[int], pattern: list[int], end: int) -> int | None:
    if not pattern:
        return None
    end = min(end, len(sequence))
    for pos in range(end - len(pattern), -1, -1):
        if sequence[pos: pos + len(pattern)] == pattern:
            return pos
    return None


def build_assistant_labels(processor, messages: list[dict], input_ids: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    expanded_ids = input_ids[0].tolist()
    search_end = len(expanded_ids)
    assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
    for msg in reversed(assistant_messages):
        content = msg["content"][0]["text"] if isinstance(msg["content"], list) else str(msg["content"])
        answer_ids = token_ids(processor, content)
        start = find_last_subsequence(expanded_ids, answer_ids, search_end)
        if start is None:
            continue
        end = min(start + len(answer_ids), input_ids.shape[1])
        labels[0, start:end] = input_ids[0, start:end]
        search_end = start
    return labels


def build_llava_inputs(
    row,
    video_root: Path,
    processor,
    max_frames,
    max_pixels,
    fps,
    device,
    model_type="qwen2vl",
):
    messages = build_llava_messages(row, video_root, max_frames, max_pixels, fps)
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    processor_video_kwargs = {"do_sample_frames": False}
    if model_type == "qwen3vl":
        video_infos = [info for info in extract_vision_info(messages) if "video" in info]
        processor_video_kwargs["video_metadata"] = [
            qwen3_video_metadata(info) for info in video_infos
        ]
    else:
        processor_video_kwargs["fps"] = [fps]
    inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **processor_video_kwargs,
    )
    labels = build_assistant_labels(processor, messages, inputs["input_ids"])
    if bool(labels.ne(IGNORE_INDEX).any()) is False:
        raise ValueError(f"No assistant labels found for sample {row.get('id')}")
    inputs["labels"] = labels
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    prompt_len = int(torch.nonzero(inputs["labels"][0].ne(IGNORE_INDEX), as_tuple=False).flatten()[0].item())
    return inputs, prompt_len


def build_text_only_inputs(row, processor, device):
    """Build the B=0 answer sequence used by the state-only ablation."""
    messages = [
        {
            "role": role_name(turn.get("from")),
            "content": [{"type": "text", "text": strip_media_marker(turn.get("value", ""))}],
        }
        for turn in row["conversations"]
    ]
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor.tokenizer([full_text], padding=True, return_tensors="pt")
    labels = build_assistant_labels(processor, messages, inputs["input_ids"])
    if not bool(labels.ne(IGNORE_INDEX).any()):
        raise ValueError(f"No text-only assistant labels found for sample {row.get('id')}")
    inputs["labels"] = labels
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = int(torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()[0].item())
    return inputs, prompt_len


def qa_modulation_gradient_norm(memory) -> float:
    """Return the combined q/o steer gradient norm used for fail-fast validation."""
    heads = list(memory.query_steer_heads) + list(memory.output_steer_heads)
    for group in memory.group_query_steer_heads:
        heads.extend(group)
    for group in memory.group_output_steer_heads:
        heads.extend(group)
    squared = 0.0
    for head in heads:
        grad = head.weight.grad
        if grad is None:
            continue
        if not torch.isfinite(grad).all():
            return float("nan")
        squared += float(grad.float().square().sum().detach().cpu())
    return squared**0.5


def warmup_steps_from_ratio(total_updates: int, epochs: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return max(1, int(total_updates * epochs * ratio))




def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA out of memory" in str(exc)


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_type",
        choices=("qwen2vl", "qwen2_5vl", "qwen3vl", "llava_video"),
        default="qwen2vl",
        help="Backbone API to use.",
    )
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--llava_data_file", default="data/llava-video-178k/trainset_9k.json")
    ap.add_argument("--llava_video_root", default="data/llava-video-178k/frames")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--filter_missing_media", action="store_true")
    ap.add_argument("--stream_jsonl", action="store_true", help="Use byte-offset streaming for large JSONL datasets")
    ap.add_argument("--out_ckpt", default=None)
    ap.add_argument("--dev_ratio", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None, help="Cap source video rows before QA expansion")
    ap.add_argument("--split_log_id_limit", type=int, default=5000, help="Max ids to store in split.json; -1 stores all")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear LR warmup ratio (0 = disabled, 0.03 = Qwen official default)")
    ap.add_argument("--num_slots", type=int, default=4)
    ap.add_argument("--mem_dim", type=int, default=128)
    ap.add_argument("--prem_layer_groups", type=int, default=1, help="1 keeps static correction; >1 uses grouped contextual reads")
    ap.add_argument("--alpha", type=float, default=1.0, help="Initial learned PREM steer scale")
    ap.add_argument(
        "--prem_modulation",
        type=str,
        default="attention",
        choices=["attention", "attention_kv", "attention_k", "attention_v"],
        help="Attention projections receiving memory steering: QO, KV, K-only, or V-only",
    )
    ap.add_argument("--max_frames", type=int, default=OFFICIAL_MAX_FRAMES)
    ap.add_argument(
        "--visual_buffer_frames",
        type=int,
        default=16,
        help="Fixed visual frame budget B shown to the decoder (0=state-only ablation)",
    )
    ap.add_argument("--max_pixels", type=int, default=OFFICIAL_MAX_PIXELS)
    ap.add_argument("--max_memory_tokens", type=int, default=128)
    ap.add_argument("--router_gamma", type=float, default=0.05)
    ap.add_argument("--pred_weight", type=float, default=0.0, help="Weight for evidence prediction loss")
    ap.add_argument("--pred_tokens", type=int, default=8, help="Held-out temporal writer tokens used by evidence prediction")
    ap.add_argument("--disable_anti_distractor", action="store_true", help="Protected write ablation: set anti_distractor=1")
    ap.add_argument("--disable_novelty", action="store_true", help="Protected write ablation: set novelty=1")
    ap.add_argument("--disable_stability", action="store_true", help="Protected write ablation: set update stability=1")
    ap.add_argument("--disable_evidence_gate_write", action="store_true", help="Protected write ablation: set write evidence_gate=1")
    ap.add_argument("--uniform_write_route", action="store_true", help="Protected write ablation: set write route=1/num_slots")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200, help="Write a resumable latest checkpoint every N steps")
    ap.add_argument("--init_ckpt", default=None, help="Initialize memory weights from a checkpoint without resuming state")
    ap.add_argument("--resume", action="store_true", help="Resume from --resume_ckpt or <out_ckpt>.latest if present")
    ap.add_argument("--resume_ckpt", default=None, help="Explicit resumable checkpoint path")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=False)
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--skip_oom", action="store_true", default=False)
    ap.add_argument("--no_skip_oom", dest="skip_oom", action="store_false")
    ap.add_argument("--skip_bad_samples", action="store_true", default=False)
    ap.add_argument("--no_skip_bad_samples", dest="skip_bad_samples", action="store_false")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.model_path is None:
        default_models = {
            "qwen2vl": "ckpt/Qwen2-VL-7B-Instruct",
            "qwen2_5vl": "ckpt/Qwen2.5-VL-7B-Instruct",
            "qwen3vl": "ckpt/Qwen3-VL-8B-Instruct",
            "llava_video": "ckpt/LLaVA-Video-7B-Qwen2",
        }
        args.model_path = default_models[args.model_type]
    if args.out_ckpt is None:
        default_outputs = {
            "qwen2vl": "outputs/prem_attention/qwen2_7b/prem.pt",
            "qwen2_5vl": "outputs/prem_attention/qwen25_7b/prem.pt",
            "qwen3vl": "outputs/prem_attention/qwen3_8b/prem.pt",
            "llava_video": "outputs/prem_attention/llava_video_7b/prem.pt",
        }
        args.out_ckpt = default_outputs[args.model_type]
    if args.prem_layer_groups < 1:
        ap.error("--prem_layer_groups must be positive")
    if args.visual_buffer_frames < 0:
        ap.error("--visual_buffer_frames must be non-negative")
    if args.model_type != "llava_video" and args.visual_buffer_frames > 0 and (
        args.visual_buffer_frames < 2 or args.visual_buffer_frames % 2
    ):
        ap.error("--visual_buffer_frames must be 0 or an even value >= 2 for Qwen video patches")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        ap.error("--warmup_ratio must be in [0, 1]")
    if args.gradient_checkpointing:
        ap.error(
            "Gradient checkpointing is incompatible with temporary q/o forward hooks: "
            "backward recomputation would run after the hooks are removed"
        )


    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    is_main = rank == 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "eager"
    if is_main:
        print(
            f"[load] model={args.model_path} attn={attn} device={device} "
            f"max_frames={args.max_frames} max_pixels={args.max_pixels}",
            flush=True,
        )
    load_stagger_seconds = float(os.environ.get("LOAD_STAGGER_SECONDS", "0"))
    if distributed and load_stagger_seconds > 0:
        time.sleep(local_rank * load_stagger_seconds)
    llava_resources = None
    if args.model_type == "llava_video":
        from models.prem_llava_video_model import load_prem_llava_video

        llava_resources = load_prem_llava_video(
            args.model_path,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa",
        )
        model = llava_resources.model
        processor = None
    else:
        if args.model_type == "qwen2vl":
            from models.prem_qwen2vl_model import PReMQwen2VLForConditionalGeneration

            model_cls = PReMQwen2VLForConditionalGeneration
        elif args.model_type == "qwen2_5vl":
            from models.prem_qwen2_5_vl_model import PReMQwen2_5_VLForConditionalGeneration

            model_cls = PReMQwen2_5_VLForConditionalGeneration
        else:
            from models.prem_qwen3_vl_model import PReMQwen3VLForConditionalGeneration

            model_cls = PReMQwen3VLForConditionalGeneration
        # Override config.architectures so HF loads our PREM wrapper, not the base class
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
        config.architectures = [model_cls.__name__]
        model = model_cls.from_pretrained(
            args.model_path,
            config=config,
            device_map={"": device},
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        )
        processor_kwargs = {"trust_remote_code": True}
        if args.model_type != "qwen3vl":
            processor_kwargs["use_fast"] = False
        processor = AutoProcessor.from_pretrained(args.model_path, **processor_kwargs)

    for p in model.parameters():
        p.requires_grad_(False)
    mem = model.build_prem_memory(
        num_slots=args.num_slots,
        alpha=args.alpha,
        mem_dim=args.mem_dim,
        num_layer_groups=args.prem_layer_groups,
        disable_anti_distractor=args.disable_anti_distractor,
        disable_novelty=args.disable_novelty,
        disable_stability=args.disable_stability,
        disable_evidence_gate_write=args.disable_evidence_gate_write,
        uniform_write_route=args.uniform_write_route,
    )
    mem.to(device=model.device, dtype=torch.float32)
    for p in mem.parameters():
        p.requires_grad_(True)
    model.train()
    mem.train()

    n_train = sum(p.numel() for p in mem.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if is_main:
        print(
            f"[dist] distributed={distributed} rank={rank}/{world_size} local_rank={local_rank}",
            flush=True,
        )
        print(f"[params] prem_trainable={n_train:,} frozen_backbone={n_frozen:,}", flush=True)

    media_filter_root = Path(args.llava_video_root) if args.filter_missing_media else None
    train_rows, dev_rows = load_llava_split(
        Path(args.llava_data_file),
        args.dev_ratio,
        args.seed,
        args.limit,
        video_root=media_filter_root,
        stream_jsonl=args.stream_jsonl,
    )
    source_train_sample_count = len(train_rows)
    source_dev_sample_count = len(dev_rows)
    source_train_qa_count = count_llava_qa_pairs(train_rows)
    source_dev_qa_count = count_llava_qa_pairs(dev_rows)
    qa_pairs_flattened = True
    data_semantics = "unified_bounded_visual_memory_qa_v3"
    train_rows = flatten_llava_qa_rows(train_rows)
    dev_rows = flatten_llava_qa_rows(dev_rows)
    train_sample_count = len(train_rows)
    train_ids = split_id_payload(train_rows, args.split_log_id_limit)
    if distributed:
        train_rows = train_rows[rank::world_size]
    if is_main:
        print(
            f"[data] source_train={source_train_sample_count} source_dev={source_dev_sample_count} "
            f"source_train_qa={source_train_qa_count} source_dev_qa={source_dev_qa_count} "
            f"qa_pairs_flattened={qa_pairs_flattened} train_per_rank={len(train_rows)} dev={len(dev_rows)} "
            f"world_size={world_size} "
            f"data_file={args.llava_data_file} video_root={args.llava_video_root} "
            f"filter_missing_media={args.filter_missing_media}",
            flush=True,
        )
    out_dir = Path(args.out_ckpt).parent
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "split.json", "w") as f:
            json.dump(
                {
                    "llava_data_file": args.llava_data_file,
                    "llava_video_root": args.llava_video_root,
                    "fps": args.fps,
                    "filter_missing_media": args.filter_missing_media,
                    "stream_jsonl": args.stream_jsonl,
                    "qa_pairs_flattened": qa_pairs_flattened,
                    "data_semantics": data_semantics,
                    "source_train_samples": source_train_sample_count,
                    "source_dev_samples": source_dev_sample_count,
                    "source_train_qa": source_train_qa_count,
                    "source_dev_qa": source_dev_qa_count,
                    "train_samples": train_sample_count,
                    "dev_samples": len(dev_rows),
                    "seed": args.seed,
                    "dev_ratio": args.dev_ratio,
                    "split_log_id_limit": args.split_log_id_limit,
                    "world_size": world_size,
                    "train_ids": train_ids,
                    "train_ids_rank0": split_id_payload(train_rows, args.split_log_id_limit),
                    "dev_ids": split_id_payload(dev_rows, args.split_log_id_limit),
                },
                f,
                indent=2,
            )
    if distributed:
        dist.barrier()

    opt = torch.optim.AdamW([p for p in mem.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    mem.train()
    step = 0
    start_epoch = 0
    checkpoint_epoch = 0
    seen_in_epoch = 0
    t0 = time.time()
    running = []
    skipped = 0
    latest_ckpt = Path(str(args.out_ckpt) + ".latest")

    def prem_state_dict_for_checkpoint():
        return {key: value.detach().cpu() for key, value in mem.state_dict().items()}

    def make_ckpt(final: bool = False):
        checkpoint = {
            "prem_state_dict": prem_state_dict_for_checkpoint(),
            "optimizer_state_dict": opt.state_dict(),
            "num_slots": args.num_slots,
            "mem_dim": args.mem_dim,
            "prem_layer_groups": args.prem_layer_groups,
            "alpha": args.alpha,
            "learned_alpha": float(torch.exp(mem.log_alpha).detach().cpu()),
            "prem_modulation": args.prem_modulation,
            "prem_placement": "layer_contextual" if args.prem_layer_groups > 1 else "per_position",
            "forget_multiplier": float(torch.sigmoid(mem.forget_logit).mean().detach().cpu()),
            "answer_uses_video_inputs": args.visual_buffer_frames > 0,
            "answer_uses_visual_tokens": args.visual_buffer_frames > 0,
            "answer_context": "bounded_visual_buffer_plus_side_memory",
            "qa_objective": "single_ce",
            "visual_buffer_frames": args.visual_buffer_frames,
            "visual_buffer_policy": "uniform_training_reservoir_streaming",
            "router_gamma": args.router_gamma,
            "pred_weight": args.pred_weight,
            "pred_tokens": args.pred_tokens,
            "writer_mode": "temporal_mean_per_step",
            "disable_anti_distractor": args.disable_anti_distractor,
            "disable_novelty": args.disable_novelty,
            "disable_stability": args.disable_stability,
            "disable_evidence_gate_write": args.disable_evidence_gate_write,
            "uniform_write_route": args.uniform_write_route,
            "protected_write_ablation": {
                "anti_distractor": ("disabled" if args.disable_anti_distractor else "enabled"),
                "novelty": "disabled" if args.disable_novelty else "enabled",
                "stability": "disabled" if args.disable_stability else "enabled",
                "evidence_gate_write": ("disabled" if args.disable_evidence_gate_write else "enabled"),
                "write_route": "uniform" if args.uniform_write_route else "learned",
            },
            "fps": args.fps,
            "train_steps": step,
            "qa_pairs_flattened": qa_pairs_flattened,
            "data_semantics": data_semantics,
            "source_train_samples": source_train_sample_count,
            "source_train_qa": source_train_qa_count,
            "train_samples": train_sample_count,
            "train_samples_per_rank": len(train_rows),
            "skipped": skipped,
            "epoch": checkpoint_epoch,
            "seen_in_epoch": seen_in_epoch,
            "resume_cursor": "next_unprocessed",
            "final_train_loss": (sum(running[-50:]) / min(len(running), 50)) if running else None,
            "first_train_loss": (sum(running[:50]) / min(len(running), 50)) if running else None,
            "max_frames": args.max_frames,
            "max_pixels": args.max_pixels,
            "model_type": args.model_type,
            "model_path": args.model_path,
            "seed": args.seed,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "llava_data_file": args.llava_data_file,
            "llava_video_root": args.llava_video_root,
            "train_wall_seconds": float(time.time() - t0),
            "train_peak_gpu_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "complete": final,
        }
        if args.model_type == "llava_video":
            checkpoint["llava_mm_spatial_pool_mode"] = str(
                model.config.mm_spatial_pool_mode
            )
            checkpoint["llava_force_sample"] = True
        checkpoint["max_memory_tokens"] = args.max_memory_tokens
        return checkpoint

    def validate_training_checkpoint(checkpoint: dict, path: Path):
        if checkpoint.get("data_semantics") != data_semantics:
            raise ValueError(f"Incompatible training checkpoint semantics: {path}")
        if checkpoint.get("qa_objective") != "single_ce":
            raise ValueError(f"Training checkpoint does not use the single-CE objective: {path}")
        if checkpoint.get("prem_modulation") != args.prem_modulation:
            raise ValueError("Training checkpoint modulation mode does not match this run")
        if (
            args.model_type == "llava_video"
            and checkpoint.get("llava_mm_spatial_pool_mode") != "average"
        ):
            raise ValueError(
                "LLaVA-Video checkpoint was not trained with official average pooling: "
                f"{path}"
            )
        if (
            args.model_type == "llava_video"
            and checkpoint.get("llava_force_sample") is not True
        ):
            raise ValueError(
                f"LLaVA-Video checkpoint does not use force_sample=true: {path}"
            )
        if int(checkpoint.get("seed", 13)) != args.seed:
            raise ValueError(
                "Training checkpoint random seed does not match this run: "
                f"checkpoint={checkpoint.get('seed', 13)} requested={args.seed} path={path}"
            )
        if int(checkpoint.get("visual_buffer_frames", -1)) != args.visual_buffer_frames:
            raise ValueError(
                "Training checkpoint visual buffer budget does not match this run: "
                f"checkpoint={checkpoint.get('visual_buffer_frames')} "
                f"requested={args.visual_buffer_frames} path={path}"
            )

    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.exists():
            raise FileNotFoundError(f"--init_ckpt does not exist: {init_path}")
        init = torch.load(init_path, map_location="cpu")
        validate_training_checkpoint(init, init_path)
        missing, unexpected = mem.load_state_dict(init.get("prem_state_dict", init), strict=False)
        if is_main:
            print(
                f"[init] ckpt={init_path} missing={list(missing)} unexpected={list(unexpected)}",
                flush=True,
            )

    if args.resume:
        resume_path = Path(args.resume_ckpt) if args.resume_ckpt else latest_ckpt
        if resume_path.exists():
            resume = torch.load(resume_path, map_location="cpu")
            validate_training_checkpoint(resume, resume_path)
            missing, unexpected = mem.load_state_dict(resume.get("prem_state_dict", resume), strict=False)
            if "optimizer_state_dict" in resume:
                try:
                    opt.load_state_dict(resume["optimizer_state_dict"])
                    for state in opt.state.values():
                        for key, value in state.items():
                            if torch.is_tensor(value):
                                state[key] = value.to(device=device)
                except ValueError as exc:
                    print(f"[resume] optimizer state skipped: {exc}", flush=True)
            step = int(resume.get("train_steps", 0))
            skipped = int(resume.get("skipped", 0))
            start_epoch, seen_in_epoch = checkpoint_resume_cursor(resume)
            checkpoint_epoch = start_epoch
            print(
                f"[resume] ckpt={resume_path} step={step} epoch={start_epoch} "
                f"seen_in_epoch={seen_in_epoch} skipped={skipped} "
                f"missing={list(missing)} unexpected={list(unexpected)}",
                flush=True,
            )
        else:
            print(f"[resume] no checkpoint found at {resume_path}; starting fresh", flush=True)

    trainable_params = [p for p in mem.parameters() if p.requires_grad]
    qa_gradients_validated = False

    if args.warmup_ratio > 0:
        local_total = len(train_rows)
        if distributed:
            total = torch.tensor([local_total], device=device, dtype=torch.long)
            dist.all_reduce(total, op=dist.ReduceOp.MAX)
            total_samples = int(total.item())
        else:
            total_samples = local_total
        warmup_steps = warmup_steps_from_ratio(total_samples, args.epochs, args.warmup_ratio)
    else:
        warmup_steps = 0
    if is_main:
        print(
            f"[schedule] lr={args.lr:g} warmup_ratio={args.warmup_ratio:g} "
            f"warmup_steps={warmup_steps}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs):
        checkpoint_epoch = epoch
        local_total = len(train_rows)
        if distributed:
            total_tensor = torch.tensor([local_total], device=device, dtype=torch.long)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.MAX)
            epoch_total = int(total_tensor.item())
        else:
            epoch_total = local_total

        local_start = seen_in_epoch if epoch == start_epoch else 0
        for row_idx in range(local_start, epoch_total):
            row = train_rows[row_idx] if row_idx < local_total else None
            opt.zero_grad(set_to_none=True)
            active = False
            gradient_weight = 0.0
            loss_value = None
            stats = {}

            if row is not None:
                writer_inputs = None
                answer_inputs = None
                stream_memory = None
                model_inputs = None
                out = None
                pred_loss = None
                loss = None
                try:
                    prem_kwargs = {
                        "prem_modulation": args.prem_modulation,
                        # The memory parameter is initialized from --alpha;
                        # forward uses a unit multiplier so log_alpha remains trainable.
                        "prem_alpha": 1.0,
                        "prem_num_slots": args.num_slots,
                        "prem_mem_dim": args.mem_dim,
                        "prem_layer_groups": args.prem_layer_groups,
                        "prem_router_gamma": args.router_gamma,
                        "prem_pred_weight": 0.0,
                        "prem_pred_tokens": args.pred_tokens,
                        "prem_disable_anti_distractor": args.disable_anti_distractor,
                        "prem_disable_novelty": args.disable_novelty,
                        "prem_disable_stability": args.disable_stability,
                        "prem_disable_evidence_gate_write": args.disable_evidence_gate_write,
                        "prem_uniform_write_route": args.uniform_write_route,
                    }
                    prem_kwargs["prem_max_memory_tokens"] = args.max_memory_tokens
                    if args.model_type == "llava_video":
                        from scripts.llava_video_utils import (
                            build_training_inputs as build_llava_video_training_inputs,
                            load_llava_writer_and_decoder_frames,
                            numeric_duration_hint,
                            prefix_query_embeddings,
                            preprocess_frames,
                            resolve_video_path,
                        )

                        video_name = row.get("video") or row.get("video_name")
                        if not video_name and row.get("videos"):
                            video_name = str(row["videos"][0])
                            if not Path(video_name).suffix:
                                video_name += ".mp4"
                        video_name = str(video_name or "")
                        media_path = resolve_video_path(Path(args.llava_video_root), video_name)
                        writer_frames, _, answer_frames, _, _ = (
                            load_llava_writer_and_decoder_frames(
                                media_path,
                                writer_max_frames=args.max_frames,
                                writer_fps=args.fps,
                                decoder_budget=args.visual_buffer_frames,
                                duration_hint=numeric_duration_hint(row.get("duration")),
                                force_sample=True,
                            )
                        )
                        writer_pixels = preprocess_frames(
                            llava_resources.image_processor,
                            writer_frames,
                            device,
                        )
                        answer_pixels = preprocess_frames(
                            llava_resources.image_processor,
                            answer_frames,
                            device,
                        )
                        answer_inputs, answer_prompt_len = build_llava_video_training_inputs(
                            row, llava_resources, answer_pixels, device
                        )
                        writer_query = prefix_query_embeddings(
                            model,
                            answer_inputs["input_ids"],
                            answer_prompt_len,
                            llava_resources.image_token_index,
                        )
                        stream_memory, pred_loss, writer_stats = model.build_prem_state_from_video(
                            images=writer_pixels,
                            query_embeddings=writer_query,
                            prem_alpha=1.0,
                            prem_num_slots=args.num_slots,
                            prem_mem_dim=args.mem_dim,
                            prem_layer_groups=args.prem_layer_groups,
                            prem_max_memory_tokens=args.max_memory_tokens,
                            prem_pred_weight=args.pred_weight,
                            prem_pred_tokens=args.pred_tokens,
                            prem_disable_anti_distractor=args.disable_anti_distractor,
                            prem_disable_novelty=args.disable_novelty,
                            prem_disable_stability=args.disable_stability,
                            prem_disable_evidence_gate_write=args.disable_evidence_gate_write,
                            prem_uniform_write_route=args.uniform_write_route,
                        )
                        model_inputs = {
                            **answer_inputs,
                            "use_cache": False,
                            "prem_prompt_lengths": torch.tensor(
                                [answer_prompt_len], device=device
                            ),
                            "prem_stream_state": stream_memory,
                            **prem_kwargs,
                        }
                    else:
                        writer_inputs, writer_prompt_len = build_llava_inputs(
                            row,
                            Path(args.llava_video_root),
                            processor,
                            args.max_frames,
                            args.max_pixels,
                            args.fps,
                            device,
                            args.model_type,
                        )
                        stream_memory, pred_loss, writer_stats = model.build_prem_state_from_video(
                            input_ids=writer_inputs["input_ids"],
                            attention_mask=writer_inputs["attention_mask"],
                            pixel_values_videos=writer_inputs["pixel_values_videos"],
                            video_grid_thw=writer_inputs["video_grid_thw"],
                            prem_prompt_lengths=torch.tensor([writer_prompt_len], device=device),
                            prem_alpha=1.0,
                            prem_num_slots=args.num_slots,
                            prem_mem_dim=args.mem_dim,
                            prem_layer_groups=args.prem_layer_groups,
                            prem_max_memory_tokens=args.max_memory_tokens,
                            prem_pred_weight=args.pred_weight,
                            prem_pred_tokens=args.pred_tokens,
                            prem_disable_anti_distractor=args.disable_anti_distractor,
                            prem_disable_novelty=args.disable_novelty,
                            prem_disable_stability=args.disable_stability,
                            prem_disable_evidence_gate_write=args.disable_evidence_gate_write,
                            prem_uniform_write_route=args.uniform_write_route,
                        )
                        if args.visual_buffer_frames > 0:
                            answer_inputs, answer_prompt_len = build_llava_inputs(
                                row,
                                Path(args.llava_video_root),
                                processor,
                                args.visual_buffer_frames,
                                args.max_pixels,
                                args.fps,
                                device,
                                args.model_type,
                            )
                        else:
                            answer_inputs, answer_prompt_len = build_text_only_inputs(row, processor, device)
                        model_inputs = {
                            "input_ids": answer_inputs["input_ids"],
                            "attention_mask": answer_inputs["attention_mask"],
                            "labels": answer_inputs["labels"],
                            "use_cache": False,
                            "prem_prompt_lengths": torch.tensor([answer_prompt_len], device=device),
                            "prem_stream_state": stream_memory,
                            **prem_kwargs,
                        }
                        if args.visual_buffer_frames > 0:
                            model_inputs["pixel_values_videos"] = answer_inputs["pixel_values_videos"]
                            model_inputs["video_grid_thw"] = answer_inputs["video_grid_thw"]
                        if args.model_type == "qwen2_5vl":
                            model_inputs["second_per_grid_ts"] = answer_inputs.get("second_per_grid_ts")
                    out = model(**model_inputs)
                    qa_loss = out.loss
                    answer_stats = dict(model.prem_last_stats or {})
                    router_balance_loss = float(answer_stats.get("router_balance_loss", 0.0))
                    qa_ce_loss = qa_loss - args.router_gamma * router_balance_loss
                    loss = qa_loss + args.pred_weight * pred_loss
                    gradient_weight = 1.0

                    if torch.isfinite(loss):
                        loss.backward()
                        if not qa_gradients_validated:
                            steer_grad_norm = qa_modulation_gradient_norm(mem)
                            if not torch.isfinite(torch.tensor(steer_grad_norm)) or steer_grad_norm <= 0:
                                raise RuntimeError(
                                    "PREM q/o steer gradients are missing. Check frozen-backbone "
                                    "gradient checkpointing before continuing training."
                                )
                            qa_gradients_validated = True
                            print(
                                f"[grad-check][rank={rank}] q_o_steer_grad_norm={steer_grad_norm:.6e}",
                                flush=True,
                            )
                        active = True
                        loss_value = float(loss.detach().item())
                        stats = {
                            **writer_stats,
                            **answer_stats,
                            "visual_buffer_frames": args.visual_buffer_frames,
                            "qa_ce_loss": float(qa_ce_loss.detach().cpu()),
                            "qa_objective_loss": float(qa_loss.detach().cpu()),
                            "evidence_prediction_loss": float(pred_loss.detach().cpu()),
                            "total_loss": loss_value,
                        }
                    else:
                        if args.skip_bad_samples:
                            skipped += 1
                            print(f"[skip][rank={rank}] {row.get('id')} non-finite loss", flush=True)
                        else:
                            raise FloatingPointError(f"Non-finite loss for sample {row.get('id')}")
                except Exception as exc:
                    opt.zero_grad(set_to_none=True)
                    if is_cuda_oom(exc) and args.skip_oom:
                        skipped += 1
                        print(f"[skip][rank={rank}] {row.get('id')} CUDA OOM", flush=True)
                        cleanup_cuda()
                    elif args.skip_bad_samples:
                        skipped += 1
                        row_id = row.get("id") if isinstance(row, dict) else None
                        print(
                            f"[skip][rank={rank}] {row_id} bad sample: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        traceback.print_exc()
                        cleanup_cuda()
                    else:
                        row_id = row.get("id") if isinstance(row, dict) else None
                        print(
                            f"[fatal][rank={rank}] {row_id} {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        traceback.print_exc()
                        if distributed:
                            dist.destroy_process_group()
                        raise
                finally:
                    model.prem_last_stream_state = None
                    del writer_inputs, answer_inputs, stream_memory, model_inputs, out, pred_loss, loss

            for param in trainable_params:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

            active_tensor = torch.tensor([1.0 if active else 0.0], device=device)
            gradient_weight_tensor = torch.tensor([gradient_weight if active else 0.0], device=device)
            if distributed:
                dist.all_reduce(active_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(gradient_weight_tensor, op=dist.ReduceOp.SUM)
                for param in trainable_params:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            active_count = float(active_tensor.item())
            global_gradient_weight = float(gradient_weight_tensor.item())
            if global_gradient_weight > 0:
                for param in trainable_params:
                    param.grad.div_(global_gradient_weight)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if warmup_steps > 0:
                    warmup_factor = min(1.0, (step + 1) / warmup_steps)
                    for pg in opt.param_groups:
                        pg["lr"] = args.lr * warmup_factor
                opt.step()

            if active and loss_value is not None:
                running.append(loss_value)
            step += 1
            seen_in_epoch = row_idx + 1

            if is_main and step % args.log_every == 0:
                denom = min(len(running), args.log_every)
                avg = sum(running[-denom:]) / denom if denom else float("nan")
                print(
                    f"[train] ep={epoch} step={step} loss={avg:.4f} "
                    f"active={active_count:.0f}/{world_size} supervised_qa={global_gradient_weight:.0f} "
                    f"skipped_rank0={skipped} "
                    f"writer={stats.get('stream_writer_mode')} "
                    f"write_gate={stats.get('mean_write_gate')} "
                    f"memory_norm={stats.get('mean_memory_norm')} "
                    f"query_steer={stats.get('mean_query_steer_norm')} "
                    f"output_steer={stats.get('mean_output_steer_norm')} "
                    f"buffer_frames={stats.get('visual_buffer_frames')} "
                    f"qa_ce={stats.get('qa_ce_loss')} "
                    f"router={stats.get('router_balance_loss')} "
                    f"pred={stats.get('evidence_prediction_loss')} "
                    f"router_max={stats.get('max_router_weight')} dt={time.time() - t0:.0f}s",
                    flush=True,
                )
            if is_main and args.save_every > 0 and step % args.save_every == 0:
                torch.save(make_ckpt(final=False), latest_ckpt)
                print(f"[save] step={step} -> {latest_ckpt}", flush=True)

        seen_in_epoch = 0
        checkpoint_epoch = epoch + 1

    ckpt = make_ckpt(final=True)
    if is_main:
        torch.save(ckpt, args.out_ckpt)
        torch.save(ckpt, latest_ckpt)
        print(
            f"[done] steps={step} first_loss={ckpt['first_train_loss']} "
            f"final_loss={ckpt['final_train_loss']} "
            f"train_wall={ckpt['train_wall_seconds']:.0f}s "
            f"train_peak_gpu={ckpt['train_peak_gpu_memory_allocated_bytes']} "
            f"-> {args.out_ckpt}",
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
