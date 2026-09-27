# Based on https://github.com/haotian-liu/LLaVA.

import os
import json
import math
import re
import time
import cv2
import torch
import argparse
import importlib.util
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")
from decord import VideoReader, cpu

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_utils import process_vision_info, qwen3_video_metadata
from scripts.eval.data_io import get_sample_media_name, load_records, normalize_mcq_sample
from scripts.eval.prem_checkpoint import artifact_identity, make_eval_signature
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration

from models.flash_memory_constants import DEFAULT_FLASH_MEMORY_CONFIG
from models.vstream_qwen2vl_model import FlashVStreamQwen2VLConfig, FlashVStreamQwen2VLModel
from models.vstream_qwen2vl_processor import FlashVStreamQwen2VLProcessor

import warnings
from peft import AutoPeftModelForCausalLM  # depend on models.vstream_qwen2vl

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm")
FRAME_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _synchronize_cuda(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()


def _efficiency_stats(enabled: bool) -> dict[str, int]:
    if not enabled or not torch.cuda.is_available():
        return {}
    return {
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    # chunk_size = math.ceil(len(lst) / n)  # integer division
    # return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]
    res = [[] for i in range(n)]
    for i, x in enumerate(lst):
        res[i % n].append(x)
    return res

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video(video_path):
    last_exc = None
    for attempt in range(3):
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            frame_idx = [i for i in range(0, len(vr), fps)]
            return vr.get_batch(frame_idx).asnumpy()
        except Exception as exc:
            last_exc = exc
            text = str(exc)
            if "DECORD_EOF_RETRY_MAX" not in text and "Unable to handle EOF" not in text:
                raise
            if attempt == 2:
                break
            time.sleep(0.5 * (attempt + 1))
    raise last_exc

def ranki_print(s, verbose_only=False):
    if verbose_only and not getattr(args, "verbose", False):
        return
    print(f'[cuda:{args.chunk_idx}] {s}')

def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def load_subtitles(subtitle_path):
    subtitles = {}
    with open(subtitle_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
        for section in content:
            if section.strip():
                lines = section.split("\n")
                if len(lines) >= 3:
                    time_range = lines[1].split(" --> ")
                    start_time = parse_subtitle_time(time_range[0])
                    end_time = parse_subtitle_time(time_range[1])
                    text = " ".join(line for line in lines[2:])
                    subtitles[(start_time, end_time)] = text
    return subtitles

def convert_time_to_frame(time_in_seconds, fps):
    return int(time_in_seconds * fps)

def extract_subtitles(video_path, subtitle_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    subtitles = load_subtitles(subtitle_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        start_frame = convert_time_to_frame(start_time, fps)
        end_frame = convert_time_to_frame(end_time, fps)
        subtitle_frames.append((start_frame, end_frame, text))

    return subtitle_frames, total_frame


def get_subtitle(video_id, frame_num = 180):
    video_name = video_id
    SUBTITLE_PROMPT = "This video's subtitles are listed below: \n"
    QUESTION_PROMPT = "Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option."
    data_path = os.path.join("./data/eval_video/videomme/Video-MME/data", video_name + ".mp4")
    subtitle_path = os.path.join("./data/eval_video/videomme/Video-MME/subtitle", video_name + ".srt")
    
    if os.path.exists(subtitle_path):  # Denote have subtitle
        print(f'try to open{subtitle_path}')
        subtitle = open(subtitle_path, encoding='utf-8').readlines()
    else:
        subtitle = ""
    if subtitle == "":
        subtitle = "No subtitles available"
    else:
        subtitle_by_frame, total_frame = extract_subtitles(data_path, subtitle_path)
        uniform_sampled_frames = np.linspace(0, total_frame - 1, frame_num, dtype=int).tolist()

        subtitle_by_frame_idx = []
        for frame_idx in uniform_sampled_frames:
            for idx, title in enumerate(subtitle_by_frame):
                if frame_idx < title[1] and frame_idx >= title[0]:
                    subtitle_by_frame_idx.append(idx)
        subtitle_by_frame_idx = list(set(subtitle_by_frame_idx))

        textlist = []
        for idx in subtitle_by_frame_idx:
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            raw_text = re.findall(pattern, subtitle_by_frame[idx][2])
            try:
                textlist.append(raw_text[0])
            except:
                continue
        subtitle = "\n".join(textlist)
    return subtitle


def resolve_video_source(video_dir, video_name):
    base = os.path.join(video_dir, video_name)
    candidates = [base]
    root, ext = os.path.splitext(base)
    if ext:
        candidates.append(root)
    else:
        candidates.extend(root + suffix for suffix in VIDEO_EXTS)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Video source for {video_name} does not exist under {video_dir}. Tried: {candidates}")


def frame_sort_key(path):
    name = os.path.basename(path)
    numbers = re.findall(r"\d+", name)
    if numbers:
        return int(numbers[-1])
    return name


def list_frame_paths(video_path):
    frame_paths = [
        os.path.join(video_path, frame_path)
        for frame_path in os.listdir(video_path)
        if frame_path.lower().endswith(FRAME_EXTS)
    ]
    return sorted(frame_paths, key=frame_sort_key)


def extract_mcq_labels(question):
    labels = []
    for line in str(question).splitlines():
        match = re.match(r"\s*[\(\[]?([A-Z])[\)\].:]\s+", line.upper())
        if match and match.group(1) not in labels:
            labels.append(match.group(1))
    if not labels:
        raise ValueError(f"MCQ question has no parseable options: {question!r}")
    return labels

def mcq_prefix_allowed_tokens(processor, labels, prompt_length):
    tokenizer = getattr(processor, "tokenizer", processor)
    option_ids = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            return None
        option_ids.append(int(ids[0]))
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    def allowed_tokens(_batch_id, input_ids):
        if input_ids.shape[-1] <= prompt_length:
            return option_ids
        return [int(eos_id)] if eos_id is not None else option_ids
    return allowed_tokens

def get_model_type(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path) as f:
        return json.load(f).get("model_type")


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    use_flash_attn = importlib.util.find_spec("flash_attn") is not None
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    if not use_flash_attn:
        warnings.warn("flash_attn is not installed; falling back to eager attention.")
    qwen_path = 'ckpt/Qwen2-VL-7B-Instruct'
    model_kind = "flash_vstream"
    if args.lora_path:
        model_config = FlashVStreamQwen2VLConfig.from_pretrained(
            qwen_path,
            trust_remote_code=True,
        )
        if getattr(model_config.vision_config, 'flash_memory_config', None) is None:
            warnings.warn(f'Qwen2VLVisionConfig.flash_memory_config is not set. Set it to default, sample 10000')
            model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG
        # use lora
        lora_path = args.lora_path
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path, 
            config=model_config,
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        ).eval()
        processor = FlashVStreamQwen2VLProcessor.from_pretrained(qwen_path)
    else:
        # use full model
        model_path = args.model_path
        model_type = get_model_type(model_path)
        if model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
            model_kind = model_type
            if model_type == "qwen2_vl":
                model_cls = Qwen2VLForConditionalGeneration
            elif model_type == "qwen2_5_vl":
                model_cls = Qwen2_5_VLForConditionalGeneration
            else:
                from transformers import Qwen3VLForConditionalGeneration

                model_cls = Qwen3VLForConditionalGeneration
            model = model_cls.from_pretrained(
                model_path,
                device_map="cuda",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            ).eval()
            processor_kwargs = {"trust_remote_code": True}
            if model_type != "qwen3_vl":
                processor_kwargs["use_fast"] = False
            processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        else:
            model_config = FlashVStreamQwen2VLConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if getattr(model_config.vision_config, 'flash_memory_config', None) is None:
                warnings.warn(f'Qwen2VLVisionConfig.flash_memory_config is not set. Set it to default, sample 10000')
                model_config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG
            model = FlashVStreamQwen2VLModel.from_pretrained(
                model_path, 
                config=model_config,
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
            ).eval()
            processor = FlashVStreamQwen2VLProcessor.from_pretrained(qwen_path)

    ranki_print(f"Loaded model_kind={model_kind}, attn_implementation={attn_implementation}")
    profile = bool(getattr(args, "profile_efficiency", False))
    if profile:
        torch.cuda.reset_peak_memory_stats()
    signature_payload = {
        "protocol": "text_only" if args.text_only else "full_video",
        "model_kind": model_kind,
        "model": artifact_identity(args.model_path),
        "dataset": args.dataset,
        "max_frames": args.max_frames,
        "fps": args.fps,
        "max_pixels": args.max_pixels,
        "mcq_max_new_tokens": args.mcq_max_new_tokens,
        "max_new_tokens": args.max_new_tokens,
    }
    if profile:
        signature_payload["profile_efficiency"] = True
    eval_signature = make_eval_signature(signature_payload)
    flash_memory_config = None
    if model_kind == "flash_vstream":
        flash_memory_config = model.config.vision_config.flash_memory_config
        ranki_print(f"Load processor success!, processor with flash_memory_config={flash_memory_config}", verbose_only=True)
        for k, v in DEFAULT_FLASH_MEMORY_CONFIG.items():
            if k not in flash_memory_config:
                flash_memory_config[k] = v

    # Load both ground truth file containing questions and answers
    gt_questions = [normalize_mcq_sample(sample) for sample in load_records(args.gt_file)]
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
        except Exception as e:
            ranki_print(f'mkdir Except: {e}')
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    if os.path.exists(answers_file) and not args.overwrite:
        with open(answers_file, "r") as f:
            existing_rows = [json.loads(row) for row in f if row.strip()]
            if any(row.get("eval_signature") != eval_signature for row in existing_rows):
                raise RuntimeError(
                    f"Existing predictions use a different evaluator/checkpoint: {answers_file}. "
                    "Use --overwrite or a new output directory."
                )
            id_set = {row['id'] for row in existing_rows}
            gt_questions = [sample for sample in gt_questions if sample['id'] not in id_set]
    ans_file = open(answers_file, "w" if args.overwrite else "a")

    for sample in tqdm(gt_questions, desc=f"cuda:{args.chunk_idx} "):
        try:
            if 'question' in sample:
                q_base_list = [sample['question']]
            else:
                q_base_list = [sample['question1'], sample['question2']]
            out_list = []
            question_list = []
            sample_update_seconds = 0.0
            sample_answer_seconds = 0.0
            for q_base in q_base_list:
                option_labels = None
                if args.dataset not in ["rvs_ego", "rvs_movie", "actnet", "nextoe", "videochatgpt"]:
                    option_labels = extract_mcq_labels(q_base)
                    label_text = ", ".join(option_labels)
                    QUESTION_PROMPT = f"Select the best answer to the following multiple-choice question based on the video. Respond with only the letter ({label_text}) of the correct option."
                else:
                    QUESTION_PROMPT = "Answer the following open-ended question based on the video. "
                video_name = get_sample_media_name(sample)
                question = QUESTION_PROMPT + q_base
                is_mcq_flag = option_labels is not None
                if 'videommesub' in args.dataset:
                    subtitle = get_subtitle(str(sample.get('video_id', video_name)), args.subtitle_frames)
                    SUBTITLE_PROMPT = "This video's subtitles are listed below: \n"
                    label_text = ", ".join(option_labels)
                    QUESTION_PROMPT = f"Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter ({label_text}) of the correct option."
                    question = SUBTITLE_PROMPT + subtitle + '\n' + QUESTION_PROMPT + q_base
                elif args.dataset in ['rvs_ego', 'rvs_movie', 'actnet', 'nextoe', 'videochatgpt']:
                    QUESTION_PROMPT = "Answer the following open-ended question based on the video. "
                    question = QUESTION_PROMPT + q_base
                    is_mcq_flag = False

                if args.text_only:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": question}],
                        }
                    ]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if is_mcq_flag:
                        text += 'Best option: ('
                    _synchronize_cuda(profile)
                    encode_start = time.perf_counter()
                    inputs = processor(
                        text=[text],
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)
                    _synchronize_cuda(profile)
                    sample_update_seconds += time.perf_counter() - encode_start
                    _synchronize_cuda(profile)
                    answer_start = time.perf_counter()
                    with torch.inference_mode():
                        max_new_tokens = args.mcq_max_new_tokens if is_mcq_flag else args.max_new_tokens
                        generate_kwargs = dict(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=max_new_tokens,
                            top_k=1,
                            do_sample=False,
                        )
                        if is_mcq_flag and option_labels is not None:
                            allowed_tokens = mcq_prefix_allowed_tokens(
                                processor, option_labels, inputs.input_ids.shape[1]
                            )
                            if allowed_tokens is not None:
                                generate_kwargs["prefix_allowed_tokens_fn"] = allowed_tokens
                        generated_ids = model.generate(**generate_kwargs)
                    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
                    output = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    _synchronize_cuda(profile)
                    sample_answer_seconds += time.perf_counter() - answer_start
                    out_list.append(output)
                    question_list.append(question)
                    continue

                video_path = resolve_video_source(args.video_dir, video_name)
                frame_paths = []
                if os.path.isdir(video_path):
                    frame_paths = list_frame_paths(video_path)
                    if not frame_paths:
                        raise FileNotFoundError(f"Frame directory {video_path} contains no image frames.")
                    if args.reproduce:
                        frame_paths = frame_paths[::4]  # set to fps=2, only for egoschema
                        # frame_paths = frame_paths[::2]  # set to fps=2, for other
                        max_frames = None
                    else:
                        max_frames = args.max_frames
                        if args.fps is None:
                            total_time = len(frame_paths)
                            ranki_print(f'Use max_frames mode, limit max_frames={args.max_frames}, real frames={total_time}', verbose_only=True)
                            if 'frames_fps4' in args.video_dir and total_time > max_frames:  # tight sample, (0, 0.5) (4, 4.5) (8, 8.5) (12, 12.5)
                                assert max_frames % 2 == 0, f"max_frames must be even, now is {max_frames}"
                                nframes = max_frames // 2
                                indices = torch.linspace(0, total_time - 1, nframes).round().long().tolist()
                                new_paths = []
                                for i in indices:  # tight sample
                                    if i < total_time - 1:
                                        new_paths.append(frame_paths[i])
                                        new_paths.append(frame_paths[i + 1])
                                    else:
                                        new_paths.append(frame_paths[i - 1])
                                        new_paths.append(frame_paths[i])
                                # for i in indices:  # 2 tight sample
                                #     if i < total_time - 2:
                                #         new_paths.append(frame_paths[i])
                                #         new_paths.append(frame_paths[i + 2])
                                #     else:
                                #         new_paths.append(frame_paths[i - 2])
                                #         new_paths.append(frame_paths[i])
                                assert len(new_paths) == max_frames
                                frame_paths = new_paths
                            elif 'rvs_movie' in args.dataset:
                                new_paths = []
                                nframes = min(total_time, max_frames // 2)
                                indices = torch.linspace(0, total_time - 1, nframes).round().long().tolist()
                                new_paths = []
                                for i in indices:  # twice sample
                                    new_paths.append(frame_paths[i])
                                    new_paths.append(frame_paths[i])
                                frame_paths = new_paths
                        else:
                            ranki_print(f'Use fps mode, target fps={args.fps}', verbose_only=True)
                            total_time = len(frame_paths)
                            source_fps = 4.0 if "frames_fps4" in args.video_dir else 1.0
                            nframes = max(1, round(total_time / source_fps * args.fps))
                            nframes = min(nframes, args.max_frames) if args.max_frames else nframes
                            indices = torch.linspace(0, total_time - 1, nframes).round().long().tolist()
                            frame_paths = [frame_paths[i] for i in indices]
                            max_frames = None
                    video_input = frame_paths
                else:
                    max_frames = args.max_frames
                    video_input = video_path
                    ranki_print(f'Use raw video file, max_frames={max_frames}, path={video_path}', verbose_only=True)
                content_video = {
                    "type": "video",
                    "video": video_input,
                }
                if args.fps is not None:
                    content_video["fps"] = args.fps
                if args.reproduce:
                    if args.reproduce_total_pixels is not None:
                        content_video['total_pixels'] = args.reproduce_total_pixels
                    # content_video['max_frames'] = 16  # only for 480 test
                    content_video['min_pixels'] = 32 * 28 * 28  # 1 / 8 224 * 224
                else:
                    if max_frames is not None:
                        content_video['max_frames'] = max_frames
                    if args.max_pixels is not None:
                        content_video['max_pixels'] = args.max_pixels
                    if args.resized_height is not None:
                        content_video['resized_height'] = args.resized_height
                    if args.resized_width is not None:
                        content_video['resized_width'] = args.resized_width
                messages = [
                    {
                        "role": "user",
                        "content": [
                            content_video,
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                if is_mcq_flag == True:
                    text += 'Best option: ('
                _synchronize_cuda(profile)
                encode_start = time.perf_counter()
                image_inputs, video_inputs = process_vision_info(messages)
                processor_kwargs = dict(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                if model_kind == "qwen3_vl":
                    processor_kwargs["video_metadata"] = [
                        qwen3_video_metadata(content_video)
                    ]
                    processor_kwargs["do_sample_frames"] = False
                elif model_kind in ("qwen2_vl", "qwen2_5_vl") and args.fps is not None:
                    processor_kwargs["fps"] = [args.fps]
                if flash_memory_config is not None:
                    processor_kwargs["flash_memory_config"] = flash_memory_config
                inputs = processor(**processor_kwargs)
                input_ids = inputs.input_ids.cuda()
                attention_mask = inputs.attention_mask.cuda()
                pixel_values_videos = inputs.pixel_values_videos.cuda()
                video_grid_thw = inputs.video_grid_thw.cuda()
                visual_position_ids = getattr(inputs, "visual_position_ids", None)
                if visual_position_ids is not None:
                    visual_position_ids = visual_position_ids.cuda()
                _synchronize_cuda(profile)
                sample_update_seconds += time.perf_counter() - encode_start

                _synchronize_cuda(profile)
                answer_start = time.perf_counter()
                with torch.inference_mode():
                    max_new_tokens = args.mcq_max_new_tokens if is_mcq_flag else args.max_new_tokens
                    generate_kwargs = dict(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                        max_new_tokens=max_new_tokens,
                        top_k=1,
                        do_sample=False,
                    )
                    if visual_position_ids is not None:
                        generate_kwargs["visual_position_ids"] = visual_position_ids
                    if is_mcq_flag and option_labels is not None:
                        allowed_tokens = mcq_prefix_allowed_tokens(processor, option_labels, input_ids.shape[1])
                        if allowed_tokens is not None:
                            generate_kwargs["prefix_allowed_tokens_fn"] = allowed_tokens
                    generated_ids = model.generate(**generate_kwargs)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    outputs = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    output = outputs[0].strip()
                _synchronize_cuda(profile)
                sample_answer_seconds += time.perf_counter() - answer_start
                input_color = ""  # Blue
                output_color = ""  # Green
                reset_color = ""  # Reset to default color
                ranki_print(f'{input_color}input={text}{reset_color} {output_color}output={output}{reset_color}', verbose_only=True)
                out_list.append(output)
                question_list.append(question)
        except Exception as e:
            ranki_print(f'Except: {e}')
            continue

        if len(question_list) == 0:
            continue

        sample_set = {
            'id': sample['id'],
            'question': question,
            'answer': sample['answer'],
            'eval_signature': eval_signature,
        }
        if profile:
            sample_set["stream_update_seconds"] = sample_update_seconds
            sample_set["answer_seconds"] = sample_answer_seconds
            sample_set.update(_efficiency_stats(True))
        for key in (
            "variant",
            "stress_variant",
            "question_category",
            "level",
            "topic_category",
            "duration_group",
            "duration",
            "num_frames",
            "stress_num_true_evidence_frames",
            "stress_num_distractor_frames",
            "episode_id",
            "query_time",
            "question_type",
            "question_subtype",
            "a_type",
            "source_answer_index",
            "option_permutation",
        ):
            if key in sample:
                sample_set[key] = sample[key]
        if 'question' in sample:
            sample_set['question'] = question_list[0]
            sample_set['pred'] = out_list[0]
        else:
            sample_set['question1'] = question_list[0]
            sample_set['question2'] = question_list[1]
            sample_set['pred1'] = out_list[0]
            sample_set['pred2'] = out_list[1]
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()


def main():
    global args

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=20)
    parser.add_argument("--subtitle_frames", type=int, default=1080)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    parser.add_argument("--resized_width", type=int, default=None)
    parser.add_argument("--resized_height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--mcq_max_new_tokens", type=int, default=1)
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--profile_efficiency", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--reproduce", action="store_true", default=False) 
    parser.add_argument("--reproduce_total_pixels", type=int, default=None)

    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
