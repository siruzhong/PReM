#
#    Licensed under the Apache License, Version 2.0 (the "License"); 
#    you may not use this file except in compliance with the License. 
#    You may obtain a copy of the License at 
#
#        http://www.apache.org/licenses/LICENSE-2.0 
#
#    Unless required by applicable law or agreed to in writing, software 
#    distributed under the License is distributed on an "AS IS" BASIS, 
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#    See the License for the specific language governing permissions and 
#    limitations under the License. 

from collections import defaultdict
import csv
import json
import os
import argparse
import re
import subprocess
import multiprocessing
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import load_records, normalize_mcq_sample


def detect_cuda_devices():
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if value:
        return value
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=False, capture_output=True, text=True,
        )
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return ",".join(ids) or None
    except OSError:
        return None


def choose_existing(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


FRAME_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm')


def has_media(root, video_id):
    base = os.path.join(root, video_id)
    candidates = [base]
    stem, ext = os.path.splitext(base)
    if ext:
        candidates.append(stem)
    else:
        candidates.extend(stem + suffix for suffix in VIDEO_EXTS)

    for candidate in candidates:
        if os.path.isfile(candidate):
            return True
        if os.path.isdir(candidate):
            try:
                return any(name.lower().endswith(FRAME_EXTS) for name in os.listdir(candidate))
            except OSError:
                return False
    return False


def sample_media_name(row):
    for key in ('video_path', 'video', 'video_id', 'vid'):
        value = row.get(key)
        if value:
            return str(value)
    raise KeyError('missing video media key')


def choose_complete_media_root(data_file, *paths):
    existing = [path for path in paths if path and os.path.exists(path)]
    if not existing:
        return paths[0]
    try:
        rows = [normalize_mcq_sample(row) for row in load_records(data_file)]
        video_ids = {sample_media_name(row) for row in rows}
    except (OSError, KeyError, json.JSONDecodeError):
        return existing[0]
    for path in existing:
        if all(has_media(path, video_id) for video_id in video_ids):
            return path
    return existing[0]


def run_command(cmd):
    print(f"exec: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def launch_multi_gpu_eval(args, dataset_name, frame_dir, data_file, evaluation_name='evaluation'):
    model_path = args.model_path
    device_ids = None
    if getattr(args, "cuda_devices", None):
        device_ids = [device.strip() for device in args.cuda_devices.split(",") if device.strip()]
        if not device_ids:
            raise ValueError("--cuda_devices was provided but no device ids were parsed")
        num_chunks = len(device_ids) if args.num_chunks is None or args.num_chunks <= 0 else args.num_chunks
        if num_chunks > len(device_ids):
            raise ValueError(
                f"num_chunks={num_chunks} exceeds available cuda_devices={device_ids}"
            )
    else:
        num_chunks = args.num_chunks
    python_bin = sys.executable
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f'launch_multi_gpu_eval: args={args}')
    output_base = os.path.join(args.output_dir, evaluation_name, dataset_name)
    if not args.test:
        if 'videochatgpt' in dataset_name:
            split_list = ["generic", "temporal", "consistency"]
            data_file_list = data_file
        else:
            split_list = [""]
            data_file_list = [data_file]
        for data_file, split in zip(data_file_list, split_list):
            output_dir = output_base + split
            processes = []
            for idx in range(0, num_chunks):
                cmd = [python_bin, str(REPO_ROOT / "scripts/eval/inference_mcq_vqa.py"),
                        "--dataset", dataset_name,
                        "--model-path", model_path,
                        "--video_dir", frame_dir,
                        "--gt_file", data_file,
                        "--output_dir", output_dir,
                        "--output_name", "pred",
                        "--num-chunks", str(num_chunks),
                        "--chunk-idx", str(idx),
                ]
                if args.reproduce:
                    cmd += ["--reproduce"]
                    if args.reproduce_total_pixels:
                        cmd += ["--reproduce_total_pixels", str(args.reproduce_total_pixels)]
                if args.fps:
                    cmd += ["--fps", str(args.fps)]
                if args.max_pixels:
                    cmd += ["--max_pixels", str(args.max_pixels)]
                if args.max_frames:
                    cmd += ["--max_frames", str(args.max_frames)]
                if args.resized_height:
                    cmd += ["--resized_height", str(args.resized_height)]
                if args.resized_width:
                    cmd += ["--resized_width", str(args.resized_width)]
                if args.lora_path:
                    cmd += ["--lora-path", args.lora_path]
                if args.max_new_tokens:
                    cmd += ["--max_new_tokens", str(args.max_new_tokens)]
                if args.mcq_max_new_tokens:
                    cmd += ["--mcq_max_new_tokens", str(args.mcq_max_new_tokens)]
                if args.verbose:
                    cmd += ["--verbose"]
                if args.overwrite:
                    cmd += ["--overwrite"]
                if args.text_only:
                    cmd += ["--text_only"]
                if getattr(args, "profile_efficiency", False):
                    cmd += ["--profile_efficiency"]
                if dataset_name == 'videommesub':
                    cmd += ["--subtitle_frames", str(args.subtitle_frames)]
                logging.debug(f"Starting subprocess with command: {' '.join(cmd)}")
                # Start subprocess and capture output
                my_env = os.environ.copy()
                if device_ids is not None:
                    my_env["CUDA_VISIBLE_DEVICES"] = str(device_ids[idx])
                else:
                    my_env["CUDA_VISIBLE_DEVICES"] = str(idx)
                p = subprocess.Popen(cmd, env=my_env)
                processes.append(p)
            failed_chunks = []
            for idx, p in enumerate(processes):
                stdout, stderr = p.communicate()
                logging.debug(f"Subprocess {idx} stdout: {stdout}")
                if stderr:
                    logging.error(f"Subprocess {idx} stderr: {stderr}")
                if p.returncode != 0:
                    logging.error(f"Subprocess {idx} failed with return code {p.returncode}")
                    failed_chunks.append((idx, p.returncode))
                else:
                    logging.debug(f"Subprocess {idx} completed successfully")
            if failed_chunks:
                failed_msg = ", ".join(f"chunk {idx}: exit {code}" for idx, code in failed_chunks)
                raise RuntimeError(f"Inference subprocess failed for {dataset_name}: {failed_msg}")
    return output_base

def get_dataset_info(args):
    dataset = args.dataset
    if dataset == 'overwrite':
        if args.stress_variant is None:
            raise ValueError("--stress_variant is required when --dataset overwrite")
        if args.stress_frame_dir is None:
            raise ValueError("--stress_frame_dir is required when --dataset overwrite")
        if args.stress_data_file is None:
            raise ValueError("--stress_data_file is required when --dataset overwrite")
        return {
            'type': 'mc',
            'dataset_name': f'overwrite_{args.stress_variant}',
            'frame_dir': args.stress_frame_dir,
            'data_file': args.stress_data_file,
        }
    lvb_data_file = 'data/eval_video/LongVideoBench/raw/lvb_val.json'
    egoschema_data_file = 'data/eval_video/EgoSchema/test_qa.json'
    videomme_data_file = 'data/eval_video/videomme/test_qa.json'
    mlvu_data_file = 'data/eval_video/mlvu/test_qa.json'
    dataset_list = [
        {'type': 'mc', 'dataset_name': 'storm', 'frame_dir': 'data/eval_video/storm_real/video', 'data_file': 'data/eval_video/storm_real/qa_results/questions.jsonl'},
        {'type': 'mc', 'dataset_name': 'longvideobench', 'frame_dir': choose_complete_media_root(lvb_data_file, 'data/eval_video/LongVideoBench/frames', 'data/eval_video/LongVideoBench/videos'), 'data_file': lvb_data_file},
        {'type': 'mc', 'dataset_name': 'egoschema', 'frame_dir': choose_existing('data/eval_video/EgoSchema/frames', 'data/eval_video/EgoSchema/videos'), 'data_file': egoschema_data_file},
        {'type': 'mc', 'dataset_name': 'egoschema_all', 'frame_dir': choose_existing('data/eval_video/EgoSchema/frames', 'data/eval_video/EgoSchema/videos'), 'data_file': 'data/eval_video/EgoSchema/all_qa.json'},
        {'type': 'mc', 'dataset_name': 'videomme', 'frame_dir': choose_existing('data/eval_video/videomme/frames', 'data/eval_video/videomme/Video-MME/data'), 'data_file': videomme_data_file},
        {'type': 'mc', 'dataset_name': 'videommesub', 'frame_dir': choose_existing('data/eval_video/videomme/frames', 'data/eval_video/videomme/Video-MME/data'), 'data_file': videomme_data_file},
        {'type': 'mc', 'dataset_name': 'videommewo', 'frame_dir': choose_existing('data/eval_video/videomme/frames', 'data/eval_video/videomme/Video-MME/data'), 'data_file': videomme_data_file},
        {'type': 'mc', 'dataset_name': 'mvbench', 'frame_dir': choose_complete_media_root('data/eval_video/mvbench/test_qa.json', 'data/eval_video/mvbench/frames', 'data/eval_video/mvbench/videos'), 'data_file': 'data/eval_video/mvbench/test_qa.json'},
        {'type': 'mc', 'dataset_name': 'lvbench', 'frame_dir': choose_complete_media_root('data/eval_video/lvbench/test_qa.json', 'data/eval_video/lvbench/frames', 'data/eval_video/lvbench/videos'), 'data_file': 'data/eval_video/lvbench/test_qa.json'},
        {'type': 'mc', 'dataset_name': 'mlvu', 'frame_dir': choose_existing('data/eval_video/mlvu/frames', 'data/eval_video/mlvu/videos'), 'data_file': mlvu_data_file},
        {'type': 'oe', 'dataset_name': 'rvs_ego', 'frame_dir': 'data/eval_video/vstream-realtime/ego4d_frames', 'data_file': 'data/eval_video/vstream-realtime/test_qa_ego4d.json'},
        {'type': 'oe', 'dataset_name': 'rvs_movie', 'frame_dir': 'data/eval_video/vstream-realtime/movienet_frames', 'data_file': 'data/eval_video/vstream-realtime/test_qa_movienet.json'},
        {'type': 'oe', 'dataset_name': 'actnet', 'frame_dir': 'data/eval_video/ActivityNet-QA/test_frames', 'data_file': 'data/eval_video/ActivityNet-QA/test_qa.json'},
        {'type': 'oe', 'dataset_name': 'nextoe', 'frame_dir': 'data/eval_video/nextoe/nextoe_frames', 'data_file': 'data/eval_video/nextoe/test_qa.json'},
    ]
    dataset_list.append({'type':'oe', 
        'dataset_name': f'videochatgpt', 
        'frame_dir': 'data/eval_video/VideoChatGPTBench/video_10000frames_high_fps1',
        'data_file': ['data/eval_video/VideoChatGPTBench/test_{split}_qa.json'.format(split=split) for split in ["generic", "temporal", "consistency"]]
    })
    for d in dataset_list:
        if d['dataset_name'] == dataset:
            if args.use_high_fps:
                d['frame_dir'] = d['frame_dir'].replace('frames', 'frames_fps4')
            return d
    return None

def extract_answer(llm_message):
    matches = re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", str(llm_message).upper())
    if not matches:
        return None
    return {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}[matches[0]]

def calc_eval_result(output_path, num_chunks, data_path):
    expected_count = len(load_records(data_path))
    if num_chunks > 1:
        pred_contents = []
        for _idx in range(num_chunks):
            file = os.path.join(output_path, f"{num_chunks}_{_idx}.json")
            if not os.path.exists(file):
                raise FileNotFoundError(f"Missing prediction chunk: {file}")
            # pred_contents += [json.loads(line) for line in open(file)]
            for line in open(file):
                pred_contents += [json.loads(line)]
    else:
        file = os.path.join(output_path, f"pred.json")
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing prediction file: {file}")
        pred_contents = [json.loads(line) for line in open(file)]

    if not pred_contents:
        raise RuntimeError(f"No predictions were produced under {output_path}")
    signatures = {row.get("eval_signature") for row in pred_contents}
    if len(signatures) != 1 or None in signatures:
        raise RuntimeError(
            f"Predictions under {output_path} are missing provenance or mix evaluation signatures"
        )
    if len(pred_contents) != expected_count:
        raise RuntimeError(
            f"Incomplete predictions under {output_path}: got {len(pred_contents)}, expected {expected_count}. "
            "Check inference exceptions in subprocess logs."
        )

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    invalid_count = 0
    for sample in pred_contents:
        res = extract_answer(sample['pred'])
        invalid_count += int(res is None)
        if res == sample['answer']:
            acc = "yes"
            score = 1.0
        else:
            acc = "no"
            score = 0.0
        prediction_set[str(sample['id'])] = {
            'acc': acc,
            'score': score,
            **sample
        }
    
    json_path = os.path.join(output_path, 'result.json')
    with open(json_path, "w") as f:
        json.dump(prediction_set, f, indent=4)
    print("[main] All evaluation completed!")

    class ScoreMeter:
        def __init__(self):
            self.score_sum = 0
            self.count = 0
            self.yes_count = 0
            self.no_count = 0
            self.score_dict = {'yes': defaultdict(int), 'no': defaultdict(int)}

        def add_score(self, score, pred):
            self.score_sum += score
            self.count += 1
            pred_lower = pred.lower()
            if 'yes' in pred_lower:
                self.yes_count += 1
                self.score_dict['yes'][score] += 1
            elif 'no' in pred_lower:
                self.no_count += 1
                self.score_dict['no'][score] += 1

        def get_average_score(self):
            res = (self.score_sum / self.count) if self.count else 0
            return f"{res * 100:.6f}"

        def get_accuracy(self, response_type):
            if response_type == 'yes':
                res =  (self.yes_count / self.count) if self.count else 0
            elif response_type == 'no':
                res = (self.no_count / self.count) if self.count else 0
            else:
                res = 0
            return f"{res * 100:.6f}"

    meter_dic = {'total': ScoreMeter()}
    for key, result in prediction_set.items():
        # Computing score
        score = result['score']
        acc = result['acc']
        meter_dic["total"].add_score(score, acc)
        if 'a_type' in result and result['a_type'] is not None:
            # typ = str(result[1]['a_type'])
            typ = str(result['a_type'])
            if typ not in meter_dic:
                meter_dic[typ] = ScoreMeter()
            meter_dic[typ].add_score(score, acc)
            if 'next' in output_path:
                typ = typ[0]
                if typ not in meter_dic:
                    meter_dic[typ] = ScoreMeter()
                meter_dic[typ].add_score(score, acc)

    csv_dic = {
        'acc': meter_dic["total"].get_accuracy('yes'),
        'score': meter_dic["total"].get_average_score(),
        'count': len(pred_contents),
        'expected': expected_count,
        'invalid': invalid_count,
    }
    output = ""
    output += "Yes count: " + str(meter_dic["total"].yes_count) + "\n"
    output += "No count: " + str(meter_dic["total"].no_count) + "\n"
    output += "Invalid count: " + str(invalid_count) + "\n"
    output += "Accuracy: " + str(meter_dic["total"].get_accuracy('yes')) + "\n"
    output += "Average score: " + str(meter_dic["total"].get_average_score()) + "\n"
    output += "\n"
    output += "Total Score Yes/No distribution:\n"
    for key, value in meter_dic["total"].score_dict.items():
        output += f"{key}:\n"
        for k in range(0, 6):
            v = value[k]
            output += f"{k}: {v}\n"
    output += "\n"
    output += "Answer Type Score distribution:\n"
    output += 'Type, Accuracy, Avg_score\n'
    # key_list = sorted([k for k in meter_dic.keys()])
    key_list = [k for k in meter_dic.keys()]
    for key in key_list:
        output += f"{key}, {meter_dic[key].get_accuracy('yes')}, {meter_dic[key].get_average_score()}\n"
        csv_dic[key] = meter_dic[key].get_accuracy('yes')

    output += "\n"
    for k in csv_dic.keys():
        output += f"{k}, "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    for k in csv_dic.keys():
        output += str(csv_dic[k]) + ", "
    output = output.rstrip(', ')  # Remove the trailing comma and space
    output += "\n"

    # kaggle upload
    if 'egoschema' in output_path:
        upload_path = json_path.replace(".json", "_upload.csv")
        with open(upload_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['q_uid', 'answer'])
            all_qa = json.load(open('data/eval_video/EgoSchema/all_qa.json'))
            info_dic = {}
            for qa in all_qa:
                info_dic[str(qa['id'])] = qa['video_id']
            for key, result in prediction_set.items():
                pred = result['pred']
                q_uid = info_dic[key.split('_')[0]]
                res = extract_answer(pred)
                writer.writerow([q_uid, res if res is not None else -1])
    elif 'videomme' in output_path:
        score_dic = {
            "duration": {"short": ScoreMeter(), "medium": ScoreMeter(), "long": ScoreMeter()},
            "domain": {"Knowledge": ScoreMeter(), "Film & Television": ScoreMeter(), "Sports Competition": ScoreMeter(), "Artistic Performance": ScoreMeter(), "Life Record": ScoreMeter(), "Multilingual": ScoreMeter()}, 
            "sub_category": {"Humanity & History": ScoreMeter(), "Literature & Art": ScoreMeter(), "Biology & Medicine": ScoreMeter(), "Finance & Commerce": ScoreMeter(), "Astronomy": ScoreMeter(), "Geography": ScoreMeter(), "Law": ScoreMeter(), "Life Tip": ScoreMeter(), "Technology": ScoreMeter(), "Animation": ScoreMeter(), "Movie & TV Show": ScoreMeter(), "Documentary": ScoreMeter(), "News Report": ScoreMeter(), "Esports": ScoreMeter(), "Basketball": ScoreMeter(), "Football": ScoreMeter(), "Athletics": ScoreMeter(), "Other Sports": ScoreMeter(), "Stage Play": ScoreMeter(), "Magic Show": ScoreMeter(), "Variety Show": ScoreMeter(), "Acrobatics": ScoreMeter(), "Handicraft": ScoreMeter(), "Food": ScoreMeter(), "Fashion": ScoreMeter(), "Daily Life": ScoreMeter(), "Travel": ScoreMeter(), "Pet & Animal": ScoreMeter(), "Exercise": ScoreMeter(), "Multilingual": ScoreMeter()}, 
            "task_type": {"Temporal Perception": ScoreMeter(), "Spatial Perception": ScoreMeter(), "Attribute Perception": ScoreMeter(), "Action Recognition": ScoreMeter(), "Object Recognition": ScoreMeter(), "OCR Problems": ScoreMeter(), "Counting Problem": ScoreMeter(), "Temporal Reasoning": ScoreMeter(), "Spatial Reasoning": ScoreMeter(), "Action Reasoning": ScoreMeter(), "Object Reasoning": ScoreMeter(), "Information Synopsis": ScoreMeter()},
        }
        total_dic = ScoreMeter()
        test_qa = json.load(open(data_path))
        info_dic = {}
        for qa in test_qa:
            info_dic[str(qa['id'])] = qa
        level_list = ['duration', 'domain', 'sub_category', 'task_type']
        for key, result in prediction_set.items():
            acc = result['acc']
            qa = info_dic[key.split('_')[0]]
            for level in level_list:
                score_dic[level][qa[level]].add_score(0, acc)
            total_dic.add_score(0, acc)
        output += '\n'
        output += 'Type, Accuracy\n'
        for level in level_list:
            for key, meter_dic in score_dic[level].items():
                output += f"{key}, {float(meter_dic.get_accuracy('yes')):.02f}\n"
        output += f"Overall, {float(total_dic.get_accuracy('yes')):.02f}\n"
    elif 'lvbench' in output_path:
        score_dic = {
            "key information retrieval": ScoreMeter(),
            "event understanding": ScoreMeter(),
            "summarization": ScoreMeter(),
            "entity recognition": ScoreMeter(),
            "reasoning": ScoreMeter(),
            "temporal grounding": ScoreMeter(),
        }
        total_dic = ScoreMeter()
        test_qa = json.load(open(data_path))
        info_dic = {}
        for qa in test_qa:
            info_dic[str(qa['id'])] = qa
        for key, result in prediction_set.items():
            acc = result['acc']
            qa = info_dic[key]
            for typ in qa['question_type']:
                score_dic[typ].add_score(0, acc)
            total_dic.add_score(0, acc)
        output += '\n'
        output += 'Type, Accuracy\n'
        for key, meter_dic in score_dic.items():
            output += f"{key}, {float(meter_dic.get_accuracy('yes')):.02f}\n"
        output += f"Overall, {float(total_dic.get_accuracy('yes')):.02f}\n"
    elif 'scalelong' in output_path:
        Granularities = ["Video Clip", "Video Shot", "Video Event", "Video Story"]
        question_types = ["Causal Reasoning", "Object Recognition", "Action Understanding", "Information Summary", "Counting Problem"]
        all_tags = Granularities + question_types
        score_dic = {}
        total_dic = ScoreMeter()
        for tag in all_tags:
            if tag not in score_dic:
                score_dic[tag] = ScoreMeter()
        test_qa = json.load(open(data_path))
        info_dic = {}
        for qa in test_qa:
            info_dic[str(qa['id'])] = qa
        for key, result in prediction_set.items():
            acc = result['acc']
            qa = info_dic[key]
            typ = qa['question_type']
            # All tags: {'Objective Recognition', 'Information Inference', 'Counting Problem', 'Information Summary', 'Counting Problems', 'Action Understanding', 'Casual Reasoning', 'Causal Reasoning', 'Object Recognition', 'information inference'}
            if typ == "information inference" or typ == "Information Inference":
                typ = "Information Summary"
            elif typ == "Counting Problems":
                typ = "Counting Problem"
            elif typ == "Objective Recognition":
                typ = "Object Recognition"
            elif typ == "Casual Reasoning":
                typ = "Causal Reasoning"
            score_dic[typ].add_score(0, acc)
            granu = qa['granularity']
            score_dic[granu].add_score(0, acc)
            total_dic.add_score(0, acc)
        output += '\n'
        output += 'Type, Accuracy\n'
        for key, meter_dic in score_dic.items():
            output += f"{key}, {float(meter_dic.get_accuracy('yes')):.02f}\n"
        output += f"Overall, {float(total_dic.get_accuracy('yes')):.02f}\n"

    print(output)
    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, 'w') as f:
        f.write(output)

def calc_gpt_based_eval_result(args, output_path, num_chunks, data_path):
    print(f'[main] cal gpt-based eval result')
    print(f'output_path={output_path}')
    print(f'num_chunks={num_chunks}')
    print(f'data_path={data_path}')

    if 'videochatgpt' in output_path:
        assert args.api_key is not None
        assert args.api_base is not None
        assert args.api_type is not None
        assert args.api_version is not None
        code_list = [
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_1_correctness.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_2_detailed_orientation.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_3_context.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_4_temporal.py",
            "data/eval_video/VideoChatGPTBench/evaluate_benchmark_5_consistency.py",
        ]
        split_list = ["generic", "generic", "generic", "temporal", "consistency", ]
        short_list = ["1_CI", "2_DO", "3_CU", "4_TU", "5_CO"]
        for split, code, short in zip(split_list, code_list, short_list):
            cmd = [sys.executable, code,
                "--pred_path", output_path + split,
                "--output_dir", os.path.join(output_path + short, "results"),
                "--output_json", os.path.join(output_path + short, "results.json"),
                "--num_chunks", str(num_chunks),
                "--num_tasks", "16",
                "--api_key", args.api_key,
                "--api_base", args.api_base,
                "--api_type", args.api_type,
                "--api_version", args.api_version,
            ]
            run_command(cmd)
    else:
        assert args.api_key is not None
        assert args.api_base is not None
        assert args.api_type is not None
        assert args.api_version is not None
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/external/flash_vstream/eval/eval_activitynet_qa.py"),
           "--pred_path", os.path.join(output_path),
           "--output_dir", os.path.join(output_path, "results"),
           "--output_json", os.path.join(output_path, "results.json"),
           "--num_chunks", str(num_chunks),
           "--num_tasks", "16",
           "--api_key", args.api_key,
           "--api_base", args.api_base,
           "--api_type", args.api_type,
           "--api_version", args.api_version,
           ]
        run_command(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default='~')
    parser.add_argument("--evaluation_name", type=str, default='evaluation')
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--cuda_devices", type=str, default=None, help="Comma-separated physical GPU ids to use for subprocess eval, e.g. 1,3,4,5")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--testtest", action="store_true")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subtitle_frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--resized_width", type=int, default=None)
    parser.add_argument("--resized_height", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--mcq_max_new_tokens", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--use_high_fps", action="store_true", default=False)
    parser.add_argument("--reproduce", action="store_true", default=False)
    parser.add_argument("--reproduce_total_pixels", type=int, default=None)
    parser.add_argument("--stress_variant", type=str, default=None)
    parser.add_argument("--stress_frame_dir", type=str, default=None)
    parser.add_argument("--stress_data_file", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--profile_efficiency", action="store_true")
    parser.add_argument("--api_key", default=None, type=str, help="OpenAI API key")
    parser.add_argument("--api_type", default=None, type=str, help="OpenAI API type")
    parser.add_argument("--api_version", default=None, type=str, help="OpenAI API version")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI API base")
    args = parser.parse_args()
    args.cuda_devices = args.cuda_devices or detect_cuda_devices()
    if args.num_chunks is None or args.num_chunks <= 0:
        args.num_chunks = len(args.cuda_devices.split(",")) if args.cuda_devices else 1

    info = get_dataset_info(args)
    if info is None:
        print(f'[main] ERROR {args.dataset} dataset was not found!')
        exit(0)
    typ = info.pop('type')
    out_dir = launch_multi_gpu_eval(args, **info, evaluation_name=args.evaluation_name)
    if typ == 'mc':
        print(f'[main] Execute {args.dataset} mcq evaluation')
        calc_eval_result(out_dir, args.num_chunks, info['data_file'])
    elif typ =='oe':
        print(f'[main] Execute {args.dataset} open-ended evaluation')
        calc_gpt_based_eval_result(args, out_dir, args.num_chunks, info['data_file'])
