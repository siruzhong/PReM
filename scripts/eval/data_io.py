"""Dataset loading and normalization helpers for evaluation scripts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    """Load either a JSON array or a JSONL file."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                return []
            if not char.isspace():
                first = char
                break
        handle.seek(0)
        if first == "[":
            records = json.load(handle)
        else:
            records = [json.loads(line) for line in handle if line.strip()]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a list of JSON objects in {path}")
    return records


def _storm_video_path(episode_id: str) -> tuple[str, str]:
    match = re.match(r"^(P\d+)", episode_id)
    if match:
        participant = match.group(1)
        return f"{participant}/{episode_id}.mp4", participant
    match = re.search(r"(FloorPlan\d+)", episode_id)
    if match:
        floorplan = match.group(1)
        return f"{floorplan}/{episode_id}.mp4", floorplan
    match = re.match(r"^([a-z]+)_bike", episode_id)
    if match:
        site = match.group(1)
        return f"{episode_id}.mp4", site
    match = re.match(r"^(iiith|indiana)_", episode_id)
    if match:
        collection = match.group(1)
        return f"{episode_id}.mp4", collection
    # Ego-Exo4D Sports exports keep all site-specific clips directly under
    # one video root rather than in participant subdirectories.
    match = re.match(
        r"^(cmu|sfu|unc|uniandes|utokyo|minnesota|georgiatech|nus)_",
        episode_id,
    )
    if match:
        collection = match.group(1)
        return f"{episode_id}.mp4", collection
    raise ValueError(f"Cannot infer STORM-Bench video directory from episode_id={episode_id!r}")


def _stable_option_permutation(sample_id: str, count: int) -> list[int]:
    return sorted(
        range(count),
        key=lambda index: hashlib.sha256(f"{sample_id}:{index}".encode("utf-8")).digest(),
    )


def normalize_mcq_sample(sample: dict) -> dict:
    """Convert supported MCQ schemas to the evaluator's canonical schema."""
    row = dict(sample)
    if "episode_id" in row and "options" in row and "answer_index" in row:
        options = list(row["options"])
        answer_index = int(row["answer_index"])
        if not 2 <= len(options) <= 26:
            raise ValueError(f"Sample {row.get('id')} has {len(options)} options")
        if not 0 <= answer_index < len(options):
            raise ValueError(f"Sample {row.get('id')} has invalid answer_index={answer_index}")

        sample_id = str(row.get("id", ""))
        permutation = _stable_option_permutation(sample_id, len(options))
        shuffled_options = [options[index] for index in permutation]
        row["source_answer_index"] = answer_index
        row["option_permutation"] = permutation
        row["answer"] = permutation.index(answer_index)
        row["question"] = str(row["question"]).rstrip() + "\n" + "\n".join(
            f"({chr(65 + index)}) {option}"
            for index, option in enumerate(shuffled_options)
        )

        episode_id = str(row.get("episode_id", ""))
        if not any(row.get(key) for key in ("video_path", "video", "video_id", "vid")):
            row["video_path"], participant = _storm_video_path(episode_id)
            row.setdefault("participant", participant)
            row.setdefault("level", participant)
        if row.get("question_type") is not None:
            row.setdefault("a_type", row["question_type"])
        return row

    if "question" in row and "answer" in row:
        return row
    if "candidates" not in row or "correct_choice" not in row:
        return row

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    question = str(row["question"]).rstrip()
    row["question"] = question + "\n" + "\n".join(
        f"({letters[index]}) {candidate}"
        for index, candidate in enumerate(row["candidates"])
    )
    row["answer"] = int(row["correct_choice"])
    return row


def get_sample_media_name(sample: dict) -> str:
    for key in ("video_path", "video", "video_id", "vid"):
        value = sample.get(key)
        if value:
            return str(value)
    raise KeyError(f"Sample {sample.get('id')} has no video media key")


def dataset_manifest(path: str | Path) -> dict[str, int | str]:
    """Return reproducibility metadata for the exact ground-truth manifest."""
    path = Path(path)
    records = load_records(path)
    ids = [str(row.get("id", "")) for row in records]
    if any(not sample_id for sample_id in ids):
        raise ValueError(f"Manifest contains a row without an id: {path}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"Manifest contains duplicate ids: {path}")
    episodes = {
        str(row.get("episode_id") or get_sample_media_name(row))
        for row in records
    }
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    id_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return {
        "dataset_sha256": file_digest,
        "dataset_ids_sha256": id_digest,
        "dataset_count": len(records),
        "dataset_episode_count": len(episodes),
    }
