from __future__ import annotations

import json

from scripts.eval.data_io import get_sample_media_name, load_records, normalize_mcq_sample


def test_load_records_accepts_jsonl(tmp_path):
    path = tmp_path / "questions.jsonl"
    rows = [{"id": "q1"}, {"id": "q2"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert load_records(path) == rows


def test_normalize_storm_sample_is_stable_and_resolves_video():
    sample = {
        "id": "P02-example_q01",
        "episode_id": "P02-example",
        "question": "What happened?",
        "options": ["first", "second", "third", "fourth"],
        "answer_index": 0,
        "question_type": "object_tracking",
    }

    first = normalize_mcq_sample(sample)
    second = normalize_mcq_sample(sample)

    assert first == second
    assert sorted(first["option_permutation"]) == [0, 1, 2, 3]
    assert first["answer"] == first["option_permutation"].index(0)
    assert first["level"] == "P02"
    assert first["participant"] == "P02"
    assert first["a_type"] == "object_tracking"
    assert get_sample_media_name(first) == "P02/P02-example.mp4"
    assert "(A)" in first["question"]


def test_normalize_storm_sim_sample_resolves_floorplan_video():
    sample = {
        "id": "STORM_FloorPlan26_seed3_6bf67c1b845a_q01",
        "episode_id": "STORM_FloorPlan26_seed3_6bf67c1b845a",
        "question": "Which object returns?",
        "options": ["bread", "pan", "bowl", "cup"],
        "answer_index": 2,
        "question_type": "object_tracking",
    }

    row = normalize_mcq_sample(sample)

    assert row["participant"] == "FloorPlan26"
    assert row["level"] == "FloorPlan26"
    assert get_sample_media_name(row) == (
        "FloorPlan26/STORM_FloorPlan26_seed3_6bf67c1b845a.mp4"
    )


def test_normalize_storm_bike_sample_resolves_flat_video():
    sample = {
        "id": "cmu_bike01_window001_revisit_q01",
        "episode_id": "cmu_bike01_window001_revisit",
        "question": "What color was the tool?",
        "options": ["Silver", "Blue", "Red", "Yellow"],
        "answer_index": 3,
        "question_type": "factual_retrieval",
    }

    row = normalize_mcq_sample(sample)

    assert row["participant"] == "cmu"
    assert row["level"] == "cmu"
    assert get_sample_media_name(row) == "cmu_bike01_window001_revisit.mp4"

    georgia = normalize_mcq_sample(
        {
            **sample,
            "id": "georgiatech_bike_01_window001_revisit_q01",
            "episode_id": "georgiatech_bike_01_window001_revisit",
        }
    )
    assert georgia["participant"] == "georgiatech"
    assert get_sample_media_name(georgia) == "georgiatech_bike_01_window001_revisit.mp4"


def test_normalize_storm_healthy_music_sports_resolves_flat_video():
    healthy = normalize_mcq_sample(
        {
            "id": "georgiatech_covid_03_window001_revisit_q01",
            "episode_id": "georgiatech_covid_03_window001_revisit",
            "question": "Where is the box?",
            "options": ["Open", "Closed", "Tilted", "Missing"],
            "answer_index": 0,
            "question_type": "current_state",
        }
    )
    music = normalize_mcq_sample(
        {
            "id": "iiith_guitar_002_window001_revisit_q01",
            "episode_id": "iiith_guitar_002_window001_revisit",
            "question": "Where is the guitar?",
            "options": ["Lap", "Stand", "Floor", "Unknown"],
            "answer_index": 1,
            "question_type": "current_state",
        }
    )
    sports = normalize_mcq_sample(
        {
            "id": "cmu_soccer03_window001_revisit_q01",
            "episode_id": "cmu_soccer03_window001_revisit",
            "question": "Where is the ball?",
            "options": ["Goal", "Sideline", "Midfield", "Unknown"],
            "answer_index": 2,
            "question_type": "current_state",
        }
    )

    assert get_sample_media_name(healthy) == "georgiatech_covid_03_window001_revisit.mp4"
    assert get_sample_media_name(music) == "iiith_guitar_002_window001_revisit.mp4"
    assert get_sample_media_name(sports) == "cmu_soccer03_window001_revisit.mp4"
    assert healthy["participant"] == "georgiatech"
    assert music["participant"] == "iiith"
    assert sports["participant"] == "cmu"
