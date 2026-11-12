import pytest

from scripts.train.run_rebuttal_campaign import gpu_memory_ready, has_launch_capacity


@pytest.mark.parametrize(
    "candidate,running,expected",
    [
        ("train", [{"kind": "train", "status": "running"}], False),
        ("eval", [{"kind": "train", "status": "running"}], True),
        ("train", [{"kind": "eval", "status": "running"}], True),
        ("eval", [{"kind": "train", "status": "running"},
                  {"kind": "eval", "status": "running"}], False),
        ("train", [{"kind": "eval", "status": "running"},
                   {"kind": "eval", "status": "running"}], False),
        ("train", [{"kind": "train", "status": "verified"}], True),
    ],
)
def test_shared_gpu_queue_reserves_one_training_slot(candidate, running, expected):
    assert has_launch_capacity({"kind": candidate}, running, 2, 1) is expected


def test_gpu_admission_restarts_window_after_one_card_loses_headroom():
    task = {"name": "mean_train", "kind": "train"}
    limits = {"train": 40000, "eval": 20000}
    since = {}
    assert not gpu_memory_ready(task, [45000] * 8, limits, 120, since, 0)
    assert not gpu_memory_ready(task, [45000] * 7 + [39999], limits, 120, since, 119)
    assert not gpu_memory_ready(task, [45000] * 8, limits, 120, since, 120)
    assert not gpu_memory_ready(task, [45000] * 8, limits, 120, since, 239)
    assert gpu_memory_ready(task, [45000] * 8, limits, 120, since, 240)


def test_gpu_admission_uses_task_specific_thresholds_and_timers():
    limits = {"train": 40000, "eval": 20000}
    since = {}
    train = {"name": "ema_train", "kind": "train"}
    evaluation = {"name": "mean_eval", "kind": "eval"}
    assert not gpu_memory_ready(train, [30000] * 8, limits, 120, since, 0)
    assert not gpu_memory_ready(evaluation, [30000] * 8, limits, 120, since, 0)
    assert gpu_memory_ready(evaluation, [30000] * 8, limits, 120, since, 120)
    assert not gpu_memory_ready(train, [45000] * 8, limits, 120, since, 120)


def test_gpu_admission_preserves_immediate_launch_when_no_window_is_set():
    assert gpu_memory_ready({"name": "train", "kind": "train"}, [20000] * 8,
                            {"train": 20000}, 0, {}, 10)


def test_gpu_admission_rejects_incomplete_device_readings():
    with pytest.raises(RuntimeError, match="device readings"):
        gpu_memory_ready({"name": "train", "kind": "train"}, [60000] * 7,
                         {"train": 40000}, 120, {}, 0)
