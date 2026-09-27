import pytest

from scripts.eval.prem_checkpoint import (
    EXPECTED_DATA_SEMANTICS,
    EXPECTED_WRITER_MODE,
    validate_prem_checkpoint,
)


def valid_checkpoint():
    return {
        "complete": True,
        "data_semantics": EXPECTED_DATA_SEMANTICS,
        "writer_mode": EXPECTED_WRITER_MODE,
        "qa_objective": "single_ce",
        "prem_modulation": "attention",
        "answer_context": "bounded_visual_buffer_plus_side_memory",
        "visual_buffer_frames": 16,
    }


def test_current_checkpoint_contract_accepts_offline_and_streaming():
    checkpoint = valid_checkpoint()
    validate_prem_checkpoint(checkpoint)


def test_state_only_ablation_accepts_b_zero_checkpoint():
    validate_prem_checkpoint({**valid_checkpoint(), "visual_buffer_frames": 0})


@pytest.mark.parametrize(
    "modulation",
    ["attention", "attention_kv", "attention_k", "attention_v"],
)
def test_checkpoint_contract_accepts_all_ablation_targets(modulation):
    validate_prem_checkpoint({**valid_checkpoint(), "prem_modulation": modulation})


@pytest.mark.parametrize(
    "override",
    [
        {"complete": False},
        {"data_semantics": "unified_per_qa"},
        {"writer_mode": "all_tokens_grouped_temporal_decay"},
        {"qa_objective": "dual_ce"},
        dict(prem_modulation="unknown"),
        dict(prem_modulation=None),
        {"visual_buffer_frames": -1},
    ],
)
def test_streaming_rejects_incompatible_checkpoint(override):
    checkpoint = {**valid_checkpoint(), **override}
    with pytest.raises(ValueError):
        validate_prem_checkpoint(checkpoint)
