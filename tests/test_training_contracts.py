import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.prem_memory import PReMAttentionMemory
from models.prem_qwen2_5_vl_model import PReMQwen2_5_VLForConditionalGeneration
from models.prem_qwen2vl_model import PReMQwen2VLForConditionalGeneration
from models.prem_qwen3_vl_model import PReMQwen3VLForConditionalGeneration
from scripts.train.train import (
    checkpoint_resume_cursor,
    qa_modulation_gradient_norm,
    warmup_steps_from_ratio,
)


TRAIN_SOURCE = Path(__file__).resolve().parents[1] / "scripts/train/train.py"
MODEL_SOURCE = Path(__file__).resolve().parents[1] / "models/prem_qwen2vl_model.py"
MODEL_CLASSES = [
    PReMQwen2VLForConditionalGeneration,
    PReMQwen2_5_VLForConditionalGeneration,
    PReMQwen3VLForConditionalGeneration,
]


def test_checkpoint_resume_cursor_accepts_current_key():
    current = {"prem_state_dict": {"weight": "current"}}
    checkpoint = {"prem_state_dict": current, "epoch": 0, "seen_in_epoch": 5, "complete": False}
    assert checkpoint_resume_cursor(checkpoint) == (0, 5)


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ({"epoch": 0, "seen_in_epoch": 7, "complete": False}, (0, 7)),
        ({"epoch": 0, "seen_in_epoch": 0, "complete": True}, (1, 0)),
        ({"epoch": 1, "seen_in_epoch": 0, "complete": True, "resume_cursor": "next_unprocessed"}, (1, 0)),
    ],
)
def test_checkpoint_resume_cursor(checkpoint, expected):
    assert checkpoint_resume_cursor(checkpoint) == expected


def test_warmup_ratio_uses_total_update_count():
    assert warmup_steps_from_ratio(total_updates=2255, epochs=1, ratio=0.03) == 67
    assert warmup_steps_from_ratio(total_updates=2255, epochs=1, ratio=0.0) == 0


def test_model_calls_do_not_use_legacy_keyword_arguments():
    tree = ast.parse(TRAIN_SOURCE.read_text(encoding="utf-8"))
    stale_keywords = [
        keyword.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg and keyword.arg.startswith("phase" + "2_")
    ]
    assert stale_keywords == []


def test_online_reader_receives_prompt_lengths():
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_build_prem_modulation_from_stream_state"
    ]
    assert len(calls) == 2
    for call in calls:
        keyword = next(item for item in call.keywords if item.arg == "prem_prompt_lengths")
        assert isinstance(keyword.value, ast.Name)
        assert keyword.value.id == "prem_prompt_lengths"


def test_training_uses_one_qa_loss_with_bounded_visual_buffer():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    assert "visual_buffer_frames" in source
    assert "state_only_weight" not in source
    assert "loss = qa_loss + args.pred_weight * pred_loss" in source
    assert "writer_only=True" in MODEL_SOURCE.read_text(encoding="utf-8")
    assert "validate_training_checkpoint(init, init_path)" in source
    assert "validate_training_checkpoint(resume, resume_path)" in source




def test_dynamic_stream_reader_dead_code_is_removed():
    source = MODEL_SOURCE.read_text(encoding="utf-8")
    assert "_prem_stream_attention_hook_stack" not in source
    assert "_finalize_prem_stream_attention_stats" not in source
    assert "stream_dynamic_qo_attention" not in source
    assert "prem_decoder_max_frames" not in source


@pytest.mark.parametrize(
    "model_class",
    MODEL_CLASSES,
)
def test_writer_only_state_build_skips_reader(model_class):
    memory = PReMAttentionMemory(hidden_size=8, num_slots=2, key_dim=4, val_dim=4)

    def fail_if_read(*_args, **_kwargs):
        raise AssertionError("writer-only state construction called the answer reader")

    memory.stream_read_per_position = fail_if_read

    class Harness:
        prem_memory = memory
        prem_aux_losses = None
        prem_last_stats = None
        prem_last_stream_state = None

        def build_prem_memory(self, **_kwargs):
            return memory

        @staticmethod
        def _prem_text_mask(input_ids, attention_mask, prompt_lengths):
            del prompt_lengths
            return input_ids.ne(99) & attention_mask.bool()

        _prem_valid_mask = _prem_text_mask

        @staticmethod
        def temporal_pool_video_embeds(video_embeds, video_grid_thw):
            del video_grid_thw
            return video_embeds

        @staticmethod
        def _downsample_tokens(sequence, _max_tokens):
            return sequence

    harness = Harness()
    input_ids = torch.tensor([[99, 99, 1, 2]])
    attention_mask = torch.ones_like(input_ids)
    inputs_embeds = torch.randn(1, 4, 8)
    query_corr, output_corr, fired, context = model_class._build_prem_modulation(
        harness,
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        video_token_mask=input_ids.eq(99),
        video_grid_thw=torch.tensor([[2, 1, 1]]),
        prem_alpha=1.0,
        prem_num_slots=2,
        prem_mem_dim=4,
        prem_layer_groups=1,
        prem_prompt_lengths=torch.tensor([4]),
        prem_max_memory_tokens=8,
        prem_router_gamma=0.05,
        prem_disable_anti_distractor=False,
        prem_disable_novelty=False,
        prem_disable_stability=False,
        prem_disable_evidence_gate_write=False,
        prem_uniform_write_route=False,
        writer_only=True,
    )

    assert query_corr is None
    assert output_corr is None
    assert fired == 1
    assert context is None
    assert harness.prem_last_stream_state.shape[0] == 1
    assert harness.prem_aux_losses["router_balance"].item() == 0.0


def test_hook_based_training_rejects_gradient_checkpointing():
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    assert "Gradient checkpointing is incompatible with temporary q/o forward hooks" in source
    assert "model.gradient_checkpointing_enable" not in source


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize(
    ("modulation", "expected"),
    [
        ("attention", "qo"),
        ("attention_kv", "kv"),
        ("attention_k", "k"),
        ("attention_v", "v"),
    ],
)
def test_modulation_names_map_to_projection_targets(model_class, modulation, expected):
    assert model_class._parse_modulation_mode(modulation) == expected


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("qo", {"q_proj", "o_proj"}),
        ("kv", {"k_proj", "v_proj"}),
        ("k", {"k_proj"}),
        ("v", {"v_proj"}),
    ],
)
def test_modulation_hook_stack_targets_only_requested_projections(model_class, mode, expected):
    attention = SimpleNamespace(
        q_proj=torch.nn.Linear(4, 4, bias=False),
        k_proj=torch.nn.Linear(4, 4, bias=False),
        v_proj=torch.nn.Linear(4, 4, bias=False),
        o_proj=torch.nn.Linear(4, 4, bias=False),
    )
    harness = SimpleNamespace(
        _prem_text_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)]
        )
    )
    query_correction = torch.zeros(1, 1, 4)
    output_correction = torch.zeros(1, 1, 4)

    with model_class._prem_hook_stack(
        harness, query_correction, output_correction, modulation_mode=mode
    ):
        hooked = {
            name
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
            if getattr(attention, name)._forward_hooks
        }
        assert hooked == expected

    assert all(
        not getattr(attention, name)._forward_hooks
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    )


def test_qa_gradient_validator_observes_steer_gradients():
    memory = PReMAttentionMemory(hidden_size=8, num_slots=2, key_dim=4, val_dim=4)
    loss = sum(head.weight.square().sum() for head in memory.query_steer_heads)
    loss = loss + sum(head.weight.square().sum() for head in memory.output_steer_heads)
    loss.backward()
    assert qa_modulation_gradient_norm(memory) > 0


@pytest.mark.parametrize(
    "filename",
    [
        "prem_qwen2vl_model.py",
        "prem_qwen2_5_vl_model.py",
        "prem_qwen3_vl_model.py",
    ],
)
def test_modern_qwen_cached_generation_disables_prem_modulation(filename):
    source = (Path(__file__).resolve().parents[1] / "models" / filename).read_text(
        encoding="utf-8"
    )
    assert "if cache_position is not None and cache_position[0] != 0:" in source
    assert "prem_modulation = None" in source
    assert "prem_stream_state = None" in source
    assert "prem_stream_stats = None" in source


def test_qwen3_wrapper_preserves_native_deepstack_path():
    if PReMQwen3VLForConditionalGeneration is None:
        pytest.skip("transformers does not provide Qwen3-VL")
    source = (
        Path(__file__).resolve().parents[1] / "models/prem_qwen3_vl_model.py"
    ).read_text(encoding="utf-8")
    assert "deepstack_visual_embeds=deepstack_visual_embeds" in source
    assert "self.model.get_rope_index" in source
    assert "self.model.language_model" in source
    assert "second_per_grid_ts" not in source
