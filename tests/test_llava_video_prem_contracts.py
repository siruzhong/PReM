from pathlib import Path
from types import SimpleNamespace
import ast

import numpy as np
import torch
import torch.nn as nn

from models.prem_llava_video_model import PReMLlavaVideoForCausalLM
from scripts.eval.eval_prem_llava_video import load_checkpoint
from scripts.llava_video_utils import (
    llava_official_frame_indices,
    load_llava_official_frames,
    load_llava_writer_and_decoder_frames,
    materialized_llava_qwen_loader,
    uniformly_select,
    uniformly_select_with_timestamps,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()


class FakeTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.model = FakeTextModel()
        self.lm_head = nn.Linear(8, 32, bias=False)

    @property
    def device(self):
        return self.lm_head.weight.device

    def encode_images(self, images):
        return images.reshape(images.shape[0], 3, 8)


def test_llava_loader_disables_meta_loading_for_the_full_checkpoint():
    calls = []

    class ModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "model"

    original = ModelClass.from_pretrained
    with materialized_llava_qwen_loader(ModelClass):
        assert ModelClass.from_pretrained("checkpoint", device_map="auto") == "model"
    assert ModelClass.from_pretrained == original
    assert calls == [
        (("checkpoint",), {"low_cpu_mem_usage": False}),
    ]


def test_llava_writer_builds_fixed_shape_state_and_prediction_loss():
    wrapper = PReMLlavaVideoForCausalLM(FakeBackbone())
    wrapper.build_prem_memory(num_slots=4, mem_dim=4)
    images = torch.randn(6, 1, 3, 8)
    query = torch.randn(1, 8)

    state, prediction_loss, stats = wrapper.build_prem_state_from_video(
        images=images,
        query_embeddings=query,
        prem_num_slots=4,
        prem_mem_dim=4,
        prem_pred_weight=0.1,
        prem_pred_tokens=2,
        encode_chunk_frames=2,
    )

    assert state.shape == (1, 4, 4, 4)
    assert prediction_loss.ndim == 0
    assert torch.isfinite(prediction_loss)
    assert stats["stream_writer_mode"] == "temporal_mean_per_step"
    assert stats["stream_memory_tokens_per_step"] == 1
    assert stats["num_stream_updates"] == 6


def test_llava_offline_and_online_writers_match_without_downsampling():
    torch.manual_seed(13)
    wrapper = PReMLlavaVideoForCausalLM(FakeBackbone())
    images = torch.randn(6, 1, 3, 8)
    config = {"prem_num_slots": 4, "prem_mem_dim": 4}

    offline_state, _, _ = wrapper.build_prem_state_from_video(
        images=images,
        encode_chunk_frames=2,
        prem_max_memory_tokens=0,
        **config,
    )
    online = wrapper.stream_reset(batch_size=1, **config)
    for chunk in images.split(2):
        online = wrapper.stream_update_from_images(
            online,
            chunk,
            encode_chunk_frames=2,
        )

    torch.testing.assert_close(online["memory"], offline_state)
    assert online["stats"]["stream_writer_mode"] == "temporal_mean_per_step"
    assert online["stats"]["stream_memory_tokens_per_step"] == 1
    assert online["num_updates"] == 6


def test_llava_kv_hooks_map_hidden_corrections_to_grouped_kv_width():
    wrapper = PReMLlavaVideoForCausalLM(FakeBackbone())
    query_correction = torch.randn(1, 5, 8)
    value_correction = torch.randn(1, 5, 8)
    hidden = torch.randn(1, 5, 8)
    layer = wrapper.backbone.model.layers[0]

    with wrapper._hook_stack(query_correction, value_correction, "kv"):
        keys = layer.self_attn.k_proj(hidden)
        values = layer.self_attn.v_proj(hidden)

    assert keys.shape == (1, 5, 4)
    assert values.shape == (1, 5, 4)

    decode_hidden = torch.randn(1, 1, 8)
    with wrapper._hook_stack(query_correction, value_correction, "kv"):
        decode_keys = layer.self_attn.k_proj(decode_hidden)
    assert decode_keys.shape == (1, 1, 4)


def test_llava_prem_answer_buffer_force_samples_exact_budget():
    frames = torch.arange(3).numpy()
    selected, timestamps = uniformly_select_with_timestamps(
        frames,
        [0.0, 1.0, 2.0],
        5,
        force_sample=True,
    )
    assert selected.tolist() == [0, 0, 1, 2, 2]
    assert timestamps == [0.0, 0.0, 1.0, 2.0, 2.0]


def test_llava_eval_rejects_checkpoint_without_average_pool_contract(tmp_path: Path):
    checkpoint = tmp_path / "old.pt"
    torch.save(
        {
            "model_type": "llava_video",
            "writer_mode": "temporal_mean_per_step",
        },
        checkpoint,
    )
    args = SimpleNamespace(prem_modulation="attention_kv")

    try:
        load_checkpoint(object(), str(checkpoint), args)
    except ValueError as exc:
        assert "official average pooling" in str(exc)
    else:
        raise AssertionError("Old LLaVA checkpoint should have been rejected")


def test_llava_eval_rejects_checkpoint_without_force_sample_contract(tmp_path: Path):
    checkpoint = tmp_path / "old.pt"
    torch.save(
        {
            "model_type": "llava_video",
            "llava_mm_spatial_pool_mode": "average",
            "writer_mode": "temporal_mean_per_step",
        },
        checkpoint,
    )
    args = SimpleNamespace(prem_modulation="attention_kv")

    try:
        load_checkpoint(object(), str(checkpoint), args)
    except ValueError as exc:
        assert "force_sample=true" in str(exc)
    else:
        raise AssertionError("Old LLaVA checkpoint should have been rejected")
