from unittest.mock import patch

import pytest
import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLTextConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from models.prem_qwen2_5_vl_model import PReMQwen2_5_VLForConditionalGeneration
from models.prem_qwen3_vl_model import PReMQwen3VLForConditionalGeneration


def tiny_qwen25():
    text_config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    vision_config = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        window_size=4,
        out_hidden_size=16,
        fullatt_block_indexes=[0],
    )
    config = Qwen2_5_VLConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    return PReMQwen2_5_VLForConditionalGeneration(config).eval()


def tiny_qwen3():
    text_config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        rope_scaling={
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 2],
            "rope_type": "default",
        },
    )
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    config = Qwen3VLConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
    )
    return PReMQwen3VLForConditionalGeneration(config).eval()


@pytest.mark.parametrize("model_factory", [tiny_qwen25, tiny_qwen3])
def test_logits_to_keep_matches_native_integer_and_tensor_semantics(model_factory):
    model = model_factory()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)

    assert model._supports_logits_to_keep()
    with torch.no_grad():
        last = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        selected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=torch.tensor([0, 2, 4]),
        )

    assert last.logits.shape == (1, 1, 32)
    assert selected.logits.shape == (1, 3, 32)


@pytest.mark.parametrize("model_factory", [tiny_qwen25, tiny_qwen3])
def test_generate_only_projects_last_prefill_token(model_factory):
    model = model_factory()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    projected_lengths = []
    original_forward = model.lm_head.forward

    def record_projected_length(hidden_states):
        projected_lengths.append(hidden_states.shape[1])
        return original_forward(hidden_states)

    with patch.object(model.lm_head, "forward", side_effect=record_projected_length):
        model.generate(input_ids, max_new_tokens=1, do_sample=False)

    assert projected_lengths == [1]
