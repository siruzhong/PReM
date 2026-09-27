import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

from models.prem_memory import PReMAttentionMemory
from models.same_backbone_baselines import (
    H2ODynamicCache,
    clone_dynamic_cache,
    compress_infinipot_v_cache,
    dynamic_cache_state_bytes,
    infinipot_v_indices,
    insert_prompt_embeddings,
    qwen25_lora_parameter_count,
    token_readout_count,
    token_readout_embeddings,
)
from scripts.eval.eval_prem_online import update_visual_buffer
from scripts.eval.eval_same_backbone_online import (
    h2o_template_parts,
    make_reservoir_state,
)
from scripts.eval import eval_same_backbone_online as same_backbone_eval


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_qwen25_rank20_lora_is_parameter_matched():
    text_config = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=2,
        num_hidden_layers=36,
    )
    config = SimpleNamespace(text_config=text_config)
    assert qwen25_lora_parameter_count(config, rank=20) == 9_216_000


def test_token_readout_uses_all_matched_heads_and_propagates_gradients():
    memory = PReMAttentionMemory(
        hidden_size=8,
        num_slots=1,
        key_dim=4,
        val_dim=4,
        num_layer_groups=2,
    )
    state = torch.randn(1, 1, 4, 4)
    query = torch.randn(1, 3, 8)
    reference = torch.randn(1, 5, 8)
    tokens = token_readout_embeddings(memory, state, query, reference)
    assert token_readout_count(memory) == 6
    assert tokens.shape == (1, 6, 8)
    tokens.square().sum().backward()
    assert memory.log_alpha.grad is not None
    assert memory.query_to_video.weight.grad is not None
    assert memory.query_steer_heads[0].weight.grad is not None
    assert memory.group_output_steer_heads[1][0].weight.grad is not None


def test_insert_prompt_embeddings_shifts_positions_and_masks_labels():
    embeddings = torch.randn(1, 4, 8)
    inserted = torch.randn(1, 2, 8)
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    position_ids = torch.arange(4).view(1, 1, 4).expand(3, 1, -1).clone()
    labels = torch.tensor([[-100, -100, 7, 8]])
    result = insert_prompt_embeddings(
        embeddings,
        attention_mask,
        position_ids,
        inserted,
        insert_at=2,
        labels=labels,
    )
    assert result["inputs_embeds"].shape == (1, 6, 8)
    assert result["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1]]
    assert result["position_ids"][0, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert result["labels"].tolist() == [[-100, -100, -100, -100, 7, 8]]


def test_infinipot_v_combines_dissimilar_history_recent_anchors_and_value_norm():
    keys = torch.tensor(
        [[[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]]
    )
    values = torch.tensor(
        [[[[10.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]]]]
    )
    selected = infinipot_v_indices(
        keys,
        values,
        token_per_frame=1,
        frames_to_keep=3,
        tar_ratio=2.0 / 3.0,
        query_ratio=0.25,
    )
    # TaR forces the key opposite to the recent anchor (index 1) and the
    # recent anchor itself (index 3); VaN fills the final slot with index 0.
    assert selected.tolist() == [[0, 1, 3]]


def test_infinipot_v_compresses_only_visual_kv_and_clones_answer_state():
    config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    cache = DynamicCache(config=config)
    prefix_keys = torch.tensor([[[[100.0, 0.0], [101.0, 0.0]]]])
    visual_keys = torch.tensor(
        [[[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]]
    )
    prefix_values = prefix_keys + 100.0
    visual_values = torch.tensor(
        [[[[10.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 0.0]]]]
    )
    cache.update(
        torch.cat([prefix_keys, visual_keys], dim=2),
        torch.cat([prefix_values, visual_values], dim=2),
        layer_idx=0,
    )
    selections = compress_infinipot_v_cache(
        cache,
        system_tokens=2,
        token_per_frame=1,
        frames_to_keep=3,
        tar_ratio=2.0 / 3.0,
        query_ratio=0.25,
    )
    assert selections[0].tolist() == [[0, 1, 3]]
    assert cache.get_seq_length() == 5
    assert torch.equal(cache.layers[0].keys[:, :, :2], prefix_keys)
    assert cache.layers[0].values[0, 0, 2:, 0].tolist() == [10.0, 1.0, 0.5]

    cloned = clone_dynamic_cache(cache, config)
    assert dynamic_cache_state_bytes(cloned) == dynamic_cache_state_bytes(cache)
    cloned.layers[0].keys.add_(1000)
    assert not torch.equal(cloned.layers[0].keys, cache.layers[0].keys)


def test_infinipot_final_stream_block_is_marked_for_compression():
    source = Path(same_backbone_eval.__file__).read_text()
    assert "at_stream_end = consumed >= usable_frames" in source
    assert "remaining_units or at_stream_end" in source


def test_h2o_keeps_heavy_hitters_and_recent_tokens_and_clones_state():
    config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    cache = H2ODynamicCache(config, heavy_size=2, recent_size=2)
    token_ids = torch.arange(6, dtype=torch.float32).view(1, 1, 6, 1)
    keys = token_ids.expand(-1, -1, -1, 4).clone()
    values = keys + 10
    cache.update(keys, values, layer_idx=0)
    attention = torch.tensor(
        [[[[0.0, 10.0, 1.0, 5.0, 0.0, 0.0]], [[0.0, 9.0, 1.0, 4.0, 0.0, 0.0]]]]
    )
    cache.update_scores_and_evict(0, attention)
    assert cache.get_seq_length() == 4
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [1.0, 3.0, 4.0, 5.0]

    cloned = cache.clone()
    assert cloned.state_bytes() == cache.state_bytes()
    cloned.layers[0].keys.add_(100)
    assert not torch.equal(cloned.layers[0].keys, cache.layers[0].keys)


def test_h2o_pre_evicts_room_for_incoming_tokens_before_attention():
    config = Qwen2_5_VLTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    cache = H2ODynamicCache(config, heavy_size=2, recent_size=2)
    token_ids = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)
    cache.update(token_ids.expand(-1, -1, -1, 4), token_ids, layer_idx=0)
    scores = torch.tensor([[[10.0, 9.0, 1.0, 100.0]]])
    cache.hh_scores[0] = scores
    cache.prepare_for_tokens(1)
    assert cache.get_seq_length() == 3
    # Both heavy-hitter slots remain; one old recent token is displaced to
    # reserve the final recent slot for the incoming token.
    assert cache.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 1.0, 3.0]
    assert cache.hh_scores[0].shape[-1] == 3


def test_h2o_template_keeps_video_state_prefix_question_independent():
    class DummyTokenizer:
        def convert_tokens_to_ids(self, token):
            assert token == "<|video_pad|>"
            return 99

        def __call__(self, text, **kwargs):
            del kwargs
            # The placeholder is fixed in the chat template; only the suffix
            # changes when a question is supplied.
            suffix = [100 + value for value in range(len(text) + 1)]
            ids = torch.tensor([[10, 11, 12, 99, *suffix]], dtype=torch.long)
            return type("Tokenized", (), {"input_ids": ids})()

    class DummyProcessor:
        tokenizer = DummyTokenizer()

        @staticmethod
        def apply_chat_template(messages, **kwargs):
            del kwargs
            text = messages[0]["content"]
            question = text[1].get("text", "") if len(text) > 1 else ""
            return "fixed-prefix<|video_pad|>" + question

    processor = DummyProcessor()
    prefix_none, suffix_none = h2o_template_parts(processor)
    prefix_one, suffix_one = h2o_template_parts(processor, "question one")
    prefix_two, suffix_two = h2o_template_parts(processor, "a much longer question two")
    assert torch.equal(prefix_none, prefix_one)
    assert torch.equal(prefix_one, prefix_two)
    assert not torch.equal(suffix_one, suffix_two)
    assert suffix_none.shape[1] > 0


def test_frozen_base_reuses_the_recurrent_methods_decoder_buffer_policy():
    frames = [Image.new("RGB", (8, 8), color=(index, 0, 0)) for index in range(12)]
    args = SimpleNamespace(buffer_frames=4, max_pixels=64)
    base_state = make_reservoir_state(None, None, frames, "ignored-video-key", args)
    recurrent_state = {}
    update_visual_buffer(
        recurrent_state,
        frames,
        args.buffer_frames,
        args.max_pixels,
    )

    base_indices = [index for index, _ in base_state["visual_buffer"]]
    recurrent_indices = [index for index, _ in recurrent_state["visual_buffer"]]
    assert base_indices == recurrent_indices
    assert base_state["stats"]["visual_buffer_policy"] == "deterministic_reservoir"
    assert base_state["stats"]["visual_buffer_frames"] == args.buffer_frames


def test_same_backbone_entry_registers_five_intro_mechanism_rows_and_fixed_global_batch():
    environment = dict(os.environ, CUDA_DEVICES="0,1,2,3", TRAIN_CUDA_DEVICES="0,1,2,3")
    completed = subprocess.run(
        ["bash", "scripts/run_same_backbone_comparison.sh", "--dry-run", "--rerun-prem"],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert output.count("scripts/train/train_same_backbone.py") == 2
    assert output.count("scripts/eval/run_same_backbone_online.py") == 5
    assert "--global_batch_size 8" in output
    for method in ("base", "lora", "token_readout", "infinipot_v", "prem"):
        assert f"--method {method}" in output
    assert "--method h2o" not in output


def test_same_backbone_entry_can_reuse_completed_prem_without_inference():
    environment = dict(os.environ, CUDA_DEVICES="0,1,2,3")
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_same_backbone_comparison.sh",
            "--dry-run",
            "--eval-only",
            "--methods",
            "prem",
            "--reuse-prem",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert "scripts/eval/import_same_backbone_prem.py" in output
    assert "scripts/eval/run_same_backbone_online.py" not in output


def test_same_backbone_entry_reuses_completed_prem_by_default():
    environment = dict(os.environ, CUDA_DEVICES="0,1,2,3")
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_same_backbone_comparison.sh",
            "--dry-run",
            "--eval-only",
            "--methods",
            "prem",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    assert "scripts/eval/import_same_backbone_prem.py" in output
    assert "scripts/eval/run_same_backbone_online.py" not in output
