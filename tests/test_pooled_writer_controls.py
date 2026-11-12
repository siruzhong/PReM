import pytest
import torch

from models.prem_memory import PReMAttentionMemory
from models.prem_writer_controls import PooledWriterMemory


def make_memory(variant):
    torch.manual_seed(13)
    return PooledWriterMemory(hidden_size=8, num_slots=1, key_dim=4, val_dim=4,
                              num_layer_groups=2, writer_variant=variant, ema_decay=0.8)


@pytest.mark.parametrize("variant", ["temporal_mean", "ema"])
def test_pooling_matches_reference_with_mask_and_stream_boundaries(variant):
    mem = make_memory(variant)
    tokens = torch.randn(2, 7, 8)
    salience = torch.rand(2, 7)
    blocked = torch.tensor([[True, False, False, True, False, False, False],
                            [False, True, False, False, False, True, False]])
    state, count, _ = mem.stream_write_sequence(tokens, salience, write_block_mask=blocked)
    ref = []
    for batch in range(2):
        observed = tokens[batch, ~blocked[batch]]
        pooled = observed[0]
        if variant == "temporal_mean":
            pooled = observed.mean(0)
        else:
            for token in observed[1:]:
                pooled = 0.8 * pooled + 0.2 * token
        ref.append(pooled)
    torch.testing.assert_close(state.flatten(1)[:, :8], torch.stack(ref))
    torch.testing.assert_close(count[:, 0], (~blocked).sum(1).float())
    first, first_count, _ = mem.stream_write_sequence(tokens[:, :3], salience[:, :3],
                                                     write_block_mask=blocked[:, :3])
    streamed, streamed_count, _ = mem.stream_write_sequence(
        tokens[:, 3:], salience[:, 3:], state=first, confidence=first_count,
        write_block_mask=blocked[:, 3:],
    )
    torch.testing.assert_close(streamed, state)
    torch.testing.assert_close(streamed_count, count)


def test_pooled_read_uses_original_question_read_and_has_writer_gradients():
    mem = make_memory("temporal_mean")
    tokens = torch.randn(1, 5, 8, requires_grad=True)
    state, _, _ = mem.stream_write_sequence(tokens, torch.ones(1, 5))
    query = torch.randn(1, 3, 8)
    actual = mem.read_sequence(state, query)
    expected = PReMAttentionMemory.read_sequence(mem, mem.pooled_binding(state), query)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)
    actual[0].square().sum().backward()
    assert tokens.grad is not None and tokens.grad.norm() > 0
    assert mem.key_proj.weight.grad is not None
    assert mem.value_proj.weight.grad is not None
