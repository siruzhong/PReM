import pytest
import torch

from models.prem_memory import PReMAttentionMemory


def _zero_update_memory() -> PReMAttentionMemory:
    memory = PReMAttentionMemory(
        hidden_size=4,
        num_slots=1,
        key_dim=2,
        val_dim=2,
    )
    with torch.no_grad():
        memory.key_proj.weight.zero_()
        memory.value_proj.weight.zero_()
    return memory


def test_grouped_stream_decay_runs_once_per_temporal_step():
    memory = _zero_update_memory()
    state = torch.ones(1, 1, 2, 2)
    confidence = torch.ones(1, 1)
    grouped = torch.zeros(1, 2, 5, 4)
    salience = torch.ones(1, 2, 5)

    new_state, new_confidence, mean_gate = memory.stream_write_grouped_sequence(
        grouped,
        state=state,
        confidence=confidence,
        salience=salience,
    )

    forget = torch.sigmoid(memory.forget_logit).view(1, 1, 1, 1)
    torch.testing.assert_close(new_state, state * forget.square())
    torch.testing.assert_close(new_confidence, confidence * (0.95**2))
    torch.testing.assert_close(mean_gate, torch.zeros_like(mean_gate))


def test_grouped_stream_decay_is_independent_of_spatial_token_count():
    memory = _zero_update_memory()
    state = torch.ones(1, 1, 2, 2)
    confidence = torch.ones(1, 1)

    results = []
    for spatial_tokens in (1, 7):
        grouped = torch.zeros(1, 1, spatial_tokens, 4)
        salience = torch.ones(1, 1, spatial_tokens)
        results.append(
            memory.stream_write_grouped_sequence(
                grouped,
                state=state,
                confidence=confidence,
                salience=salience,
            )[:2]
        )

    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])


def test_temporal_writer_is_invariant_to_chunk_boundaries():
    torch.manual_seed(7)
    memory = PReMAttentionMemory(hidden_size=8, num_slots=2, key_dim=4, val_dim=4)
    raw = torch.randn(1, 6, 5, 8)
    temporal = raw.mean(dim=2)

    conditioned, salience = memory.condition_stream_chunk(temporal)
    full_state, full_confidence, _ = memory.stream_write_sequence(conditioned, salience)

    chunk_state = None
    chunk_confidence = None
    for chunk in temporal.split(2, dim=1):
        chunk_conditioned, chunk_salience = memory.condition_stream_chunk(chunk)
        chunk_state, chunk_confidence, _ = memory.stream_write_sequence(
            chunk_conditioned,
            chunk_salience,
            state=chunk_state,
            confidence=chunk_confidence,
        )

    torch.testing.assert_close(chunk_state, full_state)
    torch.testing.assert_close(chunk_confidence, full_confidence)


def test_write_trace_preserves_update_and_exposes_slot_terms():
    torch.manual_seed(11)
    memory = PReMAttentionMemory(hidden_size=8, num_slots=3, key_dim=4, val_dim=4)
    state = memory.reset_memory(1, "cpu", torch.float32)
    confidence = memory.reset_confidence(1, "cpu", torch.float32)
    token = torch.randn(1, 8)
    salience = torch.tensor([0.7])

    plain_state, plain_confidence, plain_gate = memory.stream_write_step(
        state, confidence, token, salience
    )
    traced_state, traced_confidence, traced_gate, details = memory.stream_write_step(
        state, confidence, token, salience, return_details=True
    )

    torch.testing.assert_close(traced_state, plain_state)
    torch.testing.assert_close(traced_confidence, plain_confidence)
    torch.testing.assert_close(traced_gate, plain_gate)
    assert set(details) == {
        "route",
        "stability",
        "novelty",
        "salience",
        "anti_overwrite",
        "gate",
        "residual_norm",
        "write_update",
        "write_update_norm",
        "state_norm",
        "confidence",
    }
    assert details["route"].shape == (1, 3)
    assert details["write_update"].shape == (1, 3, 4, 4)
    assert details["write_update_norm"].shape == (1, 3)
    torch.testing.assert_close(details["route"].sum(dim=-1), torch.ones(1))


def test_disable_stability_sets_gate_factor_to_one():
    torch.manual_seed(4)
    kwargs = dict(hidden_size=8, num_slots=2, key_dim=4, val_dim=4)
    enabled = PReMAttentionMemory(**kwargs)
    disabled = PReMAttentionMemory(**kwargs, disable_stability=True)
    disabled.load_state_dict(enabled.state_dict())
    state = enabled.reset_memory(1, "cpu", torch.float32)
    confidence = enabled.reset_confidence(1, "cpu", torch.float32)
    token = torch.randn(1, 8)
    salience = torch.tensor([0.4])

    _, _, _, on_details = enabled.stream_write_step(
        state, confidence, token, salience, return_details=True
    )
    _, _, _, off_details = disabled.stream_write_step(
        state, confidence, token, salience, return_details=True
    )

    torch.testing.assert_close(off_details["stability"], torch.ones_like(off_details["stability"]))
    assert not torch.allclose(on_details["stability"], off_details["stability"])


def test_alpha_multiplier_keeps_learned_alpha_trainable():
    memory = PReMAttentionMemory(
        hidden_size=4,
        num_slots=1,
        key_dim=2,
        val_dim=2,
        alpha=2.0,
    )
    rho = torch.ones(1, 1, 1)
    per_slot = torch.randn(1, 1, 1, 2)

    effective_alpha = memory._alpha_scale(alpha_override=0.5) * memory.num_slots
    torch.testing.assert_close(effective_alpha, torch.tensor(1.0))

    query_steer, output_steer, _, _ = memory.steer_from_sequence_read(
        rho,
        per_slot,
        alpha_override=1.0,
    )
    (query_steer.square().sum() + output_steer.square().sum()).backward()

    assert memory.log_alpha.grad is not None
    assert torch.isfinite(memory.log_alpha.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_grouped_stream_cuda_forward_backward():
    device = torch.device("cuda")
    memory = PReMAttentionMemory(
        hidden_size=8,
        num_slots=2,
        key_dim=4,
        val_dim=4,
    ).to(device)
    grouped = torch.randn(1, 2, 3, 8, device=device)
    conditioned, salience = memory.condition_stream_chunk(grouped)
    state = torch.ones(1, 2, 4, 4, device=device)
    confidence = torch.zeros(1, 2, device=device)

    new_state, new_confidence, mean_gate = memory.stream_write_grouped_sequence(
        conditioned,
        state=state,
        confidence=confidence,
        salience=salience,
    )
    loss = new_state.square().mean() + new_confidence.mean() + mean_gate
    loss.backward()

    assert torch.isfinite(loss)
    assert memory.forget_logit.grad is not None
    assert torch.isfinite(memory.forget_logit.grad).all()
    assert memory.stream_salience_gate[0].weight.grad is not None
    assert torch.isfinite(memory.stream_salience_gate[0].weight.grad).all()


def test_contextual_reader_uses_independent_group_heads():
    memory = PReMAttentionMemory(
        hidden_size=8,
        num_slots=2,
        key_dim=4,
        val_dim=4,
        num_layer_groups=4,
    )
    state = torch.randn(1, 2, 4, 4)
    hidden = torch.randn(1, 5, 8)
    query = torch.randn(1, 8)

    q0, o0, *_ = memory.contextual_read_per_position(state, hidden, query, layer_group=0)
    q3, o3, *_ = memory.contextual_read_per_position(state, hidden, query, layer_group=3)

    assert q0.shape == hidden.shape
    assert o0.shape == hidden.shape
    assert q3.shape == hidden.shape
    assert o3.shape == hidden.shape
    (q0.square().mean() + o3.square().mean()).backward()
    assert memory.group_query_steer_heads[0][0].weight.grad is not None
    assert memory.group_output_steer_heads[3][0].weight.grad is not None


def test_one_group_keeps_static_reader_parameters_only():
    memory = PReMAttentionMemory(hidden_size=8, num_slots=2, key_dim=4, val_dim=4)
    assert len(memory.group_query_steer_heads) == 0
    assert len(memory.group_output_steer_heads) == 0
