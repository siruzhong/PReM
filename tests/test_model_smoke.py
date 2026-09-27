import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "ckpt/Qwen2-VL-7B-Instruct"
ONLINE_CKPT = REPO_ROOT / "outputs/prem/prem_qwen2vl.pt"


@pytest.mark.model_smoke
@pytest.mark.skipif(
    os.environ.get("PREM_RUN_MODEL_SMOKE") != "1",
    reason="set PREM_RUN_MODEL_SMOKE=1 to load the 7B checkpoint",
)
def test_online_checkpoint_model_forward():
    import torch

    from models.prem_qwen2vl_model import PReMQwen2VLForConditionalGeneration

    assert MODEL_PATH.is_dir()
    assert ONLINE_CKPT.is_file()

    model = PReMQwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    checkpoint = torch.load(ONLINE_CKPT, map_location="cpu", weights_only=False)
    num_slots = int(checkpoint.get("num_slots", 4))
    mem_dim = int(checkpoint.get("mem_dim", 128))
    alpha = float(checkpoint.get("alpha", 1.0))
    memory = model.build_prem_memory(
        num_slots=num_slots,
        mem_dim=mem_dim,
        alpha=alpha,
        disable_anti_distractor=bool(checkpoint.get("disable_anti_distractor", False)),
        disable_novelty=bool(checkpoint.get("disable_novelty", False)),
        disable_stability=bool(checkpoint.get("disable_stability", False)),
        disable_evidence_gate_write=bool(checkpoint.get("disable_evidence_gate_write", False)),
        uniform_write_route=bool(checkpoint.get("uniform_write_route", False)),
    )
    state_dict = checkpoint.get("prem_state_dict", checkpoint)
    missing, unexpected = memory.load_state_dict(state_dict, strict=False)
    assert list(missing) == []
    assert list(unexpected) == []

    model.eval()
    hidden_size = int(model.config.hidden_size)
    frame_embeds = torch.randn(2, 2, hidden_size, device=model.device, dtype=torch.bfloat16)
    stream_state = model.stream_reset(
        prem_num_slots=num_slots,
        prem_mem_dim=mem_dim,
        prem_alpha=1.0,
    )
    stream_state = model.stream_update(stream_state, frame_embeds)
    input_ids = torch.tensor([[1, 2, 3, 4]], device=model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            prem_modulation="attention",
            prem_alpha=1.0,
            prem_num_slots=num_slots,
            prem_mem_dim=mem_dim,
            prem_stream_state=stream_state["memory"],
            prem_stream_stats=stream_state["stats"],
        )
