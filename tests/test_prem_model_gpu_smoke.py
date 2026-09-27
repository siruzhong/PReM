import importlib.util
import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from models.prem_qwen2vl_model import PReMQwen2VLForConditionalGeneration


RUN_MODEL_SMOKE = os.environ.get("PREM_RUN_MODEL_SMOKE") == "1"
MODEL_PATH = Path(os.environ.get("PREM_MODEL_PATH", "ckpt/Qwen2-VL-7B-Instruct"))
CHECKPOINT_PATH = Path(
    os.environ.get("PREM_ONLINE_CKPT", "outputs/prem_v3/qwen2/prem.pt")
)


@pytest.mark.model_smoke
@pytest.mark.skipif(not RUN_MODEL_SMOKE, reason="set PREM_RUN_MODEL_SMOKE=1")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_online_checkpoint_bounded_visual_memory_generation_smoke():
    assert MODEL_PATH.exists()
    assert CHECKPOINT_PATH.exists()

    attention = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "eager"
    model = PReMQwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    memory = model.build_prem_memory(
        num_slots=int(checkpoint.get("num_slots", 4)),
        mem_dim=int(checkpoint.get("mem_dim", 128)),
        alpha=float(checkpoint.get("alpha", 1.0)),
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

    config = {
        "prem_num_slots": memory.num_slots,
        "prem_mem_dim": memory.key_dim,
        "prem_alpha": 1.0,
        "prem_disable_anti_distractor": memory.disable_anti_distractor,
        "prem_disable_novelty": memory.disable_novelty,
        "prem_disable_stability": memory.disable_stability,
        "prem_disable_evidence_gate_write": memory.disable_evidence_gate_write,
        "prem_uniform_write_route": memory.uniform_write_route,
    }
    stream_state = model.stream_reset(**config)
    frame_embeds = torch.randn(2, 2, model.config.hidden_size, device=model.device)
    with torch.no_grad():
        stream_state = model.stream_update(stream_state, frame_embeds, update_block_size=2)

        frames = [Image.new("RGB", (224, 224), color=(index * 32, 0, 0)) for index in range(2)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "max_frames": 2, "max_pixels": 200704},
                    {"type": "text", "text": "Which option is correct? A. first B. second"},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], videos=video_inputs, padding=True, return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        generated = model.stream_answer(
            stream_state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=inputs.pixel_values_videos.to(model.device),
            video_grid_thw=inputs.video_grid_thw.to(model.device),
            prem_prompt_lengths=torch.tensor([input_ids.shape[1]], device=model.device),
            max_new_tokens=1,
            do_sample=False,
        )

    assert generated.shape == (1, input_ids.shape[1] + 1)
    assert torch.isfinite(stream_state["memory"]).all()
    assert stream_state["stats"]["forget_unit"] == "temporal_step"
    assert stream_state["stats"]["stream_writer_mode"] == "temporal_mean_per_step"
    assert stream_state["stats"]["stream_memory_tokens_per_step"] == 1
