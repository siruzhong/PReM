"""Shared official-evaluation semantics for LLaVA-Video."""

from __future__ import annotations


LLAVA_VIDEO_ADAPTER_VERSION = "official_eval_v2_average_pool_time_instruction"
LLAVA_VIDEO_POOL_MODE = "average"
LLAVA_VIDEO_DIRECTORY_FPS = 1.0
LLAVA_VIDEO_TIME_INSTRUCTION_VERSION = "official_add_time_instruction_v1"


def format_time_instruction(
    duration: float,
    timestamps: list[float],
    num_frames: int,
) -> str:
    if num_frames != len(timestamps):
        raise ValueError(
            f"Frame/timestamp length mismatch: {num_frames} != {len(timestamps)}"
        )
    frame_times = ",".join(f"{timestamp:.2f}s" for timestamp in timestamps)
    return (
        f"The video lasts for {duration:.2f} seconds, and {num_frames} frames "
        "are uniformly sampled from it. These frames are located at "
        f"{frame_times}. Please answer the following questions related to this video."
    )
