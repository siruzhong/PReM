import torch

from qwen_vl_utils import qwen3_video_metadata


def test_qwen3_frame_metadata_preserves_noncontiguous_stream_indices():
    metadata = qwen3_video_metadata(
        {"video": [object()] * 4, "fps": 2.0},
        frame_indices=[0, 3, 7, 11],
        total_num_frames=12,
    )

    assert metadata["fps"] == 2.0
    assert torch.equal(metadata["frames_indices"], torch.tensor([0, 3, 7, 11]))
    assert metadata["total_num_frames"] == 12
    assert metadata["video_backend"] == "frames"


def test_qwen3_frame_metadata_derives_total_frame_count():
    metadata = qwen3_video_metadata(
        {"video": [object()] * 2, "fps": 1.0},
        frame_indices=[2, 8],
    )
    assert metadata["total_num_frames"] == 9
