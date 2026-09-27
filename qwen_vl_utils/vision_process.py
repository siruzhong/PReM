from __future__ import annotations

import base64
import math
import warnings
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

try:
    from decord import VideoReader, cpu as decord_cpu
except ImportError:  # pragma: no cover - keep torchvision fallback for minimal envs
    VideoReader = None
    decord_cpu = None


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def _sample_video_frame_indices(ele: dict, total_frames: int, video_fps: float, size_factor: int) -> torch.Tensor:
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    start_frame = max(0, int(math.floor(ele.get("video_start", 0.0) * video_fps)))
    if ele.get("video_end", None) is None:
        end_frame = total_frames - 1
    else:
        end_frame = min(total_frames - 1, int(math.ceil(ele["video_end"] * video_fps)))
    if end_frame < start_frame:
        raise ValueError(f"Invalid video range: start_frame={start_frame}, end_frame={end_frame}")
    available_frames = end_frame - start_frame + 1

    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], size_factor)
    else:
        fps = ele.get("fps", FPS)
        nframes = available_frames / video_fps * fps
        nframes = round_by_factor(nframes, size_factor)
        if "min_frames" in ele:
            min_frames = ele["min_frames"]
            if nframes < min_frames:
                nframes = ceil_by_factor(min_frames, size_factor)
        else:
            min_frames = FPS_MIN_FRAMES
            if nframes < min_frames:
                warnings.warn(f"nframes is less than DEFAULT_MIN_FRAMES {min_frames}, set to {nframes}.")
                nframes = ceil_by_factor(min_frames, size_factor)
        if "max_frames" in ele:
            max_frames = ele["max_frames"]
            if nframes > max_frames:
                nframes = floor_by_factor(max_frames, size_factor)
        else:
            max_frames = FPS_MAX_FRAMES
            if nframes > max_frames:
                warnings.warn(f"nframes is greater than DEFAULT_MAX_FRAMES {max_frames}, set to {nframes}.")
                nframes = floor_by_factor(max_frames, size_factor)

    max_available = floor_by_factor(available_frames, size_factor)
    if not (size_factor <= nframes and nframes <= max_available):
        raise ValueError(f"nframes should in interval [{size_factor}, {max_available}], but got {nframes}.")
    return torch.linspace(start_frame, end_frame, nframes).round().long()


def qwen3_video_metadata(
    ele: dict,
    *,
    frame_indices: torch.Tensor | list[int] | None = None,
    total_num_frames: int | None = None,
) -> dict:
    """Return timestamp metadata expected by the Qwen3-VL video processor."""
    source = ele["video"]
    if frame_indices is not None:
        indices = torch.as_tensor(frame_indices, dtype=torch.long)
        if indices.numel() == 0:
            raise ValueError("Qwen3-VL video metadata requires at least one frame index")
        fps = float(ele.get("fps", FPS))
        return {
            "fps": fps,
            "frames_indices": indices,
            "total_num_frames": int(total_num_frames or (int(indices.max().item()) + 1)),
            "video_backend": "frames",
        }

    if not isinstance(source, str):
        count = len(source)
        return qwen3_video_metadata(
            ele,
            frame_indices=torch.arange(count),
            total_num_frames=count,
        )
    if VideoReader is None:
        raise RuntimeError("Qwen3-VL timestamp metadata requires decord for path-based videos")
    if source.startswith("file://"):
        source = source[7:]
    reader = VideoReader(source, ctx=decord_cpu(0))
    raw_fps = float(reader.get_avg_fps())
    return {
        "fps": raw_fps,
        "frames_indices": _sample_video_frame_indices(
            ele,
            len(reader),
            raw_fps,
            FRAME_FACTOR,
        ),
        "total_num_frames": len(reader),
        "video_backend": "decord",
    }


def _fetch_video_with_decord(ele: dict, size_factor: int) -> torch.Tensor:
    video = ele["video"]
    if video.startswith("file://"):
        video = video[7:]

    vr = VideoReader(video, ctx=decord_cpu(0))
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    idx = _sample_video_frame_indices(ele, total_frames, video_fps, size_factor)
    video_np = vr.get_batch(idx.tolist()).asnumpy()
    video = torch.from_numpy(video_np).permute(0, 3, 1, 2).contiguous()

    nframes = video.size(0)
    height, width = video.shape[2:]
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * size_factor), min_pixels * 1.05)
    max_pixels = ele.get("max_pixels", max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


def fetch_video(ele: dict, size_factor: int = FRAME_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        # TODO: support http url

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]

        if VideoReader is not None:
            try:
                return _fetch_video_with_decord(ele, size_factor)
            except Exception:
                pass  # fall back to torchvision for VP9/broken videos

        video, audio, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )

        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], size_factor)
        else:
            fps = ele.get("fps", FPS)
            nframes = video.size(0) / info["video_fps"] * fps
            nframes = round_by_factor(nframes, size_factor)
            if "min_frames" in ele:
                min_frames = ele["min_frames"]
                if nframes < min_frames:
                    nframes = ceil_by_factor(min_frames, size_factor)
            else:
                min_frames = FPS_MIN_FRAMES
                if nframes < min_frames:
                    warnings.warn(f"nframes is less than DEFAULT_MIN_FRAMES {min_frames}, set to {nframes}.")
                    nframes = ceil_by_factor(min_frames, size_factor)
            if "max_frames" in ele:
                max_frames = ele["max_frames"]
                if nframes > max_frames:
                    nframes = floor_by_factor(max_frames, size_factor)
            else:
                max_frames = FPS_MAX_FRAMES
                if nframes > max_frames:
                    warnings.warn(f"nframes is greater than DEFAULT_MAX_FRAMES {max_frames}, set to {nframes}.")
                    nframes = floor_by_factor(max_frames, size_factor)

        if not (size_factor <= nframes and nframes <= video.size(0)):
            raise ValueError(f"nframes should in interval [{size_factor}, {video.size(0)}], but got {nframes}.")

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        height, width = video.shape[2:]
        video = video[idx]

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * size_factor), min_pixels * 1.05)
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=size_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        nframes = len(ele["video"])
        if "max_frames" in ele:
            max_frames = ele["max_frames"]
            if nframes > max_frames:
                nframes = floor_by_factor(max_frames, size_factor)
        else:
            max_frames = FPS_MAX_FRAMES
            if nframes > max_frames:
                warnings.warn(f"in else, nframes is greater than DEFAULT_MAX_FRAMES {max_frames}, set to {nframes}.")
                nframes = floor_by_factor(max_frames, size_factor)
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        # print(f'In fetch_video, total_pixels={total_pixels}')
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * size_factor), min_pixels * 1.05)
        max_pixels = ele.get("max_pixels", max_pixels)
        process_info.update({
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        })
        # print(f'In fetch_video, nframes={nframes}, max_frames={max_frames}, max_pixels={max_pixels}, video len={len(ele["video"])}')
        idx = torch.linspace(0, len(ele["video"]) - 1, nframes).round().long()
        images = [fetch_image({"image": video_element, **process_info}) for i, video_element in enumerate(ele["video"]) if i in idx]
        nframes = ceil_by_factor(len(images), size_factor)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs
