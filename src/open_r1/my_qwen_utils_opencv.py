from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO

import requests
import torch
import torchvision

# OpenCV is used as the default local-video decoder. It is more tolerant of
# unusual MP4 metadata, seek failures, and extreme aspect ratios than the
# torchvision/decord path used previously.
try:
    import cv2
except ImportError:  # Keep image-only usage available even without OpenCV.
    cv2 = None
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Optional


logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 1
FPS_MIN_FRAMES = 10
FPS_MAX_FRAMES = 60

# Set the maximum number of video token inputs.
# Here, 128K represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
logger.info(f"set VIDEO_TOTAL_PIXELS: {VIDEO_TOTAL_PIXELS}")


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


def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
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


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long()
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps
    video = video[idx]
    return video, sample_fps


def is_decord_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
    client
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]
    st = time.time()

    if 's3://' in video_path:
        video_bytes = client.get(video_path)
        if video_bytes is None or len(video_bytes) == 0:
            raise ValueError(f"Can't read byte from {video_path}!")
        byteio = BytesIO(video_bytes)
        vr = decord.VideoReader(byteio, num_threads=1)
    else:
        byteio = None
        vr = decord.VideoReader(video_path)

    # TODO: support start_pts and end_pts
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    vr.seek(0)

    if byteio != None:
        byteio.close()
    return video, sample_fps



def _opencv_target_nframes(
    ele: dict,
    available_frames: int,
    source_fps: float,
) -> int:
    """Choose an even number of frames, matching the Qwen temporal-patch rule.

    Short videos are allowed: one decoded frame is duplicated so the final
    tensor still contains at least two frames.
    """
    if available_frames <= 0:
        raise ValueError(f"available_frames must be positive, got {available_frames}")
    if "fps" in ele and "nframes" in ele:
        raise ValueError("Only accept either `fps` or `nframes`")

    if "nframes" in ele:
        target_count = int(round(float(ele["nframes"])))
    else:
        requested_fps = float(ele.get("fps", FPS))
        min_frames = int(ele.get("min_frames", FPS_MIN_FRAMES))
        max_frames = int(ele.get("max_frames", min(FPS_MAX_FRAMES, available_frames)))
        max_frames = max(1, max_frames)

        duration = available_frames / max(source_fps, 1e-6)
        target_count = int(round(duration * requested_fps))
        target_count = max(min_frames, target_count)
        target_count = min(max_frames, target_count)

    # Do not ask OpenCV for more unique frames than are available. Padding is
    # applied below when a very short video needs two temporal frames.
    target_count = min(target_count, FPS_MAX_FRAMES, available_frames)
    target_count = max(FRAME_FACTOR, target_count)
    target_count = (target_count // FRAME_FACTOR) * FRAME_FACTOR
    return max(FRAME_FACTOR, target_count)


def _opencv_resize_shape(
    height: int,
    width: int,
    max_pixels: int,
    factor: int = IMAGE_FACTOR,
) -> tuple[int, int]:
    """Resize shape for videos without rejecting extreme aspect ratios.

    This mirrors the inference script: shrink to the pixel budget, align both
    dimensions to Qwen's 28-pixel grid, and reduce the longer side further if
    rounding overshoots the budget.
    """
    if height <= 0 or width <= 0:
        raise RuntimeError(f"Invalid frame size: height={height}, width={width}")
    if max_pixels < factor * factor:
        max_pixels = factor * factor

    scale = min(1.0, math.sqrt(max_pixels / float(height * width)))
    new_h = max(factor, int(round(height * scale / factor)) * factor)
    new_w = max(factor, int(round(width * scale / factor)) * factor)

    while new_h * new_w > max_pixels:
        if new_h >= new_w and new_h > factor:
            new_h -= factor
        elif new_w > factor:
            new_w -= factor
        else:
            break

    return new_h, new_w


def _smart_resize_video(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = VIDEO_MIN_PIXELS,
    max_pixels: int = VIDEO_MAX_PIXELS,
) -> tuple[int, int]:
    """Video-specific resize wrapper.

    For normal frames it preserves the project's original smart_resize
    behaviour. For aspect ratios above MAX_RATIO, it deliberately avoids the
    original hard failure and uses the OpenCV-compatible safe resize rule.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid frame size: height={height}, width={width}")

    ratio = max(height, width) / min(height, width)
    if ratio <= MAX_RATIO:
        return smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    logger.warning(
        "Extreme video aspect ratio %.3f for frame %dx%d; "
        "using safe OpenCV-compatible resize instead of rejecting it.",
        ratio,
        height,
        width,
    )
    return _opencv_resize_shape(
        height=height,
        width=width,
        max_pixels=max_pixels,
        factor=factor,
    )


def _read_video_opencv(
    ele: dict,
    client=None,
) -> tuple[torch.Tensor, float]:
    """Decode a local video with OpenCV and sample frames uniformly.

    Unlike the old decoder path, a failed seek for one sampled frame does not
    abort the full video. Successful frames are retained and the final frame is
    repeated only when padding is needed for the requested even frame count.
    """
    if cv2 is None:
        raise ImportError(
            "OpenCV is required for the 'opencv' video backend. "
            "Install it with `pip install opencv-python-headless`."
        )

    path = ele.get("video")
    if not isinstance(path, str):
        raise TypeError(f"OpenCV backend expects a local string video path, got {type(path)!r}")
    if path.startswith(("http://", "https://", "s3://", "data:")):
        raise ValueError(f"OpenCV backend only supports local video files, got: {path}")

    video_path = path[7:] if path.startswith("file://") else path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video with OpenCV: {video_path}")

    started_at = time.time()
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (source_fps > 0):
            source_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        full_duration = total_frames / source_fps
        video_start = float(ele.get("video_start", 0.0) or 0.0)
        video_end = ele.get("video_end")
        video_end = full_duration if video_end is None else float(video_end)

        video_start = max(0.0, min(video_start, full_duration))
        video_end = max(video_start, min(video_end, full_duration))

        start_frame = min(total_frames - 1, max(0, int(video_start * source_fps)))
        end_frame = min(
            total_frames - 1,
            max(start_frame, int(video_end * source_fps) - 1),
        )
        available_frames = end_frame - start_frame + 1
        available_duration = available_frames / source_fps

        target_count = _opencv_target_nframes(
            ele=ele,
            available_frames=available_frames,
            source_fps=source_fps,
        )

        # Keep the decoder memory footprint bounded before stacking frames.
        min_pixels = int(ele.get("min_pixels", VIDEO_MIN_PIXELS))
        total_pixels = int(ele.get("total_pixels", VIDEO_TOTAL_PIXELS))
        allowed_max_pixels = max(
            min(VIDEO_MAX_PIXELS, int(total_pixels / target_count * FRAME_FACTOR)),
            int(min_pixels * 1.05),
        )
        requested_max_pixels = int(ele.get("max_pixels", allowed_max_pixels))
        if requested_max_pixels > allowed_max_pixels:
            logger.warning(
                "The given max_pixels[%s] exceeds the frame budget[%s].",
                requested_max_pixels,
                allowed_max_pixels,
            )
        effective_max_pixels = min(requested_max_pixels, allowed_max_pixels)

        sample_indices = (
            torch.linspace(start_frame, end_frame, steps=target_count)
            .round()
            .long()
            .tolist()
        )

        frames: list[torch.Tensor] = []
        target_h = None
        target_w = None
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                logger.warning(
                    "OpenCV could not decode frame %s from %s; skipping this frame.",
                    frame_idx,
                    video_path,
                )
                continue

            if frame.ndim == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.ndim == 3 and frame.shape[-1] == 4:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if target_h is None:
                target_h, target_w = _opencv_resize_shape(
                    height=rgb.shape[0],
                    width=rgb.shape[1],
                    max_pixels=effective_max_pixels,
                )

            if rgb.shape[:2] != (target_h, target_w):
                interpolation = (
                    cv2.INTER_AREA
                    if rgb.shape[0] > target_h or rgb.shape[1] > target_w
                    else cv2.INTER_CUBIC
                )
                rgb = cv2.resize(
                    rgb,
                    (target_w, target_h),
                    interpolation=interpolation,
                )

            frames.append(
                torch.from_numpy(rgb.copy())
                .permute(2, 0, 1)
                .contiguous()
            )

    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Failed to decode any frame from: {video_path}")

    # Match Qwen's even-frame requirement and preserve the requested length
    # when a few random seeks failed.
    while len(frames) < target_count:
        frames.append(frames[-1].clone())

    video = torch.stack(frames[:target_count], dim=0).float()
    sample_fps = target_count / max(available_duration, 1e-6)
    logger.info(
        "opencv: video_path=%s, total_frames=%s, source_fps=%.3f, "
        "sampled_frames=%s, sampled_fps=%.3f, time=%.3fs",
        video_path,
        total_frames,
        source_fps,
        target_count,
        sample_fps,
        time.time() - started_at,
    )
    return video, sample_fps


VIDEO_READER_BACKENDS = {
    "opencv": _read_video_opencv,
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    """Default to OpenCV; FORCE_QWENVL_VIDEO_READER can explicitly override."""
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER.lower()
        if video_reader_backend not in VIDEO_READER_BACKENDS:
            valid = ", ".join(sorted(VIDEO_READER_BACKENDS))
            raise ValueError(
                f"Unsupported FORCE_QWENVL_VIDEO_READER={video_reader_backend!r}. "
                f"Expected one of: {valid}."
            )
    else:
        video_reader_backend = "opencv"

    print(
        f"qwen-vl-utils using {video_reader_backend} to read video.",
        file=sys.stderr,
    )
    return video_reader_backend


def _read_with_backend(
    backend: str,
    ele: dict,
    client=None,
) -> tuple[torch.Tensor, float]:
    if backend == "torchvision":
        return VIDEO_READER_BACKENDS[backend](ele)
    return VIDEO_READER_BACKENDS[backend](ele, client=client)


def fetch_video(
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
    client=None,
) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        preferred_backend = get_video_reader_backend()

        # OpenCV is deliberately first by default, but retain decord and
        # torchvision as emergency fallbacks for unusual codecs/environments.
        backend_order = [preferred_backend]
        for candidate in ("opencv", "decord", "torchvision"):
            if candidate not in backend_order:
                backend_order.append(candidate)

        errors: list[str] = []
        video = None
        sample_fps = None
        used_backend = None
        for backend in backend_order:
            if backend == "decord" and not is_decord_available():
                errors.append("decord: package is not installed")
                continue
            try:
                video, sample_fps = _read_with_backend(backend, ele, client=client)
                used_backend = backend
                break
            except Exception as exc:
                errors.append(f"{backend}: {type(exc).__name__}: {exc}")
                logger.warning(
                    "Video backend %s failed for %s: %s",
                    backend,
                    ele["video"],
                    exc,
                )

        if video is None or sample_fps is None or used_backend is None:
            detail = " | ".join(errors)
            raise RuntimeError(
                f"All video backends failed for {ele['video']}. Details: {detail}"
            )

        nframes, _, height, width = video.shape
        min_pixels = int(ele.get("min_pixels", VIDEO_MIN_PIXELS))
        total_pixels = int(ele.get("total_pixels", VIDEO_TOTAL_PIXELS))
        max_pixels = max(
            min(VIDEO_MAX_PIXELS, int(total_pixels / nframes * FRAME_FACTOR)),
            int(min_pixels * 1.05),
        )
        max_pixels_supposed = int(ele.get("max_pixels", max_pixels))
        if max_pixels_supposed > max_pixels:
            logger.warning(
                "The given max_pixels[%s] exceeds limit[%s].",
                max_pixels_supposed,
                max_pixels,
            )
        max_pixels = min(max_pixels_supposed, max_pixels)

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = _smart_resize_video(
                int(ele["resized_height"]),
                int(ele["resized_width"]),
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        else:
            resized_height, resized_width = _smart_resize_video(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if (height, width) != (resized_height, resized_width):
            video = transforms.functional.resize(
                video,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
        else:
            video = video.float()

        if return_video_sample_fps:
            return video, sample_fps
        return video

    assert isinstance(ele["video"], (list, tuple))
    process_info = ele.copy()
    process_info.pop("type", None)
    process_info.pop("video", None)
    images = [
        fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
        for video_element in ele["video"]
    ]
    nframes = ceil_by_factor(len(images), FRAME_FACTOR)
    if len(images) < nframes:
        images.extend([images[-1]] * (nframes - len(images)))
    if return_video_sample_fps:
        return images, process_info.pop("fps", 2.0)
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
    return_video_kwargs: bool = False,
    client=None
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, Optional[dict]]:

    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video(vision_info, return_video_sample_fps=True, client=client)
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs