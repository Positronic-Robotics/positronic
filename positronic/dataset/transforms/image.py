from collections.abc import Sequence
from enum import Enum
from functools import partial

import cv2
import numpy as np
from PIL import Image as PilImage

from positronic.utils.lazy import LazySequence

from ..signal import Signal
from .signals import Elementwise


class Resample(Enum):
    """Pixel interpolation: Pillow bilinear or OpenCV area averaging."""

    BILINEAR = 'bilinear'
    AREA = 'area'


class ResizeMode(Enum):
    """How image dimensions relate to the requested size."""

    BOUND = 'bound'
    PAD = 'pad'
    STRETCH = 'stretch'


def transform_frame(
    img: np.ndarray, width: int, height: int, *, mode: ResizeMode, resample: Resample = Resample.BILINEAR
) -> np.ndarray:
    """Bound without enlarging, fit and zero-pad, or stretch to the exact requested size."""
    h, w = img.shape[:2]
    if mode is ResizeMode.PAD:
        ratio = max(w / width, h / height)
        tw, th = int(w / ratio), int(h / ratio)
    elif mode is ResizeMode.BOUND:
        scale = min(1.0, width / w, height / h)
        tw, th = int(w * scale), int(h * scale)
    elif mode is ResizeMode.STRETCH:
        tw, th = width, height
    else:
        raise ValueError(f'Unknown resize mode: {mode!r}')
    if (tw, th) == (w, h) and (mode is not ResizeMode.PAD or (w, h) == (width, height)):
        return img
    if resample is Resample.AREA:
        resized = cv2.resize(np.ascontiguousarray(img), (tw, th), interpolation=cv2.INTER_AREA)
    else:
        resized = np.array(PilImage.fromarray(img).resize((tw, th), resample=PilImage.Resampling.BILINEAR))
    if mode is not ResizeMode.PAD:
        return resized
    output = np.zeros((height, width, *resized.shape[2:]), dtype=resized.dtype)
    x, y = (width - tw) // 2, (height - th) // 2
    output[y : y + th, x : x + tw] = resized
    return output


def resize(
    width: int, height: int, signal: Signal[np.ndarray], interpolation: int = cv2.INTER_LINEAR
) -> Signal[np.ndarray]:
    """Return a Signal view with frames resized using OpenCV.

    Args:
        width: Target width (W).
        height: Target height (H).
        signal: Input image Signal with frames shaped (H, W, 3), dtype uint8.
        interpolation: OpenCV interpolation flag (e.g., cv2.INTER_LINEAR).
    """
    interp_flag = int(interpolation)

    def per_frame(img: np.ndarray) -> np.ndarray:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f'Expected frame shape (H, W, 3), got {img.shape}')
        return cv2.resize(img, dsize=(width, height), interpolation=interp_flag)

    def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return LazySequence(x, per_frame)

    return Elementwise(signal, fn)


def resize_with_pad_per_frame(width: int, height: int, method: Resample, img: np.ndarray) -> np.ndarray:
    return transform_frame(img, width, height, mode=ResizeMode.PAD, resample=method)


def resize_with_pad(
    width: int, height: int, signal: Signal[np.ndarray], method: Resample = Resample.BILINEAR
) -> Signal[np.ndarray]:
    """Return a Signal view with frames resized and padded.

    Args:
        width: Target width (W).
        height: Target height (H).
        signal: Input image Signal with frames shaped (H, W, 3), dtype uint8.
        method: Pixel interpolation filter.
    """

    def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return LazySequence(x, partial(resize_with_pad_per_frame, width, height, method))

    return Elementwise(signal, fn)
