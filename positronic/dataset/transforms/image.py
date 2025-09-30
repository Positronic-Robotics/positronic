from collections.abc import Sequence
from functools import partial

import cv2
import numpy as np
from PIL import Image as PilImage

from ..signal import Signal
from .signals import Elementwise, LazySequence


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

    return Elementwise(signal, fn, names=['height', 'width', 'channel'])


def _resize_with_pad_pil(image: PilImage.Image, height: int, width: int, method: int) -> PilImage.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.
    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = PilImage.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


def resize_with_pad_per_frame(width: int, height: int, method, img: np.ndarray) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img  # No need to resize if the image is already the correct size.

    return np.array(_resize_with_pad_pil(PilImage.fromarray(img), height, width, method=method))


def resize_with_pad(
    width: int, height: int, signal: Signal[np.ndarray], method=PilImage.Resampling.BILINEAR
) -> Signal[np.ndarray]:
    """Return a Signal view with frames resized-with-pad using PIL.

    Args:
        width: Target width (W).
        height: Target height (H).
        signal: Input image Signal with frames shaped (H, W, 3), dtype uint8.
        method: PIL resampling method (e.g., PilImage.Resampling.BILINEAR).
    """

    def fn(x: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        return LazySequence(x, partial(resize_with_pad_per_frame, width, height, method))

    return Elementwise(signal, fn, names=['height', 'width', 'channel'])
