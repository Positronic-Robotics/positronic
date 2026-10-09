import collections.abc as cabc
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import numpy as np
from PIL import Image as PilImage
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, JpegEncoding, encode_images, encode_jpeg
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic.dataset.transforms.episode import EpisodeTransform

from .base import Codec


class EncodeImages(Codec):
    """JPEG-encode uint8 RGB observations recursively, or restrict encoding to explicit paths.

    Automatic selection matches ``(..., H, W, 3)`` arrays with positive height and width. ``paths=[]``
    selects nothing. Decoded native results pass through unchanged.
    """

    WIRE_NAME = 'encode_images'

    def __init__(self, paths: list[list[str | int]] | None = None, quality: int = DEFAULT_JPEG_QUALITY):
        if type(quality) is not int or not 0 <= quality <= 100:
            raise ValueError('JPEG quality must be an integer between 0 and 100')
        self._images = None if paths is None else tuple(JpegEncoding(tuple(path), quality) for path in paths)
        self._quality = quality

    def _encode_value(self, value: Any) -> Any:
        if (
            isinstance(value, np.ndarray)
            and value.dtype == np.uint8
            and value.ndim >= 3
            and value.shape[-1] == 3
            and min(value.shape[-3:-1]) > 0
        ):
            return encode_jpeg(value, self._quality)
        if isinstance(value, cabc.Mapping):
            return {key: self._encode_value(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self._encode_value(item) for item in value)
        return value

    def encode(self, data: dict) -> dict:
        return self._encode_value(data) if self._images is None else encode_images(data, self._images)

    def decode(self, data: Any) -> Any:
        return data

    def to_spec(self) -> dict[str, Any]:
        args: dict[str, Any] = {'quality': self._quality}
        if self._images is not None:
            args['paths'] = [list(image.path) for image in self._images]
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}


def _scaled(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, width / w, height / h)
    tw, th = int(w * scale), int(h * scale)
    if (tw, th) == (w, h):
        return image
    return np.array(PilImage.fromarray(image).resize((tw, th), resample=PilImage.Resampling.BILINEAR))


class RestrictImageSize(Codec):
    """Bound image dimensions while preserving aspect ratio.

    Images larger than ``width`` x ``height`` are scaled down to fit it, keeping aspect ratio; anything
    already within it passes through untouched.
    """

    WIRE_NAME = 'restrict_image_size'

    # Below this a stack scales quicker in one thread than a pool costs to raise.
    _PARALLEL_FROM = 4
    _MAX_WORKERS = 8

    def __init__(self, width: int = 640, height: int = 640):
        self._width = width
        self._height = height

    def encode(self, data):
        return {key: self._restrict(value) for key, value in data.items()}

    def _restrict(self, value: Any) -> Any:
        # Codecs nest images inside dicts and lists (e.g. GR00T), so recurse to reach every image array.
        if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
            # A TemporalStack emits a (T, H, W, 3) stack, so bound each frame rather than the stack's first axis.
            if value.ndim == 4:
                return np.stack(self._scaled_frames(value))
            return _scaled(value, self._width, self._height)
        if isinstance(value, cabc.Mapping):
            return {k: self._restrict(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self._restrict(v) for v in value)
        return value

    @staticmethod
    def _usable_cpus() -> int:
        """CPUs this process may run on — its affinity mask where the platform publishes one, else the
        host's count. FOOTGUN: neither reads a cgroup CPU quota, so a container limited by `--cpus` and
        not by a mask still reads the host's cores.
        """
        if hasattr(os, 'sched_getaffinity'):
            return len(os.sched_getaffinity(0))
        return os.cpu_count() or 1

    def _workers(self, frames: int) -> int:
        """Threads to scale ``frames`` on. One means the serial path: a pool wins nothing on a single
        usable CPU, and below ``_PARALLEL_FROM`` it costs more to raise than the frames take."""
        if frames < self._PARALLEL_FROM:
            return 1
        return max(1, min(frames, self._MAX_WORKERS, self._usable_cpus()))

    def _scaled_frames(self, stack: np.ndarray) -> list[np.ndarray]:
        """Every frame of one stack, scaled. Pillow drops the GIL for a resize and the frames are
        independent, so more than one may run at a time."""
        scale = partial(_scaled, width=self._width, height=self._height)
        workers = self._workers(len(stack))
        if workers == 1:
            return [scale(frame) for frame in stack]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(scale, stack))

    def decode(self, data):
        return data

    @property
    def training_encoder(self) -> EpisodeTransform:
        raise NotImplementedError(
            'RestrictImageSize bounds what goes on the wire and is not part of the model contract: '
            'training derives its columns from full-resolution episodes.'
        )

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'width': self._width, 'height': self._height}}
