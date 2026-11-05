import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import combinations
from typing import Any, Self

import numpy as np
from positronic_model_server.serialization import DEFAULT_JPEG_QUALITY, JpegEncoding, encode_images, encode_jpeg
from positronic_model_server.spec import ARGS, NAME, VERSION
from positronic_model_server.tree import map_leaves, update_at

from positronic.dataset.transforms import image
from positronic.dataset.transforms.image import Resample, ResizeMode

from .base import Codec
from .observation import SelectObservationFields


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
        return value

    def encode(self, data: dict) -> dict:
        return map_leaves(data, self._encode_value) if self._images is None else encode_images(data, self._images)

    def decode(self, data: Any) -> Any:
        return data

    def to_spec(self) -> dict[str, Any]:
        args: dict[str, Any] = {'quality': self._quality}
        if self._images is not None:
            args['paths'] = [list(image.path) for image in self._images]
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}


class TransformImages(Codec):
    """Transform RGB images in nested observations, including temporal frame stacks.

    ``BOUND`` only shrinks to fit. ``PAD`` fits and pads to the exact size, preserving aspect ratio.
    ``STRETCH`` resizes to the exact size, allowing distortion. Width, height and mode are required.
    ``paths=None`` finds arrays automatically; explicit paths require images and leave other values untouched.
    Field names and decoded results pass through unchanged.
    """

    WIRE_NAME = 'transform_images'

    _MIN_PARALLEL_PIXELS = 256 * 256
    _MAX_WORKERS = 8

    def __init__(
        self,
        width: int,
        height: int,
        *,
        mode: ResizeMode,
        resample: Resample = Resample.BILINEAR,
        paths: list[list[str | int]] | None = None,
    ):
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError('Image dimensions must be positive integers')
        if paths is not None and any(type(part) not in (str, int) for path in paths for part in path):
            raise ValueError('Image paths contain only mapping keys and list indices')
        self._width = width
        self._height = height
        self._mode = mode
        self._resample = resample
        self._paths = None if paths is None else tuple(tuple(path) for path in paths)

    @classmethod
    def from_spec_args(cls, **args: Any) -> Self:
        args['resample'] = Resample(args.get('resample', Resample.BILINEAR.value))
        if 'mode' in args:
            args['mode'] = ResizeMode(args['mode'])
        return cls(**args)

    @staticmethod
    def _usable_cpus() -> int:
        """CPUs this process may run on — its affinity mask where the platform publishes one, else the
        host's count. FOOTGUN: neither reads a cgroup CPU quota, so a container limited by `--cpus` and
        not by a mask still reads the host's cores.
        """
        if hasattr(os, 'sched_getaffinity'):
            return len(os.sched_getaffinity(0))
        return os.cpu_count() or 1

    def _workers(self, frames: Sequence[np.ndarray]) -> int:
        pixels = sum(
            frame.shape[0] * frame.shape[1]
            for frame in frames
            if frame.shape[:2] != (self._height, self._width)
            and (self._mode is not ResizeMode.BOUND or frame.shape[0] > self._height or frame.shape[1] > self._width)
        )
        if pixels < self._MIN_PARALLEL_PIXELS:
            return 1
        return max(1, min(len(frames), self._MAX_WORKERS, self._usable_cpus()))

    def _scaled_frames(self, frames: Sequence[np.ndarray]) -> list[np.ndarray]:
        scale = partial(
            image.transform_frame, width=self._width, height=self._height, mode=self._mode, resample=self._resample
        )
        workers = self._workers(frames)
        if workers == 1:
            return [scale(frame) for frame in frames]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(scale, frames))

    @staticmethod
    def _frames(value: np.ndarray) -> list[np.ndarray]:
        if value.ndim not in (3, 4) or value.shape[-1] != 3 or min(value.shape[-3:-1]) < 1:
            raise ValueError(f'Expected HWC with 3 channels or a THWC frame stack, got {value.shape}')
        return [value] if value.ndim == 3 else list(value)

    def _map_images(self, data: Any, transform: Callable[[np.ndarray], np.ndarray]) -> Any:
        if self._paths is None:

            def transform_value(value: Any) -> Any:
                if isinstance(value, np.ndarray) and value.ndim in (3, 4) and value.shape[-1] == 3:
                    return transform(value)
                return value

            return map_leaves(data, transform_value)
        for path in self._paths:
            data = update_at(data, path, lambda value: transform(np.asarray(value)))
        return data

    def encode(self, data):
        if self._paths is not None and any(
            a == b[: len(a)] or b == a[: len(b)] for a, b in combinations(self._paths, 2)
        ):
            # Overlapping paths transform the same data in the declared order.
            def transform(value: np.ndarray) -> np.ndarray:
                frames = self._scaled_frames(self._frames(value))
                return frames[0] if value.ndim == 3 else np.stack(frames)

            return self._map_images(data, transform)

        frames: list[np.ndarray] = []

        def collect(value: np.ndarray) -> np.ndarray:
            frames.extend(self._frames(value))
            return value

        self._map_images(data, collect)
        scaled = iter(self._scaled_frames(frames))

        def replace(value: np.ndarray) -> np.ndarray:
            return next(scaled) if value.ndim == 3 else np.stack([next(scaled) for _ in value])

        return self._map_images(data, replace)

    def decode(self, data):
        return data

    @property
    def training_encoder(self):
        raise NotImplementedError('Training recipes transform full-resolution signals with image.transform_frame')

    def to_spec(self):
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'width': self._width,
                'height': self._height,
                'mode': self._mode.value,
                'resample': self._resample.value,
                'paths': None if self._paths is None else [list(path) for path in self._paths],
            },
        }


def image_observation(images: dict[str, tuple[str, tuple[int, int]]], resample: Resample = Resample.BILINEAR) -> Codec:
    """Select camera fields and resize each image with padding."""
    result = SelectObservationFields({name: source for name, (source, _) in images.items()})
    for name, (_, (width, height)) in images.items():
        result = result | TransformImages(width, height, mode=ResizeMode.PAD, resample=resample, paths=[[name]])
    return result
