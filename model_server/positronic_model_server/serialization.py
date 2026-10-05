"""Wire serialization helpers for numpy arrays, images and standard Python types.

Supports:
- built-in scalars: `str`, `int`, `float`, `bool`, `None`
- containers: `dict` / `list` / `tuple` recursively composed of supported values
- numeric numpy values: `numpy.ndarray` and `numpy` scalar types
- images JPEG-encoded with ``encode_jpeg``

Nothing here knows a domain type. A boundary that carries one writes its own msgpack hooks and
delegates to ``pack`` / ``unpack`` for the rest.
"""

import collections.abc as cabc
import functools
import io
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
from PIL import Image as PilImage

# The envelope each encoded value travels in: a marker naming the type, and the fields carrying it.
_NDARRAY = b'__ndarray__'
_NPGENERIC = b'__npgeneric__'
_JPEG = b'__jpeg__'
_DATA = b'data'
_DTYPE = b'dtype'
_SHAPE = b'shape'
_FRAMES = b'frames'
_NDIM = b'ndim'

DEFAULT_JPEG_QUALITY = 90


@dataclass(frozen=True)
class JpegEncoding:
    """An explicit mapping/list path to an RGB image; an empty path selects the whole value."""

    path: tuple[str | int, ...]
    quality: int = DEFAULT_JPEG_QUALITY

    def __post_init__(self) -> None:
        if any(type(part) not in (str, int) for part in self.path):
            raise ValueError('Image paths contain only mapping keys and list indices')
        if type(self.quality) is not int or not 0 <= self.quality <= 100:
            raise ValueError('JPEG quality must be an integer between 0 and 100')

    def encode(self, value: Any) -> Any:
        """Copy the selected container path and encode its leaf, preserving unrelated values."""

        def at(node: Any, path: tuple[str | int, ...]) -> Any:
            if not path:
                if not isinstance(node, np.ndarray):
                    raise ValueError(f'JPEG path {self.path!r} does not select an array')
                return encode_jpeg(node, self.quality)
            key, *rest = path
            if isinstance(node, cabc.Mapping):
                return {**node, key: at(node[key], tuple(rest))}
            if isinstance(node, list | tuple) and type(key) is int:
                copy = list(node)
                copy[key] = at(node[key], tuple(rest))
                return tuple(copy) if isinstance(node, tuple) else copy
            raise ValueError(f'JPEG path {self.path!r} does not match the value')

        return at(value, self.path)


def encode_images(value: Any, encodings: cabc.Sequence[JpegEncoding]) -> Any:
    """Encode only the explicitly selected images, in observations or model results."""
    for encoding in encodings:
        value = encoding.encode(value)
    return value


def encode_jpeg(image: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> dict[bytes, Any]:
    """JPEG-encode RGB images shaped ``(..., H, W, 3)``, casting pixels to uint8.

    Nonempty 3D/4D inputs use the v1/v2 marker. Extra leading dimensions and empty batches carry
    their full shape. JPEG changes pixels; ordinary arrays passed to ``serialise`` stay lossless.
    """
    if image.ndim < 3 or image.shape[-1] != 3 or min(image.shape[-3:-1]) < 1:
        raise ValueError('JPEG images must have shape (..., H, W, 3) with positive height and width')
    if type(quality) is not int or not 0 <= quality <= 100:
        raise ValueError('JPEG quality must be an integer between 0 and 100')
    frames = image.reshape((-1, *image.shape[-3:]))
    bufs = []
    for frame in frames:
        buf = io.BytesIO()
        PilImage.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(buf, format='JPEG', quality=quality)
        bufs.append(buf.getvalue())
    if image.ndim > 4 or not bufs:
        return {_JPEG: True, _FRAMES: bufs, _SHAPE: image.shape}
    return {_JPEG: True, _FRAMES: bufs, _NDIM: int(image.ndim)}


def _decode_jpeg(marker: dict) -> np.ndarray:
    """Inverse of ``encode_jpeg``: decode per-frame JPEGs and restore the original shape."""
    if _SHAPE in marker and not marker[_FRAMES]:
        shape = tuple(marker[_SHAPE])
        if len(shape) < 4 or shape[-1] != 3 or min(shape[-3:-1]) < 1 or 0 not in shape[:-3]:
            raise ValueError('An empty JPEG batch must have an empty leading dimension')
        return np.empty(shape, dtype=np.uint8)
    frames = np.stack([np.asarray(PilImage.open(io.BytesIO(buf))) for buf in marker[_FRAMES]])
    if _SHAPE in marker:
        return frames.reshape(marker[_SHAPE])
    return frames if marker[_NDIM] == 4 else frames[0]


def pack(obj):
    """msgpack's ``default`` hook: one value in its wire form, or unchanged when msgpack handles it."""
    if isinstance(obj, cabc.Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')
    if isinstance(obj, np.ndarray):
        return {_NDARRAY: True, _DATA: obj.tobytes(), _DTYPE: obj.dtype.str, _SHAPE: obj.shape}
    if isinstance(obj, np.generic):
        return {_NPGENERIC: True, _DATA: obj.item(), _DTYPE: obj.dtype.str}
    return obj


def unpack(obj):
    """msgpack's ``object_hook``: one decoded mapping restored to the value it encodes."""
    if _NDARRAY in obj:
        return np.ndarray(buffer=obj[_DATA], dtype=np.dtype(obj[_DTYPE]), shape=obj[_SHAPE])
    if _NPGENERIC in obj:
        return np.dtype(obj[_DTYPE]).type(obj[_DATA])
    if _JPEG in obj:
        return _decode_jpeg(obj)
    return obj


def serialise(obj: Any) -> bytes:
    packed = msgpack.packb(obj, default=pack)
    assert packed is not None
    return packed


deserialise = functools.partial(msgpack.unpackb, object_hook=unpack)

# Aliases for consistency
serialize = serialise
deserialize = deserialise
