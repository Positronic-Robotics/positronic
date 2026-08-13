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
from typing import Any

import msgpack
import numpy as np
from PIL import Image as PilImage

# JPEG quality for images on the wire. A single HD frame — and especially a (T, H, W, 3) stack — is many
# MB raw, over the ~2 MB websocket message cap of a Modal-fronted endpoint. Per-frame JPEG keeps a
# 25-frame two-camera stack around 1-2 MB and cuts upload latency; q=90 is visually lossless here.
_JPEG_QUALITY = 90


def encode_jpeg(image: np.ndarray) -> dict[bytes, Any]:
    """JPEG-encode a single ``(H, W, 3)`` image or a ``(T, H, W, 3)`` stack to a compact wire marker.

    Sends one JPEG per frame plus the original ``ndim`` so ``unpack`` restores the exact shape.
    """
    frames = image if image.ndim == 4 else image[None]
    bufs = []
    for frame in frames:
        buf = io.BytesIO()
        PilImage.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(buf, format='JPEG', quality=_JPEG_QUALITY)
        bufs.append(buf.getvalue())
    return {b'__jpeg__': True, b'frames': bufs, b'ndim': int(image.ndim)}


def _decode_jpeg(marker: dict) -> np.ndarray:
    """Inverse of ``encode_jpeg``: decode per-frame JPEGs and restore the original shape."""
    frames = np.stack([np.asarray(PilImage.open(io.BytesIO(buf))) for buf in marker[b'frames']])
    return frames if marker[b'ndim'] == 4 else frames[0]


def pack(obj):
    """msgpack's ``default`` hook: one value in its wire form, or unchanged when msgpack handles it."""
    if isinstance(obj, cabc.Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')
    if isinstance(obj, np.ndarray):
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    return obj


def unpack(obj):
    """msgpack's ``object_hook``: one decoded mapping restored to the value it encodes."""
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    if b'__jpeg__' in obj:
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
