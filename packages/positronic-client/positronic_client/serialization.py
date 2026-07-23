"""Wire serialization for numpy arrays and standard Python types, with extension hooks for domain envelopes.

The base wire supports:
- built-in scalars: `str`, `int`, `float`, `bool`, `None`
- containers: `dict` / `list` / `tuple` recursively composed of supported values
- numeric numpy values: `numpy.ndarray` and `numpy` scalar types
- JPEG-compressed images (``encode_jpeg`` markers, decoded transparently)

Domain types ride as extension hooks: ``make_wire`` builds a ``(serialise, deserialise)`` pair whose hooks run
after the base handling, so a dialect (e.g. positronic's roboarm-command envelopes) layers on top without this
module knowing about it. A consumer of the base wire receives unknown envelopes as the plain dicts they are on
the wire.
"""

import collections.abc as cabc
import functools
import io
from collections.abc import Callable
from typing import Any

import msgpack
import numpy as np
from PIL import Image as PilImage

# JPEG quality for images on the wire. A single HD frame — and especially a (T, H, W, 3) stack — is many
# MB raw, over the ~2 MB websocket message cap of a Modal-fronted endpoint. Per-frame JPEG keeps a
# 25-frame two-camera stack around 1-2 MB and cuts upload latency; q=90 is visually lossless here.
_JPEG_QUALITY = 90

# A pack hook returns the wire form of a domain object, or None when the object isn't its to handle.
# An unpack hook returns the domain object for a wire dict, or None to pass it to the next hook.
PackHook = Callable[[Any], Any | None]
UnpackHook = Callable[[dict], Any | None]


def encode_jpeg(image: np.ndarray) -> dict[bytes, Any]:
    """JPEG-encode a single ``(H, W, 3)`` image or a ``(T, H, W, 3)`` stack to a compact wire marker.

    Sends one JPEG per frame plus the original ``ndim`` so ``_unpack`` restores the exact shape.
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


def make_wire(
    pack_hooks: tuple[PackHook, ...] = (), unpack_hooks: tuple[UnpackHook, ...] = ()
) -> tuple[Callable[[Any], bytes], Callable[[bytes], Any]]:
    """Build a ``(serialise, deserialise)`` pair extending the base wire with domain hooks."""

    def _pack(obj):
        if isinstance(obj, cabc.Mapping):
            return dict(obj)
        if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ('V', 'O', 'c'):
            raise ValueError(f'Unsupported dtype: {obj.dtype}')
        if isinstance(obj, np.ndarray):
            return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
        if isinstance(obj, np.generic):
            return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
        for hook in pack_hooks:
            wire = hook(obj)
            if wire is not None:
                return wire
        return obj

    def _unpack(obj):
        if b'__ndarray__' in obj:
            return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
        if b'__npgeneric__' in obj:
            return np.dtype(obj[b'dtype']).type(obj[b'data'])
        if b'__jpeg__' in obj:
            return _decode_jpeg(obj)
        for hook in unpack_hooks:
            value = hook(obj)
            if value is not None:
                return value
        return obj

    return functools.partial(msgpack.packb, default=_pack), functools.partial(msgpack.unpackb, object_hook=_unpack)


serialise, deserialise = make_wire()
