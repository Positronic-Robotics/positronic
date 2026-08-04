"""Wire serialization helpers for numpy arrays, robot commands, and standard Python types.

Supports:
- built-in scalars: `str`, `int`, `float`, `bool`, `None`
- containers: `dict` / `list` / `tuple` recursively composed of supported values
- numeric numpy values: `numpy.ndarray` and `numpy` scalar types
- robot commands: ``positronic.drivers.roboarm.command.CommandType`` instances —
  transparently round-tripped via ``to_wire`` / ``from_wire``.
"""

# TODO: This module currently knows about ``roboarm.command`` directly. If we
# accumulate more domain types that need wire treatment (gripper commands,
# observation packets, etc.), replace the inline dispatch with a generic
# registry / ``__to_wire__`` protocol so utils stays domain-agnostic.

import collections.abc as cabc
import functools
import io
from typing import Any

import msgpack
import numpy as np
from PIL import Image as PilImage

from positronic.drivers import roboarm as _roboarm
from positronic.drivers.roboarm import command as _roboarm_command

# JPEG quality for images on the wire. A single HD frame — and especially a (T, H, W, 3) stack — is many
# MB raw, over the ~2 MB websocket message cap of a Modal-fronted endpoint. Per-frame JPEG keeps a
# 25-frame two-camera stack around 1-2 MB and cuts upload latency; q=90 is visually lossless here.
_JPEG_QUALITY = 90


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


def _pack(obj):
    if isinstance(obj, cabc.Mapping):
        return dict(obj)
    if isinstance(obj, np.ndarray | np.generic) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')
    if isinstance(obj, np.ndarray):
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    if isinstance(obj, _roboarm_command.CommandType):
        return {b'__cmd__': _roboarm_command.to_wire(obj)}
    if isinstance(obj, _roboarm.RobotStatus):
        # A str key, unlike the bytes keys above: a server that leaves the envelope undecoded passes a plain
        # dict to a recorder that does ``key.endswith(...)``, which TypeErrors on bytes and not on str.
        return {'__robotstatus__': obj.value}
    return obj


def _unpack(obj):
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    if b'__jpeg__' in obj:
        return _decode_jpeg(obj)
    if b'__cmd__' in obj:
        return _roboarm_command.from_wire(obj[b'__cmd__'])
    # Accept both the str key (current wire form, see _pack) and the bytes key, so the wire
    # can later migrate to the bytes form — consistent with the envelopes above — without
    # breaking any server already deployed against this version. Both round-trip to the enum.
    if '__robotstatus__' in obj:
        return _roboarm.RobotStatus(obj['__robotstatus__'])
    if b'__robotstatus__' in obj:
        return _roboarm.RobotStatus(obj[b'__robotstatus__'])
    return obj


def serialise(obj: Any) -> bytes:
    packed = msgpack.packb(obj, default=_pack)
    assert packed is not None
    return packed


deserialise = functools.partial(msgpack.unpackb, object_hook=_unpack)

# Aliases for consistency
serialize = serialise
deserialize = deserialise
