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
        # NOTE: str key, unlike the bytes keys above. A pre-PR server that doesn't decode
        # this type leaves the envelope as a plain dict in the observation; its recorder does
        # `key.endswith(...)` on dict keys, which TypeErrors on a bytes key but is harmless on
        # a str one. New servers reconstruct the enum in `_unpack` below.
        return {'__robotstatus__': obj.value}
    return obj


# TODO(remove-pre-PR-server-compat): drop once all deployed inference servers
# are rebuilt against the new client. Pre-PR vendor codecs returned commands as
# unwrapped ``to_wire(...)`` dicts (no ``__cmd__`` envelope), which the new
# client otherwise forwards to drivers as plain dicts and the driver ``match``
# falls through. The legacy ``type`` strings are specific enough that collision
# with arbitrary payloads is unlikely in practice.
_LEGACY_COMMAND_TYPES = frozenset({
    _roboarm_command.Reset.TYPE,
    _roboarm_command.Recover.TYPE,
    _roboarm_command.CartesianPosition.TYPE,
    _roboarm_command.JointPosition.TYPE,
    _roboarm_command.JointDelta.TYPE,
    _roboarm_command.CartesianDelta.TYPE,
})


def _unpack(obj):
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    if b'__jpeg__' in obj:
        return _decode_jpeg(obj)
    if b'__cmd__' in obj:
        inner = obj[b'__cmd__']
        # The legacy shim below decodes the inner ``to_wire`` dict to a Command
        # before this outer hook fires — accept either shape.
        if isinstance(inner, _roboarm_command.CommandType):
            return inner
        return _roboarm_command.from_wire(inner)
    # Accept both the str key (current wire form, see _pack) and the bytes key, so the wire
    # can later migrate to the bytes form — consistent with the envelopes above — without
    # breaking any server already deployed against this version. Both round-trip to the enum.
    if '__robotstatus__' in obj:
        return _roboarm.RobotStatus(obj['__robotstatus__'])
    if b'__robotstatus__' in obj:
        return _roboarm.RobotStatus(obj[b'__robotstatus__'])
    # TODO(remove-pre-PR-server-compat): see _LEGACY_COMMAND_TYPES above.
    if obj.get('type') in _LEGACY_COMMAND_TYPES:
        return _roboarm_command.from_wire(obj)
    return obj


serialise = functools.partial(msgpack.packb, default=_pack)
deserialise = functools.partial(msgpack.unpackb, object_hook=_unpack)

# Aliases for consistency
serialize = serialise
deserialize = deserialise
