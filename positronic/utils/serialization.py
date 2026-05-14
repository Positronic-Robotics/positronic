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

import msgpack
import numpy as np

from positronic.drivers.roboarm import command as _roboarm_command


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
    return obj


def _unpack(obj):
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    if b'__cmd__' in obj:
        return _roboarm_command.from_wire(obj[b'__cmd__'])
    return obj


serialise = functools.partial(msgpack.packb, default=_pack)
deserialise = functools.partial(msgpack.unpackb, object_hook=_unpack)

# Aliases for consistency
serialize = serialise
deserialize = deserialise
