import collections.abc as cabc
import functools

import msgpack
import numpy as np

"""
Offboard wire serialization helpers.

This protocol intentionally supports only transport-friendly ("wire-serializable") values:
- built-in scalars: `str`, `int`, `float`, `bool`, `None`
- containers: `dict` / `list` / `tuple` recursively composed of supported values
- numeric numpy values: `numpy.ndarray` and `numpy` scalar types

Avoid sending arbitrary Python objects across the wire. If you need to transmit domain objects
(e.g., robot commands), transmit a plain-data representation (for example a tagged dict) and
reconstruct objects at the boundary.
"""


def pack_numpy(obj):
    # Accept any Mapping (e.g. MappingProxyType) and normalize to a plain dict
    # before msgpack sees it. This keeps internal "frozen view" protections
    # while ensuring the wire format stays transport-friendly.
    if isinstance(obj, cabc.Mapping):
        return dict(obj)

    if (isinstance(obj, np.ndarray | np.generic)) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')

    if isinstance(obj, np.ndarray):
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}

    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}

    return obj


def unpack_numpy(obj):
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])

    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])

    return obj


serialise = functools.partial(msgpack.packb, default=pack_numpy)
deserialise = functools.partial(msgpack.unpackb, object_hook=unpack_numpy)
