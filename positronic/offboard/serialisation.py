import functools

import msgpack
import numpy as np


def pack_numpy(obj):
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
