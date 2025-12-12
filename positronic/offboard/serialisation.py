from typing import Any

import msgpack
import numpy as np


def serialise(obj: Any) -> bytes:
    if (isinstance(obj, np.ndarray | np.generic)) and obj.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype: {obj.dtype}')

    if isinstance(obj, np.ndarray):
        obj = {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    elif isinstance(obj, np.generic):
        obj = {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}

    return msgpack.packb(obj)


def deserialise(data: bytes) -> Any:
    obj = msgpack.unpackb(data)
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])

    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])

    return obj
