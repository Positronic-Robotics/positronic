"""Env-server wire names and msgpack encoding for numpy arrays and plain data.

This module must work in an isolated interpreter without Positronic installed.
Arrays use raw bytes, ``dtype.str``, and shape for compatibility between numpy versions.
"""

import functools
from collections.abc import Callable
from enum import Enum
from typing import Any, cast

import msgpack
import numpy as np

CMD = 'cmd'
OK = 'ok'
TASKS = 'tasks'
SPEC = 'spec'
TOKEN = 'token'
ACTION = 'action'
ERROR = 'error'


class Command(Enum):
    TASKS = 'tasks'
    RESET = 'reset'
    STEP = 'step'
    CLOSE = 'close'


def _pack(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ('V', 'O', 'c'):
            raise ValueError(f'Unsupported dtype: {obj.dtype}')
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    return obj


def _unpack(obj):
    if b'__ndarray__' in obj:
        # A bytearray keeps the decoded array writable.
        return np.ndarray(buffer=bytearray(obj[b'data']), dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    return obj


# msgpack's default autoreset=True makes packb return bytes.
encode = cast(Callable[[Any], bytes], functools.partial(msgpack.packb, default=_pack))
decode = functools.partial(msgpack.unpackb, object_hook=_unpack)
