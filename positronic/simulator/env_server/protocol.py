"""Wire codec for the remote env-server boundary: msgpack with a numpy envelope.

This module is **positronic-free** — it imports only ``msgpack`` and ``numpy`` — so it can be
imported (or copied) into a benchmark's isolated interpreter alongside the dumb server without
dragging in pimm or the rest of positronic. Only raw numpy arrays and plain-data dicts cross the
wire; every canonical<->raw mapping lives client-side in the ``EnvAdapter``.

Arrays travel as raw bytes plus their ``dtype.str`` and shape, so a numpy-2 server round-trips a
numpy-1 client unchanged.
"""

import functools

import msgpack
import numpy as np


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
        # ``bytearray`` (not the raw msgpack ``bytes``) backs a writable array, so the socket path
        # matches the in-process path for envs/adapters that mutate a decoded buffer in place.
        return np.ndarray(buffer=bytearray(obj[b'data']), dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    return obj


encode = functools.partial(msgpack.packb, default=_pack)
decode = functools.partial(msgpack.unpackb, object_hook=_unpack)
