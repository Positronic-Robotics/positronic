"""The offboard wire's contract: its message keys, the shared msgpack encoding, and the robot command
this boundary carries.

A served command arrives either inside the ``__cmd__`` envelope or as the bare ``to_wire`` mapping at a
command channel, and nothing but the channel tells that mapping from any other dict.
"""

import collections.abc as cabc
import functools
from enum import StrEnum
from typing import Any

import msgpack
import numpy as np

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.utils import serialization

# The top-level keys of every server-to-client message: ``STATUS`` until the server reports itself ready
# and hands over its ``META``, then one ``RESULT`` or ``ERROR`` per inference.
STATUS = 'status'
MESSAGE = 'message'
META = 'meta'
RESULT = 'result'
ERROR = 'error'

# The server's own entries in the ``META`` it hands over: where it serves, which checkpoint it resolved, and
# what the rig builds and obeys — the local stack spec, image compression, the positronic version it runs.
HOST = 'host'
PORT = 'port'
CHECKPOINT_ID = 'checkpoint_id'
LOCAL_STACK = 'local_stack'
COMPRESS_IMAGES = 'compress_images'
POSITRONIC_VERSION = 'positronic_version'


class ServerStatus(StrEnum):
    READY = 'ready'
    WAITING = 'waiting'
    LOADING = 'loading'
    ERROR = 'error'


_CMD = b'__cmd__'


def _pack(obj):
    if isinstance(obj, command.CommandType):
        return {_CMD: command.to_wire(obj)}
    return serialization.pack(obj)


def _unpack(obj):
    if _CMD in obj:
        return command.from_wire(obj[_CMD])
    return serialization.unpack(obj)


def serialise(obj: Any) -> bytes:
    packed = msgpack.packb(obj, default=_pack)
    assert packed is not None
    return packed


deserialise = functools.partial(msgpack.unpackb, object_hook=_unpack)


def _as_wire(value: Any) -> Any:
    """A wire field as ``from_wire`` reads it: vectors as arrays, strings and nested mappings as they are."""
    if isinstance(value, cabc.Mapping):
        return {k: _as_wire(v) for k, v in value.items()}
    return value if isinstance(value, str) else np.asarray(value)


def _typed(value: Any) -> Any:
    """One command channel's value, typed."""
    if not isinstance(value, cabc.Mapping):
        return value
    return command.from_wire(_as_wire(value))


def typed_commands(result: Any) -> Any:
    """A served result — one action, a list of them, or ``None`` — with every command channel typed."""
    if isinstance(result, cabc.Mapping):
        return {k: _typed(v) if keys.is_robot_command(k) else v for k, v in result.items()}
    if isinstance(result, list):
        return [typed_commands(action) for action in result]
    return result
