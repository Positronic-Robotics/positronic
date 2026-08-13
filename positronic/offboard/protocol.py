"""The offboard wire's contract: the frame envelope both ends spell, the shared msgpack encoding, and the
robot command this boundary carries.

A served command reaches the rig in one of two shapes, and both arrive typed:
- ``serialise`` writes a ``CommandType`` sitting anywhere in the payload as the ``__cmd__`` envelope, and
  ``deserialise`` reads it back
- an endpoint that does not speak that envelope answers with the bare ``to_wire`` mapping at its command
  channel; nothing but the channel tells that from any other dict, so ``typed_commands`` reads it instead

The env-server boundary (``positronic.simulator.env_server.protocol``) names the same commands differently
and its far end must not import positronic, so it cannot share this.
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

# The frame envelope every message travels in: ``STATUS`` frames until the server reports itself ready and
# hands over its ``META``, then one ``RESULT`` or ``ERROR`` per inference. The server writes these and the
# client reads them, so both import them from here.
STATUS = 'status'
MESSAGE = 'message'
META = 'meta'
RESULT = 'result'
ERROR = 'error'


class Status(StrEnum):
    """What a frame reports about the session, in its ``STATUS`` field."""

    READY = 'ready'
    WAITING = 'waiting'
    LOADING = 'loading'
    ERROR = 'error'


# The command envelope, written and read by the two hooks below and nowhere else.
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


def _typed(value: Any) -> Any:
    """One command channel's value, typed. A non-mapping — a decoded command, a ``.pose`` vector — is
    returned as it came."""
    if not isinstance(value, cabc.Mapping):
        return value
    # ``from_wire`` reads the vectors as arrays; the wire carries sequences, and ``type`` is its one string.
    return command.from_wire({k: v if isinstance(v, str) else np.asarray(v) for k, v in value.items()})


def typed_commands(result: Any) -> Any:
    """A served result — one action, a list of them, or ``None`` — with every command channel typed."""
    if isinstance(result, cabc.Mapping):
        return {k: _typed(v) if keys.is_robot_command(k) else v for k, v in result.items()}
    if isinstance(result, list):
        return [typed_commands(action) for action in result]
    return result
