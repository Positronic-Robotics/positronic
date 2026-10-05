"""Legacy v1/v2 encoding and robot-command interpretation.

A served command arrives either inside the ``__cmd__`` envelope or as the bare ``to_wire`` mapping at a
command channel, and nothing but the channel tells that mapping from any other dict.
"""

import collections.abc as cabc
import functools
from typing import Any

import msgpack
import numpy as np
from positronic_model_server import serialization
from positronic_model_server.protocol import ProtocolVersion

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.utils.versions import Version

CURRENT_VERSION = ProtocolVersion.V2
VERSIONS = {version.value: Version(version) for version in (ProtocolVersion.V1, ProtocolVersion.V2)}


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
