"""The offboard wire's contract: its message keys, the shared msgpack encoding, and the robot command
this boundary carries.

A served command arrives either inside the ``__cmd__`` envelope or as the bare ``to_wire`` mapping at a
command channel, and nothing but the channel tells that mapping from any other dict.
"""

import collections.abc as cabc
import functools
from enum import StrEnum
from types import MappingProxyType
from typing import Any, NamedTuple, Self

import msgpack
import numpy as np

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.offboard import keys as offboard_keys
from positronic.utils import serialization

# The top-level keys of every server-to-client message: ``STATUS`` until the server reports itself ready
# and hands over its ``META``, then one ``RESULT`` or ``ERROR`` per inference.
STATUS = 'status'
MESSAGE = 'message'
META = 'meta'
RESULT = 'result'
ERROR = 'error'
# What the server spent on one inference, beside the ``RESULT`` it answers with: durations in
# milliseconds on the server's own clock. A server that sends none leaves the round trip undivided.
TIMING = 'timing'

# The phases ``TIMING`` reports. `SERVED` brackets the others: it opens on the observation
# arriving and closes before the answer is encoded.
TIMING_SERVED = 'served_ms'
TIMING_DECODE = 'decode_ms'
TIMING_INFER = 'infer_ms'
# Time the observation waited for the inference slot, inside `SERVED`.
TIMING_QUEUED = 'queued_ms'


def timing_key(name: str) -> str:
    return f'{name}_ms'


# What a blocking session's call is timed as: the heavy work it waits out.
MODEL_CALL = 'model'
# Time the model's own call took, inside `INFER`.
TIMING_MODEL = timing_key(MODEL_CALL)


class ServerStatus(StrEnum):
    READY = 'ready'
    WAITING = 'waiting'
    LOADING = 'loading'
    ERROR = 'error'


# How many inferences the loaded checkpoint has answered, in the record the unary verbs return.
INFERENCES = 'inferences'


class Readiness(NamedTuple):
    """What a server says about itself when a caller asks, outside any session.

    Every field holds only for the moment it answers: a server that answers ``READY`` may load another
    checkpoint and answer ``LOADING`` again, with its port bound throughout.
    """

    status: ServerStatus
    message: str = ''
    checkpoint_id: str | None = None
    inferences: int = 0
    timing: cabc.Mapping[str, float] = MappingProxyType({})
    positronic_version: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            STATUS: str(self.status),
            MESSAGE: self.message,
            offboard_keys.CHECKPOINT_ID: self.checkpoint_id,
            INFERENCES: self.inferences,
            TIMING: dict(self.timing),
            offboard_keys.POSITRONIC_VERSION: self.positronic_version,
        }

    @classmethod
    def from_wire(cls, answer: cabc.Mapping[str, Any]) -> Self:
        """The record ``answer`` carries. Raises ``ValueError`` when it names no status this protocol has."""
        return cls(
            status=ServerStatus(answer.get(STATUS)),
            message=answer.get(MESSAGE) or '',
            checkpoint_id=answer.get(offboard_keys.CHECKPOINT_ID),
            inferences=int(answer.get(INFERENCES) or 0),
            timing=MappingProxyType(dict(answer.get(TIMING) or {})),
            positronic_version=answer.get(offboard_keys.POSITRONIC_VERSION),
        )


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
