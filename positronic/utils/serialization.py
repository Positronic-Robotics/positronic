"""positronic's wire dialect: the base client wire plus roboarm command / robot status envelopes.

The base wire (numpy arrays, scalars, containers, JPEG markers) lives in ``positronic_client.serialization``;
this module layers the domain envelopes on top via its extension hooks, so servers and positronic-side clients
round-trip ``CommandType`` / ``RobotStatus`` objects while a bare ``positronic_client`` consumer receives the
same payloads as plain wire dicts.
"""

from typing import Any

from positronic_client.serialization import make_wire

from positronic.drivers import roboarm as _roboarm
from positronic.drivers.roboarm import command as _roboarm_command


def _pack_domain(obj: Any) -> Any | None:
    if isinstance(obj, _roboarm_command.CommandType):
        return {b'__cmd__': _roboarm_command.to_wire(obj)}
    if isinstance(obj, _roboarm.RobotStatus):
        # NOTE: str key, unlike the bytes keys above. A pre-PR server that doesn't decode
        # this type leaves the envelope as a plain dict in the observation; its recorder does
        # `key.endswith(...)` on dict keys, which TypeErrors on a bytes key but is harmless on
        # a str one. New servers reconstruct the enum in `_unpack_domain` below.
        return {'__robotstatus__': obj.value}
    return None


# TODO(remove-pre-PR-server-compat): drop once all deployed inference servers
# are rebuilt against the new client. Pre-PR vendor codecs returned commands as
# unwrapped ``to_wire(...)`` dicts (no ``__cmd__`` envelope), which the new
# client otherwise forwards to drivers as plain dicts and the driver ``match``
# falls through. The legacy ``type`` strings are specific enough that collision
# with arbitrary payloads is unlikely in practice.
_LEGACY_COMMAND_TYPES = frozenset({
    _roboarm_command.Reset.TYPE,
    _roboarm_command.CartesianPosition.TYPE,
    _roboarm_command.JointPosition.TYPE,
    _roboarm_command.JointDelta.TYPE,
    _roboarm_command.CartesianDelta.TYPE,
})


def _unpack_domain(obj: dict) -> Any | None:
    if b'__cmd__' in obj:
        inner = obj[b'__cmd__']
        # The legacy shim below decodes the inner ``to_wire`` dict to a Command
        # before this outer hook fires — accept either shape.
        if isinstance(inner, _roboarm_command.CommandType):
            return inner
        return _roboarm_command.from_wire(inner)
    # Accept both the str key (current wire form, see _pack_domain) and the bytes key, so the wire
    # can later migrate to the bytes form — consistent with the envelopes above — without
    # breaking any server already deployed against this version. Both round-trip to the enum.
    if '__robotstatus__' in obj:
        return _roboarm.RobotStatus(obj['__robotstatus__'])
    if b'__robotstatus__' in obj:
        return _roboarm.RobotStatus(obj[b'__robotstatus__'])
    # TODO(remove-pre-PR-server-compat): see _LEGACY_COMMAND_TYPES above.
    if obj.get('type') in _LEGACY_COMMAND_TYPES:
        return _roboarm_command.from_wire(obj)
    return None


serialise, deserialise = make_wire(pack_hooks=(_pack_domain,), unpack_hooks=(_unpack_domain,))

# Aliases for consistency
serialize = serialise
deserialize = deserialise
