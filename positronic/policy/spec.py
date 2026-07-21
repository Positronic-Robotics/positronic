"""The policy-definition split marker and the wire format for the rig-side half.

A policy definition is one wrapper pipeline with a ``remote`` marker naming the client/server
border::

    definition = TemporalStack(...) | ChunkedSchedule() | remote | codec

Everything left of the marker is the *local* half — the stack the rig runs in front of the
connection; everything right of it is the *remote* half — what the inference server runs around
the model. ``split`` separates the two; ``inline`` drops the marker for serving the whole policy
in one process.

The server publishes the local half in its ``ready`` handshake as a plain-data spec tree:
``{'name': ..., 'args': {...}}`` leaves composed by ``{'seq': [...]}`` (the ``|`` operator) and
``{'par': [...]}`` (the ``&`` operator). ``RemotePolicy`` rebuilds the stack via ``from_spec``.

``WIRE_WRAPPERS`` is the closed vocabulary and the security boundary: names resolve only against
this table, so a server can select which of our components the rig runs but can never execute
foreign code. Wire names follow the command wire format's discipline — stable, decoupled from
import paths; new constructor arguments must have defaults; changing an entry's meaning means a
new name.
"""

import functools
import operator
from typing import Any

from positronic.policy.base import Policy, PolicyWrapper
from positronic.policy.wrappers import ChunkedSchedule, TemporalStack


class _RemoteMarker(PolicyWrapper):
    """The client/server border in a policy definition. Only ever split on, never applied."""

    def wrap(self, policy: Policy) -> Policy:
        raise TypeError('`remote` marks the client/server border of a definition; split() it instead of wrapping')


remote = _RemoteMarker()

WIRE_WRAPPERS: dict[str, type[PolicyWrapper]] = {'chunked_schedule': ChunkedSchedule, 'temporal_stack': TemporalStack}


def _join(components: tuple) -> PolicyWrapper | None:
    return functools.reduce(operator.or_, components) if components else None


def split(definition: PolicyWrapper) -> tuple[PolicyWrapper | None, PolicyWrapper | None]:
    """Split a definition on the ``remote`` marker into its ``(local, remote)`` halves.

    An empty half is ``None``; a definition of just the marker means "no glue on either side".
    """
    components = definition._pipeline_components()
    markers = [i for i, c in enumerate(components) if isinstance(c, _RemoteMarker)]
    if len(markers) != 1:
        raise ValueError(f'A policy definition needs exactly one `remote` marker, found {len(markers)}')
    idx = markers[0]
    return _join(components[:idx]), _join(components[idx + 1 :])


def inline(definition: PolicyWrapper) -> PolicyWrapper | None:
    """The definition with the marker dropped — both halves composed for in-process serving."""
    local, rem = split(definition)
    return _join(tuple(part for part in (local, rem) if part is not None))


def from_spec(node: dict[str, Any]) -> PolicyWrapper | None:
    """Rebuild a declared local stack from its wire spec; ``None`` for an empty declaration.

    Unknown entry names raise ``ValueError`` and unknown arguments ``TypeError`` — a declaration
    this build cannot honor fails before anything moves.
    """
    if 'seq' in node:
        parts = tuple(part for part in (from_spec(child) for child in node['seq']) if part is not None)
        return _join(parts)
    if 'par' in node:
        return functools.reduce(operator.and_, (from_spec(child) for child in node['par']))
    name = node.get('name')
    if name not in WIRE_WRAPPERS:
        raise ValueError(f'Unknown local-stack entry {name!r}; this build knows {sorted(WIRE_WRAPPERS)}')
    return WIRE_WRAPPERS[name](**node.get('args', {}))
