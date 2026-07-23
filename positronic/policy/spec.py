"""The policy-pipeline split marker and the wire format for the rig-side half.

A policy pipeline is one wrapper chain with a ``remote`` marker naming the client/server
border::

    pipeline = TemporalStack(...) | ChunkedSchedule() | remote | codec

Everything left of the marker is the *local* half — the stack the rig runs in front of the
connection; everything right of it is the *remote* half — what the inference server runs around
the model. ``split`` separates the two; ``inline`` drops the marker for serving the whole policy
in one process.

The server publishes the local half in its ``ready`` handshake as a plain-data spec tree:
``{'name': ..., 'args': {...}}`` leaves composed by ``{SEQ: [...]}`` (the ``|`` operator) and
``{PAR: [...]}`` (the ``&`` operator). ``RemotePolicy`` rebuilds the stack via ``from_spec``.

``WIRE_WRAPPERS`` is the closed vocabulary and the security boundary: names resolve only against
this table, so a server can select which of our components the rig runs but can never execute
foreign code. Wire names follow the command wire format's discipline — stable, decoupled from
import paths; new constructor arguments must have defaults; changing an entry's meaning means a
new name.
"""

import functools
import operator
from typing import Any

from positronic.policy.action import (
    AbsoluteJointsAction,
    AbsolutePositionAction,
    JointDeltaAction,
    RelativePositionAction,
)
from positronic.policy.base import PAR, SEQ, Policy, PolicyWrapper
from positronic.policy.codec import (
    ActionHorizon,
    ActionTimestamp,
    BinarizeGripInference,
    BinarizeGripTraining,
    FlipGrip,
)
from positronic.policy.observation import ObservationCodec
from positronic.policy.wrappers import ChunkedSchedule, TemporalStack


class _RemoteMarker(PolicyWrapper):
    """The client/server border in a policy pipeline. Only ever split on, never applied."""

    def wrap(self, policy: Policy) -> Policy:
        raise TypeError('`remote` marks the client/server border of a pipeline; split() it instead of wrapping')


remote = _RemoteMarker()

WIRE_WRAPPERS: dict[str, type[PolicyWrapper]] = {
    'chunked_schedule': ChunkedSchedule,
    'temporal_stack': TemporalStack,
    'action_timestamp': ActionTimestamp,
    'action_horizon': ActionHorizon,
    'binarize_grip_training': BinarizeGripTraining,
    'binarize_grip_inference': BinarizeGripInference,
    'flip_grip': FlipGrip,
    'observation_codec': ObservationCodec,
    'absolute_position_action': AbsolutePositionAction,
    'absolute_joints_action': AbsoluteJointsAction,
    'relative_position_action': RelativePositionAction,
    'joint_delta_action': JointDeltaAction,
}


def _join(components: tuple) -> PolicyWrapper | None:
    return functools.reduce(operator.or_, components) if components else None


def split(pipeline: PolicyWrapper) -> tuple[PolicyWrapper | None, PolicyWrapper | None]:
    """Split a pipeline on the ``remote`` marker into its ``(local, remote)`` halves.

    An empty half is ``None``; a pipeline of just the marker means "no glue on either side".
    """
    components = pipeline._pipeline_components()
    markers = [i for i, c in enumerate(components) if isinstance(c, _RemoteMarker)]
    if len(markers) != 1:
        raise ValueError(f'A policy pipeline needs exactly one `remote` marker, found {len(markers)}')
    idx = markers[0]
    return _join(components[:idx]), _join(components[idx + 1 :])


def inline(pipeline: PolicyWrapper) -> PolicyWrapper | None:
    """The pipeline with the marker dropped — both halves composed for in-process serving."""
    local, rem = split(pipeline)
    return _join(tuple(part for part in (local, rem) if part is not None))


def from_spec(node: dict[str, Any]) -> PolicyWrapper | None:
    """Rebuild a declared local stack from its wire spec; ``None`` for an empty declaration.

    Unknown entry names raise ``ValueError`` and unknown arguments ``TypeError`` — a declaration
    this build cannot honor fails before anything moves.
    """
    if SEQ in node:
        parts = tuple(part for part in (from_spec(child) for child in node[SEQ]) if part is not None)
        return _join(parts)
    if PAR in node:
        return functools.reduce(operator.and_, (from_spec(child) for child in node[PAR]))
    name = node.get('name')
    if name not in WIRE_WRAPPERS:
        raise ValueError(f'Unknown local-stack entry {name!r}; this build knows {sorted(WIRE_WRAPPERS)}')
    return WIRE_WRAPPERS[name](**node.get('args', {}))
