"""The policy-pipeline algebra: the split marker, the model-source terminal, and the rig-side wire format.

A policy pipeline is one wrapper chain with a ``remote`` marker naming the client/server border,
closed by a ``ModelSource`` terminal::

    pipeline = TemporalStack(...) | ChunkedSchedule() | remote | codec | source

Everything left of the marker is the *local* half — the stack the rig runs in front of the
connection; everything right of it is the *remote* half — what the inference server runs around
the model the source loads. ``split`` separates the two halves; ``inline`` composes the whole
pipeline around the source's model into one in-process ``Policy``.

The server publishes the local half in its ``ready`` handshake as a plain-data spec tree:
``{'name': ..., 'args': {...}}`` leaves composed by ``{SEQ: [...]}`` (the ``|`` operator) and
``{PAR: [...]}`` (the ``&`` operator). ``RemotePolicy`` rebuilds the stack via ``from_spec``.

``WIRE_WRAPPERS`` is the closed vocabulary and the security boundary: names resolve only against
this table, so a server can select which of our components the rig runs but can never execute
foreign code. Model sources are never wire-deliverable — they exist only server-side. Wire names
follow the command wire format's discipline — stable, decoupled from import paths; new constructor
arguments must have defaults; changing an entry's meaning means a new name.
"""

import abc
import functools
import operator
from collections.abc import Callable
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
    RestrictImageSize,
)
from positronic.policy.observation import ObservationCodec
from positronic.policy.wrappers import ChunkedSchedule, TemporalStack


class _RemoteMarker(PolicyWrapper):
    """The client/server border in a policy pipeline. Only ever split on, never applied."""

    def wrap(self, policy: Policy) -> Policy:
        raise TypeError('`remote` marks the client/server border of a pipeline; split() it instead of wrapping')


remote = _RemoteMarker()


class ModelSource(abc.ABC):
    """Serving-side terminal of a policy pipeline: a stateless factory of models.

    Construction is cheap and side-effect-free; all runtime state (weights, subprocesses) belongs to
    the Policy returned by ``load``, torn down by its ``close()``. On a server the source is fixed at
    launch: the server rejects session params that would change it (structural equality below).
    """

    @abc.abstractmethod
    def get_models(self) -> list[str]:
        """Ids of the models this source can load, oldest first."""

    def resolve(self, model_id: str | None) -> str:
        """Explicit id validated against ``get_models()``; ``None`` picks the last entry (latest)."""
        models = self.get_models()
        if model_id is None:
            return models[-1]
        if model_id not in models:
            raise ValueError(f'Unknown model {model_id!r}. Available: {models}')
        return model_id

    @abc.abstractmethod
    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        """Load the model into a ready ``Policy``, reporting progress messages through ``on_progress``."""

    def meta(self, model_id: str) -> dict[str, Any]:
        """Static per-model handshake facts (model type, config name, ...)."""
        return {}

    def __eq__(self, other):
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __ror__(self, left):
        if isinstance(left, PolicyWrapper):
            return Pipeline(left._wrappers(), self)
        return NotImplemented


class Pipeline:
    """A policy pipeline closed by a model source: the full description of a policy server."""

    def __init__(self, components: tuple[PolicyWrapper, ...], source: ModelSource):
        self.components = tuple(components)
        self.source = source


class PolicySource(ModelSource):
    """Serves one ready in-process Policy — for in-process pipelines and tests."""

    def __init__(self, policy: Policy, name: str = 'default'):
        self._policy = policy
        self._name = name

    def get_models(self) -> list[str]:
        return [self._name]

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        return self._policy


WIRE_WRAPPERS: dict[str, type[PolicyWrapper]] = {
    'chunked_schedule': ChunkedSchedule,
    'temporal_stack': TemporalStack,
    'action_timestamp': ActionTimestamp,
    'action_horizon': ActionHorizon,
    'binarize_grip_training': BinarizeGripTraining,
    'binarize_grip_inference': BinarizeGripInference,
    'flip_grip': FlipGrip,
    'restrict_image_size': RestrictImageSize,
    'observation_codec': ObservationCodec,
    'absolute_position_action': AbsolutePositionAction,
    'absolute_joints_action': AbsoluteJointsAction,
    'relative_position_action': RelativePositionAction,
    'joint_delta_action': JointDeltaAction,
}


def _join(components: tuple) -> PolicyWrapper | None:
    return functools.reduce(operator.or_, components) if components else None


def split(pipeline: Pipeline | PolicyWrapper) -> tuple[PolicyWrapper | None, PolicyWrapper | None]:
    """Split a pipeline's wrapper chain on the ``remote`` marker into its ``(local, remote)`` halves.

    An empty half is ``None``; a chain of just the marker means "no glue on either side".
    """
    components = pipeline.components if isinstance(pipeline, Pipeline) else pipeline._wrappers()
    markers = [i for i, c in enumerate(components) if isinstance(c, _RemoteMarker)]
    if len(markers) != 1:
        raise ValueError(f'A policy pipeline needs exactly one `remote` marker, found {len(markers)}')
    idx = markers[0]
    return _join(components[:idx]), _join(components[idx + 1 :])


def inline(pipeline: Pipeline) -> Policy:
    """The whole pipeline in one process: components (marker dropped) wrapped around the source's latest model."""
    components = tuple(c for c in pipeline.components if not isinstance(c, _RemoteMarker))
    policy = pipeline.source.load(pipeline.source.resolve(None))
    joined = _join(components)
    return joined.wrap(policy) if joined is not None else policy


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
