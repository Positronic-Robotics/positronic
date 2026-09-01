"""The policy-pipeline algebra: the split marker, the model-source terminal, and the rig-side wire format.

A policy pipeline is one layer chain with a ``remote`` marker naming the client/server border,
closed by a ``ModelSource`` terminal::

    pipeline = TemporalStack(...) | ChunkedSchedule() | remote | codec | source

Everything left of the marker is the *local* half — the stack the rig runs in front of the
connection; everything right of it is the *remote* half — what the inference server runs around
the model the source loads. The marker itself carries the settings of the wire between them.
``split`` separates the halves and the border; ``inline`` composes the whole pipeline around the
source's model into one in-process ``Policy``.

The server publishes the local half in its ``ready`` handshake as a plain-data spec tree:
``{'name': ..., 'args': {...}}`` leaves composed by ``{SEQ: [...]}`` (the ``|`` operator) and
``{PAR: [...]}`` (the ``&`` operator). ``RemotePolicy`` rebuilds the stack via ``from_spec``.

``WIRE_LAYERS`` is the closed vocabulary and the security boundary: names resolve only against
this table, so a server can select which components the rig runs but can never make it execute
foreign code. Model sources are never wire-deliverable — they exist only server-side. Names are
stable and decoupled from import paths: new constructor arguments need defaults, and changing an
entry's meaning means a new name.
"""

import abc
import functools
import operator
from collections.abc import Callable
from typing import Any

from positronic.drivers.roboarm.models import EE_FRAME
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, JointDeltaAction
from positronic.policy.base import PAR, SEQ, Layer, Policy
from positronic.policy.codec import (
    ActionHorizon,
    ActionTimestamp,
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    RestrictImageSize,
)
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.observation import ObservationCodec


class RemoteMarker(Layer):
    """The client/server border in a policy pipeline. Only ever split on, never applied.

    Its arguments describe the wire rather than the policy, so they belong to the border itself:
    ``compress_images`` has the rig JPEG-encode frames before sending, for an endpoint behind a proxy
    with a message-size cap. The server declares them in its handshake and the rig obeys.

    ``remote`` is the plain border; call it to describe the wire::

        ChunkedSchedule() | remote(compress_images=True) | codec | source
    """

    def __init__(self, compress_images: bool = False):
        self.compress_images = compress_images

    def __call__(self, *, compress_images: bool = False) -> 'RemoteMarker':
        return RemoteMarker(compress_images=compress_images)

    def wrap(self, policy: Policy) -> Policy:
        raise TypeError('`remote` marks the client/server border of a pipeline; split() it instead of wrapping')


remote = RemoteMarker()


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
        if isinstance(left, Layer):
            return Pipeline(left._layers(), self)
        return NotImplemented


class Pipeline:
    """A policy pipeline closed by a model source: the full description of a policy server."""

    def __init__(self, components: tuple[Layer, ...], source: ModelSource):
        self.components = tuple(components)
        self.source = source
        # Which side of ``remote`` the conversion sits on is a choice; doing it on both sides is not. Two
        # declarations convert twice, leaving poses at the product while the handshake still names one of them.
        declared = [c for c in self.components if isinstance(c, Codec) and EE_FRAME in c.meta]
        if len(declared) > 1:
            raise ValueError(
                f'{len(declared)} components of this pipeline declare {EE_FRAME}; each converts, so poses '
                'end up at the product of their transforms. Declare the frame once.'
            )


class PolicySource(ModelSource):
    """Serves one ready in-process Policy — for in-process pipelines and tests."""

    def __init__(self, policy: Policy, name: str = 'default'):
        self._policy = policy
        self._name = name

    def get_models(self) -> list[str]:
        return [self._name]

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        return self._policy


WIRE_LAYERS: dict[str, type[Layer]] = {
    layer.WIRE_NAME: layer
    for layer in (
        ChunkedSchedule,
        StopOnFault,
        TemporalStack,
        ActionTimestamp,
        ActionHorizon,
        BinarizeGripTraining,
        BinarizeGripInference,
        FlipGrip,
        RestrictImageSize,
        ChangeEEFrame,
        ObservationCodec,
        AbsolutePositionAction,
        AbsoluteJointsAction,
        JointDeltaAction,
    )
}


def _join(components: tuple) -> Layer | None:
    return functools.reduce(operator.or_, components) if components else None


def split(pipeline: Pipeline | Layer) -> tuple[Layer | None, RemoteMarker, Layer | None]:
    """Split a pipeline's layer chain at the ``remote`` marker into ``(local, border, remote)``.

    An empty half is ``None``; a chain of just the marker means "no glue on either side". The border
    carries the wire's own settings for the server to declare.
    """
    components = pipeline.components if isinstance(pipeline, Pipeline) else pipeline._layers()
    markers = [(i, c) for i, c in enumerate(components) if isinstance(c, RemoteMarker)]
    if len(markers) != 1:
        raise ValueError(f'A policy pipeline needs exactly one `remote` marker, found {len(markers)}')
    idx, border = markers[0]
    return _join(components[:idx]), border, _join(components[idx + 1 :])


def inline(pipeline: Pipeline) -> Policy:
    """The whole pipeline in one process: components (marker dropped) wrapped around the source's latest model."""
    components = tuple(c for c in pipeline.components if not isinstance(c, RemoteMarker))
    policy = pipeline.source.load(pipeline.source.resolve(None))
    joined = _join(components)
    return joined.wrap(policy) if joined is not None else policy


def from_spec(node: dict[str, Any]) -> Layer | None:
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
    if name not in WIRE_LAYERS:
        raise ValueError(f'Unknown local-stack entry {name!r}; this build knows {sorted(WIRE_LAYERS)}')
    return WIRE_LAYERS[name](**node.get('args', {}))
