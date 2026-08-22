"""Policies, sessions, and the wrappers that compose them. The target design is in DESIGN.md."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

# Structural keys of the wire spec: ``|`` serializes as ``{SEQ: [...]}``, ``&`` as ``{PAR: [...]}``.
SEQ = 'seq'
PAR = 'par'


class NotAnswered(RuntimeError):
    """The call has not answered yet. A read of an ``Answer`` raises this rather than waiting."""


class Answer(ABC):
    """The caller's handle on one call.

    pimm has an ``Answer`` of the same shape. The two are not interchangeable.
    """

    @abstractmethod
    def done(self) -> bool: ...

    @abstractmethod
    def result(self) -> Any:
        """What the function returned. Raises what the function raised, or ``NotAnswered`` before it answers."""


# A call to an ``Fn`` starts the work and returns at once.
Fn = Callable[..., Answer]


class Runtime(ABC):
    """What the framework offers one session. Every session gets its own.

    Closed before the session it serves: a call still in flight is using what that session holds.
    """

    @property
    @abstractmethod
    def fns(self) -> Mapping[str, Fn]:
        """The policy's functions, under the names it declared them by."""


class Session(ABC):
    """Per-episode inference session. Created by ``Policy.new_session()``.

    Sessions hold per-episode state (trajectory buffers, latency tracking, etc.)
    and are the primary interface for running inference. Call the session like
    a function to get actions::

        session = policy.new_session(context)
        trajectory = session(obs, time_ns)

    **Plain-data contract**: sessions accept and return only plain data
    (dicts, lists, numpy arrays, scalars). No tensors or custom objects.

    **Return contract**: ``list[dict] | None``. ``None`` means "no new
    trajectory, keep executing the current one" — what a scheduling layer
    answers while its chunk plays, and what a session answers while the
    function it asked is still in flight.
    An empty list means "stop whatever is executing now". A non-empty list is
    a new trajectory. Single-action returns must be wrapped into a 1-element
    list by the producer.
    """

    @abstractmethod
    def __call__(self, obs: Mapping[str, Any], time_ns: int) -> list[dict[str, Any]] | None:
        """Predict actions for the given observation, without waiting: heavy work belongs in
        ``Policy.functions``.

        ``time_ns`` is the caller's clock reading in nanoseconds. A session reads no clock of its own.
        """

    @property
    def meta(self) -> dict[str, Any]:
        """What this session reports about its model and its episode."""
        return {}

    def cancel(self):
        """Drop any in-flight trajectory state. Layers that buffer/schedule a
        trajectory (e.g. ``ChunkedSchedule``) should reset so the next call
        triggers a fresh inference. Override and propagate via ``super().cancel()``.
        """
        return None

    def close(self):
        """End this session and release per-episode resources."""
        return None


class DelegatingSession(Session):
    """Session that delegates all methods to an inner session. Subclass and override what you need."""

    def __init__(self, inner: Session):
        self._inner = inner

    def __call__(self, obs, time_ns):
        return self._inner(obs, time_ns)

    @property
    def meta(self):
        return self._inner.meta

    def cancel(self):
        self._inner.cancel()

    def close(self):
        self._inner.close()


class Policy(ABC):
    """Factory for inference sessions.

    A Policy holds shared resources (model weights, connections) and creates
    per-episode ``Session`` instances. One Policy can serve multiple robots
    by creating independent sessions.
    """

    @abstractmethod
    def new_session(self, context: dict[str, Any] | None = None, rt: Runtime | None = None) -> Session:
        """Create a new inference session for an episode.

        Args:
            context: The episode's task description.
            rt: This session's runtime, serving ``functions``. ``None`` only where no caller supplied one.
                A session that needs one refuses to open without it.
        """

    @property
    def functions(self) -> Mapping[str, Callable[..., Any]]:
        """The work this policy runs off the session's thread, by name. The framework serves it as ``rt.fns``."""
        return {}

    def close(self):  # noqa: B027
        """Release shared resources (model weights, connections, etc.)."""


class DelegatingPolicy(Policy):
    """Policy that delegates all methods to an inner policy. Subclass and override what you need."""

    def __init__(self, inner: Policy):
        self._inner = inner

    def new_session(self, context=None, rt=None):
        return self._inner.new_session(context, rt)

    @property
    def functions(self):
        return self._inner.functions

    def close(self):
        self._inner.close()


class Layer:
    """Recipe for wrapping a session, fixed at configuration time and applied to a policy via ``wrap()``.

    Layers may be stateful, may control flow (skip the inner call), and have no
    training-time dual. They compose with ``|`` (sequential, left is outermost).
    Unlike Codecs, they do NOT support ``&`` (parallel).

    ``|`` works across types: ``layer | layer``, ``layer | codec``, and
    ``codec | layer`` all produce a Layer pipeline that ``wrap(policy)``
    applies right-to-left::

        pipeline = TemporalStack(...) | ChunkedSchedule() | codec
        wrapped = pipeline.wrap(RemotePolicy(...))

    **Extension points**: subclasses override *one* of ``make_session`` (the
    common case — transform one session's ``__call__``) or ``wrap`` (for
    policy-level state across sessions, like composition).
    """

    def wrap(self, policy: Policy) -> Policy:
        """Apply this layer to a policy. Default: wrap every session it creates via ``make_session``."""
        return _LayerPolicy(policy, self)

    def make_session(self, inner: Session) -> Session:
        """Make this layer's session around ``inner``."""
        raise NotImplementedError('Override make_session or wrap')

    # The name this layer travels under, set by every deliverable subclass. ``WIRE_LAYERS`` is keyed by
    # it, so the name is written once and both sides of the wire read the same attribute.
    WIRE_NAME: ClassVar[str]

    def to_spec(self) -> dict[str, Any]:
        """Plain-data wire spec of this layer, for a server's local-stack declaration.

        Only layers registered in ``positronic.policy.spec.WIRE_LAYERS`` are deliverable to a rig.
        The spec is ``{'name': WIRE_NAME}`` plus ``{'args': {...}}`` when the layer takes any;
        ``args`` are constructor keywords, since the rig rebuilds by calling the constructor with them.
        """
        raise NotImplementedError(f'{type(self).__name__} is not deliverable to a rig (no wire spec)')

    def __or__(self, other: Layer) -> Layer:
        if isinstance(other, Layer):
            return _ComposedLayer((*self._layers(), *other._layers()))
        return NotImplemented

    # Used for flattening nested | compositions into a single _ComposedLayer
    def _layers(self) -> tuple:
        return (self,)


class _LayerPolicy(DelegatingPolicy):
    """Policy produced by ``Layer.wrap()``."""

    def __init__(self, inner: Policy, layer: Layer):
        super().__init__(inner)
        self._layer = layer

    def new_session(self, context=None, rt=None):
        return self._layer.make_session(self._inner.new_session(context, rt))


class _ComposedLayer(Layer):
    """Composed pipeline of layers. Applies right-to-left."""

    def __init__(self, components: tuple):
        self._components = components

    def wrap(self, policy: Policy) -> Policy:
        for component in reversed(self._components):
            policy = component.wrap(policy)
        return policy

    def to_spec(self) -> dict[str, Any]:
        return {SEQ: [component.to_spec() for component in self._components]}

    def _layers(self) -> tuple:
        return self._components
