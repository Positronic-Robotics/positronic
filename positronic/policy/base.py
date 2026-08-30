from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, ClassVar

# Structural keys of the wire spec: ``|`` serializes as ``{SEQ: [...]}``, ``&`` as ``{PAR: [...]}``.
SEQ = 'seq'
PAR = 'par'

# The name a policy's inference travels under. It takes the observation and answers a chunk: a list of
# actions, each naming command channels and carrying its own ``keys.ACTION_TIMESTAMP`` in seconds from the
# call. An empty list drops what is playing, and one action may come back bare.
INFER = 'infer'


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
    """What the framework offers one episode: its work, started off the loop thread.

    Closed before the work it serves is released: a call still in flight is using what that work holds.
    """

    @property
    @abstractmethod
    def fns(self) -> Mapping[str, Fn]:
        """The episode's work, under the names the policy declared it by."""


class Session(ABC):
    """Per-episode control session, created by ``Policy.new_session``.

    A session holds what one episode plays — the chunk in flight, the history a stack samples — and is
    what a caller drives a robot through::

        session = policy.new_session(rt)
        commands, resume_at_ns = session(obs, time_ns)

    **Plain-data contract**: sessions accept and return only plain data (dicts, lists, numpy arrays,
    scalars). No tensors or custom objects.
    """

    @abstractmethod
    def __call__(self, obs: Mapping[str, Any], time_ns: int) -> tuple[Mapping[str, Any], int]:
        """The commands to run now and the time this session wants its next call, without waiting:
        heavy work belongs in ``Policy.episode``.

        ``time_ns`` is the caller's clock reading in nanoseconds. A session reads no clock of its own.
        ``commands`` names a command channel per entry; an empty mapping asks for nothing this call.
        ``resume_at_ns`` is after ``time_ns``, on the same clock. The caller aims at it and may call
        earlier. A session that waits for work it cannot time names a poll period of its own.
        """

    def cancel(self):
        """Drop what is in flight, so the next call plans afresh. Propagate with ``super().cancel()``."""
        return None

    def close(self):
        """End this session and release what it holds."""
        return None


class DelegatingSession(Session):
    """Session that delegates all methods to an inner ``Session``. Subclass and override what you need."""

    def __init__(self, inner: Session):
        self._inner = inner

    def __call__(self, obs, time_ns):
        return self._inner(obs, time_ns)

    def cancel(self):
        self._inner.cancel()

    def close(self):
        self._inner.close()


class Policy:
    """Factory for episodes.

    A Policy holds what every episode shares — model weights, a connection, a subprocess — and opens the
    per-episode work each one runs. One Policy serves several robots through independent episodes. It
    declares the work of an episode, the session that plays one, or both.
    """

    @contextmanager
    def episode(self, context: dict[str, Any] | None = None) -> Iterator[Mapping[str, Callable[..., Any]]]:
        """The work of one episode, by name, and what it holds for as long as the episode runs.

        The framework serves the work as ``Runtime.fns`` and closes it after the episode. A policy that
        answers a chunk declares that work under ``INFER``. ``context`` is the episode's task description.
        """
        yield {}

    def new_session(self, rt: Runtime) -> Session:
        """The session that plays this episode, over the work ``rt`` serves."""
        raise NotImplementedError(f'{type(self).__name__} answers a chunk; put a ChunkPlayer above it to play one')

    @property
    def meta(self) -> dict[str, Any]:
        """Static metadata about this policy/model."""
        return {}

    def close(self):
        """Release shared resources (model weights, connections, etc.)."""


class DelegatingPolicy(Policy):
    """Policy that delegates all methods to an inner policy. Subclass and override what you need."""

    def __init__(self, inner: Policy):
        self._inner = inner

    def episode(self, context=None):
        return self._inner.episode(context)

    def new_session(self, rt):
        return self._inner.new_session(rt)

    @property
    def meta(self):
        return self._inner.meta

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

        pipeline = TemporalStack(...) | ChunkPlayer() | codec
        wrapped = pipeline.wrap(RemotePolicy(...))

    **Extension points**: subclasses override *one* of ``make_session`` (the
    common case — transform one session's ``__call__``) or ``wrap`` (for
    policy-level state across sessions, like composition). A layer that sits under a ``ChunkPlayer``
    subclasses ``ChunkLayer`` instead: there is no session there, only the work ``INFER`` names.
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

    # Whether this layer turns the chunk work below it into a session. A ``ChunkLayer`` wraps that work, so
    # a composition puts every one of them under the layer that plays it.
    PLAYS_CHUNKS: ClassVar[bool] = False

    def to_spec(self) -> dict[str, Any]:
        """Plain-data wire spec of this layer, for a server's local-stack declaration.

        Only layers registered in ``positronic.policy.spec.WIRE_LAYERS`` are deliverable to a rig.
        The spec is ``{'name': WIRE_NAME}`` plus ``{'args': {...}}`` when the layer takes any;
        ``args`` are constructor keywords, since the rig rebuilds by calling the constructor with them.
        """
        raise NotImplementedError(f'{type(self).__name__} is not deliverable to a rig (no wire spec)')

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata contributed by this layer (merged into the wrapped policy's meta)."""
        return {}

    def __or__(self, other: Layer) -> Layer:
        if isinstance(other, Layer):
            return _ComposedLayer((*self._layers(), *other._layers()))
        return NotImplemented

    # Used for flattening nested | compositions into a single _ComposedLayer
    def _layers(self) -> tuple:
        return (self,)


class ChunkLayer(Layer):
    """Layer under a ``ChunkPlayer``: it wraps the work ``INFER`` names, for one episode.

    Below the player there is no session, so a chunk layer has nothing to cancel and no round to pace. It
    sees the observation on the way down and the chunk on the way up.
    """

    def wrap(self, policy: Policy) -> Policy:
        return _ChunkLayerPolicy(policy, self)

    @contextmanager
    def episode_fn(self, infer: Callable[..., Any]) -> Iterator[Callable[..., Any]]:
        """``infer`` with this layer's work around it, and what that takes for as long as the episode runs."""
        raise NotImplementedError('Override episode_fn or wrap')


class _WrappedPolicy(DelegatingPolicy):
    """Policy produced by ``Layer.wrap()``: the layer's own metadata joins what it wraps."""

    def __init__(self, inner: Policy, layer: Layer):
        super().__init__(inner)
        self._layer = layer

    @property
    def meta(self):
        return self._inner.meta | self._layer.meta


class _LayerPolicy(_WrappedPolicy):
    """Every session it creates goes through the layer's ``make_session``."""

    def new_session(self, rt):
        return self._layer.make_session(self._inner.new_session(rt))


class _ChunkLayerPolicy(_WrappedPolicy):
    """Its episode serves the layer's ``INFER`` in place of the one below it."""

    _layer: ChunkLayer

    @contextmanager
    def episode(self, context=None):
        with self._inner.episode(context) as fns, self._layer.episode_fn(fns[INFER]) as infer:
            yield {**fns, INFER: infer}


class _ComposedLayer(Layer):
    """Composed pipeline of layers. Applies right-to-left."""

    def __init__(self, components: tuple):
        self._components = components

    def wrap(self, policy: Policy) -> Policy:
        played = False
        for component in reversed(self._components):
            assert not (played and isinstance(component, ChunkLayer)), (
                f'compose {type(component).__name__} under the layer that plays the chunk, not above it'
            )
            played = played or component.PLAYS_CHUNKS
            policy = component.wrap(policy)
        return policy

    def to_spec(self) -> dict[str, Any]:
        return {SEQ: [component.to_spec() for component in self._components]}

    def _layers(self) -> tuple:
        return self._components
