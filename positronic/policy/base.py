"""Policy API Design requirements

Policies and sessions:
- A `Policy` is the algorithm that controls a robot from observed data.
- A `Policy` makes one `Session` per control session.
- Sessions are independent, and several may exist at the same moment.
- A `Policy` is told which robot it is to control, so the session comes ready for it.
- A session is cancellable from outside at any moment, and lives until it is cancelled.

The control loop:
- One pass: `(observations, time) -> (commands, due)` — the current observations and time in, the
  commands to execute now (possibly none) out.
- The `time` argument is the only clock a session has.
- Observations and commands may carry domain types (a robot command, an image).
- Returned commands are emitted towards the robot driver in the same pass.
- `due` is when the session wants control back: an absolute instant on the clock of `time`, strictly in
  the future.
- The framework returns control best-effort at `due`: it may be earlier or later.

Pure functions:
- A policy may define pure functions that are available to sessions via the framework.
- Invoking one starts the work off the control pass and returns a handle at once.
- The session may read the handle when it next has control.
- A function's inputs and outputs are types the framework can serialize: plain types and numpy, selected domain classes.
  An unsupported type fails at the call.
- A session computes only while in control or inside a served call.
- The framework makes no stateful guarantees: a function may cache, but dropping the cache must not
  change what it computes.

Composability:
- A session is a chain of layers; the innermost answers on its own, with or without pure functions.
- Every link of the chain speaks the session protocol.
- Layers are the toolbox: a serving algorithm is built by chaining them, without touching the framework.
- A chain of layers is a layer.
- Layers communicate only through the observations and commands flowing through them.
- A layer knows nothing of its neighbours or its position.
- What an inner layer sees is its outer's choice.
- Chain order fully determines behavior.
- A `Codec` transforms data at a link: a pure function's inputs and outputs, or a layer's observations
  down and commands up. It never touches `due`.

Logging:
- The framework records the top-level exchange — observations in, commands out — on its own.
- A layer can log named data of its own; the framework records it with the control session, on the same
  clock.
- Logging never affects control: it adds no waiting and no failure path to the loop.

Remote policies:
- The framework natively supports remote policies: the whole policy lives on a server.
- The server describes the full policy: the pure functions it serves and the stack around them.
- One URL is enough to fully specify the policy, given that the server returns a proper declaration.
- Declarations are backward compatible: the framework evolves without changing what an already-served
  policy does.
- A declaration the rig cannot honor is refused at the handshake.

TODO: the Python shape of the pass — an object protocol, with generators as a supported authoring style
  the framework adapts.
TODO: how a layer logs — a handle given at construction, or part of the return value.
TODO: the shape of the robot description, and a server's ability to refuse one.
TODO: the protocol delivering the declared stack to the rig — versioning and compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

Now = Callable[[], float]


# Structural keys of the wire spec: ``|`` serializes as ``{SEQ: [...]}``, ``&`` as ``{PAR: [...]}``.
SEQ = 'seq'
PAR = 'par'


class Session(ABC):
    """Per-episode inference session. Created by ``Policy.new_session()``.

    Sessions hold per-episode state (trajectory buffers, latency tracking, etc.)
    and are the primary interface for running inference. Call the session like
    a function to get actions::

        session = policy.new_session(context)
        trajectory = session(obs)

    **Plain-data contract**: sessions accept and return only plain data
    (dicts, lists, numpy arrays, scalars). No tensors or custom objects.

    **Return contract**: ``list[dict] | None``. ``None`` means "no new
    trajectory, keep executing the current one" (used by scheduling wrappers).
    An empty list means "stop whatever is executing now". A non-empty list is
    a new trajectory. Single-action returns must be wrapped into a 1-element
    list by the producer.
    """

    @abstractmethod
    def __call__(self, obs: Mapping[str, Any]) -> list[dict[str, Any]] | None:
        """Predict actions for the given observation."""

    @property
    def meta(self) -> dict[str, Any]:
        """Session metadata (may include policy meta + per-session info)."""
        return {}

    def cancel(self):
        """Drop any in-flight trajectory state. Wrappers that buffer/schedule a
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

    def __call__(self, obs):
        return self._inner(obs)

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
    def new_session(self, context: dict[str, Any] | None = None, now: Now | None = None) -> Session:
        """Create a new inference session for an episode.

        Args:
            context: The episode's task description.
            now: The runtime clock (current time in seconds), supplied by the harness and passed down
                to every wrapped session. ``None`` where no runtime clock exists (server-side, warmup).
        """

    @property
    def meta(self) -> dict[str, Any]:
        """Static metadata about this policy/model."""
        return {}

    def close(self):  # noqa: B027
        """Release shared resources (model weights, connections, etc.)."""


class DelegatingPolicy(Policy):
    """Policy that delegates all methods to an inner policy. Subclass and override what you need."""

    def __init__(self, inner: Policy):
        self._inner = inner

    def new_session(self, context=None, now=None):
        return self._inner.new_session(context, now)

    @property
    def meta(self):
        return self._inner.meta

    def close(self):
        self._inner.close()


class PolicyWrapper:
    """Composable wrapper recipe — created without an inner policy, applied via ``wrap()``.

    PolicyWrappers may be stateful, may control flow (skip the inner call),
    and have no training-time dual. They compose with ``|`` (sequential, left
    is outermost). Unlike Codecs, they do NOT support ``&`` (parallel).

    ``|`` works across types: ``wrapper | wrapper``, ``wrapper | codec``,
    and ``codec | wrapper`` all produce a PolicyWrapper pipeline that
    ``wrap(policy)`` applies right-to-left::

        pipeline = TemporalStack(...) | ChunkedSchedule() | codec
        wrapped = pipeline.wrap(RemotePolicy(...))

    **Extension points**: subclasses override *one* of ``wrap_session`` (the
    common case — transform one session's ``__call__``) or ``wrap`` (for
    policy-level state across sessions, like composition).
    """

    def wrap(self, policy: Policy) -> Policy:
        """Apply this wrapper to a policy. Default: wrap every session it creates via ``wrap_session``.

        Composition happens at config time; the runtime clock reaches the wrapped
        sessions through ``new_session``.
        """
        return _WrapperPolicy(policy, self)

    def wrap_session(self, inner: Session, context: dict[str, Any] | None, now: Now | None) -> Session:
        """Wrap a single session. Subclasses override this for per-session wrapping."""
        raise NotImplementedError('Override wrap_session or wrap')

    # The name this wrapper travels under, set by every deliverable subclass. ``WIRE_WRAPPERS`` is keyed by
    # it, so the name is written once and both sides of the wire read the same attribute.
    WIRE_NAME: ClassVar[str]

    def to_spec(self) -> dict[str, Any]:
        """Plain-data wire spec of this wrapper, for a server's local-stack declaration.

        Only wrappers registered in ``positronic.policy.spec.WIRE_WRAPPERS`` are deliverable to a rig.
        The spec is ``{'name': WIRE_NAME}`` plus ``{'args': {...}}`` when the wrapper takes any;
        ``args`` are constructor keywords, since the rig rebuilds by calling the constructor with them.
        """
        raise NotImplementedError(f'{type(self).__name__} is not deliverable to a rig (no wire spec)')

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata contributed by this wrapper (merged into the wrapped policy's meta)."""
        return {}

    def __or__(self, other: PolicyWrapper) -> PolicyWrapper:
        if isinstance(other, PolicyWrapper):
            return _ComposedWrapper((*self._wrappers(), *other._wrappers()))
        return NotImplemented

    # Used for flattening nested | compositions into a single _ComposedWrapper
    def _wrappers(self) -> tuple:
        return (self,)


class _WrapperPolicy(DelegatingPolicy):
    """Generic policy wrapper produced by ``PolicyWrapper.wrap()``.

    Delegates session creation to the wrapper's ``wrap_session`` and merges meta.
    """

    def __init__(self, inner: Policy, wrapper: PolicyWrapper):
        super().__init__(inner)
        self._wrapper = wrapper

    def new_session(self, context=None, now=None):
        return self._wrapper.wrap_session(self._inner.new_session(context, now), context, now)

    @property
    def meta(self):
        return self._inner.meta | self._wrapper.meta


class _ComposedWrapper(PolicyWrapper):
    """Composed pipeline of wrappers and codecs. Applies right-to-left."""

    def __init__(self, components: tuple):
        self._components = components

    def wrap(self, policy: Policy) -> Policy:
        for component in reversed(self._components):
            policy = component.wrap(policy)
        return policy

    def to_spec(self) -> dict[str, Any]:
        return {SEQ: [component.to_spec() for component in self._components]}

    def _wrappers(self) -> tuple:
        return self._components
