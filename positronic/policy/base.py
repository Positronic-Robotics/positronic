from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

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
    A list is a new trajectory, replacing whatever is playing. Single-action
    returns must be wrapped into a 1-element list by the producer.
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


class LatencyMode(Enum):
    """When the platform lets a scheduling wrapper have the model's answer."""

    # At completion, and the world runs on meanwhile: the cost is whatever the model really took.
    LIVE = 'live'
    # A fixed delay after the call started, whatever the model really took. A delay of zero holds the world
    # still for the whole call.
    DECLARED = 'declared'
    # The call's own wall duration after it started, charged on the world clock.
    MEASURED = 'measured'


class InferenceGate:
    """The platform's hold on a scheduling wrapper's path to the model.

    Installed around the wrapper's inner session, so no wrapper can reach a result before the mode's cost
    has been paid. The wrapper resumes at the release instant and anchors there.
    """

    def __init__(self, now: Now, mode: LatencyMode, delay_sec: float = 0.0):
        self._now = now
        self._mode = mode
        self._delay_sec = delay_sec
        self._wall_t0 = 0.0
        self._cancelled = False
        # True while a call is inside the model — a wrapper that answered on its own never sets it.
        # ``t0`` is the world instant that call started, valid once ``entered``.
        self.t0 = 0.0
        self.entered = False

    def wrap(self, inner: Session) -> Session:
        return InferenceGate._Session(inner, self)

    def cancel(self) -> None:
        """Release a parked call, whose result is on its way to a harness that no longer wants it."""
        self._cancelled = True

    def hold(self) -> float | None:
        """Wall seconds the world must not advance for, or ``None`` to hold until the call completes."""
        match self._mode:
            case LatencyMode.LIVE:
                return 0.0
            case LatencyMode.DECLARED:
                return None if self._now() >= self.t0 + self._delay_sec else 0.0
            case LatencyMode.MEASURED:
                # The world may run no further ahead of the call's start than wall time has: measured
                # charging only means anything with the world at or below real time during the call.
                return max(0.0, (self._now() - self.t0) - (time.monotonic() - self._wall_t0))

    def _release_at(self) -> float:
        match self._mode:
            case LatencyMode.LIVE:
                return self.t0
            case LatencyMode.DECLARED:
                return self.t0 + self._delay_sec
            case LatencyMode.MEASURED:
                return self.t0 + (time.monotonic() - self._wall_t0)

    class _Session(DelegatingSession):
        """Charges the inner call, on whatever thread the harness dispatched it to."""

        def __init__(self, inner: Session, gate: InferenceGate):
            super().__init__(inner)
            self._gate = gate

        def __call__(self, obs):
            gate = self._gate
            gate.t0 = gate._now()
            gate._wall_t0 = time.monotonic()
            gate.entered = True
            result = self._inner(obs)
            release = gate._release_at()
            # The world clock is advanced by the harness's thread, so the park has to poll it; sleeping
            # zero hands over the GIL without adding a wake-up granularity to the release instant.
            while not gate._cancelled and gate._now() < release:
                time.sleep(0)
            gate.entered = False
            return result


class Policy(ABC):
    """Factory for inference sessions.

    A Policy holds shared resources (model weights, connections) and creates
    per-episode ``Session`` instances. One Policy can serve multiple robots
    by creating independent sessions.
    """

    @abstractmethod
    def new_session(
        self, context: dict[str, Any] | None = None, now: Now | None = None, gate: InferenceGate | None = None
    ) -> Session:
        """Create a new inference session for an episode.

        Args:
            context: Episode context (task description, eval metadata, etc.).
            now: The runtime clock (current time in seconds), supplied by the harness and passed down
                to every wrapped session. ``None`` where no runtime clock exists (server-side, warmup).
            gate: The platform's hold on the path to the model, supplied by the harness and installed
                around the inner session of every ``SchedulingWrapper`` in the stack. ``None`` where no
                runtime imposes inference cost (server-side, warmup).
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

    def new_session(self, context=None, now=None, gate=None):
        return self._inner.new_session(context, now, gate)

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

    def to_spec(self) -> dict[str, Any]:
        """Plain-data wire spec of this wrapper, for a server's local-stack declaration.

        Only wrappers registered in ``positronic.policy.spec.WIRE_WRAPPERS`` are deliverable to a rig.
        The spec is ``{'name': <wire name>}`` plus ``{'args': {...}}`` when the wrapper takes any;
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


class SchedulingWrapper(PolicyWrapper):
    """A wrapper that owns the plan: it decides when to call the model and returns the trajectory the
    harness plays, rather than one action for the moment.

    Being one is what earns the wrapper an ``InferenceGate`` around its inner session, so the inference
    cost is imposed below it instead of trusted to it.
    """


class _WrapperPolicy(DelegatingPolicy):
    """Generic policy wrapper produced by ``PolicyWrapper.wrap()``.

    Delegates session creation to the wrapper's ``wrap_session`` and merges meta.
    """

    def __init__(self, inner: Policy, wrapper: PolicyWrapper):
        super().__init__(inner)
        self._wrapper = wrapper

    def new_session(self, context=None, now=None, gate=None):
        inner = self._inner.new_session(context, now, gate)
        if gate is not None and isinstance(self._wrapper, SchedulingWrapper):
            inner = gate.wrap(inner)
        return self._wrapper.wrap_session(inner, context, now)

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
