from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from positronic.policy.sampler import EpisodeCounter, Sampler, UniformSampler

Now = Callable[[], float]


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
    def __call__(self, obs: dict[str, Any]) -> list[dict[str, Any]] | None:
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
    def new_session(self, context: dict[str, Any] | None = None) -> Session:
        """Create a new inference session for an episode.

        Args:
            context: Episode context (task description, eval metadata, etc.).
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

    def new_session(self, context=None):
        return self._inner.new_session(context)

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

        pipeline = ErrorRecovery() | ChunkedSchedule() | codec
        wrapped = pipeline.wrap(RemotePolicy(...))

    **Extension points**: subclasses override *one* of ``wrap_session`` (the
    common case — transform one session's ``__call__``) or ``wrap`` (for
    policy-level state across sessions, like composition).
    """

    def wrap(self, policy: Policy, now: Now | None = None) -> Policy:
        """Apply this wrapper to a policy. Default: wrap every session it creates via ``wrap_session``.

        ``now`` is the runtime clock (current time in seconds), supplied by the harness when it
        applies the pipeline. Wrappers whose sessions need wall time read it; codecs and recording
        taps, applied at config time, leave it ``None``.
        """
        return _WrapperPolicy(policy, self, now)

    def wrap_session(self, inner: Session, context: dict[str, Any] | None, now: Now | None) -> Session:
        """Wrap a single session. Subclasses override this for per-session wrapping."""
        raise NotImplementedError('Override wrap_session or wrap')

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata contributed by this wrapper (merged into the wrapped policy's meta)."""
        return {}

    def __or__(self, other: PolicyWrapper) -> PolicyWrapper:
        if isinstance(other, PolicyWrapper):
            return _Pipeline((*self._pipeline_components(), *other._pipeline_components()))
        return NotImplemented

    # Used for flattening nested | compositions into a single _Pipeline
    def _pipeline_components(self) -> tuple:
        return (self,)


class _WrapperPolicy(DelegatingPolicy):
    """Generic policy wrapper produced by ``PolicyWrapper.wrap()``.

    Delegates session creation to the wrapper's ``wrap_session`` and merges meta.
    """

    def __init__(self, inner: Policy, wrapper: PolicyWrapper, now: Now | None):
        super().__init__(inner)
        self._wrapper = wrapper
        self._now = now

    def new_session(self, context=None):
        return self._wrapper.wrap_session(self._inner.new_session(context), context, self._now)

    @property
    def meta(self):
        return self._inner.meta | self._wrapper.meta


class _Pipeline(PolicyWrapper):
    """Composed pipeline of wrappers and codecs. Applies right-to-left."""

    def __init__(self, components: tuple):
        self._components = components

    def wrap(self, policy: Policy, now: Now | None = None) -> Policy:
        for component in reversed(self._components):
            policy = component.wrap(policy, now)
        return policy

    def _pipeline_components(self) -> tuple:
        return self._components


class _KeyedSession(DelegatingSession):
    """Wraps the selected sub-policy's session so sampled episodes keep the selected policy's
    metadata. ``SampledPolicy.meta`` is ``{}``, so without this the selected policy's static
    fields (type, checkpoint, codec) would vanish from episode/handshake metadata when a
    sub-policy exposes them only via ``Policy.meta`` (with an empty ``Session.meta``). The
    sampler's chosen key is merged last so completion counting always records the exact key
    that was sampled, even if it differs from a per-session ``key_field``."""

    def __init__(self, inner: Session, policy_meta: dict[str, Any], key_field: str, key: str):
        super().__init__(inner)
        self._policy_meta = policy_meta
        self._key_field = key_field
        self._key = key

    @property
    def meta(self):
        return {**self._policy_meta, **self._inner.meta, self._key_field: self._key}


class SampledPolicy(Policy):
    """Selects a sub-policy on each new_session using a pluggable sampling strategy."""

    def __init__(
        self,
        *policies: Policy,
        sampler: Sampler | None = None,
        weights: list[float] | None = None,
        key_field: str = 'server.checkpoint_path',
        group_fields: list[str] | None = None,
    ):
        self._policies = policies
        self._sampler = sampler
        self._weights = weights
        self._key_field = key_field
        self._keys: tuple[str, ...] | None = None
        # The harness bumps this on each completed episode (via ``counter.record``);
        # the sampler reads it to balance. Seeded from prior recordings by the caller.
        self.counter = EpisodeCounter(key_field, group_fields)

    @property
    def sampler(self) -> Sampler | None:
        if self._sampler is None and self._keys is not None:
            weight_map = dict(zip(self._keys, self._weights, strict=True)) if self._weights else None
            self._sampler = UniformSampler(weight_map)
        return self._sampler

    def _get_keys(self) -> tuple[str, ...]:
        if self._keys is None:
            self._keys = tuple(p.meta.get(self._key_field, str(i)) for i, p in enumerate(self._policies))
        return self._keys

    def new_session(self, context=None):
        keys = self._get_keys()
        ctx = context or {}
        key = self.sampler.sample(keys, ctx, self.counter.counts(keys, ctx))
        policy = self._policies[keys.index(key)]
        session = policy.new_session(context)
        return _KeyedSession(session, policy.meta, self._key_field, key)

    @property
    def meta(self):
        return {}
