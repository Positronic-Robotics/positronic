from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from positronic.policy.sampler import Sampler, UniformSampler


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

    def on_episode_complete(self):
        """Called when the episode using this session is finalized (successful stop).

        Distinct from ``close()``: fires only on successful episode completion
        (e.g. used by samplers to count completed runs), whereas ``close()``
        releases resources for any session end, successful or aborted.
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

    def on_episode_complete(self):
        self._inner.on_episode_complete()

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

    def wrap(self, policy: Policy) -> Policy:
        """Apply this wrapper to a policy. Default: wrap every session it creates via ``wrap_session``."""
        return _WrapperPolicy(policy, self)

    def wrap_session(self, inner: Session, context: dict[str, Any] | None) -> Session:
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

    def __init__(self, inner: Policy, wrapper: PolicyWrapper):
        super().__init__(inner)
        self._wrapper = wrapper

    def new_session(self, context=None):
        return self._wrapper.wrap_session(self._inner.new_session(context), context)

    @property
    def meta(self):
        return self._inner.meta | self._wrapper.meta


class _Pipeline(PolicyWrapper):
    """Composed pipeline of wrappers and codecs. Applies right-to-left."""

    def __init__(self, components: tuple):
        self._components = components

    def wrap(self, policy: Policy) -> Policy:
        for component in reversed(self._components):
            policy = component.wrap(policy)
        return policy

    def _pipeline_components(self) -> tuple:
        return self._components


class SampledPolicy(Policy):
    """Selects a sub-policy on each new_session using a pluggable sampling strategy."""

    def __init__(
        self,
        *policies: Policy,
        sampler: Sampler | None = None,
        weights: list[float] | None = None,
        key_field: str = 'server.checkpoint_path',
    ):
        self._policies = policies
        self._sampler = sampler
        self._weights = weights
        self._key_field = key_field
        self._keys: tuple[str, ...] | None = None

    class _SampledSession(DelegatingSession):
        """Session created by SampledPolicy — wraps an inner session + tracks sampling."""

        def __init__(self, inner: Session, key: str, sampler: Sampler, context: dict):
            super().__init__(inner)
            self._key = key
            self._sampler = sampler
            self._context = context

        def on_episode_complete(self):
            self._sampler.count(self._key, self._context)
            super().on_episode_complete()

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
        key = self.sampler.sample(keys, ctx)
        sub_policy = self._policies[keys.index(key)]
        return SampledPolicy._SampledSession(sub_policy.new_session(context), key, self.sampler, ctx)

    @property
    def meta(self):
        return {}
