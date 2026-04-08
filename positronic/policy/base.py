from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from positronic.policy.sampler import Sampler, UniformSampler

if TYPE_CHECKING:
    from positronic.policy.codec import Codec


class Session(ABC):
    """Per-episode inference session. Created by ``Policy.new_session()``.

    Sessions hold per-episode state (trajectory buffers, latency tracking, etc.)
    and are the primary interface for running inference. Call the session like
    a function to get actions::

        session = policy.new_session(context)
        trajectory = session(obs)

    **Plain-data contract**: sessions accept and return only plain data
    (dicts, lists, numpy arrays, scalars). No tensors or custom objects.
    """

    @abstractmethod
    def __call__(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Predict actions for the given observation.

        Returns a single action dict, a trajectory (list of dicts), or None
        to signal "keep executing the current trajectory" (used by scheduling wrappers).
        """

    @property
    def meta(self) -> dict[str, Any]:
        """Session metadata (may include policy meta + per-session info)."""
        return {}

    def on_episode_complete(self):
        """Called when the episode using this session is finalized."""
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


class PolicyWrapper(ABC):
    """Composable wrapper recipe — created without an inner policy, applied via ``wrap()``.

    PolicyWrappers may be stateful, may control flow (skip the inner call),
    and have no training-time dual. They compose with ``|`` (sequential, left
    is outermost). Unlike Codecs, they do NOT support ``&`` (parallel).

    ``|`` works across types: ``wrapper | wrapper``, ``wrapper | codec``,
    and ``codec | wrapper`` all produce a PolicyWrapper pipeline that
    ``wrap(policy)`` applies right-to-left::

        pipeline = ErrorRecovery() | ChunkedSchedule() | codec
        wrapped = pipeline.wrap(RemotePolicy(...))
    """

    @abstractmethod
    def wrap(self, policy: Policy) -> Policy:
        """Apply this wrapper to a concrete policy, returning a wrapped Policy."""

    def __or__(self, other: PolicyWrapper | Codec) -> PolicyWrapper:
        from positronic.policy.codec import Codec  # noqa — circular dep

        mine = self._pipeline_components()
        if isinstance(other, PolicyWrapper):
            return _Pipeline((*mine, *other._pipeline_components()))
        if isinstance(other, Codec):
            return _Pipeline((*mine, other))
        return NotImplemented

    def __ror__(self, other: Codec) -> PolicyWrapper:
        from positronic.policy.codec import Codec  # noqa — circular dep

        if isinstance(other, Codec):
            return _Pipeline((other, *self._pipeline_components()))
        return NotImplemented

    def _pipeline_components(self) -> tuple:
        return (self,)


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
        return _SampledSession(sub_policy.new_session(context), key, self.sampler, ctx)

    @property
    def meta(self):
        return {}
