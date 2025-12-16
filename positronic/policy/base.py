import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class Policy(ABC):
    """Abstract base class for all policies."""

    @abstractmethod
    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Computes an action for the given observation.

        **Plain-data contract**
        Policies should accept and return only "plain" data structures:
        - built-in scalars: `str`, `int`, `float`, `bool`, `None`
        - containers: `dict` / `list` / `tuple` recursively composed of supported values
        - numeric numpy values: `numpy.ndarray` and `numpy` scalar types

        Avoid returning arbitrary Python objects (custom classes, sockets, file handles, etc.).
        Keeping inputs/outputs as plain data makes policies easy to compose (wrappers/ensemblers),
        record/replay, and run in different execution contexts.
        """
        pass

    def reset(self):
        """Resets the policy state."""
        return None

    @property
    def meta(self) -> dict[str, Any]:
        """Returns metadata about the policy configuration."""
        return {}

    def close(self):
        """Closes the policy and releases any resources."""
        return None


class EncodedPolicy(Policy):
    """A policy that encodes the observation before selecting an action."""

    def __init__(self, policy: Policy, encoder: Callable[[dict[str, Any]], dict[str, Any]], extra_meta=None):
        self._policy = policy
        self._encoder = encoder
        self._extra_meta = extra_meta or {}

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        encoded_obs = self._encoder(obs)
        return self._policy.select_action(encoded_obs)

    def reset(self):
        self._policy.reset()

    @property
    def meta(self) -> dict[str, Any]:
        return self._policy.meta | self._extra_meta

    def close(self):
        self._policy.close()


class DecodedPolicy(Policy):
    """A policy that decodes the action after selecting it from the policy."""

    def __init__(
        self, policy: Policy, decoder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]], extra_meta=None
    ):
        self._policy = policy
        self._decoder = decoder
        self._extra_meta = extra_meta or {}

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        action = self._policy.select_action(obs)
        return self._decoder(action, obs)

    def reset(self):
        self._policy.reset()

    @property
    def meta(self) -> dict[str, Any]:
        return self._policy.meta | self._extra_meta

    def close(self):
        self._policy.close()


class SampledPolicy(Policy):
    """Randomly selects a policy from a list on each reset."""

    def __init__(self, *policies: Policy, weights: list[float] | None = None):
        self._policies = policies
        self._weights = weights
        self._current_policy = self._select_policy()

    def _select_policy(self) -> Policy:
        return random.choices(self._policies, self._weights)[0]

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._current_policy.select_action(obs)

    def reset(self):
        """Resets the policy and selects a new active sub-policy."""
        self._current_policy = self._select_policy()
        self._current_policy.reset()

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the metadata of the currently active sub-policy."""
        return self._current_policy.meta
