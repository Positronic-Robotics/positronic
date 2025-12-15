import random
from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """Abstract base class for all policies."""

    @abstractmethod
    def select_action(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Computes an action for the given observation."""
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
