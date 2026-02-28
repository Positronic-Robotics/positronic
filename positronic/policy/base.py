import random
from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """Abstract base class for all policies."""

    @abstractmethod
    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
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

    # TODO: Add `dummy_input() -> dict | None` method here. Each vendor Policy subclass
    # (Gr00tPolicy, OpenpiPolicy, DreamZeroPolicy) knows its model's expected input format
    # and can provide a zero-filled dummy for warmup. This removes the need for
    # `Codec.dummy_encoded()` and lets servers warm up without requiring a codec.

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
        index = random.choices(range(len(self._policies)), self._weights)[0]
        return self._policies[index]

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        return self._current_policy.select_action(obs)

    def reset(self):
        """Resets the policy and selects a new active sub-policy."""
        self._current_policy = self._select_policy()
        self._current_policy.reset()

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the metadata of the currently active sub-policy."""
        return self._current_policy.meta
