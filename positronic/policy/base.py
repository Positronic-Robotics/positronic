import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from positronic.policy.action import ActionDecoder
from positronic.policy.observation import ObservationEncoder


@dataclass
class Codec:
    """Pair of observation encoder and action decoder."""

    observation: ObservationEncoder
    action: ActionDecoder


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


class DecodedEncodedPolicy(Policy):
    """A policy wrapper that optionally encodes observations and decodes actions.

    **Important**: `decoder(action, obs)` is called with original observation.
    """

    def __init__(
        self,
        policy: Policy,
        encoder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        decoder: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
        extra_meta=None,
        action_horizon_sec: float | None = None,
        action_fps: float | None = None,
    ):
        self._policy = policy
        self._encoder = encoder
        self._decoder = decoder
        self._extra_meta = extra_meta or {}
        self._action_horizon_sec = action_horizon_sec
        self._action_fps = action_fps

    def _encode(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._encoder:
            return self._encoder(obs)
        return obs

    def _decode(self, action: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
        if self._decoder:
            return self._decoder(action, obs)
        return action

    def select_action(self, obs: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        encoded_obs = self._encode(obs)
        action = self._policy.select_action(encoded_obs)
        if isinstance(action, list):
            if self._action_horizon_sec is not None and self._action_fps is not None:
                max_count = round(self._action_horizon_sec * self._action_fps)
                action = action[:max_count]
            # NOTE: Decoding happens relative to the original observation!
            decoded = [self._decode(a, obs) for a in action]
            if self._action_fps is not None:
                dt = 1.0 / self._action_fps
                for i, d in enumerate(decoded):
                    d['timestamp'] = i * dt
            return decoded
        decoded = self._decode(action, obs)
        if self._action_fps is not None:
            decoded['timestamp'] = 0.0
        return decoded

    def reset(self):
        self._policy.reset()

    @property
    def meta(self) -> dict[str, Any]:
        result = self._policy.meta | self._extra_meta
        if self._action_fps is not None:
            result['action_fps'] = self._action_fps
        if self._action_horizon_sec is not None:
            result['action_horizon_sec'] = self._action_horizon_sec
        return result

    def close(self):
        self._policy.close()


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
