"""A client's overlap with the preceding prediction in the same session."""

import math
from dataclasses import dataclass, fields
from typing import Any

ACTION_PREFIX = 'action_prefix'


@dataclass
class ActionPrefix:
    """Steps to reuse from the preceding answer, sampled on the new prediction's time grid.

    ``sequence`` counts successful calls from zero within a session. ``positions`` indexes the
    preceding answer; fractional positions describe interpolation. ``actions`` carries the targets
    the client scheduled, in model units. The first ``frozen_steps`` must remain fixed; later steps
    may be revised. A first call, or an exhausted plan, has no positions or actions.
    """

    sequence: int
    positions: list[float]
    frozen_steps: int
    actions: list[dict[str, Any]]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('RTC sequence must be a non-negative integer')
        if not isinstance(self.positions, list) or any(
            type(position) not in (int, float) or not math.isfinite(position) or position < 0
            for position in self.positions
        ):
            raise ValueError('RTC positions must be finite non-negative numbers')
        if type(self.frozen_steps) is not int or not 0 <= self.frozen_steps <= len(self.positions):
            raise ValueError('RTC frozen_steps must lie within the overlap')
        if not isinstance(self.actions, list) or len(self.actions) != len(self.positions):
            raise ValueError('RTC positions and actions must have the same length')
        if any(not isinstance(action, dict) for action in self.actions):
            raise ValueError('RTC actions must be dictionaries')
        if self.sequence == 0 and self.positions:
            raise ValueError('The first RTC call cannot reuse an earlier prediction')

    def to_observation(self) -> dict[str, Any]:
        return {ACTION_PREFIX: {field.name: getattr(self, field.name) for field in fields(self)}}

    @classmethod
    def from_observation(cls, observation: dict[str, Any]) -> 'ActionPrefix':
        return cls(**observation[ACTION_PREFIX])
