"""The takeover of a rollout with a helper: the press fields of the helper switch the driver of the rig.

The policy drives from the start. Each element of the helper holds its commands and the press fields. A press field
holds the time of the last press, or ``None`` before the first press, so a new value is a new press::

    {'robot_command': CartesianPosition(...), 'take_over': 1_200_000_000, 'hand_back': None}

Take over gives the rig to the helper, and Hand back gives it back to the policy, in the step of the press.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_serializer

from positronic.drivers.roboarm.command import CartesianDelta, JointDelta
from positronic.policy.base import Commands
from positronic.policy.codecs import PolicyCommandError
from positronic.policy.keys import HELPER_FIELDS, HELPER_HAND_BACK, HELPER_TAKE_OVER, POLICY_SOURCE, TELEOPERATOR
from positronic.policy.record import Event, EventKind, StepRecord


class ControlObservation(BaseModel):
    """The ``CONTROL`` observation of the helper: the driver whose commands pass to the rig, and the last event."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    source: str
    event: Event | None

    @field_serializer('event')
    def _event(self, event: Event | None) -> dict[str, Any] | None:
        return None if event is None else event.model_dump(mode='json', by_alias=True, exclude_none=True)


def helper_commands(element: Mapping[str, Any]) -> Commands:
    """The commands of ``element``, an element of the helper: each field but the press fields."""
    return {name: value for name, value in element.items() if name not in HELPER_FIELDS}


class Driver(StrEnum):
    """The top-level source whose commands pass to the rig."""

    POLICY = POLICY_SOURCE
    TELEOPERATOR = TELEOPERATOR


class _Presses(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    take_over: StrictInt | None = Field(default=None, alias=HELPER_TAKE_OVER)
    hand_back: StrictInt | None = Field(default=None, alias=HELPER_HAND_BACK)


# The driver that each press gives the rig to.
_PRESS_TO = {HELPER_TAKE_OVER: Driver.TELEOPERATOR, HELPER_HAND_BACK: Driver.POLICY}
# The commands that move the arm once from where it is.
_DELTAS = (CartesianDelta, JointDelta)


class Takeover:
    """The driver of one episode with a helper. The presses of the first element of the helper are not new: they
    came before the episode. When the first element has no press field, each later press is new. An element with no
    press field keeps the last presses. When both presses are new, the later press wins, and at one time Hand back
    wins. A step with no element repeats the absolute targets of the newest element, and not its deltas, which move
    the arm once."""

    def __init__(self):
        self._driver = Driver.POLICY
        self._presses: _Presses | None = None
        self._event: Event | None = None
        self._held: Commands = {}

    def control(self) -> dict[str, Any]:
        """The ``CONTROL`` observation of the helper, as plain data."""
        return ControlObservation(source=self._driver, event=self._event).model_dump(mode='json')

    @staticmethod
    def _parse_presses(element: Mapping[str, Any]) -> _Presses | None:
        """The press fields of ``element``, or ``None`` when it has none."""
        if not (fields := {name: value for name, value in element.items() if name in HELPER_FIELDS}):
            return None
        try:
            return _Presses.model_validate(fields)
        except ValidationError as e:
            raise PolicyCommandError(f'The press fields of an element of the helper are not well formed: {e}') from e

    @staticmethod
    def _new_press(presses: _Presses, last: _Presses) -> str | None:
        """The name of the newest press in ``presses`` that ``last`` does not have, or ``None``."""
        new: dict[str, int] = {}
        if presses.take_over is not None and presses.take_over != last.take_over:
            new[HELPER_TAKE_OVER] = presses.take_over
        if presses.hand_back is not None and presses.hand_back != last.hand_back:
            new[HELPER_HAND_BACK] = presses.hand_back
        return max(new, key=lambda name: (new[name], name == HELPER_HAND_BACK), default=None)

    def step(self, policy: Commands, element: Mapping[str, Any]) -> tuple[Commands, StepRecord]:
        """The commands that the rig gets and the record of the step, for the commands of the policy and the element
        of the helper. While the helper drives, the record keeps the commands of the policy as its shadow actions."""
        events: tuple[Event, ...] = ()
        if element:
            commands = helper_commands(element)
            self._held = {name: value for name, value in commands.items() if not isinstance(value, _DELTAS)}
        else:
            commands = self._held
        presses = self._parse_presses(element)
        if self._presses is None:
            self._presses = presses if presses is not None else _Presses()
        elif presses is not None:
            press = self._new_press(presses, self._presses)
            if press is not None and _PRESS_TO[press] is not self._driver:
                self._event = Event(kind=EventKind.SWITCH, from_=self._driver, to=_PRESS_TO[press], cause=press)
                self._driver = _PRESS_TO[press]
                events = (self._event,)
            self._presses = presses
        match self._driver:
            case Driver.TELEOPERATOR:
                shadow: dict[str, Commands] = {Driver.POLICY: policy} if policy else {}
                return commands, StepRecord(self._driver, frozenset({self._driver}), events, shadow)
            case Driver.POLICY:
                return policy, StepRecord(self._driver, frozenset({self._driver}), events)
