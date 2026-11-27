"""The takeover of a rollout with a helper: the press fields of the helper switch the driver of the rig.

The policy drives from the start. Each element of the helper holds its commands and the press fields. A press field
holds the time of the last press, or ``None`` before the first press, so a new value is a new press::

    {'robot_command': CartesianPosition(...), 'take_over': 1_200_000_000, 'hand_back': None}

Take over gives the rig to the helper, and Hand back gives it back to the policy, in the step of the press.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_serializer

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


class _Presses(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    take_over: StrictInt | None = Field(None, alias=HELPER_TAKE_OVER)
    hand_back: StrictInt | None = Field(None, alias=HELPER_HAND_BACK)


def _presses(element: Mapping[str, Any]) -> _Presses | None:
    """The press fields of ``element``, or ``None`` when it has none."""
    if not (fields := {name: value for name, value in element.items() if name in HELPER_FIELDS}):
        return None
    try:
        return _Presses.model_validate(fields)
    except ValidationError as e:
        raise PolicyCommandError(f'The press fields of an element of the helper are not well formed: {e}') from e


def helper_commands(element: Mapping[str, Any]) -> Commands:
    """The commands of ``element``, an element of the helper: each field but the press fields."""
    return {name: value for name, value in element.items() if name not in HELPER_FIELDS}


# The source that each press gives the rig to.
_PRESS_TO = {HELPER_TAKE_OVER: TELEOPERATOR, HELPER_HAND_BACK: POLICY_SOURCE}


class Takeover:
    """The driver of one episode with a helper. The presses of the first element of the helper are not new: they
    came before the episode. An element with no press field keeps the last presses. When both presses are new, the
    later press wins, and at one time Hand back wins."""

    def __init__(self):
        self._driver = POLICY_SOURCE
        self._presses: _Presses | None = None
        self._event: Event | None = None

    def control(self) -> dict[str, Any]:
        """The ``CONTROL`` observation of the helper, as plain data."""
        return ControlObservation(source=self._driver, event=self._event).model_dump(mode='json')

    def _new_press(self, presses: _Presses) -> str | None:
        """The name of the newest press in ``presses`` that the last presses do not have, or ``None``."""
        if (last := self._presses) is None:
            return None
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
        if (presses := _presses(element)) is not None:
            if (press := self._new_press(presses)) is not None and _PRESS_TO[press] != self._driver:
                self._event = Event(kind=EventKind.SWITCH, from_=self._driver, to=_PRESS_TO[press], cause=press)
                self._driver = _PRESS_TO[press]
                events = (self._event,)
            self._presses = presses
        if self._driver == TELEOPERATOR:
            shadow = {POLICY_SOURCE: policy} if policy else {}
            return helper_commands(element), StepRecord(TELEOPERATOR, frozenset({TELEOPERATOR}), events, shadow)
        return policy, StepRecord(POLICY_SOURCE, frozenset({POLICY_SOURCE}), events)
