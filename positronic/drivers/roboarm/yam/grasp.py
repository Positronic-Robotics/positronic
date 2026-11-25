import enum
import math
from dataclasses import dataclass

from positronic.utils import check_finite_nonnegative, check_finite_positive


@dataclass(frozen=True)
class GraspTuning:
    """How the gripper of one arm closes on an object. The gripper target closes at ``close_rate_per_s`` or slower. A
    closing motor effort above ``touch_effort_nm`` shows a touch, except in the first ``launch_s`` of a close.

    The defaults are measured on one arm.
    """

    close_rate_per_s: float = 2.0  # full strokes per second; the motor then meets an object at about 10 rad/s
    touch_effort_nm: float = 0.8  # a free close at that rate peaks at 0.6 Nm; a touch passes 1.9 Nm in one 5 ms step
    launch_s: float = 0.06  # a close from rest peaks at about 1 Nm in this time

    def __post_init__(self) -> None:
        check_finite_positive(self.close_rate_per_s, 'close_rate_per_s')
        check_finite_positive(self.touch_effort_nm, 'touch_effort_nm')
        check_finite_nonnegative(self.launch_s, 'launch_s')


DEFAULT_GRASP = GraspTuning()


class _Phase(enum.Enum):
    FREE = enum.auto()  # the gripper does not close
    APPROACH = enum.auto()  # the gripper closes and has not touched an object
    HELD = enum.auto()  # the gripper touched an object


class Grasp:
    """The close of one gripper motor, in motor radians and seconds. At each step it takes the target that the motor
    gets and returns the target to send."""

    def __init__(self, tuning: GraspTuning, closed: float, opened: float):
        self.closing = math.copysign(1.0, closed - opened)  # the sign of a closing move
        self._tuning = tuning
        self._max_speed = tuning.close_rate_per_s * abs(opened - closed)
        self._phase = _Phase.FREE
        self._close_started = 0.0
        self._sent = self._sent_at = 0.0

    def target(self, now: float, q: float, request: float, effort: float, hold_limit: float | None) -> float:
        """The target to send for ``request`` at time ``now``, with the motor at ``q`` and its last ``effort`` in Nm.
        ``hold_limit`` is the farthest closing target that keeps the hold force. None sets no hold force."""
        closing, tuning = self.closing, self._tuning
        if closing * (request - q) <= 0:
            self._phase = _Phase.FREE
            return request
        if self._phase is _Phase.FREE:
            self._phase = _Phase.APPROACH
            self._close_started = self._sent_at = now
            self._sent = q
        elif (
            self._phase is _Phase.APPROACH
            and now - self._close_started > tuning.launch_s
            and closing * effort > tuning.touch_effort_nm
        ):
            self._phase = _Phase.HELD
        step = self._max_speed * (now - self._sent_at)
        target = request if closing * (request - self._sent) <= step else self._sent + closing * step
        if self._phase is _Phase.HELD and hold_limit is not None and closing * (target - hold_limit) > 0:
            target = hold_limit
        self._sent, self._sent_at = target, now
        return target
