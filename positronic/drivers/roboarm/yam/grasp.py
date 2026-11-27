import enum
from dataclasses import dataclass

from positronic.utils import check_finite_nonnegative, check_finite_positive


@dataclass(frozen=True)
class GraspTuning:
    """How the gripper of one arm closes on an object. The gripper target closes at ``close_rate_per_s`` or slower. A
    closing motor effort above ``touch_effort_nm`` shows a touch, except in the first ``launch_s`` of a close. In that
    time, the closing torque stays at ``touch_effort_nm`` or less.

    The defaults are measured on one arm.
    """

    close_rate_per_s: float = 2.0  # full strokes per second; the motor then meets an object at about 10 rad/s
    touch_effort_nm: float = 0.8  # a free close at that rate peaks at 0.6 Nm; a touch passes 1.9 Nm
    launch_s: float = 0.06  # a close from rest peaks at about 1 Nm in this time
    open_margin_rad: float = 0.02  # about 0.3 mm of jaw; i2rt's stall limiter can send a target a little behind q

    def __post_init__(self) -> None:
        check_finite_positive(self.close_rate_per_s, 'close_rate_per_s')
        check_finite_positive(self.touch_effort_nm, 'touch_effort_nm')
        check_finite_nonnegative(self.launch_s, 'launch_s')
        check_finite_nonnegative(self.open_margin_rad, 'open_margin_rad')


DEFAULT_GRASP = GraspTuning()


def capped_target(
    closing_sign: float, q: float, target: float, kp: float, feedforward_and_damping: float, cap_nm: float | None
) -> float:
    """``target``, moved back so that the P-D loop of the motor at ``q`` asks for ``cap_nm`` or less of closing torque.
    None sets no cap."""
    if cap_nm is not None and kp > 0 and closing_sign * (feedforward_and_damping + kp * (target - q)) > cap_nm:
        return q + (closing_sign * cap_nm - feedforward_and_damping) / kp
    return target


class _Phase(enum.Enum):
    NOT_CLOSING = enum.auto()
    APPROACH = enum.auto()
    HELD = enum.auto()


class Grasp:
    """The close of one gripper motor, in motor radians and seconds.

    A touch holds until the request opens past ``open_margin_rad``. If the object slips out, the fingers close on at
    the close rate under the hold cap, as intended.
    """

    _MAX_STEP_S = 0.01  # i2rt warns when its control loop runs below 100 Hz

    def __init__(self, tuning: GraspTuning, closing_sign: float, stroke_rad: float):
        self._closing_sign = closing_sign
        self._tuning = tuning
        self._max_speed = tuning.close_rate_per_s * stroke_rad
        self._phase = _Phase.NOT_CLOSING
        self._close_started = 0.0
        self._sent = self._sent_at = 0.0

    def target(
        self,
        now: float,
        q: float,
        request: float,
        effort: float,
        kp: float,
        feedforward_and_damping: float,
        hold_nm: float | None,
    ) -> float:
        """The target to send for ``request`` at time ``now``, with the motor at ``q`` and its last ``effort``.

        After a touch, the grasp caps the closing torque at ``hold_nm``; None sets no cap. In the launch, it caps it at
        the touch effort and at ``hold_nm``.
        """
        closing_sign, tuning = self._closing_sign, self._tuning
        margin = 0.0 if self._phase is _Phase.NOT_CLOSING else tuning.open_margin_rad
        if closing_sign * (request - q) <= -margin:
            self._phase = _Phase.NOT_CLOSING
            return request
        if self._phase is _Phase.NOT_CLOSING:
            self._phase = _Phase.APPROACH
            self._close_started = self._sent_at = now
            self._sent = q
        elif self._phase is _Phase.APPROACH and closing_sign * (request - self._sent) <= 0:
            self._close_started = now  # a target that rests at the request starts a new launch when it moves on
        launching = now - self._close_started <= tuning.launch_s
        if self._phase is _Phase.APPROACH and not launching and closing_sign * effort > tuning.touch_effort_nm:
            self._phase = _Phase.HELD
        step = self._max_speed * min(now - self._sent_at, self._MAX_STEP_S)
        target = request if closing_sign * (request - self._sent) <= step else self._sent + closing_sign * step
        if self._phase is _Phase.HELD:
            cap_nm = hold_nm
        elif launching:
            cap_nm = tuning.touch_effort_nm if hold_nm is None else min(tuning.touch_effort_nm, hold_nm)
        else:
            cap_nm = None
        target = capped_target(closing_sign, q, target, kp, feedforward_and_damping, cap_nm)
        self._sent, self._sent_at = target, now
        return target
