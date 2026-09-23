from dataclasses import dataclass


@dataclass(frozen=True)
class SettleTuning:
    """How a joint-space move reaches its target on one arm: its pace, its arrival test, and the correction
    that closes the servo's steady gap.

    A position servo holds its chain a steady distance short of the reference, and that distance differs
    between two arms. A move measures the gap once the chain is still and asks for a reference past the
    target, by up to ``max_correction_rad``, until every joint rests within ``tolerance_rad``.
    """

    tolerance_rad: float  # every joint must rest this close to the target
    still_velocity_rad_s: float  # every joint reads slower than this when the chain counts as still
    still_time_s: float  # how long every joint must stay still, and on target, without a break
    grip_tolerance: float  # normalized; the fingers must read this close to the asked grip
    max_speed_rad_s: float  # the ramp's pace on the joint that travels farthest
    min_ramp_s: float  # the shortest ramp, however near the target
    settle_timeout_s: float  # time after the ramp for the chain to come to rest, per pass
    attempts: int  # correction passes before the move fails
    max_correction_rad: float  # the most the reference may lie past the target, per joint


# The defaults fit the arm the driver was brought up on. The park lands tighter, because torque is cut there.
PARK_SETTLE = SettleTuning(
    tolerance_rad=0.005,
    still_velocity_rad_s=0.02,
    still_time_s=0.2,
    grip_tolerance=0.05,
    max_speed_rad_s=0.35,
    min_ramp_s=2.0,
    settle_timeout_s=8.0,
    attempts=6,
    max_correction_rad=0.05,
)
MOVE_SETTLE = SettleTuning(
    tolerance_rad=0.02,
    still_velocity_rad_s=0.02,
    still_time_s=0.2,
    grip_tolerance=0.05,
    max_speed_rad_s=0.35,
    min_ramp_s=2.0,
    settle_timeout_s=1.0,
    attempts=6,
    max_correction_rad=0.05,
)
