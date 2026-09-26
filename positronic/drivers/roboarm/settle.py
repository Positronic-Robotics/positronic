from dataclasses import dataclass


@dataclass(frozen=True)
class SettleTuning:
    """How a joint-space move settles onto its target on one arm.

    A position servo holds the chain short of its reference, by a gap that differs between arms. A move
    measures the gap once the chain is still, and moves the reference past the target by up to
    ``max_correction_rad``, until every joint rests within ``tolerance_rad``.
    """

    tolerance_rad: float  # the largest joint error at rest
    still_position_rad: float  # the largest position spread per joint over still_time_s, for the chain to be still
    still_time_s: float  # how long the chain must stay still and on target
    grip_tolerance: float  # the largest grip error, normalized
    max_speed_rad_s: float  # the ramp speed of the joint that travels farthest
    min_ramp_s: float
    settle_timeout_s: float  # time to come to rest after the ramp, per pass
    attempts: int  # correction passes before the move fails
    max_correction_rad: float  # the largest distance of the reference past the target, per joint


# Defaults measured on one arm. The park is tighter, because torque is cut there.
PARK_SETTLE = SettleTuning(
    tolerance_rad=0.01,  # the stops of joints 2 and 3 read up to 0.006 rad above zero
    still_position_rad=0.002,
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
    still_position_rad=0.002,
    still_time_s=0.2,
    grip_tolerance=0.05,
    max_speed_rad_s=0.35,
    min_ramp_s=2.0,
    settle_timeout_s=1.0,
    attempts=6,
    max_correction_rad=0.05,
)
