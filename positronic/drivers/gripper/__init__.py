"""Two-finger gripper drivers, and what they share about being placed."""

from pimm.calls import Call

ARRIVED_TOL = 0.05
# Fingers stopped by what they are holding never reach their target
ARRIVAL_TIMEOUT_S = 3.0


def answer_when_arrived(
    call: Call[float, None], grip: float, target: float, out_of_time: bool
) -> Call[float, None] | None:
    """Answer ``call`` if the fingers are there or have run out of time; returns the call still waiting, if any."""
    if abs(grip - target) < ARRIVED_TOL:
        call.set_result(None)
        return None
    if out_of_time:
        call.set_exception(TimeoutError(f'the gripper stopped at {grip:.2f}, short of {target:.2f}'))
        return None
    return call
