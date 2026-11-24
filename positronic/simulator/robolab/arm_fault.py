"""The fault of a clone's arm state, after which the clone cannot go on.

RoboLab's interpreter imports this module as a top-level ``arm_fault``, so it uses only Python and numpy.

A physics blow-up can keep the arm state finite for many steps while it grows far past anything the arm can do. The
bounds of a joint come from its own position limits: a contact at a limit overshoots by a small part of the joint
range, and an arm that chatters moves a small part of the range in one control step.
"""

import numpy as np

if __package__:
    from positronic.simulator.robolab import keys
else:
    import keys  # pyright: ignore[reportMissingImports]


def _arm_fault(joint_pos: np.ndarray, joint_vel: np.ndarray, pos_limits: np.ndarray, control_dt: float) -> str | None:
    if not (np.isfinite(joint_pos).all() and np.isfinite(joint_vel).all()):
        return keys.FAULT_NONFINITE_ARM
    lower, upper = pos_limits[:, 0], pos_limits[:, 1]
    joint_range = upper - lower
    past_limit = (joint_pos < lower - joint_range) | (joint_pos > upper + joint_range)
    if past_limit.any() or (np.abs(joint_vel) * control_dt > joint_range).any():
        return keys.FAULT_ARM_OUT_OF_RANGE
    return None


def arm_faults(
    joint_pos: np.ndarray, joint_vel: np.ndarray, pos_limits: np.ndarray, control_dt: float
) -> list[str | None]:
    """Per slot, the fault of its arm state, or ``None`` when the clone can go on.

    ``joint_pos`` and ``joint_vel`` are ``(slots, joints)``, and ``pos_limits`` is ``(slots, joints, 2)`` with the
    lower and the upper limit of each joint. A joint or joint velocity that is not finite is
    ``keys.FAULT_NONFINITE_ARM``. A joint more than its range past a limit, or a joint speed that crosses more than
    the range in one control step, is ``keys.FAULT_ARM_OUT_OF_RANGE``.
    """
    return [
        _arm_fault(q, dq, limits, control_dt) for q, dq, limits in zip(joint_pos, joint_vel, pos_limits, strict=True)
    ]
