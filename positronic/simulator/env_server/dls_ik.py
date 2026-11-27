"""The damped-least-squares IK that an env server runs to solve a pose to the joints of an arm.

This module must work in an isolated interpreter without positronic, so it uses only Python and numpy. The kinematics
of an arm give, for its joints ``q``, the position and the rotation matrix of the arm's site, and the ``6 x n``
Jacobian ``[position; rotation]`` of the site, in the frame of the target.
"""

import enum
import math
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

IK_ITERATIONS = 100
IK_DAMPING = 0.05
# The YAM driver's acceptance of an IK solution.
IK_POSITION_TOLERANCE = 1e-3  # m
IK_ROTATION_TOLERANCE = 1e-2  # rad
# The error of a near solution of a target that no start solves within the tolerances.
NEAR_POSITION_TOLERANCE = 1e-2  # m
NEAR_ROTATION_TOLERANCE = 1e-1  # rad

Kinematics = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


class Fit(enum.Enum):
    """How the joints of an IK solution meet the target."""

    CONVERGED = 'converged'  # within the tolerances
    NEAR = 'near'  # a near solution (``solve``)
    MISS = 'miss'


class IkSolution(NamedTuple):
    """The joints that an IK solve reached, the pose error at them, and how they meet the target."""

    q: np.ndarray
    residual: np.ndarray
    fit: Fit


def _quaternion(rot: np.ndarray) -> np.ndarray:
    """The unit quaternion ``[w, x, y, z]`` of a rotation matrix, with ``w >= 0``."""
    (r00, r01, r02), (r10, r11, r12), (r20, r21, r22) = rot
    trace = r00 + r11 + r22
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        quat = [s / 4, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s]
    elif r00 > r11 and r00 > r22:
        s = 2 * math.sqrt(1 + r00 - r11 - r22)
        quat = [(r21 - r12) / s, s / 4, (r01 + r10) / s, (r02 + r20) / s]
    elif r11 > r22:
        s = 2 * math.sqrt(1 + r11 - r00 - r22)
        quat = [(r02 - r20) / s, (r01 + r10) / s, s / 4, (r12 + r21) / s]
    else:
        s = 2 * math.sqrt(1 + r22 - r00 - r11)
        quat = [(r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, s / 4]
    quat = np.array(quat) / np.linalg.norm(quat)
    return -quat if quat[0] < 0 else quat


def _rotation_vector(rot: np.ndarray) -> np.ndarray:
    """The axis of a rotation matrix times its angle, in radians, from 0 to pi."""
    quat = _quaternion(rot)
    sine = np.linalg.norm(quat[1:])
    if sine < 1e-12:
        return np.zeros(3)
    return quat[1:] / sine * 2 * math.atan2(sine, quat[0])


def _pose_error(target_pos: Any, target_rot: Any, site_pos: Any, site_rot: Any) -> np.ndarray:
    """The ``[position(3), rotation(3)]`` error from the site's pose to the target, in the frame of the target."""
    turn = np.asarray(target_rot, dtype=np.float64) @ np.asarray(site_rot, dtype=np.float64).T
    return np.concatenate([np.asarray(target_pos, dtype=np.float64) - site_pos, _rotation_vector(turn)])


def _within(error: np.ndarray, position: float, rotation: float) -> bool:
    return bool(np.linalg.norm(error[:3]) < position and np.linalg.norm(error[3:]) < rotation)


def _fit(error: np.ndarray) -> Fit:
    return Fit.CONVERGED if _within(error, IK_POSITION_TOLERANCE, IK_ROTATION_TOLERANCE) else Fit.MISS


def _step(jac: np.ndarray, error: np.ndarray) -> np.ndarray:
    return jac.T @ np.linalg.solve(jac @ jac.T + IK_DAMPING**2 * np.eye(6), error)


def damped_least_squares_ik(
    kinematics: Kinematics,
    *,
    start_q: Any,
    lower: Any,
    upper: Any,
    target_pos: Any,
    target_rot: Any,
    drop_blocked: bool = False,
) -> IkSolution:
    """One damped-least-squares solve from ``start_q``, with the joints clipped to ``[lower, upper]`` at the start and
    at each step. With ``drop_blocked``, each step leaves out each joint at a limit that the step moves past it, and
    solves again with the others. Without convergence, ``q`` holds the joints after the last step."""
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    q = np.clip(np.array(start_q, dtype=np.float64), lower, upper)
    for _ in range(IK_ITERATIONS):
        site_pos, site_rot, jac = kinematics(q)
        error = _pose_error(target_pos, target_rot, site_pos, site_rot)
        if _within(error, IK_POSITION_TOLERANCE, IK_ROTATION_TOLERANCE):
            return IkSolution(q, error, Fit.CONVERGED)
        dq, free = _step(jac, error), np.ones(q.size, dtype=bool)
        while drop_blocked and (blocked := free & (((q <= lower) & (dq < 0)) | ((q >= upper) & (dq > 0)))).any():
            free &= ~blocked
            dq = _step(jac * free, error)
        q = np.clip(q + dq, lower, upper)
    site_pos, site_rot, _ = kinematics(q)
    error = _pose_error(target_pos, target_rot, site_pos, site_rot)
    return IkSolution(q, error, _fit(error))


def middle(lower: Any, upper: Any) -> np.ndarray:
    """The middle of the joint ranges ``[lower, upper]``. A joint with no limit has the middle 0."""
    low, high = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    return np.where(np.isfinite(low) & np.isfinite(high), (low + high) / 2, 0.0)


def _is_near(kinematics: Kinematics, measured_q: Any, solution: IkSolution, target_pos: Any, target_rot: Any) -> bool:
    """Whether ``solution``, solved from ``measured_q``, ends within the near tolerances, nearer the target than
    ``measured_q``, and moves the site no farther than the target is from it, within the tolerances."""
    site_pos, site_rot, _ = kinematics(np.asarray(measured_q, dtype=np.float64))
    command = _pose_error(target_pos, target_rot, site_pos, site_rot)
    reached_pos, reached_rot, _ = kinematics(solution.q)
    move = _pose_error(reached_pos, reached_rot, site_pos, site_rot)
    return bool(
        _within(solution.residual, NEAR_POSITION_TOLERANCE, NEAR_ROTATION_TOLERANCE)
        and np.linalg.norm(solution.residual) < np.linalg.norm(command)
        and np.linalg.norm(move[:3]) <= np.linalg.norm(command[:3]) + IK_POSITION_TOLERANCE
        and np.linalg.norm(move[3:]) <= np.linalg.norm(command[3:]) + IK_ROTATION_TOLERANCE
    )


def solve(
    kinematics: Kinematics, *, measured_q: Any, lower: Any, upper: Any, target_pos: Any, target_rot: Any
) -> IkSolution:
    """The first solution within the tolerances from two starts: ``measured_q``, then the middle of the joint ranges.
    Without one, the solution from ``measured_q`` as ``Fit.NEAR`` when it is near, else the solution of one more solve
    from ``measured_q`` that drops the joints at a limit, when it is within the tolerances or near. Else the solution
    from ``measured_q``, a ``Fit.MISS``."""
    target = {'lower': lower, 'upper': upper, 'target_pos': target_pos, 'target_rot': target_rot}
    first = damped_least_squares_ik(kinematics, start_q=measured_q, **target)
    if first.fit is Fit.CONVERGED:
        return first
    second = damped_least_squares_ik(kinematics, start_q=middle(lower, upper), **target)
    if second.fit is Fit.CONVERGED:
        return second
    if _is_near(kinematics, measured_q, first, target_pos, target_rot):
        return first._replace(fit=Fit.NEAR)
    third = damped_least_squares_ik(kinematics, start_q=measured_q, drop_blocked=True, **target)
    if third.fit is Fit.CONVERGED:
        return third
    if _is_near(kinematics, measured_q, third, target_pos, target_rot):
        return third._replace(fit=Fit.NEAR)
    return first
