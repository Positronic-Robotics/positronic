"""The wire command of one arm as absolute joint targets, and the IK that solves a pose command, for an
environment server to apply.

This module must work in an isolated interpreter without Positronic installed.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import mujoco
import numpy as np

if __package__:
    from positronic.simulator.env_server import protocol
else:
    import protocol


def unpack_wire_pose(vector: Any) -> tuple[np.ndarray, np.ndarray]:
    """Translation and rotation matrix from a wire pose ``[t(3), R(9) row-major]``."""
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.shape[0] != 12:
        raise ValueError(f'wire pose must be [t(3), R(9)], got {vec.shape[0]} values')
    return vec[:3].copy(), vec[3:].reshape(3, 3).copy()


def compose_reference_delta(
    cur_pos: Any, cur_rot: Any, delta_pos: Any, delta_rot: Any
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(cur_pos, dtype=np.float64).reshape(3) + np.asarray(delta_pos, dtype=np.float64).reshape(3),
        np.asarray(delta_rot, dtype=np.float64).reshape(3, 3) @ np.asarray(cur_rot, dtype=np.float64).reshape(3, 3),
    )


def wire_command_to_arm_action(
    command: dict[str, Any],
    current_q: Any,
    *,
    ik: Callable[[np.ndarray, np.ndarray], Any],
    current_eef: tuple[Any, Any],
) -> np.ndarray | None:
    """Absolute arm joint targets, or None where ``ik`` finds no solution.

    ``ik`` and ``current_eef`` use poses in the frame the env measures.
    """
    current = np.asarray(current_q, dtype=np.float32).reshape(-1)
    match command[protocol.COMMAND_TYPE]:
        case protocol.JOINT_POS:
            target = command[protocol.COMMAND_JOINT_POS]
        case protocol.JOINT_DELTA:
            dq = np.asarray(command[protocol.COMMAND_JOINT_DELTA], dtype=np.float32).reshape(-1)
            if dq.shape[0] != current.shape[0]:
                raise ValueError(f'joint delta {dq.shape[0]} vs measured joints {current.shape[0]}')
            target = current + dq
        case protocol.HOLD:
            target = current
        case protocol.CARTESIAN:
            target = ik(*unpack_wire_pose(command[protocol.COMMAND_POSE]))
        case protocol.CARTESIAN_DELTA:
            delta_pos, delta_rot = unpack_wire_pose(command[protocol.COMMAND_DELTA])
            target = ik(*compose_reference_delta(*current_eef, delta_pos, delta_rot))
        case other:
            raise ValueError(
                f'{other!r} is not a canonical command type; the contract is {list(protocol.CANONICAL_COMMAND_TYPES)}'
            )
    return None if target is None else np.asarray(target, dtype=np.float32).reshape(-1)


IK_ITERATIONS = 100
IK_DAMPING = 0.05


class IkSolution(NamedTuple):
    """The joints an IK solve reached, the pose error at them, and whether a check passed in the budget."""

    q: np.ndarray
    residual: np.ndarray
    converged: bool


def damped_least_squares_ik(
    model: mujoco.MjModel,
    *,
    live_qpos: Any,
    site_id: int,
    qpos_ids: Any,
    dof_ids: Any,
    start_q: Any,
    lower: Any,
    upper: Any,
    pose_error: Callable[[np.ndarray, np.ndarray], np.ndarray],
    converged: Callable[[np.ndarray], bool],
) -> IkSolution:
    """Damped-least-squares differential IK on a site's world-frame Jacobian.

    The solve runs on a scratch ``MjData`` seeded from ``live_qpos``, so it moves nothing in the live scene.
    ``pose_error(site_pos, site_rot)`` is the world-frame ``[position(3), rotation(3)]`` error to the target.
    Without convergence, ``q`` holds the joints after the last step.
    """
    data = mujoco.MjData(model)
    data.qpos[:] = live_qpos
    q = np.array(start_q, dtype=np.float64)

    def error_at(q: np.ndarray) -> np.ndarray:
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        site_pos = np.array(data.site_xpos[site_id])
        site_rot = np.array(data.site_xmat[site_id]).reshape(3, 3)
        return np.asarray(pose_error(site_pos, site_rot), dtype=np.float64)

    for _ in range(IK_ITERATIONS):
        error = error_at(q)
        if converged(error):
            return IkSolution(q, error, True)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp, jacr])[:, dof_ids]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + IK_DAMPING**2 * np.eye(6), error)
        q = np.clip(q + dq, lower, upper)
    return IkSolution(q, error_at(q), False)
