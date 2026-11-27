"""The damped-least-squares IK that an environment server runs on a MuJoCo model.

This module must work in an isolated interpreter without Positronic installed.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import mujoco
import numpy as np

IK_ITERATIONS = 100
IK_DAMPING = 0.05


class IkSolution(NamedTuple):
    """The joints an IK solve reached, the pose error at them, and whether a check passed in the budget."""

    q: np.ndarray
    residual: np.ndarray
    converged: bool


# TODO: Solve with `dls_ik.solve` on the kinematics of the site, as positronic-next does, so that one solve serves each
# env server.
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
