"""FK and IK on an MJCF model, for a driver that solves for its own arm.

An arm whose controller solves Cartesian goals itself does not need this. One whose controller does not,
or whose firmware solves them in a way the driver cannot accept, carries its own model and solves here.

Its own module rather than beside the placo ``Kinematics``: that one imports ``placo``, which only the
``hardware`` extra carries, and a driver that solves against an MJCF must not need it.
"""

from collections.abc import Callable, Iterable, Sequence

import mujoco as mj
import numpy as np

from positronic import geom
from positronic.utils import package_assets_path

from .ik import qpos_from_site_pose


class MjcfKinematics:
    """FK and IK on an MJCF model at one site, in the model's own base frame.

    ``mujoco`` exports every symbol below from a compiled extension, so a type checker cannot see them.

    :param mjcf_path: the model, relative to the package assets.
    :param site: the site FK reports and IK aims at.
    :param joint_names: the joints, in the order a caller states them.
    :param reach_postures: warm starts for a target, tried after the live posture. The arms this drives
        have a 6-DoF wrist, which gives LM no null space to escape a bad basin, so a seed near the goal is
        what makes limit-clamped IK reliable.
    """

    # FK-verify acceptance for a solution, after wrapping and clamping it into joint range
    _POS_TOL = 1e-3  # meters
    _ROT_TOL = 1e-2  # radians

    def __init__(
        self,
        mjcf_path: str,
        site: str,
        joint_names: Sequence[str],
        reach_postures: Callable[[geom.Transform3D], Iterable[np.ndarray]],
    ):
        # rules-allow: primitive-type — `package_assets_path(relative_path: str) -> str` owns the join, and
        # every caller spells the model this way
        self._model = mj.MjModel.from_xml_path(package_assets_path(mjcf_path))
        self._data = mj.MjData(self._model)
        self._site_id = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_SITE, site)
        if self._site_id < 0:
            # Indexing with -1 reads the model's last site, so the arm would solve for another frame.
            raise ValueError(f'{mjcf_path} names no site {site!r}')
        self._qpos_ids = np.array([self._model.joint(name).qposadr.item() for name in joint_names])
        self._dof_ids = np.array([self._model.joint(name).dofadr.item() for name in joint_names])
        ranges = np.array([self._model.joint(name).range for name in joint_names])
        self.lower, self.upper = ranges[:, 0], ranges[:, 1]
        self._reach_postures = reach_postures

    def fk(self, q: np.ndarray) -> geom.Transform3D:
        self._data.qpos[self._qpos_ids] = q
        mj.mj_kinematics(self._model, self._data)
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, self._data.site_xmat[self._site_id].copy())
        return geom.Transform3D(self._data.site_xpos[self._site_id].copy(), geom.Rotation.from_quat(quat))

    def ik(
        self, target: geom.Transform3D, current_q: np.ndarray, max_jump: float | np.ndarray | None = None
    ) -> np.ndarray | None:
        """The joints that reach ``target``, warm-started from where the arm stands, or ``None``.

        ``max_jump`` bounds how far the solution may sit from ``current_q``, per joint or over all of them,
        and the search stops at the live posture: the arm keeps the shape it has, and a pose it can reach
        only in another one comes back as nothing. Without it the reach postures are tried too, so the arm
        may change shape to get there.

        A solution is wrapped and clamped into joint range and then FK-verified, so a target the arm cannot
        reach comes back as nothing rather than as the nearest thing the solver stopped at.
        """
        seeds = (current_q,) if max_jump is not None else (current_q, *self._reach_postures(target))
        for start in seeds:
            self._data.qpos[:] = 0.0
            self._data.qpos[self._qpos_ids] = start
            qpos, _, success = qpos_from_site_pose(
                self._model,
                self._data,
                self._site_id,
                self._dof_ids,
                target.translation,
                target.rotation.as_quat,
                rot_weight=0.5,
            )
            if not success:
                continue
            q = qpos[self._qpos_ids].copy()
            # A revolute joint at q ± 2π is the same pose; wrap out-of-range entries back in when they fit.
            q = np.where(q > self.upper, q - 2 * np.pi, q)
            q = np.where(q < self.lower, q + 2 * np.pi, q)
            q = np.clip(q, self.lower, self.upper)
            if max_jump is not None and np.any(np.abs(q - current_q) > max_jump):
                continue
            reached = self.fk(q)
            rot_err = (reached.rotation.inv * target.rotation).angle
            rot_err = min(rot_err, 2 * np.pi - rot_err)
            if np.linalg.norm(reached.translation - target.translation) < self._POS_TOL and rot_err < self._ROT_TOL:
                return q
        return None
