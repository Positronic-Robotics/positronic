"""FK/IK of the i2rt YAM arm on the vendored MJCF (``assets/i2rt_yam/yam.xml``) at ``GRIPPER_SITE``.

The driver solves FK/IK here, so its joint order and control frame must match the YAM sim's.
"""

import mujoco as mj
import numpy as np

from positronic import geom
from positronic.utils import package_assets_path

from ..ik import qpos_from_site_pose
from ..models import GRIPPER_SITE, YAM_JOINT_NAMES, YAM_MJCF_PATH

IK_POS_TOL = 1e-3  # meters; FK-verify acceptance for an IK solution after limit clamping
IK_ROT_TOL = 1e-2  # radians


# TODO(#517): move this MJCF FK/IK beside `drivers/roboarm/kinematics.py`, in its vocabulary, so the MuJoCo sim
# (`simulator/mujoco/sim.py`) solves IK through it too.
class YamKinematics:
    """FK/IK on the vendored YAM MJCF at ``GRIPPER_SITE``, in the arm-base frame."""

    def __init__(self):
        model_path = package_assets_path(YAM_MJCF_PATH)
        self._model = mj.MjModel.from_xml_path(model_path)
        self._data = mj.MjData(self._model)
        site = mj.mjtObj.mjOBJ_SITE
        self._site_id = mj.mj_name2id(self._model, site, GRIPPER_SITE)
        self._qpos_ids = np.array([self._model.joint(name).qposadr.item() for name in YAM_JOINT_NAMES])
        self._dof_ids = np.array([self._model.joint(name).dofadr.item() for name in YAM_JOINT_NAMES])
        ranges = np.array([self._model.joint(name).range for name in YAM_JOINT_NAMES])
        self._lower, self._upper = ranges[:, 0], ranges[:, 1]

    @property
    def lower(self) -> np.ndarray:
        return self._lower.copy()

    @property
    def upper(self) -> np.ndarray:
        return self._upper.copy()

    def fk(self, q: np.ndarray) -> geom.Transform3D:
        self._data.qpos[self._qpos_ids] = q
        mj.mj_kinematics(self._model, self._data)
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, self._data.site_xmat[self._site_id].copy())
        return geom.Transform3D(self._data.site_xpos[self._site_id].copy(), geom.Rotation.from_quat(quat))

    @staticmethod
    def _reach_postures(x: float, y: float) -> list[np.ndarray]:
        """IK warm starts toward arm-base point (x, y): joint 1 at the target's azimuth, the elbow folded at two
        heights. The 6-DoF wrist gives LM no null space, so a seed near the goal makes limit-clamped IK reliable."""
        az = np.arctan2(y, x)
        return [np.array([az, 1.8, 2.2, 0.0, -0.9, 0.0]), np.array([az, 1.2, 1.2, 0.0, 0.6, 0.0])]

    def ik(
        self, target: geom.Transform3D, current_q: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray | None:
        """Multi-start LM IK: the live posture first, then the reach postures toward the target's azimuth.
        Each start's best answer is wrapped and clamped into ``[lower, upper]``, then FK-verified before acceptance."""
        for start in (current_q, *self._reach_postures(*target.translation[:2])):
            self._data.qpos[:] = 0.0
            self._data.qpos[self._qpos_ids] = start
            # The FK check below accepts or refuses the answer: the solver reports success only below an error of 1e-14.
            qpos, _, _ = qpos_from_site_pose(
                self._model,
                self._data,
                self._site_id,
                self._dof_ids,
                target.translation,
                target.rotation.as_quat,
                rot_weight=0.5,
            )
            q = qpos[self._qpos_ids].copy()
            # A revolute joint at q ± 2π is the same pose; wrap out-of-range entries back in when they fit.
            q = np.where((q > upper) & (q - 2 * np.pi >= lower), q - 2 * np.pi, q)
            q = np.where((q < lower) & (q + 2 * np.pi <= upper), q + 2 * np.pi, q)
            q = np.clip(q, lower, upper)
            reached = self.fk(q)
            rot_err = (reached.rotation.inv * target.rotation).angle
            rot_err = min(rot_err, 2 * np.pi - rot_err)
            if np.linalg.norm(reached.translation - target.translation) < IK_POS_TOL and rot_err < IK_ROT_TOL:
                return q
        return None
