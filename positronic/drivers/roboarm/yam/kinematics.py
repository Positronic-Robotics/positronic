"""FK/IK of the i2rt YAM arm on the vendored MJCF (``assets/i2rt_yam/yam.xml``) at ``GRIPPER_SITE``.

The driver solves FK/IK here, so its joint order and control frame must match the YAM sim's.
"""

import mujoco as mj
import numpy as np

from positronic import geom
from positronic.utils import package_assets_path

from ..ik import qpos_from_site_pose
from ..models import GRIPPER_SITE, YAM_FINGER_JOINT_NAMES, YAM_GRIPPER_BODIES, YAM_JOINT_NAMES, YAM_MJCF_PATH
from .stream import Hull

IK_POS_TOL = 1e-3  # meters; FK-verify acceptance for an IK solution after limit clamping
IK_ROT_TOL = 1e-2  # radians
# Damped least squares steps of `YamKinematics.follow`. A turn of 1 rad weighs as a shift of 0.1 m, so near a joint
# limit the tool keeps its point and gives up its turn. The damping fades in as the smallest singular value of the
# weighted Jacobian falls below FOLLOW_SINGULAR. Each step turns a joint at most FOLLOW_MAX_STEP_RAD.
FOLLOW_ITERATIONS = 5
FOLLOW_ROT_WEIGHT = 0.1
FOLLOW_DAMPING = 0.05
FOLLOW_SINGULAR = 0.08
FOLLOW_MAX_STEP_RAD = 0.05


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
        self.gripper_hulls = self._gripper_hulls()

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

    def _gripper_hulls(self) -> tuple[Hull, ...]:
        """The convex hull of each gripper mesh over the full travel of the fingers, in the frame of ``GRIPPER_SITE``.
        Each mesh is inside its hull, and each vertex of a hull is a mesh vertex."""
        # A finger slides, so each place of it on its travel is inside the hull of its closed and open places
        closed, opened = self._gripper_vertices(0.0), self._gripper_vertices(1.0)
        return tuple(Hull.of(np.vstack(places)) for places in zip(closed, opened, strict=True))

    def _gripper_vertices(self, open_share: float) -> list[np.ndarray]:
        """The vertices of each gripper mesh with the fingers open by ``open_share`` of their travel, in the frame of
        ``GRIPPER_SITE``."""
        model, data = self._model, self._data
        data.qpos[:] = 0.0
        for name in YAM_FINGER_JOINT_NAMES:
            finger = model.joint(name)
            data.qpos[finger.qposadr.item()] = open_share * finger.range[1]
        mj.mj_kinematics(model, data)
        site_pos, site_mat = data.site_xpos[self._site_id], data.site_xmat[self._site_id].reshape(3, 3)
        meshes = [
            g
            for g in range(model.ngeom)
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH
            and mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) in YAM_GRIPPER_BODIES
        ]
        points = []
        for g in meshes:
            first, count = model.mesh_vertadr[model.geom_dataid[g]], model.mesh_vertnum[model.geom_dataid[g]]
            world = model.mesh_vert[first : first + count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            points.append((world - site_pos) @ site_mat)
        return points

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        """The 6x6 Jacobian of ``GRIPPER_SITE`` at ``q``: the shift rows, then the turn rows, in the arm-base frame."""
        self._data.qpos[:] = 0.0
        self._data.qpos[self._qpos_ids] = q
        mj.mj_kinematics(self._model, self._data)
        mj.mj_comPos(self._model, self._data)
        shift, turn = np.zeros((3, self._model.nv)), np.zeros((3, self._model.nv))
        mj.mj_jacSite(self._model, self._data, shift, turn, self._site_id)
        return np.vstack([shift[:, self._dof_ids], turn[:, self._dof_ids]])

    def follow(self, target: geom.Transform3D, seed: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """The joints a few damped least squares steps from ``seed`` toward ``target``, inside ``[lower, upper]``.

        A joint at a limit that a step pushes further is locked for that step, and the other joints take its share.
        An unreachable target gives the nearest pose these steps find. The steps never leave the posture of ``seed``
        for another IK answer. A joint of ``seed`` past a limit does not go further and is not pushed back.
        """
        low, high = np.minimum(lower, seed), np.maximum(upper, seed)
        weight = np.array([1.0, 1.0, 1.0, FOLLOW_ROT_WEIGHT, FOLLOW_ROT_WEIGHT, FOLLOW_ROT_WEIGHT])
        q = np.array(seed, dtype=np.float64)
        for _ in range(FOLLOW_ITERATIONS):
            now = self.fk(q)
            error = weight * np.concatenate([
                target.translation - now.translation,
                (target.rotation * now.rotation.inv).as_rotvec,
            ])
            jacobian = weight[:, None] * self._jacobian(q)
            smallest = float(np.linalg.svd(jacobian, compute_uv=False)[-1])
            damping = FOLLOW_DAMPING**2 * max(0.0, 1.0 - (smallest / FOLLOW_SINGULAR) ** 2)
            step = self._damped_step(jacobian, error, damping)
            locked = ((q <= low) & (step < 0)) | ((q >= high) & (step > 0))
            if locked.any():
                step = self._damped_step(jacobian * ~locked, error, max(damping, 1e-6))
                step[locked] = 0.0
            q = np.clip(q + step * min(1.0, FOLLOW_MAX_STEP_RAD / max(float(np.max(np.abs(step))), 1e-12)), low, high)
        return q

    def progress_per_rad(self, target: geom.Transform3D, before: np.ndarray, after: np.ndarray) -> float:
        """How much nearer the joints ``after`` take the tool to ``target`` than ``before``, in meters a turn of 1 rad
        weighs as FOLLOW_ROT_WEIGHT m, per radian of the joint that turns farthest. Infinite where no joint turns."""
        turned = float(np.max(np.abs(np.asarray(after) - np.asarray(before))))
        if turned == 0.0:
            return np.inf
        return (self._pose_error(target, before) - self._pose_error(target, after)) / turned

    def _pose_error(self, target: geom.Transform3D, q: np.ndarray) -> float:
        now = self.fk(q)
        turn = (target.rotation * now.rotation.inv).as_rotvec
        return float(np.linalg.norm(target.translation - now.translation) + FOLLOW_ROT_WEIGHT * np.linalg.norm(turn))

    @staticmethod
    def _damped_step(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
        return jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(len(error)), error)

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
