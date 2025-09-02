"""
Offline IK debug utility for Franka Panda using MuJoCo as a kinematics/Jacobian backend.

This avoids any dependency on a physical arm or franky runtime. It adapts the
Robot._inverse_kinematics(...) API by providing a minimal "robot-like" object
exposing `.model.pose`, `.model.zero_jacobian`, and `.state` with `q`, `F_T_EE`,
and `EE_T_K`.

Usage:
    uv run python -m positronic.drivers.roboarm.franka_ik_debug --n 20
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import mujoco as mj

import configuronic as cfn
from positronic import geom
from .franka import Robot as FrankaIK


class _MjPoseJacobianBackend:
    """MuJoCo-backed FK and Jacobian provider compatible with FrankaIK._inverse_kinematics.

    - pose(frame, q, F_T_EE, EE_T_K): returns mujoco site pose (end_effector)
    - zero_jacobian(...): returns a 6x7 geometric Jacobian for that site
    """

    def __init__(self, model: mj.MjModel, site_name: str, joint_names: List[str]):
        self.model = model
        self.data = mj.MjData(model)
        self.site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Unknown site '{site_name}' in model")
        # Map joints to their position and velocity indices
        self.qpos_ids = [int(model.jnt_qposadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, j)]) for j in joint_names]
        self.dof_ids = [int(model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, j)]) for j in joint_names]

    def _set_q(self, q: np.ndarray):
        for i, qpos_id in enumerate(self.qpos_ids):
            self.data.qpos[qpos_id] = float(q[i])
        mj.mj_forward(self.model, self.data)

    @staticmethod
    def _affine_from_site(data: mj.MjData, site_id: int):
        pos = np.array(data.site_xpos[site_id])
        xmat = np.array(data.site_xmat[site_id]).reshape(3, 3)
        quat_wxyz = np.empty(4)
        mj.mju_mat2Quat(quat_wxyz, xmat.flatten())
        # Franka IK expects xyzw ordering (see _affine_to_geom in franka.py)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos, quat_xyzw

    def pose(self, frame, q: np.ndarray, F_T_EE: np.ndarray, EE_T_K: np.ndarray):
        # F_T_EE and EE_T_K are ignored; test with identity attachments
        self._set_q(q)
        pos, quat = self._affine_from_site(self.data, self.site_id)

        # Build a small struct mimicking franky.Affine
        class _A:
            pass

        a = _A()
        a.translation = pos
        a.quaternion = quat  # xyzw as expected by franka adapter
        return a

    def zero_jacobian(self, frame, q: np.ndarray, F_T_EE: np.ndarray, EE_T_K: np.ndarray) -> np.ndarray:
        self._set_q(q)
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        # Extract columns for our 7 joints in WORLD (spatial) frame
        Jp = jacp[:, self.dof_ids]
        Jr_world = jacr[:, self.dof_ids]
        J = np.vstack([Jp, Jr_world])
        return J


@dataclass
class _FakeState:
    q: np.ndarray
    F_T_EE: np.ndarray
    EE_T_K: np.ndarray


class _FakeRobot:

    def __init__(self, model: _MjPoseJacobianBackend, q0: np.ndarray):
        self.model = model
        self.state = _FakeState(q=q0.copy(), F_T_EE=np.eye(4), EE_T_K=np.eye(4))


def _random_targets_around(start_pose: geom.Transform3D, n: int = 10, pos_radius: float = 0.05):
    rng = np.random.default_rng(0)
    targets = []
    for _ in range(n):
        dp = rng.normal(0.0, pos_radius, size=3)
        # small random rotation
        axis = rng.normal(0.0, 1.0, size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        angle = rng.uniform(-0.3, 0.3)
        rot = geom.Rotation.from_rotvec(axis * angle)
        t = geom.Transform3D(start_pose.translation + dp, rot * start_pose.rotation)
        targets.append(t)
    return targets


@cfn.config()
def debug_franka_ik(
    n: int = 10,
    max_iters: int = 150,
    tol: float = 1e-4,
    pinv_reg: float = 0.3,
    nullspace_gain: float = 0.002,
    alpha: float = 1.0,
    beta: float = 0.8,
    pos_radius: float = 0.05,
    thr_pos: float = 5e-3,
    thr_rot: float = 5e-3,
    xml_path: str = "positronic/assets/mujoco/panda.xml",
):
    # Load Panda model and construct adapter
    model = mj.MjModel.from_xml_path(xml_path)
    joint_names = [f"joint{i}" for i in range(1, 8)]
    backend = _MjPoseJacobianBackend(model, site_name="end_effector", joint_names=joint_names)

    # Home configuration similar to franka.py
    q_home = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0], dtype=float)
    fake_robot = _FakeRobot(backend, q_home)

    # Current pose via backend
    cur_aff = backend.pose(None, q_home, np.eye(4), np.eye(4))
    cur_pose = geom.Transform3D(cur_aff.translation, geom.Rotation.from_quat_xyzw(cur_aff.quaternion))

    # Generate targets and evaluate IK
    targets = _random_targets_around(cur_pose, n=n, pos_radius=pos_radius)
    ok = 0
    for i, tgt in enumerate(targets):
        q_sol = FrankaIK._inverse_kinematics(
            fake_robot,
            tgt,
            tol=tol,
            max_iters=max_iters,
            pinv_reg=pinv_reg,
            nullspace_gain=nullspace_gain,
            line_search_alpha=alpha,
            line_search_beta=beta,
        )
        # Evaluate pose error
        aff_new = backend.pose(None, q_sol, np.eye(4), np.eye(4))
        pose_new = geom.Transform3D(aff_new.translation, geom.Rotation.from_quat_xyzw(aff_new.quaternion))
        pos_err = np.linalg.norm(pose_new.translation - tgt.translation)
        rot_err = np.linalg.norm((pose_new.rotation.inv * tgt.rotation).as_rotvec)
        ok += int(pos_err < thr_pos and rot_err < thr_rot)
        # Condition numbers at start and end (optional info)
        J_start = backend.zero_jacobian(None, fake_robot.state.q, np.eye(4), np.eye(4))
        J_end = backend.zero_jacobian(None, q_sol, np.eye(4), np.eye(4))
        cond_start = np.linalg.cond(J_start @ J_start.T)
        cond_end = np.linalg.cond(J_end @ J_end.T)
        print(f"[{i:02d}] pos_err={pos_err:.4f}m, rot_err={rot_err:.4f}rad | cond(start)={cond_start:.1e}, "
              f"cond(end)={cond_end:.1e}")

    print(f"Success {ok}/{n} within tight thresholds")


if __name__ == "__main__":
    cfn.cli(debug_franka_ik)
