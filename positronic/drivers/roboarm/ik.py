"""IK solvers using MuJoCo for FK/Jacobian computation.

Three solvers for reconstructing joint-space targets from recorded EE targets:
- LMIKSolver: Levenberg-Marquardt on the site pose with nullspace damping (for sim data)
- DLSIKSolver: DLS with nullspace bias and line search (ports inverse_kinematics_q0)
- DLSIKSolverWithLimits: DLS with joint limits via bounded least squares
  (ports inverse_kinematics_with_limits, using scipy instead of OSQP)
"""

import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation as ScipyRotation

from positronic.dataset import transforms


def _prepare_spec(urdf_xml, control_frame):
    """Parse URDF or MJCF into an MjSpec, stripping meshes and resolving the control frame site.

    The control frame must exist in the model as a site or body. For bodies (e.g. real URDF
    with ``end_effector`` link baked in by positronic-franka), a site is added at its origin.
    """
    root = ET.fromstring(urdf_xml)
    if root.tag == 'robot':
        for link in root.findall('.//link'):
            for elem in link.findall('visual') + link.findall('collision'):
                link.remove(elem)
        urdf_xml = ET.tostring(root, encoding='unicode')
    spec = mj.MjSpec.from_string(urdf_xml)

    all_sites = {s.name for b in spec.bodies for s in b.sites}
    if control_frame in all_sites:
        return spec

    body_names = {b.name for b in spec.bodies}
    if control_frame in body_names:
        site = spec.body(control_frame).add_site()
        site.name = control_frame
        return spec

    raise ValueError(f'Control frame {control_frame!r} not found as site or body in model')


def _parse_target(target_ee_pose_vec):
    """Parse [tx,ty,tz,w,x,y,z] into position and rotation matrix."""
    R = np.zeros(9)
    mj.mju_quat2Mat(R, target_ee_pose_vec[3:7])
    return target_ee_pose_vec[:3], R.reshape(3, 3)


def _cartesian_error(pos_cur, R_cur, t_tgt, R_tgt):
    """6D Cartesian error matching cartesian_error_ from positronic-franka robot.hpp."""
    e_pos = pos_cur - t_tgt
    e_rot = -R_cur @ ScipyRotation.from_matrix(R_cur.T @ R_tgt).as_rotvec()
    return np.concatenate([e_pos, e_rot])


def _nullspace_method(jac_joints, delta, regularization_strength):
    """Joint velocities for an end-effector delta via damped least squares."""
    hess_approx = jac_joints.T @ jac_joints
    joint_delta = jac_joints.T @ delta
    if regularization_strength > 0:
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]


def qpos_from_site_pose(
    model,
    data,
    site_id,
    dof_indices,
    target_pos,
    target_quat,
    *,
    tol=1e-14,
    rot_weight=0.5,
    regularization_threshold=0.1,
    regularization_strength=3e-2,
    max_update_norm=2.0,
    progress_thresh=20.0,
    max_steps=100,
):
    """Levenberg-Marquardt IK for a site's 6-DoF pose, searched in place on ``data``.

    Warm-starts from ``data.qpos`` and perturbs only ``dof_indices`` until site ``site_id``
    reaches ``target_pos``/``target_quat``. Returns ``(qpos, err_norm, success)`` where ``qpos``
    is the (mutated) ``data.qpos`` view.
    """
    dof_indices = np.asarray(dof_indices)
    jac = np.empty((6, model.nv))
    jac_pos, jac_rot = jac[:3], jac[3:]
    err = np.empty(6)
    err_pos, err_rot = err[:3], err[3:]
    update_nv = np.zeros(model.nv)
    site_xquat = np.empty(4)
    neg_site_xquat = np.empty(4)
    err_rot_quat = np.empty(4)

    mj.mj_fwdPosition(model, data)
    site_xpos = data.site_xpos[site_id]
    site_xmat = data.site_xmat[site_id]

    err_norm = 0.0
    success = False
    for _ in range(max_steps):
        err_pos[:] = target_pos - site_xpos
        err_norm = np.linalg.norm(err_pos)
        mj.mju_mat2Quat(site_xquat, site_xmat)
        mj.mju_negQuat(neg_site_xquat, site_xquat)
        mj.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
        mj.mju_quat2Vel(err_rot, err_rot_quat, 1.0)
        err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            success = True
            break

        mj.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        reg = regularization_strength if err_norm > regularization_threshold else 0.0
        update_joints = _nullspace_method(jac[:, dof_indices], err, reg)

        update_norm = np.linalg.norm(update_joints)
        if err_norm / update_norm > progress_thresh:
            break
        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        update_nv[dof_indices] = update_joints
        mj.mj_integratePos(model, data.qpos, update_nv, 1.0)
        mj.mj_fwdPosition(model, data)

    return data.qpos, err_norm, success


class _SolverBase:
    """Base for MuJoCo-based IK solvers. Handles model loading, FK, and Jacobian.

    MuJoCo objects are built lazily on first use so that the solver stays
    naturally picklable (only urdf_xml, joint_names, control_frame, and
    solver params are stored as instance state).
    """

    def __init__(self, urdf_xml, joint_names, control_frame):
        self.urdf_xml = urdf_xml
        self.joint_names = tuple(joint_names)
        self.control_frame = control_frame
        self._mj = None  # backing for _model property

    @property
    def _model(self):
        if self._mj is not None:
            return self._mj
        spec = _prepare_spec(self.urdf_xml, self.control_frame)
        self._mj = spec.compile()
        self._data = mj.MjData(self._mj)
        self._site_id = mj.mj_name2id(self._mj, mj.mjtObj.mjOBJ_SITE, self.control_frame)
        self._jqpos = np.array([self._mj.joint(n).qposadr.item() for n in self.joint_names])
        self._joint_dof_ids = np.array([self._mj.joint(n).dofadr.item() for n in self.joint_names])
        jnt_ids = [mj.mj_name2id(self._mj, mj.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names]
        self._jlower = self._mj.jnt_range[jnt_ids, 0].copy()
        self._jupper = self._mj.jnt_range[jnt_ids, 1].copy()
        self._init_extra(spec)
        return self._mj

    @property
    def _joint_qpos_ids(self):
        _ = self._model
        return self._jqpos

    @property
    def _joint_lower(self):
        _ = self._model
        return self._jlower

    @property
    def _joint_upper(self):
        _ = self._model
        return self._jupper

    def _init_extra(self, spec):
        """Override in subclasses that need additional setup after model loading."""

    def _fk_jac(self, q):
        """Forward kinematics + Jacobian at joint positions q."""
        model = self._model
        self._data.qpos[self._jqpos] = q
        mj.mj_forward(model, self._data)
        pos = self._data.site_xpos[self._site_id].copy()
        R = self._data.site_xmat[self._site_id].reshape(3, 3).copy()
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mj.mj_jacSite(model, self._data, jacp, jacr, self._site_id)
        J = np.vstack([jacp[:, self._joint_dof_ids], jacr[:, self._joint_dof_ids]])
        return pos, R, J

    def __getstate__(self):
        # MuJoCo objects (_mj, _data, etc.) are not picklable; exclude all
        # _-prefixed attrs so properties rebuild lazily after unpickling.
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mj = None


class LMIKSolver(_SolverBase):
    """IK via Levenberg-Marquardt on the site pose. For sim data.

    Ports qpos_from_site_pose from dm_control.
    """

    def solve(self, current_q, target_ee_pose_vec):
        model = self._model
        self._data.qpos[self._joint_qpos_ids] = current_q
        qpos, _, _ = qpos_from_site_pose(
            model,
            self._data,
            self._site_id,
            self._joint_dof_ids,
            target_ee_pose_vec[:3],
            target_ee_pose_vec[3:7],
            rot_weight=0.5,
        )
        return qpos[self._joint_qpos_ids]


class DLSIKSolver(_SolverBase):
    """DLS IK with nullspace bias and backtracking line search.

    Ports inverse_kinematics_q0 from positronic-franka robot.hpp.
    """

    def __init__(
        self,
        urdf_xml,
        joint_names,
        control_frame,
        *,
        tol=1e-4,
        max_iters=150,
        min_step=1e-8,
        pinv_reg=0.03,
        nullspace_gain=0.002,
        line_search_alpha=1.0,
        line_search_beta=0.5,
        line_search_max_steps=20,
    ):
        super().__init__(urdf_xml, joint_names, control_frame)
        self.tol = tol
        self.max_iters = max_iters
        self.min_step = min_step
        self.pinv_reg = pinv_reg
        self.nullspace_gain = nullspace_gain
        self.line_search_alpha = line_search_alpha
        self.line_search_beta = line_search_beta
        self.line_search_max_steps = line_search_max_steps

    def solve(self, current_q, target_ee_pose_vec):  # noqa: C901
        t_tgt, R_tgt = _parse_target(target_ee_pose_vec)
        q = current_q.astype(np.float64).copy()
        n = len(q)

        for _ in range(self.max_iters):
            pos, R_cur, J = self._fk_jac(q)
            e = _cartesian_error(pos, R_cur, t_tgt, R_tgt)
            err_norm = np.linalg.norm(e)
            if err_norm < self.tol:
                break

            # DLS pseudoinverse + nullspace bias toward zero
            J_pinv = J.T @ np.linalg.inv(J @ J.T + self.pinv_reg * np.eye(6))
            dq = -J_pinv @ e + (np.eye(n) - J_pinv @ J) @ (-self.nullspace_gain * np.exp(err_norm) * q)

            # Backtracking line search
            step = self.line_search_alpha
            best_err, best_q = err_norm, q.copy()
            for _ in range(self.line_search_max_steps):
                q_trial = q + step * dq
                pos_t, R_t, _ = self._fk_jac(q_trial)
                err_trial = np.linalg.norm(_cartesian_error(pos_t, R_t, t_tgt, R_tgt))
                if err_trial < best_err:
                    best_err = err_trial
                    best_q = q_trial
                step *= self.line_search_beta
                if step < self.min_step:
                    break

            if best_err >= err_norm - 1e-9:
                break
            q = best_q

        return q


class DLSIKSolverWithLimits(_SolverBase):
    """DLS IK with joint limits via bounded least squares.

    Ports inverse_kinematics_with_limits from positronic-franka robot.hpp,
    replacing OSQP with scipy.optimize.lsq_linear.
    """

    def __init__(
        self,
        urdf_xml,
        joint_names,
        control_frame,
        *,
        tol=1e-4,
        max_iters=150,
        min_step=1e-8,
        pinv_reg=0.03,
        line_search_alpha=1.0,
    ):
        super().__init__(urdf_xml, joint_names, control_frame)
        self.tol = tol
        self.max_iters = max_iters
        self.min_step = min_step
        self.pinv_reg = pinv_reg
        self.line_search_alpha = line_search_alpha

    def solve(self, current_q, target_ee_pose_vec):
        t_tgt, R_tgt = _parse_target(target_ee_pose_vec)
        q = current_q.astype(np.float64).copy()
        n = len(q)
        sqrt_lam = np.sqrt(max(self.pinv_reg, 1e-6))

        for _ in range(self.max_iters):
            pos, R_cur, J = self._fk_jac(q)
            e = _cartesian_error(pos, R_cur, t_tgt, R_tgt)
            if np.linalg.norm(e) < self.tol:
                break

            # min ||[J; √λ I] dq - [-e; 0]||² s.t. joint_lower - q ≤ dq ≤ joint_upper - q
            A = np.vstack([J, sqrt_lam * np.eye(n)])
            b = np.concatenate([-e, np.zeros(n)])
            dq = lsq_linear(A, b, bounds=(self._joint_lower - q, self._joint_upper - q)).x

            dq_norm = np.linalg.norm(dq)
            if dq_norm < self.min_step:
                break

            step_scale = min(1.0, self.line_search_alpha / dq_norm) if self.line_search_alpha > 0 else 1.0
            q_next = np.clip(q + step_scale * dq, self._joint_lower, self._joint_upper)
            if np.linalg.norm(q_next - q) < self.min_step:
                break
            q = q_next

        return q


def ik_joints_from_episode(episode, solver_cls, tgt_ee_pose_key, current_q_key):
    """Episode -> Signal. Computes target joints from EE targets via IK.

    Reads 'urdf', 'joint_names', and 'control_frame' from episode statics.
    """
    solver = solver_cls(episode['urdf'], episode['joint_names'], episode['control_frame'])
    return transforms.pairwise(episode[current_q_key], episode[tgt_ee_pose_key], solver.solve)
