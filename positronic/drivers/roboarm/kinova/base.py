# This file contains code derived from tidybot2, which is:
# Copyright (c) 2024 Jimmy Wu
# Released under MIT License (https://github.com/jimmyyhwu/tidybot2/blob/main/LICENSE)
#
# All modifications and additions are:
# Copyright (c) 2025 Positronic Robotics Inc.
# All rights reserved.
# This code may not be used, copied, modified, merged, published, distributed,
# sublicensed, and/or sold without explicit permission from the copyright holder.

import math

import mujoco
import numpy as np
import pinocchio as pin
from cvxopt import matrix, solvers
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from positronic import geom
from positronic.utils.rerun_compat import log_numeric_series

K_r = np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18])
K_l = np.diag([75.0, 75.0, 75.0, 75.0, 40.0, 40.0, 40.0])
K_lp = np.diag([5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0])
K_p = np.diag([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
K_d = np.diag([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
K_r_inv = np.linalg.inv(K_r)
K_r_K_l = K_r @ K_l
_DT = 0.001

_DAMPING_COEFF = 1e-12
_MAX_ANGLE_CHANGE = np.deg2rad(45)


def wrap_joint_angle(q, q_base):
    return q_base + np.mod(q - q_base + np.pi, 2 * np.pi) - np.pi


class KinematicsSolver:
    """Solves forward and inverse kinematics for the Kinova arm."""

    def __init__(self, path: str = 'positronic/drivers/roboarm/kinova/gen3.xml', ee_offset=0.0, site_name='pinch_site'):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0

        # Cache references
        self.qpos0 = np.zeros(self.model.nq)
        self.site_id = self.model.site(site_name).id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Add end effector offset for gripper
        # 0.061525 comes from the Kinova URDF
        self.model.site(self.site_id).pos = np.array([0.0, 0.0, -0.061525 - ee_offset])

        # Preallocate arrays
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = _DAMPING_COEFF * np.eye(6)
        self.eye = np.eye(self.model.nv)

        self.joint_limits_idx = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        for i, row in enumerate(self.model.jnt_range):
            if not (row[0] == row[1] == 0):
                self.joint_limits_idx.append(i)
                self.joint_limits_lower.append(row[0])
                self.joint_limits_upper.append(row[1])

        self.joint_limits_idx = np.array(self.joint_limits_idx)
        self.joint_limits_lower = np.array(self.joint_limits_lower)
        self.joint_limits_upper = np.array(self.joint_limits_upper)

        if len(self.joint_limits_idx) > 0:
            self.G = np.vstack([
                np.eye(self.model.nv)[self.joint_limits_idx],
                -np.eye(self.model.nv)[self.joint_limits_idx],
            ])
            self.G = matrix(self.G)
        else:
            self.G = None

    def forward(self, qpos):
        self.data.qpos = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        pos = self.data.site(self.site_id).xpos.copy()
        mat = self.data.site(self.site_id).xmat.copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, mat)
        return geom.Transform3D(pos, geom.Rotation.from_quat(quat))

    def inverse(self, pos: geom.Transform3D, qpos0: np.ndarray, max_iters: int = 40, err_thresh: float = 1e-5):
        self.data.qpos = qpos0

        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # Translational error
            self.err_pos[:] = pos.translation - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, pos.rotation.as_quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            if np.linalg.norm(self.err) < err_thresh:
                break

            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)
            update = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.damping, self.err)
            qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            update += (
                self.eye - (self.jac.T @ np.linalg.pinv(self.jac @ self.jac.T + self.damping)) @ self.jac
            ) @ qpos0_err

            # Enforce max angle change
            update_max = np.abs(update).max()
            if update_max > _MAX_ANGLE_CHANGE:
                update *= _MAX_ANGLE_CHANGE / update_max

            # Apply update
            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)

        return self.data.qpos.copy()

    def inverse_limits(
        self,
        pos: geom.Transform3D,
        qpos0: np.ndarray,
        max_iters: int = 20,
        err_thresh: float = 1e-3,
        clamp=True,
        debug=False,
    ):
        assert self.G is not None, 'Joint limits are not set'
        solvers.options['show_progress'] = False
        qpos0 = wrap_joint_angle(qpos0, np.zeros(7)) if clamp else qpos0
        self.data.qpos = qpos0

        iter = 0
        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            # Translational error
            self.err_pos[:] = pos.translation - self.site_pos

            # Rotational error
            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, pos.rotation.as_quat, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            if np.linalg.norm(self.err) < err_thresh:
                break

            # Get Jacobian
            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)

            # Setup QP problem
            # min_x (1/2) x^T P x + q^T x
            # s.t. G x <= h
            #      A x = b

            # Objective: min ||J·Δq - e||² + α||Δq - (q₀ - q)||²
            # where α is a small weight for the null space objective
            alpha = 1e-5
            n = self.model.nv

            # P = J^T J + α I
            P = self.jac.T @ self.jac + alpha * np.eye(n)
            P = matrix(P)

            # q = -J^T e - α (q₀ - q)
            qpos0_err = np.mod(self.qpos0 - self.data.qpos + np.pi, 2 * np.pi) - np.pi
            q = -self.jac.T @ self.err - alpha * qpos0_err
            q = matrix(q)

            # Joint limit constraints: lb - q <= Δq <= ub - q
            # Rewrite as:
            # Δq <= ub - q
            # -Δq <= q - lb

            # h = [ub - q; q - lb]
            h_upper = self.joint_limits_upper - self.data.qpos[self.joint_limits_idx]
            h_lower = self.data.qpos[self.joint_limits_idx] - self.joint_limits_lower
            h = np.concatenate([h_upper, h_lower])
            h = matrix(h)

            # No equality constraints
            A = matrix(0.0, (0, n))
            b = matrix(0.0, (0, 1))

            solution = solvers.qp(P, q, self.G, h, A, b)
            update = np.array(solution['x']).flatten()

            mujoco.mj_integratePos(self.model, self.data.qpos, update, 1.0)
            self.data.qpos = wrap_joint_angle(self.data.qpos, np.zeros(7)) if clamp else self.data.qpos

            iter += 1

        if debug:
            log_numeric_series('ik/iter', iter)
            log_numeric_series('ik/err', np.linalg.norm(self.err))
            log_numeric_series('ik/err/pos', np.linalg.norm(self.err_pos))
            log_numeric_series('ik/err/rot', np.linalg.norm(self.err_rot))
            log_numeric_series('ik/update', self.data.qpos - qpos0)
        return self.data.qpos.copy()


class JointCompliantController:
    """Implements compliant joint control for the Kinova arm."""

    class LowPassFilter:
        """Simple low-pass filter implementation."""

        def __init__(self, alpha, initial_value):
            assert 0 < alpha <= 1, 'Alpha must be between 0 and 1'
            self.alpha = alpha
            self.y = initial_value

        def filter(self, x):
            self.y = self.alpha * x + (1 - self.alpha) * self.y
            return self.y

    def __init__(
        self,
        actuator_count,
        path: str = 'positronic/drivers/roboarm/kinova/model.urdf',
        relative_dynamics_factor=0.5,
        max_velocity=(1.396, 1.396, 1.396, 1.396, 2.443, 2.443, 2.443),
        max_acceleration=(4.188, 4.188, 4.188, 4.188, 7.853, 7.853, 7.853),
    ):
        self.q_s = None
        self.q_d = None
        self.dq_d = None
        self.q_n = None
        self.dq_n = None
        self.tau_filter = None

        self.actuator_count = actuator_count
        self.otg = None
        self.otg_inp = None
        self.otg_out = None
        self.otg_res = None
        self.relative_dynamics_factor = relative_dynamics_factor

        self.target_qpos = None

        # Initialize pinocchio model and data
        self.model = pin.buildModelFromUrdf(path)
        self.joint_nq = [joint.nq for joint in self.model.joints]
        self.data = self.model.createData()
        self._q_pin = np.zeros(self.model.nq)
        self.max_velocity = np.array(max_velocity) * self.relative_dynamics_factor
        self.max_acceleration = np.array(max_acceleration) * self.relative_dynamics_factor

    def set_target_qpos(self, qpos):
        self.target_qpos = qpos

    @property
    def finished(self):
        return self.otg_res == Result.Finished

    def gravity(self, q):
        q_pin = self._q_pin  # Reuse pre-allocated q_pin
        q_pin_idx = 0
        q_idx = 0
        for joint_nq in self.joint_nq[1:]:  # skip base joint
            if joint_nq == 1:  # revolute joint
                q_pin[q_pin_idx] = q[q_idx]
                q_pin_idx += 1
                q_idx += 1
            elif joint_nq == 2:  # continuous joint
                q_pin[q_pin_idx], q_pin[q_pin_idx + 1] = math.cos(q[q_idx]), math.sin(q[q_idx])
                q_pin_idx += 2
                q_idx += 1
        return pin.computeGeneralizedGravity(self.model, self.data, q_pin)

    def compute_torque(self, q, dq, tau):
        gravity = self.gravity(q)

        # Initialize controller state if needed
        if self.q_s is None:
            self.q_s = q.copy()
            self.q_d = q.copy()
            self.dq_d = np.zeros_like(q)
            self.q_n = q.copy()
            self.dq_n = dq.copy()
            self.tau_filter = JointCompliantController.LowPassFilter(0.01, tau.copy())

            self.otg = Ruckig(self.actuator_count, _DT)
            self.otg_inp = InputParameter(self.actuator_count)
            self.otg_out = OutputParameter(self.actuator_count)
            self.otg_inp.max_velocity = self.max_velocity
            self.otg_inp.max_acceleration = self.max_acceleration
            self.otg_inp.current_position = q.copy()
            self.otg_inp.current_velocity = dq.copy()
            self.otg_inp.target_position = q.copy()
            self.otg_inp.target_velocity = np.zeros(self.actuator_count)
            self.otg_res = Result.Finished

        self.q_s = wrap_joint_angle(q, self.q_s)
        dq_s = dq.copy()  # TODO: It seems that we don't need copy here
        tau_s_f = self.tau_filter.filter(tau)

        if self.target_qpos is not None:
            qpos = wrap_joint_angle(self.target_qpos, self.q_s)
            self.otg_inp.target_position = qpos
            self.otg_res = Result.Working

            self.target_qpos = None

        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.q_d[:] = self.otg_out.new_position
            self.dq_d[:] = self.otg_out.new_velocity

        tau_task = -K_p @ (self.q_n - self.q_d) - K_d @ (self.dq_n - self.dq_d) + gravity

        # Nominal motor plant
        ddq_n = K_r_inv @ (tau_task - tau_s_f)
        self.dq_n += ddq_n * _DT
        self.q_n += self.dq_n * _DT

        tau_f = K_r_K_l @ ((self.dq_n - dq_s) + K_lp @ (self.q_n - self.q_s))  # Nominal friction

        return tau_task + tau_f
