"""The shared damped-least-squares IK, solved on a small MuJoCo arm."""

import mujoco
import numpy as np

from positronic.simulator.env_server import arm_action
from positronic.simulator.env_server.tests.mujoco_chain import hinge_chain_xml

_JOINTS = 6
_POSITION_TOLERANCE, _ROTATION_TOLERANCE = 1e-3, 1e-2


def _model() -> mujoco.MjModel:
    """A six-hinge arm, and a sliding box the solve must not move."""
    return mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="box" pos="1 0 0"><joint name="box_x" type="slide" axis="1 0 0"/><geom size="0.05"/></body>'
        f'{hinge_chain_xml("arm", 0.0, _JOINTS)}</worldbody></mujoco>'
    )


def _arm_joints(model: mujoco.MjModel) -> list[int]:
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'arm_joint{i}') for i in range(1, _JOINTS + 1)]


def _site_pose(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'arm_grasp_site')
    return np.array(data.site_xpos[site]), np.array(data.site_xmat[site]).reshape(3, 3)


def _solve(model: mujoco.MjModel, data: mujoco.MjData, target_pos: np.ndarray, target_rot: np.ndarray):
    joints = _arm_joints(model)
    qpos_ids = model.jnt_qposadr[joints]
    quat, rotation_error = np.empty(4), np.empty(3)

    def pose_error(site_pos: np.ndarray, site_rot: np.ndarray) -> np.ndarray:
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(target_rot @ site_rot.T).reshape(9))
        mujoco.mju_quat2Vel(rotation_error, quat, 1.0)
        return np.concatenate([target_pos - site_pos, rotation_error])

    return arm_action.damped_least_squares_ik(
        model,
        live_qpos=data.qpos,
        site_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'arm_grasp_site'),
        qpos_ids=qpos_ids,
        dof_ids=model.jnt_dofadr[joints],
        start_q=data.qpos[qpos_ids],
        lower=model.jnt_range[joints, 0],
        upper=model.jnt_range[joints, 1],
        pose_error=pose_error,
        converged=lambda e: bool(
            np.linalg.norm(e[:3]) < _POSITION_TOLERANCE and np.linalg.norm(e[3:]) < _ROTATION_TOLERANCE
        ),
    )


def _live_scene(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[0] = 0.25  # the box, which a solve on the live data would carry along
    data.qpos[1:] = 0.1
    mujoco.mj_forward(model, data)
    return data


def test_a_reachable_pose_converges_and_leaves_the_live_scene_unmoved():
    model = _model()
    data = _live_scene(model)
    qpos_ids = model.jnt_qposadr[_arm_joints(model)]
    goal = data.qpos.copy()
    goal[qpos_ids] = [0.4, -0.3, 0.5, 0.2, -0.4, 0.3]
    target_pos, target_rot = _site_pose(model, goal)
    live_before = data.qpos.copy()

    solution = _solve(model, data, target_pos, target_rot)

    assert solution.converged
    reached = data.qpos.copy()
    reached[qpos_ids] = solution.q
    reached_pos, reached_rot = _site_pose(model, reached)
    np.testing.assert_allclose(reached_pos, target_pos, atol=_POSITION_TOLERANCE)
    np.testing.assert_allclose(reached_rot, target_rot, atol=_ROTATION_TOLERANCE)
    np.testing.assert_array_equal(data.qpos, live_before)


def test_an_unreachable_pose_reports_no_convergence_within_the_joint_range():
    model = _model()
    data = _live_scene(model)
    live_before = data.qpos.copy()

    solution = _solve(model, data, np.array([5.0, 0.0, 0.0]), np.eye(3))

    assert not solution.converged
    assert np.linalg.norm(solution.residual[:3]) > 4.0
    assert np.all(solution.q >= -3) and np.all(solution.q <= 3)
    np.testing.assert_array_equal(data.qpos, live_before)
