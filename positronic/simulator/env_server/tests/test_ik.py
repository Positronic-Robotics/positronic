"""The shared damped-least-squares IK, solved on a small MuJoCo arm."""

import mujoco
import numpy as np
import pytest

from positronic.simulator.env_server import dls_ik, mujoco_ik
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

    return mujoco_ik.damped_least_squares_ik(
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


def _wrist(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One hinge about z at the origin, whose site turns with it and does not move."""
    turn = np.array([[np.cos(q[0]), -np.sin(q[0]), 0.0], [np.sin(q[0]), np.cos(q[0]), 0.0], [0.0, 0.0, 1.0]])
    return np.zeros(3), turn, np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])


def test_a_solve_that_stops_at_a_joint_limit_converges_from_the_middle_of_the_joint_ranges():
    _, target_rot, _ = _wrist(np.array([-1.5]))
    target = {'lower': [-2.0], 'upper': [2.0], 'target_pos': np.zeros(3), 'target_rot': target_rot}

    # From 1.9, the shorter turn to -1.5 goes up, past the limit at 2.
    stopped = dls_ik.damped_least_squares_ik(_wrist, start_q=[1.9], **target)
    solution = dls_ik.solve(_wrist, measured_q=[1.9], **target)

    assert stopped.fit is dls_ik.Fit.MISS
    np.testing.assert_allclose(stopped.q, [2.0])
    assert solution.fit is dls_ik.Fit.CONVERGED
    np.testing.assert_allclose(solution.q, [-1.5], atol=dls_ik.IK_ROTATION_TOLERANCE)


def test_a_half_turn_converges():
    _, target_rot, _ = _wrist(np.array([np.pi]))

    solution = dls_ik.solve(
        _wrist, measured_q=[0.0], lower=[-4.0], upper=[4.0], target_pos=np.zeros(3), target_rot=target_rot
    )

    assert solution.fit is dls_ik.Fit.CONVERGED
    np.testing.assert_allclose(abs(solution.q[0]), np.pi, atol=dls_ik.IK_ROTATION_TOLERANCE)


def test_a_start_past_a_joint_limit_solves_to_joints_within_the_limits():
    _, target_rot, _ = _wrist(np.array([2.0]))

    solution = dls_ik.solve(
        _wrist, measured_q=[2.0 + 1e-6], lower=[-2.0], upper=[2.0], target_pos=np.zeros(3), target_rot=target_rot
    )

    assert solution.fit is dls_ik.Fit.CONVERGED
    assert solution.q[0] <= 2.0


def test_a_target_just_past_a_joint_limit_gets_a_near_solution_at_the_limit():
    _, target_rot, _ = _wrist(np.array([2.05]))

    solution = dls_ik.solve(
        _wrist, measured_q=[1.95], lower=[-2.0], upper=[2.0], target_pos=np.zeros(3), target_rot=target_rot
    )

    assert solution.fit is dls_ik.Fit.NEAR
    np.testing.assert_allclose(solution.q, [2.0])


def test_a_target_far_past_a_joint_limit_misses():
    _, target_rot, _ = _wrist(np.array([2.5]))

    solution = dls_ik.solve(
        _wrist, measured_q=[1.95], lower=[-2.0], upper=[2.0], target_pos=np.zeros(3), target_rot=target_rot
    )

    assert solution.fit is dls_ik.Fit.MISS
    assert np.linalg.norm(solution.residual[3:]) > dls_ik.NEAR_ROTATION_TOLERANCE


def _crank(radius: float):
    """One hinge about z at the origin, whose site is ``radius`` from the axis and turns with it."""

    def kinematics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, turn, _ = _wrist(q)
        jac = np.array([[-radius * np.sin(q[0])], [radius * np.cos(q[0])], [0.0], [0.0], [0.0], [1.0]])
        return radius * turn[:, 0], turn, jac

    return kinematics


@pytest.mark.parametrize(
    ('radius', 'turn', 'shift', 'moves_too_far'),
    [(0.1, 0.05, 0.0, 'position'), (2.0, 0.0, 0.04, 'rotation')],
    ids=['turn in place', 'shift along the circle'],
)
def test_a_step_that_moves_the_site_farther_than_the_target_misses(radius, turn, shift, moves_too_far):
    # The hinge turns the site only as it moves the site along its circle. The best pose moves the site farther than
    # the target in position (a turn in place) or in rotation (a shift that keeps the rotation).
    crank = _crank(radius)
    site_pos, _, _ = crank(np.array([0.0]))
    _, target_rot, _ = _wrist(np.array([turn]))
    target_pos = site_pos + [0.0, shift, 0.0]
    target = {'lower': [-1.0], 'upper': [1.0], 'target_pos': target_pos, 'target_rot': target_rot}

    stopped = dls_ik.damped_least_squares_ik(crank, start_q=[0.0], **target)
    solution = dls_ik.solve(crank, measured_q=[0.0], **target)

    moved_pos, moved_rot, _ = crank(stopped.q)
    position_move = np.linalg.norm(moved_pos - site_pos) - np.linalg.norm(target_pos - site_pos)
    rotation_move = abs(stopped.q[0]) - turn
    assert np.linalg.norm(stopped.residual[:3]) < dls_ik.NEAR_POSITION_TOLERANCE
    assert np.linalg.norm(stopped.residual[3:]) < dls_ik.NEAR_ROTATION_TOLERANCE
    assert np.linalg.norm(stopped.residual) < np.linalg.norm([shift, turn])
    if moves_too_far == 'position':
        assert position_move > dls_ik.IK_POSITION_TOLERANCE
    else:
        assert rotation_move > dls_ik.IK_ROTATION_TOLERANCE
    assert solution.fit is dls_ik.Fit.MISS


def test_a_step_that_ends_no_nearer_the_target_misses():
    # The target is 5 mm out from the circle of the site, which the hinge cannot leave.
    crank = _crank(1.0)
    site_pos, site_rot, _ = crank(np.array([0.3]))

    solution = dls_ik.solve(
        crank, measured_q=[0.3], lower=[-1.0], upper=[1.0], target_pos=1.005 * site_pos, target_rot=site_rot
    )

    assert np.linalg.norm(solution.residual[:3]) < dls_ik.NEAR_POSITION_TOLERANCE
    assert solution.fit is dls_ik.Fit.MISS


def test_only_a_solution_from_the_measured_joints_is_near(monkeypatch):
    # The middle of the joint ranges is 1.0. Its solve ends 0.01 rad short of the target, which would be near. The
    # solves from the measured joints do not move.
    _, target_rot, _ = _wrist(np.array([0.06]))
    target = {'lower': [-1.0], 'upper': [3.0], 'target_pos': np.zeros(3), 'target_rot': target_rot}
    solved_from = []

    def solve_once(kinematics, *, start_q, drop_blocked=False, **target):
        solved_from.append(float(start_q[0]))
        q = np.array([0.05]) if start_q[0] == 1.0 else np.array([0.0])
        site_pos, site_rot, _ = kinematics(q)
        return dls_ik.IkSolution(
            q, dls_ik._pose_error(target['target_pos'], target['target_rot'], site_pos, site_rot), dls_ik.Fit.MISS
        )

    monkeypatch.setattr(dls_ik, 'damped_least_squares_ik', solve_once)
    solution = dls_ik.solve(_wrist, measured_q=[0.0], **target)

    assert solved_from == [0.0, 1.0, 0.0]
    assert solution.fit is dls_ik.Fit.MISS
    np.testing.assert_array_equal(solution.q, [0.0])


def _slides(columns: list[list[float]]):
    """Joints that each move the site in a plane along one column, with no turn."""
    plane = np.array(columns, dtype=np.float64).T

    def kinematics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        jac = np.zeros((6, plane.shape[1]))
        jac[:2] = plane
        return np.append(plane @ q, 0.0), np.eye(3), jac

    return kinematics


def test_drop_blocked_moves_a_joint_off_its_limit_when_the_step_points_into_the_range():
    solution = dls_ik.damped_least_squares_ik(
        _slides([[1.0, 0.0]]),
        start_q=[0.0],
        lower=[0.0],
        upper=[1.0],
        target_pos=[0.5, 0.0, 0.0],
        target_rot=np.eye(3),
        drop_blocked=True,
    )

    assert solution.fit is dls_ik.Fit.CONVERGED


def test_drop_blocked_also_leaves_out_a_joint_that_the_step_blocks_after_another_joint_is_left_out(monkeypatch):
    # From joints 0 and 1 at their upper limits, the step moves joint 0 past its limit. Without joint 0, the step moves
    # joint 1 past its limit. Without both, joint 2 alone takes the step: -1.5, the error (-1, -2) on its column (1, 1).
    monkeypatch.setattr(dls_ik, 'IK_ITERATIONS', 1)
    kinematics = _slides([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    start_q = np.array([1.0, 1.0, 0.0])
    site_pos, _, _ = kinematics(start_q)

    solution = dls_ik.damped_least_squares_ik(
        kinematics,
        start_q=start_q,
        lower=[-1.0, -1.0, -3.0],
        upper=[1.0, 1.0, 3.0],
        target_pos=site_pos + [-1.0, -2.0, 0.0],
        target_rot=np.eye(3),
        drop_blocked=True,
    )

    np.testing.assert_allclose(solution.q, [1.0, 1.0, -1.5], atol=0.01)
