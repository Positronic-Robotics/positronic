"""The pose IK of the RoboLab env: its Panda kinematics against MuJoCo's Panda, the solve stepped in a loop that
reaches each step's joint targets, and the self-collision model of the FR3 driver."""

import mujoco
import numpy as np
import pytest
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from positronic.simulator.robolab import pose_ik
from positronic.utils import package_assets_path

_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
_HOME = np.array([0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0])
_DOWN = np.diag([1.0, -1.0, -1.0])  # the tool z axis points down
_NO_TOOL = np.eye(4)


def _pose(position, rotation=_DOWN) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return pose


def _run(
    target: np.ndarray, steps: int, q: np.ndarray = _HOME, flange_to_tool: np.ndarray = _NO_TOOL
) -> list[np.ndarray]:
    """The measured joints after each control step."""
    trajectory = [q]
    for _ in range(steps):
        trajectory.append(pose_ik.joint_targets(target, trajectory[-1], flange_to_tool, _LOWER, _UPPER))
    return trajectory


def test_panda_flange_matches_the_mujoco_panda():
    model = mujoco.MjModel.from_xml_path(package_assets_path('assets/mujoco/panda_nohand.xml'))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'attachment_site')
    qpos_ids = [model.joint(f'joint{i}').qposadr.item() for i in range(1, 8)]
    rng = np.random.default_rng(0)
    mount = None  # the fixed rotation from the DH flange axes to the MuJoCo attachment site axes
    for _ in range(20):
        q = rng.uniform(_LOWER, _UPPER)
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        flange = pose_ik.panda_flange(q)
        site_rotation = data.site_xmat[site].reshape(3, 3)
        mount = flange[:3, :3].T @ site_rotation if mount is None else mount
        np.testing.assert_allclose(flange[:3, 3], data.site_xpos[site], atol=1e-6)
        np.testing.assert_allclose(flange[:3, :3] @ mount, site_rotation, atol=1e-6)


def test_the_tool_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    flange_to_tool = _pose([0.01, -0.02, 0.15], Rotation.from_rotvec([0.3, -1.1, 0.7]).as_matrix())
    q = rng.uniform(_LOWER, _UPPER)
    tool, jacobian = pose_ik.tool_kinematics(q, flange_to_tool)
    for joint in range(7):
        dq = np.zeros(7)
        dq[joint] = 1e-6
        moved = pose_ik.tool_kinematics(q + dq, flange_to_tool)[0]
        np.testing.assert_allclose(-pose_ik.pose_error(tool, moved) / 1e-6, jacobian[:, joint], atol=1e-4)


def test_pose_error_of_a_half_turn_has_the_norm_pi():
    half_turn = _pose([0.0, 0.0, 0.0], np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(np.linalg.norm(pose_ik.pose_error(np.eye(4), half_turn)), np.pi)


@pytest.mark.parametrize('seed', range(20))
def test_minimize_on_box_matches_a_bounded_least_squares_solve(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(10, 7))
    h = a.T @ a + 1e-3 * np.eye(7)
    g = rng.normal(size=7)
    lo, hi = -rng.uniform(0.0, 0.3, size=7), rng.uniform(0.0, 0.3, size=7)
    lo[0] = hi[0] = 0.0  # a joint that the box fixes

    d = pose_ik.minimize_on_box(h, g, lo, hi)

    chol = np.linalg.cholesky(h)
    reference = lsq_linear(chol.T, -np.linalg.solve(chol, g), bounds=(lo - 1e-12, hi + 1e-12), tol=1e-12).x
    assert np.all(d >= lo) and np.all(d <= hi)
    np.testing.assert_allclose(d, reference, atol=1e-6)


def test_a_reachable_pose_converges_inside_the_joint_limits():
    target = pose_ik.panda_flange(np.array([0.3, 0.2, -0.1, -1.9, 0.1, 2.2, 0.5]))

    trajectory = _run(target, 10)

    assert all(np.all(q >= _LOWER) and np.all(q <= _UPPER) for q in trajectory)
    np.testing.assert_allclose(pose_ik.pose_error(pose_ik.panda_flange(trajectory[-1]), target), 0.0, atol=1e-4)


def test_each_step_stays_within_the_max_joint_step():
    trajectory = _run(_pose([0.2, -0.5, 0.6]), 20)

    steps = np.diff(trajectory, axis=0)
    assert np.abs(steps).max() <= pose_ik.MAX_JOINT_STEP + 1e-9


def test_a_pose_beyond_reach_settles_with_the_elbow_off_its_limit():
    trajectory = _run(_pose([0.95, 0.5, 0.3]), 200)

    steps = np.diff(trajectory, axis=0)
    tool_positions = np.array([pose_ik.panda_flange(q)[:3, 3] for q in trajectory[-50:]])
    assert all(np.all(q >= _LOWER) and np.all(q <= _UPPER) for q in trajectory)
    assert trajectory[-1][3] < _UPPER[3] - 0.1
    assert np.ptp(tool_positions, axis=0).max() < 1e-3
    assert np.abs(steps[-50:]).max() < 0.02


# The eef control frame of the RoboLab Panda in the flange frame. The tests below start from joints and commands that
# RoboLab episodes recorded.
_EEF = _pose([0.0, 0.0, 0.155], np.eye(3))


def _position_error(q: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pose_ik.pose_error(pose_ik.tool_kinematics(q, _EEF)[0], target)[:3]))


@pytest.mark.parametrize('joint', [0, 3])
def test_a_joint_pushed_past_its_limit_gets_a_target_at_that_limit(joint):
    # SpoonInMug trial 2 at 341.6 s: j1 is 0.28 rad past its upper limit. The FruitsOrangesOnPlate trials that raised
    # on such a state recorded no joints, so the joint here is 0.6 rad past its limit, more than MAX_JOINT_STEP.
    q0 = np.array([3.17858, -0.00805, -1.72896, -2.27623, -0.01310, 2.27854, 1.03406])
    q0[0] = _UPPER[0]
    q0[joint] = _UPPER[joint] + 0.6

    q = pose_ik.joint_targets(_pose([0.27, 0.44, 0.20]), q0, _EEF, _LOWER, _UPPER)

    assert np.all(q >= _LOWER) and np.all(q <= _UPPER)
    assert q[joint] == _UPPER[joint]
    assert np.abs(np.delete(q - q0, joint)).max() <= pose_ik.MAX_JOINT_STEP + 1e-9


def test_a_target_closer_than_the_stretched_elbow_reaches_folds_the_elbow():
    # LargerObjectRaisinBoxInBin trial 1 at 295 s: j4 at its upper limit, the tool 0.31 m from the command.
    q0 = np.array([0.00120, 0.75619, -0.02001, -0.06980, 0.01862, 0.77877, -0.00073])
    target = _pose([0.36, 0.0, 0.34])

    trajectory = _run(target, 30, q0, _EEF)

    assert np.abs(np.diff(trajectory, axis=0)).max() <= pose_ik.MAX_JOINT_STEP + 1e-9
    assert _position_error(trajectory[-1], target) < 1e-3


def test_the_link_frames_match_the_mujoco_panda_links():
    model = mujoco.MjModel.from_xml_path(package_assets_path('assets/mujoco/panda_nohand.xml'))
    data = mujoco.MjData(model)
    qpos_ids = [model.joint(f'joint{i}').qposadr.item() for i in range(1, 8)]
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.uniform(_LOWER, _UPPER)
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        frames = pose_ik.link_frames(q)
        for link in range(8):
            body = model.body(f'link{link}').id
            np.testing.assert_allclose(frames[link, :3, 3], data.xpos[body], atol=1e-9)
            np.testing.assert_allclose(frames[link, :3, :3], data.xmat[body].reshape(3, 3), atol=1e-9)


# The tests below port those of the FR3 driver's self-collision model. They use the FR3 joint limits of src/pose_ik.hpp.
_FR3_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
_FR3_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])


def test_the_closest_points_of_two_segments_are_on_their_common_normal_or_at_an_end():
    o, x, z = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    a, b = pose_ik.closest_points_of_segments(-x, x, np.array([0.0, -1.0, 1.0]), np.array([0.0, 1.0, 1.0]))
    np.testing.assert_allclose(a, o, atol=1e-12)
    np.testing.assert_allclose(b, z, atol=1e-12)
    c, d = pose_ik.closest_points_of_segments(o, x, np.array([2.0, 1.0, 0.0]), np.array([2.0, -1.0, 0.0]))
    np.testing.assert_allclose(c, x, atol=1e-12)
    np.testing.assert_allclose(d, 2 * x, atol=1e-12)
    e, f = pose_ik.closest_points_of_segments(o, x, z, z + x)
    assert abs(np.linalg.norm(e - f) - 1.0) < 1e-12


def test_the_clearance_gradient_is_the_derivative_of_the_clearance():
    rng = np.random.default_rng(11)
    h = 1e-6
    for _ in range(50):
        q = pose_ik.POSTURE_JOINTS + 1.5 * rng.uniform(-1.0, 1.0, size=7)
        gradient = pose_ik.self_collision_clearances(q)[1]
        for joint, step in enumerate(h * np.eye(7)):
            above = pose_ik.self_collision_clearances(q + step)[0]
            below = pose_ik.self_collision_clearances(q - step)[0]
            np.testing.assert_allclose(gradient[:, joint], (above - below) / (2 * h), atol=1e-6)


def test_no_joint_changes_a_clearance_faster_than_its_rate_bound():
    rng = np.random.default_rng(13)
    for q in rng.uniform(_FR3_LOWER, _FR3_UPPER, size=(2000, 7)):
        gradient = pose_ik.self_collision_clearances(q)[1]
        assert np.all(np.abs(gradient) <= pose_ik.CLEARANCE_RATE_BOUNDS + 1e-12)


def _clearance(q: np.ndarray) -> float:
    return float(pose_ik.pair_clearances(q).min())


def _least_clearance_on_the_move(start: np.ndarray, end: np.ndarray) -> float:
    """The least clearance at ``start`` and at 2000 points of the straight joint move from it to ``end``."""
    parts = np.arange(2001) / 2000
    return float(pose_ik.pair_clearances(start + parts[:, None] * (end - start)).min())


def test_a_move_that_dips_between_two_of_sixteen_points_stops_before_the_dip():
    # The clearance falls to 0.084 m between the points 8/16 and 9/16 of the move, which both keep 0.09 m.
    start = np.array([-2.025573, 1.055818, 2.246552, -2.790485, 2.673202, 2.095729, -1.071398])
    end = np.array([1.167132, 0.990198, -2.123749, -2.129653, 0.255810, 1.355959, 2.292034])
    sixteenths = start + np.arange(17)[:, None] / 16 * (end - start)
    assert np.all(pose_ik.pair_clearances(sixteenths) >= pose_ik.SELF_COLLISION_CLEARANCE)
    assert _least_clearance_on_the_move(start, end) < pose_ik.SELF_COLLISION_CLEARANCE - 0.005

    stop = pose_ik.stop_before_self_collision(start, end)

    floor = pose_ik.SELF_COLLISION_CLEARANCE - pose_ik.SELF_COLLISION_PATH_TOLERANCE
    assert _least_clearance_on_the_move(start, stop) >= floor
    assert np.linalg.norm(stop - start) > 0.5 * np.linalg.norm(end - start)
    after_the_dip = start + 9 / 16 * (end - start)
    assert _least_clearance_on_the_move(after_the_dip, end) >= pose_ik.SELF_COLLISION_CLEARANCE
    assert np.array_equal(pose_ik.stop_before_self_collision(after_the_dip, end), end)


# An FR3 posture with the shoulder back and the elbow folded: link 5 is 5.1 cm from link 2, closer than the 6.4 cm at
# which the FR3 self-collision reflex stops the arm.
_SHOULDER_BACK_ELBOW_FOLDED = np.array([
    0.98877418,
    -1.15661502,
    -0.96312511,
    -2.93868756,
    -0.990777671,
    2.05664039,
    1.14565432,
])


def _towards_the_folded_elbow(part: float) -> np.ndarray:
    """Joints a part of the way from ``POSTURE_JOINTS`` to the folded elbow, at 0.123 m clearance for 0.84."""
    return pose_ik.POSTURE_JOINTS + part * (_SHOULDER_BACK_ELBOW_FOLDED - pose_ik.POSTURE_JOINTS)


def _flange_miss(q: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """The flange miss of ``q`` against the ``4 x 4`` pose ``target``, in mm and deg."""
    e = pose_ik.pose_error(pose_ik.panda_flange(q), target)
    return float(np.linalg.norm(e[:3]) * 1e3), float(np.degrees(np.linalg.norm(e[3:])))


def _fr3_joint_targets(target: np.ndarray, q0: np.ndarray) -> np.ndarray:
    return pose_ik.joint_targets(target, q0, _NO_TOOL, _FR3_LOWER, _FR3_UPPER)


def test_the_move_to_a_pose_whose_joints_are_too_close_keeps_the_self_collision_clearance():
    q0 = _towards_the_folded_elbow(0.84)
    target = pose_ik.panda_flange(_SHOULDER_BACK_ELBOW_FOLDED)
    assert _clearance(q0) > pose_ik.SELF_COLLISION_PUSH_CLEARANCE

    q = _fr3_joint_targets(target, q0)

    assert _least_clearance_on_the_move(q0, q) >= pose_ik.SELF_COLLISION_CLEARANCE - 1e-9
    assert _flange_miss(q, target)[0] < _flange_miss(q0, target)[0]


def test_a_pose_whose_joints_are_just_outside_the_push_clearance_is_reached_within_0_1_mm_and_0_1_deg():
    q_target = _towards_the_folded_elbow(0.86)
    clearance = _clearance(q_target)
    push = pose_ik.SELF_COLLISION_PUSH_CLEARANCE
    assert push < clearance < push + 0.02
    target = pose_ik.panda_flange(q_target)

    mm, deg = _flange_miss(_fr3_joint_targets(target, _towards_the_folded_elbow(0.84)), target)

    assert mm < 0.1 and deg < 0.1


def test_a_start_inside_the_clearance_moves_out_and_towards_the_target():
    # A flange target 20 cm in front of the base and 30 cm below the flange at this start. The solve from this start
    # stops at the j3 lower limit 7 cm short of it, so the test asserts progress and not arrival.
    xyzw = [0.993636106, -0.065788185, -0.091428419, 0.000218116]
    target = _pose([0.202441812, -0.005892748, 0.048565254], Rotation.from_quat(xyzw).as_matrix())
    q = _SHOULDER_BACK_ELBOW_FOLDED
    assert _clearance(q) < pose_ik.SELF_COLLISION_CLEARANCE

    for _ in range(30):
        next_q = _fr3_joint_targets(target, q)
        floor = min(pose_ik.SELF_COLLISION_CLEARANCE, _clearance(q))
        assert _least_clearance_on_the_move(q, next_q) >= floor - 1e-9, 'a move brings the links closer than its start'
        q = next_q

    assert _clearance(q) >= pose_ik.SELF_COLLISION_CLEARANCE
    assert _flange_miss(q, target)[0] < _flange_miss(_SHOULDER_BACK_ELBOW_FOLDED, target)[0]


def _pair(link_a: int, link_b: int) -> int:
    return pose_ik.SELF_COLLISION_PAIRS.index((link_a, link_b))


def _least_pair_clearances_on_the_move(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Per checked pair, the least clearance at ``start`` and at 2000 points of the straight joint move to ``end``."""
    parts = np.arange(2001) / 2000
    return pose_ik.pair_clearances(start + parts[:, None] * (end - start)).min(axis=0)


def test_a_move_that_escapes_one_close_pair_does_not_take_another_pair_below_the_floor():
    # Link 5 starts 0.051 m from link 2 and ends 0.078 m from it. Link 7 starts 0.155 m from link 2 and ends 0.073 m
    # from it. No pair comes closer than link 5 starts.
    start = _SHOULDER_BACK_ELBOW_FOLDED
    end = np.array([1.384, -1.477, -0.791, -3.032, -0.499, 1.563, 1.542])
    close, other = _pair(2, 5), _pair(2, 7)
    at_start = pose_ik.pair_clearances(start)
    assert at_start[close] < pose_ik.SELF_COLLISION_CLEARANCE
    assert at_start[other] > pose_ik.SELF_COLLISION_PUSH_CLEARANCE
    assert pose_ik.pair_clearances(end)[close] > at_start[close]
    assert _least_clearance_on_the_move(start, end) >= at_start[close]
    assert _least_pair_clearances_on_the_move(start, end)[other] < pose_ik.SELF_COLLISION_CLEARANCE - 0.01

    stop = pose_ik.stop_before_self_collision(start, end)

    floor = np.minimum(pose_ik.SELF_COLLISION_CLEARANCE, at_start) - pose_ik.SELF_COLLISION_PATH_TOLERANCE
    assert np.all(_least_pair_clearances_on_the_move(start, stop) >= floor), 'a pair goes below its floor'
    assert pose_ik.pair_clearances(stop)[close] > at_start[close]


def test_a_pair_that_starts_inside_the_clearance_keeps_its_start_clearance():
    # A move that takes link 5 away from link 2 plays to its end at 0.087 m. The reverse move brings link 5 closer and
    # stops.
    start = _SHOULDER_BACK_ELBOW_FOLDED
    out = np.array([1.173, -1.186, -1.034, -2.834, -0.818, 2.141, 1.153])
    close = _pair(2, 5)
    at_start = pose_ik.pair_clearances(start)[close]
    on_the_way_out = _least_pair_clearances_on_the_move(start, out)
    assert pose_ik.pair_clearances(out)[close] < pose_ik.SELF_COLLISION_CLEARANCE
    assert on_the_way_out[close] >= at_start
    assert np.all(np.delete(on_the_way_out, close) > pose_ik.SELF_COLLISION_CLEARANCE)
    assert np.array_equal(pose_ik.stop_before_self_collision(start, out), out)

    into = start - (out - start)
    assert pose_ik.pair_clearances(into)[close] < at_start
    stop = pose_ik.stop_before_self_collision(start, into)
    assert _least_pair_clearances_on_the_move(start, stop)[close] >= at_start - pose_ik.SELF_COLLISION_PATH_TOLERANCE
