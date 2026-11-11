"""The pose IK of the RoboLab env: its Panda kinematics against MuJoCo's Panda, and the solve stepped in a loop
that reaches each step's joint targets."""

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


def _rotation_error_deg(q: np.ndarray, target: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(pose_ik.pose_error(pose_ik.tool_kinematics(q, _EEF)[0], target)[3:])))


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


@pytest.mark.parametrize(
    'q0, target',
    [
        # TakeMeasuringSpoonOut trial 1 at 220 s: j2, j3 and j5 at their limits, the tool 0.40 m from the command.
        (
            np.array([-0.64115, 1.76280, -2.89700, -1.62606, -2.89385, 0.05128, 0.21993]),
            _pose([0.42, -0.40, 0.055], Rotation.from_quat([-0.94765, 0.31931, 0.0, 0.0]).as_matrix()),
        ),
        # OneBottleInSquarePail trial 0 at 220 s: j5, j6 and j7 at their limits, the tool 0.20 m from the command.
        (np.array([-1.91271, -0.14115, 1.83762, -2.43315, -2.89728, 3.75250, 2.87750]), _pose([0.40, 0.003, 0.32])),
    ],
    ids=['rotation', 'wrist'],
)
def test_a_pose_out_of_reach_from_the_joints_brings_the_tool_to_the_position(q0, target):
    trajectory = _run(target, 60, q0, _EEF)

    assert np.abs(np.diff(trajectory, axis=0)).max() <= pose_ik.MAX_JOINT_STEP + 1e-9
    assert _position_error(trajectory[-1], target) < 0.05
    assert _rotation_error_deg(trajectory[-1], target) < 60.0
    assert np.abs(np.diff(trajectory[-10:], axis=0)).max() < 1e-3


@pytest.mark.parametrize(
    'q0, target',
    [
        # ReorientAllMugs trial 0 at 266 s: the steps stall 9 cm from the position, with the tool turned 2 degrees off.
        (
            np.array([0.03500, 0.61112, 0.50942, -3.06172, -2.89745, 0.69585, 2.86389]),
            _pose([0.39509, 0.09899, 0.18676], Rotation.from_quat([-0.56515, -0.09556, -0.81506, 0.08456]).as_matrix()),
        ),
        # TakeMugsOffOfShelf trial 0 at 16 s: the steps stall 4 cm from the position, with the tool on the rotation.
        (
            np.array([0.00002, -1.00512, -0.00004, -3.05898, -0.00008, 2.74874, 0.00006]),
            _pose([0.36485, 0.0, 0.29747], Rotation.from_quat([-0.92388, 0.0, -0.38268, 0.0]).as_matrix()),
        ),
    ],
    ids=['reorient', 'shelf'],
)
def test_a_stall_near_the_position_keeps_the_rotation(q0, target):
    trajectory = _run(target, 60, q0, _EEF)

    assert _position_error(trajectory[-1], target) < 0.1
    assert _rotation_error_deg(trajectory[-1], target) < 3.0
