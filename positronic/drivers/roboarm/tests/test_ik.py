import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from positronic import keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm.ik import DLSIKSolver, DLSIKSolverWithLimits, LMIKSolver, ik_joints_from_episode
from positronic.utils import package_assets_path

URDF = Path(package_assets_path('assets/mujoco/panda_ik.xml')).read_text()
JOINT_NAMES = [f'joint{i}' for i in range(1, 8)]
CONTROL_FRAME = 'end_effector'

# Reachable joint configs: home, stretched, and two arbitrary
TEST_CONFIGS = [
    np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0]),
    np.array([0.5, 0.3, -0.4, -1.2, 0.8, 1.0, -0.3]),
    np.array([-0.8, -1.0, 0.6, -2.5, -0.3, 2.5, 0.9]),
]


def _fk(urdf_xml, q):
    """Compute EE pose [tx,ty,tz,w,x,y,z] via MuJoCo FK."""
    model = mj.MjModel.from_xml_string(urdf_xml)
    data = mj.MjData(model)
    qpos_ids = [model.joint(n).qposadr.item() for n in JOINT_NAMES]
    data.qpos[qpos_ids] = q
    mj.mj_forward(model, data)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, 'end_effector')
    pos = data.site_xpos[site_id].copy()
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, data.site_xmat[site_id])
    return np.concatenate([pos, quat])


def _assert_fk_matches(solver, q_start, target_pose, pos_tol=1e-3, rot_tol=1e-2):
    """Run IK from q_start toward target_pose, verify FK of result matches."""
    q_result = solver.solve(q_start, target_pose)
    result_pose = _fk(solver.urdf_xml, q_result)
    np.testing.assert_allclose(result_pose[:3], target_pose[:3], atol=pos_tol, err_msg='position mismatch')
    # Quaternion sign ambiguity: compare closest
    q_diff = min(np.linalg.norm(result_pose[3:] - target_pose[3:]), np.linalg.norm(result_pose[3:] + target_pose[3:]))
    assert q_diff < rot_tol, f'rotation mismatch: {q_diff:.4f}'
    return q_result


@pytest.mark.parametrize('q_target', TEST_CONFIGS)
def test_dls_solver(q_target):
    target_pose = _fk(URDF, q_target)
    q_start = np.zeros(7)
    solver = DLSIKSolver(URDF, JOINT_NAMES, CONTROL_FRAME)
    _assert_fk_matches(solver, q_start, target_pose)


def test_dls_solver_with_limits():
    """Test bounded IK from realistic (nearby) starting points — the actual use case.

    DLSIKSolverWithLimits uses linearized bounded least squares, which converges
    well from nearby starting points but can get stuck from far away (q=zeros).
    In practice, ik_joints_from_episode always passes the current joint state.
    """
    solver = DLSIKSolverWithLimits(URDF, JOINT_NAMES, CONTROL_FRAME)
    for q_target in TEST_CONFIGS:
        target_pose = _fk(URDF, q_target)
        # Start from a perturbed target (±0.3 rad), clamped to limits
        rng = np.random.RandomState(42)
        q_start = np.clip(q_target + rng.uniform(-0.3, 0.3, 7), solver._joint_lower, solver._joint_upper)
        q_result = _assert_fk_matches(solver, q_start, target_pose)
        # Verify joint limits respected
        assert np.all(q_result >= solver._joint_lower - 1e-6)
        assert np.all(q_result <= solver._joint_upper + 1e-6)


@pytest.mark.parametrize('q_target', TEST_CONFIGS)
def test_lm_solver(q_target):
    target_pose = _fk(URDF, q_target)
    q_start = np.zeros(7)
    solver = LMIKSolver(URDF, JOINT_NAMES, CONTROL_FRAME)
    _assert_fk_matches(solver, q_start, target_pose)


def test_ik_joints_from_episode():
    n_steps = 5
    ts = np.arange(n_steps, dtype=np.int64) * 100_000_000  # 100ms apart

    # Generate a trajectory of EE poses from known joint configs
    q_traj = np.linspace(TEST_CONFIGS[0], TEST_CONFIGS[1], n_steps)
    ee_poses = np.array([_fk(URDF, q) for q in q_traj])

    episode = EpisodeContainer(
        data={
            keys.JOINTS: DummySignal(ts, q_traj),
            'robot_command.pose': DummySignal(ts, ee_poses),
            'urdf': URDF,
            'joint_names': JOINT_NAMES,
            'control_frame': CONTROL_FRAME,
        }
    )
    result = ik_joints_from_episode(episode, DLSIKSolverWithLimits, 'robot_command.pose', keys.JOINTS)

    assert len(result) == n_steps
    for i in range(n_steps):
        reconstructed_pose = _fk(URDF, result[i][0])
        np.testing.assert_allclose(reconstructed_pose[:3], ee_poses[i, :3], atol=1e-3)


@pytest.mark.parametrize('solver_cls', [DLSIKSolver, DLSIKSolverWithLimits])
def test_pickle_roundtrip(solver_cls):
    solver = solver_cls(URDF, JOINT_NAMES, CONTROL_FRAME)
    # Force model build before pickling
    target_pose = _fk(URDF, TEST_CONFIGS[0])
    solver.solve(np.zeros(7), target_pose)
    assert solver._mj is not None

    restored = pickle.loads(pickle.dumps(solver))
    assert restored._mj is None  # cache cleared

    # Solver params preserved
    assert restored.tol == solver.tol
    assert restored.max_iters == solver.max_iters

    # Still works after unpickling (start near target for limit-aware solver)
    q_start = TEST_CONFIGS[0] + 0.05
    q_result = restored.solve(q_start, target_pose)
    result_pose = _fk(URDF, q_result)
    np.testing.assert_allclose(result_pose[:3], target_pose[:3], atol=1e-3)
