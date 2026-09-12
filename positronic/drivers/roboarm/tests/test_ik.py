import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from positronic import geom, keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import (
    DLSIKSolver,
    DLSIKSolverWithLimits,
    LMIKSolver,
    _prepare_spec,
    frame_transform,
    ik_joints_from_episode,
    pose_anchor,
)
from positronic.drivers.roboarm.models import (
    DEFAULT_FRAME,
    DROID_EE_FRAME,
    DROID_EEF_LINK,
    EE_LINK,
    FLANGE_LINK,
    GRASP_SITE_LINK,
    bundled_franka_model,
    bundled_panda_model,
)
from positronic.utils import package_assets_path

PANDA_URDF = Path(package_assets_path('assets/mujoco/panda_ik.xml')).read_text()
PANDA_JOINTS = [f'joint{i}' for i in range(1, 8)]
PANDA_FRAME = 'end_effector'

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
    qpos_ids = [model.joint(n).qposadr.item() for n in PANDA_JOINTS]
    data.qpos[qpos_ids] = q
    mj.mj_forward(model, data)
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, PANDA_FRAME)
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
    target_pose = _fk(PANDA_URDF, q_target)
    q_start = np.zeros(7)
    solver = DLSIKSolver(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME)
    _assert_fk_matches(solver, q_start, target_pose)


def test_dls_solver_with_limits():
    """Test bounded IK from realistic (nearby) starting points — the actual use case.

    DLSIKSolverWithLimits uses linearized bounded least squares, which converges
    well from nearby starting points but can get stuck from far away (q=zeros).
    In practice, ik_joints_from_episode always passes the current joint state.
    """
    solver = DLSIKSolverWithLimits(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME)
    for q_target in TEST_CONFIGS:
        target_pose = _fk(PANDA_URDF, q_target)
        # Start from a perturbed target (±0.3 rad), clamped to limits
        rng = np.random.RandomState(42)
        q_start = np.clip(q_target + rng.uniform(-0.3, 0.3, 7), solver._joint_lower, solver._joint_upper)
        q_result = _assert_fk_matches(solver, q_start, target_pose)
        # Verify joint limits respected
        assert np.all(q_result >= solver._joint_lower - 1e-6)
        assert np.all(q_result <= solver._joint_upper + 1e-6)


@pytest.mark.parametrize('q_target', TEST_CONFIGS)
def test_lm_solver(q_target):
    target_pose = _fk(PANDA_URDF, q_target)
    q_start = np.zeros(7)
    solver = LMIKSolver(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME)
    _assert_fk_matches(solver, q_start, target_pose)


def test_ik_joints_from_episode():
    n_steps = 5
    ts = np.arange(n_steps, dtype=np.int64) * 100_000_000  # 100ms apart

    # Generate a trajectory of EE poses from known joint configs
    q_traj = np.linspace(TEST_CONFIGS[0], TEST_CONFIGS[1], n_steps)
    ee_poses = np.array([_fk(PANDA_URDF, q) for q in q_traj])

    episode = EpisodeContainer(
        data={
            keys.JOINTS: DummySignal(ts, q_traj),
            keys.TARGET_EE_POSE: DummySignal(ts, ee_poses),
            roboarm_keys.URDF: PANDA_URDF,
            roboarm_keys.JOINT_NAMES: PANDA_JOINTS,
            roboarm_keys.CONTROL_FRAME: PANDA_FRAME,
        }
    )
    result = ik_joints_from_episode(episode, DLSIKSolverWithLimits, keys.TARGET_EE_POSE, keys.JOINTS)

    assert len(result) == n_steps
    for i in range(n_steps):
        reconstructed_pose = _fk(PANDA_URDF, result[i][0])
        np.testing.assert_allclose(reconstructed_pose[:3], ee_poses[i, :3], atol=1e-3)


def _fk_site(urdf_xml, q, frame):
    model = _prepare_spec(urdf_xml, frame).compile()
    data = mj.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    qpos_ids = [model.joint(n).qposadr.item() for n in PANDA_JOINTS]
    data.qpos[qpos_ids] = q
    mj.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, frame)  # pyright: ignore[reportAttributeAccessIssue]
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, data.site_xmat[sid])  # pyright: ignore[reportAttributeAccessIssue]
    return np.concatenate([data.site_xpos[sid].copy(), quat])


def test_ik_joints_from_episode_solves_targets_a_codec_moved():
    """Targets re-expressed in a policy's frame come back to the episode's anchor frame before the solve."""
    n_steps = 3
    ts = np.arange(n_steps, dtype=np.int64) * 100_000_000
    q_traj = np.linspace(TEST_CONFIGS[0], TEST_CONFIGS[1], n_steps)
    quat = geom.Rotation.Representation.QUAT
    urdf = bundled_panda_model()[roboarm_keys.URDF]
    offset = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.from_euler([0.0, 0.0, np.pi / 4]))
    anchored = [geom.Transform3D.from_vector(_fk_site(urdf, q, DEFAULT_FRAME), quat) for q in q_traj]
    moved = np.array([(pose * offset).as_vector(quat) for pose in anchored])

    episode = EpisodeContainer(
        data={
            keys.JOINTS: DummySignal(ts, q_traj),
            keys.TARGET_EE_POSE: DummySignal(ts, moved),
            roboarm_keys.URDF: urdf,
            roboarm_keys.JOINT_NAMES: PANDA_JOINTS,
            roboarm_keys.CONTROL_FRAME: DEFAULT_FRAME,
            roboarm_keys.EE_FRAME: offset.as_vector(quat),
        }
    )
    result = ik_joints_from_episode(episode, DLSIKSolverWithLimits, keys.TARGET_EE_POSE, keys.JOINTS)

    for i in range(n_steps):
        solved = _fk_site(urdf, result[i][0], DEFAULT_FRAME)
        np.testing.assert_allclose(solved[:3], anchored[i].translation, atol=1e-3)


def test_frame_transform_reproduces_droid_eef_across_configs():
    urdf = bundled_franka_model()[roboarm_keys.URDF]
    transform = frame_transform(urdf, EE_LINK, DROID_EEF_LINK)
    quat = geom.Rotation.Representation.QUAT
    for q in TEST_CONFIGS:
        got = geom.Transform3D.from_vector(_fk_site(urdf, q, EE_LINK), quat) * transform
        want = geom.Transform3D.from_vector(_fk_site(urdf, q, DROID_EEF_LINK), quat)
        np.testing.assert_allclose(got.translation, want.translation, atol=1e-9)
        assert geom.quat_closest(got.rotation, want.rotation) == want.rotation


@pytest.mark.parametrize('model', [bundled_franka_model(), bundled_panda_model()], ids=['franka', 'panda'])
def test_bundled_model_declares_the_frame_it_reports_in(model):
    assert model[roboarm_keys.CONTROL_FRAME] == DEFAULT_FRAME
    frame_transform(model[roboarm_keys.URDF], DEFAULT_FRAME, DEFAULT_FRAME)


def test_declared_droid_frame_matches_the_model_geometry():
    """The checkpoint states its frame as a constant; this pins it to where the model puts that frame."""
    urdf = bundled_franka_model()[roboarm_keys.URDF]
    measured = frame_transform(urdf, DEFAULT_FRAME, DROID_EEF_LINK)
    np.testing.assert_allclose(DROID_EE_FRAME.as_matrix, measured.as_matrix, atol=1e-9)


def test_default_frame_still_coincides_with_the_franka_tool_frame():
    """Two things rest on this and neither is reachable from the data: recordings predating ``EE_FRAME`` are
    solved at ``end_effector`` (see ``pose_anchor``), and ``DROID_EE_FRAME`` is stated from where
    ``DEFAULT_FRAME`` sits. TODO(#550): moving it to the flange invalidates both, so re-express those
    recordings and re-measure the constant in the same change.
    """
    transform = frame_transform(bundled_franka_model()[roboarm_keys.URDF], DEFAULT_FRAME, EE_LINK)
    np.testing.assert_allclose(transform.as_matrix, np.eye(4), atol=1e-12)


def test_pose_anchor_reads_a_stated_frame_and_falls_back_to_the_name():
    offset = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.identity)
    quat = geom.Rotation.Representation.QUAT
    stated = EpisodeContainer(
        data={roboarm_keys.CONTROL_FRAME: PANDA_FRAME, roboarm_keys.EE_FRAME: offset.as_vector(quat)}
    )
    legacy = EpisodeContainer(data={roboarm_keys.CONTROL_FRAME: PANDA_FRAME})

    assert pose_anchor(stated)[0] == DEFAULT_FRAME
    np.testing.assert_allclose(pose_anchor(stated)[1].as_vector(quat), offset.as_vector(quat), atol=1e-12)
    assert pose_anchor(legacy)[0] == PANDA_FRAME
    np.testing.assert_allclose(pose_anchor(legacy)[1].as_matrix, np.eye(4), atol=1e-12)


def test_frame_transform_rejects_frames_across_movable_joints():
    # rules-allow: hardcoded-keys — the base link is this test's input, not a name it shares with the code
    with pytest.raises(ValueError, match='movable joints'):
        frame_transform(bundled_franka_model()[roboarm_keys.URDF], 'link0', EE_LINK)


def test_frame_transform_identity_when_frames_match():
    transform = frame_transform(bundled_franka_model()[roboarm_keys.URDF], EE_LINK, EE_LINK)
    np.testing.assert_allclose(transform.translation, 0.0, atol=1e-12)
    assert transform.rotation == geom.Rotation.identity


@pytest.mark.parametrize('solver_cls', [DLSIKSolver, DLSIKSolverWithLimits])
def test_pickle_roundtrip(solver_cls):
    solver = solver_cls(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME)
    # Force model build before pickling
    target_pose = _fk(PANDA_URDF, TEST_CONFIGS[0])
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
    result_pose = _fk(PANDA_URDF, q_result)
    np.testing.assert_allclose(result_pose[:3], target_pose[:3], atol=1e-3)


def test_grasp_site_sits_at_the_2f85_grasp_point():
    """The 2F-85's grasp point is 155mm along the flange approach axis, in the flange's own orientation —
    where MolmoSpaces' franka_droid model places its ``gripper/grasp_site``."""
    transform = frame_transform(bundled_franka_model()[roboarm_keys.URDF], FLANGE_LINK, GRASP_SITE_LINK)
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.155], atol=1e-9)
    np.testing.assert_allclose(transform.rotation.as_rotation_matrix, np.eye(3), atol=1e-9)


def test_grasp_site_model_declares_the_frame_it_reports_in():
    """A rig measuring at the grasp point declares ``DEFAULT_FRAME`` there, so the frame it publishes poses
    in is the frame it drives."""
    model = bundled_franka_model(GRASP_SITE_LINK)
    assert model[roboarm_keys.CONTROL_FRAME] == DEFAULT_FRAME
    transform = frame_transform(model[roboarm_keys.URDF], DEFAULT_FRAME, GRASP_SITE_LINK)
    np.testing.assert_allclose(transform.as_matrix, np.eye(4), atol=1e-9)
