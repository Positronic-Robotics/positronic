import pickle
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from pimm.time import RECEIVED_WORLD
from positronic import geom, keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import (
    IK_DAMPING_SINGULAR_VALUE,
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
    EE_LINK,
    FLANGE_LINK,
    GRIPPER_SITE,
    ROBOLAB_EE_FRAME,
    ROBOLAB_EEF_LINK,
    YAM_JOINT_NAMES,
    bundled_franka_model,
    bundled_panda_model,
    bundled_yam_model,
)
from positronic.drivers.roboarm.so101.driver import Robot as So101Robot
from positronic.drivers.roboarm.yam import driver as yam
from positronic.utils import package_assets_path

PANDA_URDF = Path(package_assets_path('assets/mujoco/panda_ik.xml')).read_text()
PANDA_JOINTS = bundled_panda_model()[roboarm_keys.JOINT_NAMES]
PANDA_FRAME = EE_LINK

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


def test_dls_solver_with_limits_moves_each_joint_at_most_max_step():
    solver = DLSIKSolverWithLimits(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME, max_step=0.2)
    q_start = TEST_CONFIGS[0]
    q_result = solver.solve(q_start, _fk(PANDA_URDF, TEST_CONFIGS[1]))
    assert np.max(np.abs(q_result - q_start)) == pytest.approx(0.2)


def test_dls_solver_with_limits_refuses_when_no_joints_are_in_reach():
    solver = DLSIKSolverWithLimits(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME)
    q_start = TEST_CONFIGS[0].copy()
    q_start[3] = 0.6  # joint4 limit is -0.0698, so q_start is more than max_step outside it
    with pytest.raises(ValueError, match='max_step'):
        solver.solve(q_start, _fk(PANDA_URDF, TEST_CONFIGS[0]))


def test_dls_solver_with_limits_damps_joint_motion_at_a_singularity():
    """Along the weak direction, the joints move at most the tool change over IK_DAMPING_SINGULAR_VALUE."""
    solver = DLSIKSolverWithLimits(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME)
    q_start = np.array([0.0, -0.31, 0.0, -0.47, 0.0, 1.2886, 0.0])
    pos, R, J = solver._fk_jac(q_start)
    U, sv, _ = np.linalg.svd(J)
    assert sv[-1] < 1e-3
    tool_change = 0.005
    weak = U[:, -1] * tool_change
    quat = geom.Rotation.Representation.QUAT
    rotation = geom.Rotation.from_rotvec(weak[3:]) * geom.Rotation.from_rotation_matrix(R)
    target_pose = geom.Transform3D(pos + weak[:3], rotation).as_vector(quat)

    q_result = solver.solve(q_start, target_pose)

    assert np.linalg.norm(q_result - q_start) <= tool_change / IK_DAMPING_SINGULAR_VALUE + 1e-3


def test_dls_solver_with_limits_solves_a_five_joint_arm():
    meta = So101Robot._build_robot_meta()
    solver = DLSIKSolverWithLimits(meta[roboarm_keys.URDF], meta[roboarm_keys.JOINT_NAMES], DEFAULT_FRAME)
    q_start = np.array([0.1, -0.2, 0.3, 0.2, 0.1])
    q_target = q_start + 0.05
    pos, R, _ = solver._fk_jac(q_target)
    target_pose = geom.Transform3D(pos, geom.Rotation.from_rotation_matrix(R)).as_vector(
        geom.Rotation.Representation.QUAT
    )

    q_result = solver.solve(q_start, target_pose)

    np.testing.assert_allclose(solver._fk_jac(q_result)[0], pos, atol=1e-3)


@pytest.mark.parametrize('q_target', TEST_CONFIGS)
def test_lm_solver(q_target):
    target_pose = _fk(PANDA_URDF, q_target)
    q_start = np.zeros(7)
    solver = LMIKSolver(PANDA_URDF, PANDA_JOINTS, PANDA_FRAME)
    _assert_fk_matches(solver, q_start, target_pose)


@pytest.mark.parametrize('timeline', [HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME])
def test_ik_joints_from_episode(timeline):
    n_steps = 5
    ts = np.arange(n_steps, dtype=np.int64) * 100_000_000  # 100ms apart

    # Generate a trajectory of EE poses from known joint configs
    q_traj = np.linspace(TEST_CONFIGS[0], TEST_CONFIGS[1], n_steps)
    ee_poses = np.array([_fk(PANDA_URDF, q) for q in q_traj])

    episode = EpisodeContainer(
        data={
            keys.JOINTS: DummySignal(ts, q_traj, timelines=(timeline,)),
            keys.TARGET_EE_POSE: DummySignal(ts, ee_poses, timelines=(timeline,)),
            roboarm_keys.URDF: PANDA_URDF,
            roboarm_keys.JOINT_NAMES: PANDA_JOINTS,
            roboarm_keys.CONTROL_FRAME: PANDA_FRAME,
        }
    )
    result = ik_joints_from_episode(episode, DLSIKSolverWithLimits, keys.TARGET_EE_POSE, keys.JOINTS)

    assert len(result) == n_steps
    assert result.timelines == (timeline,)
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


def test_frame_transform_reproduces_robolab_eef_across_configs():
    urdf = bundled_franka_model()[roboarm_keys.URDF]
    transform = frame_transform(urdf, EE_LINK, ROBOLAB_EEF_LINK)
    quat = geom.Rotation.Representation.QUAT
    for q in TEST_CONFIGS:
        got = geom.Transform3D.from_vector(_fk_site(urdf, q, EE_LINK), quat) * transform
        want = geom.Transform3D.from_vector(_fk_site(urdf, q, ROBOLAB_EEF_LINK), quat)
        np.testing.assert_allclose(got.translation, want.translation, atol=1e-9)
        assert geom.quat_closest(got.rotation, want.rotation) == want.rotation


@pytest.mark.parametrize(
    'model', [bundled_franka_model(), bundled_panda_model(), bundled_yam_model()], ids=['franka', 'panda', 'yam']
)
def test_bundled_model_declares_the_frame_it_reports_in(model):
    assert model[roboarm_keys.CONTROL_FRAME] == DEFAULT_FRAME
    frame_transform(model[roboarm_keys.URDF], DEFAULT_FRAME, DEFAULT_FRAME)


@pytest.mark.parametrize(
    'declared, frame', [(DROID_EE_FRAME, FLANGE_LINK), (ROBOLAB_EE_FRAME, ROBOLAB_EEF_LINK)], ids=['droid', 'robolab']
)
def test_declared_frame_matches_the_model_geometry(declared, frame):
    """A checkpoint or an env states its frame as a constant; this pins it to where the model puts that frame."""
    urdf = bundled_franka_model()[roboarm_keys.URDF]
    measured = frame_transform(urdf, DEFAULT_FRAME, frame)
    np.testing.assert_allclose(declared.as_matrix, measured.as_matrix, atol=1e-9)


def test_default_frame_still_coincides_with_the_franka_tool_frame():
    """Two things rest on this and neither is reachable from the data: recordings predating ``EE_FRAME`` are
    solved at ``end_effector`` (see ``pose_anchor``), and ``DROID_EE_FRAME`` is stated from where
    ``DEFAULT_FRAME`` sits. TODO(#550): moving it to the flange invalidates both, so re-express those
    recordings and re-measure the constant in the same change.
    """
    transform = frame_transform(bundled_franka_model()[roboarm_keys.URDF], DEFAULT_FRAME, EE_LINK)
    np.testing.assert_allclose(transform.as_matrix, np.eye(4), atol=1e-12)


# Joint configs inside the YAM's joint range: home, and three that bend every joint.
YAM_CONFIGS = [
    np.zeros(6),
    np.array([0.5, 1.2, 1.0, -0.6, 0.4, 1.5]),
    np.array([-1.8, 2.5, 2.9, 1.2, -1.1, -0.7]),
    np.array([2.6, 0.3, 3.4, 0.0, 1.4, 2.0]),
]


@pytest.mark.parametrize('q', YAM_CONFIGS)
def test_the_bundled_yam_urdf_puts_the_default_frame_where_the_yam_driver_measures_its_pose(q):
    model = bundled_yam_model()
    urdf = model[roboarm_keys.URDF]
    compiled = _prepare_spec(urdf, DEFAULT_FRAME).compile()
    data = mj.MjData(compiled)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[[compiled.joint(name).qposadr.item() for name in model[roboarm_keys.JOINT_NAMES]]] = q
    mj.mj_kinematics(compiled, data)  # pyright: ignore[reportAttributeAccessIssue]
    site = mj.mj_name2id(compiled, mj.mjtObj.mjOBJ_SITE, DEFAULT_FRAME)  # pyright: ignore[reportAttributeAccessIssue]
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, data.site_xmat[site])  # pyright: ignore[reportAttributeAccessIssue]

    measured = yam._Kinematics().fk(q)

    np.testing.assert_allclose(data.site_xpos[site], measured.translation, atol=1e-4)
    assert (geom.Rotation.from_quat(quat).inv * measured.rotation).angle < 1e-3


def test_the_bundled_yam_urdf_has_the_joint_range_of_the_yam_driver():
    urdf = bundled_yam_model()[roboarm_keys.URDF]
    compiled = _prepare_spec(urdf).compile()
    kinematics = yam._Kinematics()

    ranges = np.array([compiled.joint(name).range for name in YAM_JOINT_NAMES])

    np.testing.assert_allclose(ranges[:, 0], kinematics.lower, atol=1e-5)
    np.testing.assert_allclose(ranges[:, 1], kinematics.upper, atol=1e-5)


def test_the_bundled_yam_model_carries_every_mesh_its_urdf_names():
    model = bundled_yam_model()

    named = {mesh.get('filename') for mesh in ET.fromstring(model[roboarm_keys.URDF]).iter('mesh')}

    assert named == set(model['meshes']) and all(model['meshes'].values())


def test_panda_flange_sits_on_link7_as_on_the_fr3():
    for urdf in (bundled_panda_model()[roboarm_keys.URDF], bundled_franka_model()[roboarm_keys.URDF]):
        # rules-allow: hardcoded-keys — the wrist link is this test's input, not a name it shares with the code
        transform = frame_transform(urdf, 'link7', FLANGE_LINK)
        np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.107], atol=1e-12)
        np.testing.assert_allclose(transform.rotation.as_rotation_matrix, np.eye(3), atol=1e-12)


def test_panda_hand_frames_keep_their_offset_from_the_flange():
    """The sim reports and solves at the hand point, 58.4 mm along the flange axis and -45deg about it."""
    hand_point = geom.Transform3D(np.array([0.0, 0.0, 0.0584]), geom.Rotation.from_euler([0.0, 0.0, -np.pi / 4]))
    urdf = bundled_panda_model()[roboarm_keys.URDF]
    for frame in (EE_LINK, DEFAULT_FRAME):
        np.testing.assert_allclose(frame_transform(urdf, FLANGE_LINK, frame).as_matrix, hand_point.as_matrix, atol=1e-9)
    mjcf_flange = frame_transform(PANDA_URDF, FLANGE_LINK, PANDA_FRAME)
    np.testing.assert_allclose(mjcf_flange.as_matrix, hand_point.as_matrix, atol=1e-7)


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

    assert restored.__getstate__() == solver.__getstate__()

    # Still works after unpickling (start near target for limit-aware solver)
    q_start = TEST_CONFIGS[0] + 0.05
    q_result = restored.solve(q_start, target_pose)
    result_pose = _fk(PANDA_URDF, q_result)
    np.testing.assert_allclose(result_pose[:3], target_pose[:3], atol=1e-3)


@pytest.mark.parametrize(
    'urdf, grip_point',
    [
        # The closed 2F-85 pads, where MolmoSpaces puts its ``gripper/grasp_site``.
        (bundled_franka_model()[roboarm_keys.URDF], 0.155),
        # The midpoint of the fingertip pads in the sim scene.
        (bundled_panda_model()[roboarm_keys.URDF], 0.10365),
        (PANDA_URDF, 0.10365),
    ],
    ids=['franka', 'panda', 'panda_ik'],
)
def test_gripper_sits_on_the_flange_approach_axis(urdf, grip_point):
    """The grip point keeps the flange axes, so z is the approach axis and x, y are those of ``link8``."""
    transform = frame_transform(urdf, FLANGE_LINK, GRIPPER_SITE)
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, grip_point], atol=1e-9)
    np.testing.assert_allclose(transform.rotation.as_rotation_matrix, np.eye(3), atol=1e-9)


def test_gripper_model_declares_the_frame_it_reports_in():
    model = bundled_franka_model(GRIPPER_SITE)
    assert model[roboarm_keys.CONTROL_FRAME] == DEFAULT_FRAME
    transform = frame_transform(model[roboarm_keys.URDF], DEFAULT_FRAME, GRIPPER_SITE)
    np.testing.assert_allclose(transform.as_matrix, np.eye(4), atol=1e-9)
