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
    ee_frames,
    frame_transform,
    ik_joints_from_episode,
    qpos_from_site_pose,
)
from positronic.drivers.roboarm.models import (
    DROID_EE_FRAME,
    EE_LINK,
    FLANGE_LINK,
    FRANKA_HAND_FRAME,
    GRIPPER_SITE,
    PANDA_HAND_FRAME,
    ROBOLAB_EE_FRAME,
    ROBOLAB_EEF_LINK,
    YAM_GRASP_SITE_FRAME,
    YAM_JOINT_NAMES,
    YAM_WRIST_LINK,
    EEFrame,
    bundled_franka_model,
    bundled_panda_model,
    bundled_yam_model,
    ee_frames_entry,
)
from positronic.drivers.roboarm.so101.driver import Robot as So101Robot
from positronic.drivers.roboarm.tests.test_yam import STRETCHED, past_full_stretch
from positronic.drivers.roboarm.yam import driver as yam
from positronic.eval import keys as eval_keys
from positronic.utils import package_assets_path

PANDA_URDF = Path(package_assets_path('assets/mujoco/panda_ik.xml')).read_text()
YAM_MJCF_PATH = package_assets_path('assets/mujoco/i2rt_yam/yam.xml')
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
    solver = DLSIKSolverWithLimits(meta[roboarm_keys.URDF], meta[roboarm_keys.JOINT_NAMES], GRIPPER_SITE)
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
            **ee_frames_entry({keys.TARGET_EE_POSE: EEFrame(site=PANDA_FRAME)}),
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
    """Targets re-expressed in a policy's frame come back to the site their ``EE_FRAMES`` entry names."""
    n_steps = 3
    ts = np.arange(n_steps, dtype=np.int64) * 100_000_000
    q_traj = np.linspace(TEST_CONFIGS[0], TEST_CONFIGS[1], n_steps)
    quat = geom.Rotation.Representation.QUAT
    urdf = bundled_panda_model()[roboarm_keys.URDF]
    offset = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.from_euler([0.0, 0.0, np.pi / 4]))
    anchored = [geom.Transform3D.from_vector(_fk_site(urdf, q, GRIPPER_SITE), quat) for q in q_traj]
    moved = np.array([(pose * offset).as_vector(quat) for pose in anchored])

    episode = EpisodeContainer(
        data={
            keys.JOINTS: DummySignal(ts, q_traj),
            keys.TARGET_EE_POSE: DummySignal(ts, moved),
            roboarm_keys.URDF: urdf,
            roboarm_keys.JOINT_NAMES: PANDA_JOINTS,
            **ee_frames_entry({keys.TARGET_EE_POSE: EEFrame(site=GRIPPER_SITE, offset=offset)}),
        }
    )
    result = ik_joints_from_episode(episode, DLSIKSolverWithLimits, keys.TARGET_EE_POSE, keys.JOINTS)

    for i in range(n_steps):
        solved = _fk_site(urdf, result[i][0], GRIPPER_SITE)
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
def test_bundled_model_has_the_site_its_embodiment_reports_at(model):
    frame_transform(model[roboarm_keys.URDF], GRIPPER_SITE, GRIPPER_SITE)


@pytest.mark.parametrize(
    'model, declared, frame',
    [
        (bundled_franka_model(), DROID_EE_FRAME, FLANGE_LINK),
        (bundled_franka_model(), ROBOLAB_EE_FRAME, ROBOLAB_EEF_LINK),
        (bundled_franka_model(), FRANKA_HAND_FRAME, EE_LINK),
        (bundled_panda_model(), PANDA_HAND_FRAME, EE_LINK),
        (bundled_yam_model(), YAM_GRASP_SITE_FRAME, EE_LINK),
    ],
    ids=['droid', 'robolab', 'fr3', 'panda', 'yam'],
)
def test_declared_frame_matches_the_model_geometry(model, declared, frame):
    """A checkpoint or an env states its frame as a constant; this pins it to where the model puts that frame."""
    measured = frame_transform(model[roboarm_keys.URDF], GRIPPER_SITE, frame)
    np.testing.assert_allclose(declared.as_matrix, measured.as_matrix, atol=1e-9)


def _yam_mjcf_wrist_to_site(site: str) -> geom.Transform3D:
    """``site`` of the YAM MJCF, in ``YAM_WRIST_LINK``."""
    model = mj.MjModel.from_xml_path(YAM_MJCF_PATH)
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)
    wrist, placed = data.body(YAM_WRIST_LINK), data.site(site)
    to_wrist = wrist.xmat.reshape(3, 3).T
    rotation = geom.Rotation.from_rotation_matrix(to_wrist @ placed.xmat.reshape(3, 3))
    return geom.Transform3D(to_wrist @ (placed.xpos - wrist.xpos), rotation)


def test_yam_declared_frame_is_the_grasp_site_of_the_yam_mjcf():
    """YAM poses older than ``GRIPPER_SITE`` sit at the MJCF ``grasp_site``, where the driver measured them."""
    # rules-allow: hardcoded-keys — the vendor's site name is this test's input, not a name it shares with the code
    measured = _yam_mjcf_wrist_to_site(GRIPPER_SITE).inv * _yam_mjcf_wrist_to_site('grasp_site')
    np.testing.assert_allclose(YAM_GRASP_SITE_FRAME.as_matrix, measured.as_matrix, atol=1e-9)


# Joint configs inside the YAM's joint range: home, and three that bend every joint.
YAM_CONFIGS = [
    np.zeros(6),
    np.array([0.5, 1.2, 1.0, -0.6, 0.4, 1.5]),
    np.array([-1.8, 2.5, 2.9, 1.2, -1.1, -0.7]),
    np.array([2.6, 0.3, 3.4, 0.0, 1.4, 2.0]),
]


@pytest.mark.parametrize('q', YAM_CONFIGS)
def test_the_bundled_yam_urdf_puts_gripper_where_the_yam_driver_measures_its_pose(q):
    model = bundled_yam_model()
    urdf = model[roboarm_keys.URDF]
    compiled = _prepare_spec(urdf, GRIPPER_SITE).compile()
    data = mj.MjData(compiled)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[[compiled.joint(name).qposadr.item() for name in model[roboarm_keys.JOINT_NAMES]]] = q
    mj.mj_kinematics(compiled, data)  # pyright: ignore[reportAttributeAccessIssue]
    site = mj.mj_name2id(compiled, mj.mjtObj.mjOBJ_SITE, GRIPPER_SITE)  # pyright: ignore[reportAttributeAccessIssue]
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, data.site_xmat[site])  # pyright: ignore[reportAttributeAccessIssue]

    measured = yam._Kinematics().fk(q)

    np.testing.assert_allclose(data.site_xpos[site], measured.translation, atol=1e-4)
    assert (geom.Rotation.from_quat(quat).inv * measured.rotation).angle < 1e-3


@pytest.mark.parametrize('q', YAM_CONFIGS[1:])
def test_the_yam_driver_solves_a_pose_at_gripper_to_joints_that_reach_it(q):
    kinematics = yam._Kinematics()
    target = kinematics.fk(q)

    solved = kinematics.ik(target, q + 0.2, kinematics.lower, kinematics.upper)

    assert solved is not None
    reached = kinematics.fk(solved)
    np.testing.assert_allclose(reached.translation, target.translation, atol=1e-3)
    assert (reached.rotation.inv * target.rotation).angle < 1e-2


LM_ROT_WEIGHT = 0.5


def _yam_lm_search(start: np.ndarray, target: geom.Transform3D) -> tuple[np.ndarray, float, bool]:
    """``qpos_from_site_pose`` on the YAM MJCF at ``GRIPPER_SITE`` from joints ``start``, with its YAM joints."""
    model = mj.MjModel.from_xml_path(YAM_MJCF_PATH)  # pyright: ignore[reportAttributeAccessIssue]
    data = mj.MjData(model)  # pyright: ignore[reportAttributeAccessIssue]
    qpos_ids = [model.joint(name).qposadr.item() for name in YAM_JOINT_NAMES]
    dof_ids = [model.joint(name).dofadr.item() for name in YAM_JOINT_NAMES]
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, GRIPPER_SITE)  # pyright: ignore[reportAttributeAccessIssue]
    data.qpos[qpos_ids] = start
    qpos, err, success = qpos_from_site_pose(
        model, data, site_id, dof_ids, target.translation, target.rotation.as_quat, rot_weight=LM_ROT_WEIGHT
    )
    return qpos[qpos_ids].copy(), float(err), success


def _lm_error(kinematics: yam._Kinematics, q: np.ndarray, target: geom.Transform3D) -> float:
    """The error that the LM search measures at joints ``q``."""
    reached = kinematics.fk(q)
    rotation_error = (reached.rotation.inv * target.rotation).angle
    return float(np.linalg.norm(reached.translation - target.translation) + LM_ROT_WEIGHT * rotation_error)


def test_an_lm_search_past_the_reach_edge_answers_no_worse_than_its_start_from_nearly_equal_starts():
    kinematics = yam._Kinematics()
    target = past_full_stretch(kinematics, 5e-4)

    errors = []
    for k in range(20):
        start = STRETCHED + k * 1e-12
        solved, err, success = _yam_lm_search(start, target)
        assert not success
        assert err == pytest.approx(_lm_error(kinematics, solved, target), rel=1e-6)
        assert _lm_error(kinematics, solved, target) <= _lm_error(kinematics, start, target)
        errors.append(err)

    assert max(errors) - min(errors) < 1e-5


@pytest.mark.parametrize('q', YAM_CONFIGS[1:])
def test_an_lm_search_that_converges_answers_the_joints_it_converged_to(q):
    kinematics = yam._Kinematics()
    target = kinematics.fk(q)

    solved, err, success = _yam_lm_search(q + 0.2, target)

    assert success
    assert err < 1e-14
    assert _lm_error(kinematics, solved, target) < 1e-12


def test_the_bundled_yam_urdf_has_the_joint_range_of_the_yam_mjcf():
    urdf = bundled_yam_model()[roboarm_keys.URDF]
    compiled = _prepare_spec(urdf).compile()
    kinematics = yam._Kinematics()

    ranges = np.array([compiled.joint(name).range for name in YAM_JOINT_NAMES])

    np.testing.assert_allclose(ranges[:, 0], kinematics.lower, atol=1e-5)
    np.testing.assert_allclose(ranges[:, 1], kinematics.upper, atol=1e-5)


def test_the_bundled_yam_model_carries_every_mesh_its_urdf_names():
    model = bundled_yam_model()

    named = {mesh.get('filename') for mesh in ET.fromstring(model[roboarm_keys.URDF]).iter('mesh')}

    assert named == set(model[roboarm_keys.MESHES]) and all(model[roboarm_keys.MESHES].values())


def test_panda_flange_sits_on_link7_as_on_the_fr3():
    for urdf in (bundled_panda_model()[roboarm_keys.URDF], bundled_franka_model()[roboarm_keys.URDF]):
        # rules-allow: hardcoded-keys — the wrist link is this test's input, not a name it shares with the code
        transform = frame_transform(urdf, 'link7', FLANGE_LINK)
        np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.107], atol=1e-12)
        np.testing.assert_allclose(transform.rotation.as_rotation_matrix, np.eye(3), atol=1e-12)


def test_panda_hand_frames_keep_their_offset_from_the_flange():
    """The panda's ``EE_LINK``, where older sim recordings sit, is 58.4 mm along the flange axis and -45deg about it."""
    hand_point = geom.Transform3D(np.array([0.0, 0.0, 0.0584]), geom.Rotation.from_euler([0.0, 0.0, -np.pi / 4]))
    urdf = bundled_panda_model()[roboarm_keys.URDF]
    np.testing.assert_allclose(frame_transform(urdf, FLANGE_LINK, EE_LINK).as_matrix, hand_point.as_matrix, atol=1e-9)
    mjcf_flange = frame_transform(PANDA_URDF, FLANGE_LINK, PANDA_FRAME)
    np.testing.assert_allclose(mjcf_flange.as_matrix, hand_point.as_matrix, atol=1e-7)


def test_ee_frames_reads_the_site_and_offset_of_each_pose_signal():
    offset = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.identity)
    stated = {
        eval_keys.POSE_SIGNALS: [keys.EE_POSE, keys.TARGET_EE_POSE],
        **ee_frames_entry({
            keys.EE_POSE: EEFrame(site=GRIPPER_SITE),
            keys.TARGET_EE_POSE: EEFrame(site=EE_LINK, offset=offset),
        }),
    }

    frames = ee_frames(stated)

    assert frames[keys.EE_POSE].site == GRIPPER_SITE
    np.testing.assert_allclose(frames[keys.EE_POSE].offset.as_matrix, np.eye(4), atol=1e-12)
    assert frames[keys.TARGET_EE_POSE].site == EE_LINK
    np.testing.assert_allclose(frames[keys.TARGET_EE_POSE].offset.as_matrix, offset.as_matrix, atol=1e-12)


def test_ee_frames_refuses_a_pose_signal_with_no_frame():
    statics = {
        eval_keys.POSE_SIGNALS: [keys.EE_POSE, keys.TARGET_EE_POSE],
        **ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)}),
    }
    with pytest.raises(ValueError, match=keys.TARGET_EE_POSE):
        ee_frames(statics)


def test_ee_frames_refuses_a_site_the_carried_model_lacks():
    statics = {
        roboarm_keys.URDF: '<robot name="r"><link name="base"/></robot>',
        eval_keys.POSE_SIGNALS: [keys.EE_POSE],
        **ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)}),
    }
    with pytest.raises(ValueError, match=GRIPPER_SITE):
        ee_frames(statics)


def test_ee_frames_skips_the_site_check_when_no_model_is_carried():
    statics = {eval_keys.POSE_SIGNALS: [keys.EE_POSE], **ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)})}
    assert ee_frames(statics)[keys.EE_POSE].site == GRIPPER_SITE


def test_ik_joints_from_episode_refuses_targets_with_no_frame():
    episode = EpisodeContainer(
        data={
            keys.JOINTS: DummySignal(np.array([0]), np.array([TEST_CONFIGS[0]])),
            keys.TARGET_EE_POSE: DummySignal(np.array([0]), np.array([_fk(PANDA_URDF, TEST_CONFIGS[0])])),
            roboarm_keys.URDF: PANDA_URDF,
            roboarm_keys.JOINT_NAMES: PANDA_JOINTS,
        }
    )
    with pytest.raises(ValueError, match=keys.TARGET_EE_POSE):
        ik_joints_from_episode(episode, DLSIKSolverWithLimits, keys.TARGET_EE_POSE, keys.JOINTS)


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


def test_yam_gripper_sits_where_the_closed_fingertips_meet_on_the_wrist_axes():
    """The fingertips meet 40.8 mm to one side of the wrist axis, so the grip point is off it, on its axes."""
    urdf = bundled_yam_model()[roboarm_keys.URDF]
    for transform in (frame_transform(urdf, YAM_WRIST_LINK, GRIPPER_SITE), _yam_mjcf_wrist_to_site(GRIPPER_SITE)):
        np.testing.assert_allclose(transform.translation, [0.0, -0.0408, 0.138], atol=1e-9)
        np.testing.assert_allclose(transform.rotation.as_rotation_matrix, np.eye(3), atol=1e-9)


def test_yam_gripper_sits_midway_between_the_fingers_at_every_pose_and_opening():
    """The fingers slide along x of the wrist, so the grip point is midway between their pads at every opening."""
    model = mj.MjModel.from_xml_path(YAM_MJCF_PATH)
    data = mj.MjData(model)
    # rules-allow: hardcoded-keys — the MJCF finger joints and pad bodies are this test's input, not names it shares
    fingers, pads = ('left_finger', 'right_finger'), ('lf_down', 'rf_down')
    arm_qpos = [model.joint(name).qposadr.item() for name in YAM_JOINT_NAMES]
    finger_qpos = [model.joint(name).qposadr.item() for name in fingers]
    rng = np.random.default_rng(0)
    for q in [*YAM_CONFIGS, *rng.uniform(-1.0, 1.0, (3, len(YAM_JOINT_NAMES)))]:
        for opening in (0.0, 0.02, 0.037):
            data.qpos[arm_qpos] = q
            data.qpos[finger_qpos] = (opening, -opening)
            mj.mj_kinematics(model, data)
            wrist = data.body(YAM_WRIST_LINK)
            to_wrist = wrist.xmat.reshape(3, 3).T
            grip = to_wrist @ (data.site(GRIPPER_SITE).xpos - wrist.xpos)
            pad_x = [(to_wrist @ (data.body(pad).xpos - wrist.xpos))[0] for pad in pads]
            assert abs((pad_x[0] + pad_x[1]) / 2 - grip[0]) < 1e-9
