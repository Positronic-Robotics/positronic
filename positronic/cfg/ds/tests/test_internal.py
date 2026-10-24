"""The robot transforms read every recording at ``GRIPPER_SITE``, from the model it was recorded on."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from positronic import geom, keys
from positronic.cfg.ds.internal import REAL_ROBOT_TRANSFORM, SIM_ROBOT_TRANSFORM, YAM_ROBOT_TRANSFORM
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.ik import ee_frames
from positronic.drivers.roboarm.models import (
    DROID_EE_FRAME,
    EE_LINK,
    FLANGE_LINK,
    FR3_EE_FRAME,
    GRIPPER_SITE,
    PANDA_EE_FRAME,
    YAM_EE_FRAME,
    YAM_WRIST_LINK,
    EEFrame,
    attach_frame,
    bundled_franka_model,
    bundled_panda_model,
    bundled_yam_model,
    ee_frames_entry,
)
from positronic.eval import keys as eval_keys
from positronic.policy.training import ChangeEEFrame
from positronic.utils import package_assets_path

QUAT = geom.Rotation.Representation.QUAT
OFFSET = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.from_euler([0.0, 0.0, 0.3]))
POSE = geom.Transform3D(np.array([0.3, 0.1, 0.4]), geom.Rotation.from_euler([0.2, -0.1, 0.5]))
# rules-allow: hardcoded-keys — the keys and the sites a recording wrote before ``gripper``, which no code writes now
CONTROL_FRAME, EE_FRAME, DEFAULT, YAM_GRASP_SITE = 'control_frame', 'ee_frame', 'default', 'grasp_site'
POSE_SIGNALS = (keys.EE_POSE, keys.TARGET_EE_POSE)
ONE_ARM_POSES: dict[str, geom.Transform3D] = dict.fromkeys(POSE_SIGNALS, POSE)


def _recording(poses: Mapping[str, geom.Transform3D] = ONE_ARM_POSES, /, **statics):
    data = {signal: DummySignal([1000, 2000], np.tile(pose.as_vector(QUAT), (2, 1))) for signal, pose in poses.items()}
    return EpisodeContainer(data={**data, **statics})


def _assert_reads_at_the_gripper(
    episode, gripper_to_recorded: geom.Transform3D, recorded: Mapping[str, geom.Transform3D] = ONE_ARM_POSES
) -> None:
    """Every pose signal of ``episode`` sits at ``GRIPPER_SITE``, moved off the frame the recording was made in."""
    for signal, pose in recorded.items():
        want = pose * gripper_to_recorded.inv
        for value, _ in episode[signal]:
            got = geom.Transform3D.from_vector(np.asarray(value), QUAT)
            np.testing.assert_allclose(got.translation, want.translation, atol=1e-9)
            assert (want.rotation.inv * got.rotation).angle < 1e-9
    for frame in ee_frames(episode).values():
        assert (frame.site, frame.offset) == (GRIPPER_SITE, geom.Transform3D.identity)


@pytest.mark.parametrize(
    'statics', [{CONTROL_FRAME: EE_LINK}, {CONTROL_FRAME: DEFAULT}, {}], ids=['before_485', 'default', 'no_frame']
)
def test_a_real_recording_at_the_franka_tool_frame_reads_at_the_gripper(statics):
    relabelled = REAL_ROBOT_TRANSFORM(_recording(**statics))
    _assert_reads_at_the_gripper(relabelled, FR3_EE_FRAME)
    assert CONTROL_FRAME not in relabelled


def test_a_real_recording_moves_by_the_offset_its_ee_frame_named():
    relabelled = REAL_ROBOT_TRANSFORM(_recording(**{CONTROL_FRAME: DEFAULT, EE_FRAME: OFFSET.as_vector(QUAT)}))
    _assert_reads_at_the_gripper(relabelled, FR3_EE_FRAME * OFFSET)
    assert EE_FRAME not in relabelled


def test_a_real_recording_moves_from_the_frames_it_states():
    stated = ee_frames_entry(dict.fromkeys(POSE_SIGNALS, EEFrame(site=EE_LINK, offset=OFFSET)))
    _assert_reads_at_the_gripper(REAL_ROBOT_TRANSFORM(_recording(**stated)), FR3_EE_FRAME * OFFSET)


def test_a_real_recording_moves_from_where_its_own_model_puts_default():
    root = ET.fromstring(bundled_franka_model()[roboarm_keys.URDF])
    attach_frame(root, DEFAULT, FLANGE_LINK)
    recording = _recording(**{roboarm_keys.URDF: ET.tostring(root, encoding='unicode'), CONTROL_FRAME: DEFAULT})
    _assert_reads_at_the_gripper(REAL_ROBOT_TRANSFORM(recording), DROID_EE_FRAME)


def test_a_real_model_with_no_gripper_gets_one_at_the_2f85_grip_point():
    urdf = Path(package_assets_path('drivers/roboarm/fr3.urdf')).read_text()
    recording = _recording(**{roboarm_keys.URDF: urdf, CONTROL_FRAME: EE_LINK})
    _assert_reads_at_the_gripper(REAL_ROBOT_TRANSFORM(recording), FR3_EE_FRAME)


@pytest.mark.parametrize(
    'transform, model',
    [
        (REAL_ROBOT_TRANSFORM, bundled_franka_model()),
        (SIM_ROBOT_TRANSFORM, bundled_panda_model()),
        (YAM_ROBOT_TRANSFORM, bundled_yam_model()),
    ],
    ids=['real', 'sim', 'yam'],
)
def test_a_recording_at_the_gripper_keeps_its_poses(transform, model):
    stated = ee_frames_entry(dict.fromkeys(POSE_SIGNALS, EEFrame(site=GRIPPER_SITE)))
    recording = _recording(**{
        roboarm_keys.URDF: model[roboarm_keys.URDF],
        eval_keys.POSE_SIGNALS: list(POSE_SIGNALS),
        **stated,
    })

    read = transform(recording)

    for signal in POSE_SIGNALS:
        for (got, _), (recorded, _) in zip(read[signal], recording[signal], strict=True):
            np.testing.assert_array_equal(got, recorded)


@pytest.mark.parametrize(
    'statics',
    [{CONTROL_FRAME: EE_LINK}, {}, ee_frames_entry(dict.fromkeys(POSE_SIGNALS, EEFrame(site=DEFAULT)))],
    ids=['before_485', 'no_frame', 'default'],
)
def test_a_sim_recording_at_the_panda_hand_reads_at_the_gripper(statics):
    _assert_reads_at_the_gripper(SIM_ROBOT_TRANSFORM(_recording(**statics)), PANDA_EE_FRAME)


def test_a_sim_recording_with_mujoco_xml_for_a_model_reads_at_the_gripper():
    """Older sim recordings store the MuJoCo scene under ``urdf``; the bundled panda replaces it."""
    scene = Path(package_assets_path('assets/mujoco/panda.xml')).read_text()
    recording = _recording(**{roboarm_keys.URDF: scene, CONTROL_FRAME: EE_LINK})
    _assert_reads_at_the_gripper(SIM_ROBOT_TRANSFORM(recording), PANDA_EE_FRAME)


def test_a_recorded_delta_frame_moves_with_the_targets():
    delta_frame = DummySignal([1000], np.array([geom.Transform3D.identity.as_vector(QUAT)]))
    recording = _recording(**{CONTROL_FRAME: EE_LINK, keys.TARGET_EE_POSE_DELTA_FRAME: delta_frame})

    read = REAL_ROBOT_TRANSFORM(recording)

    moved = geom.Transform3D.from_vector(np.asarray(read[keys.TARGET_EE_POSE_DELTA_FRAME][0][0]), QUAT)
    np.testing.assert_allclose(moved.as_matrix, FR3_EE_FRAME.as_matrix, atol=1e-9)


def test_training_moves_a_recording_from_before_485_off_the_gripper():
    trained = ChangeEEFrame(GRIPPER_SITE, OFFSET)(REAL_ROBOT_TRANSFORM(_recording(**{CONTROL_FRAME: EE_LINK})))
    want = POSE * FR3_EE_FRAME.inv * OFFSET
    got = geom.Transform3D.from_vector(np.asarray(trained[keys.EE_POSE][0][0]), QUAT)
    np.testing.assert_allclose(got.as_matrix, want.as_matrix, atol=1e-9)


def _yam_urdf_with_default() -> str:
    """The URDF a YAM recording carried before ``gripper``: ``default`` at the MJCF ``grasp_site``, no ``gripper``."""
    root = ET.fromstring(Path(package_assets_path('assets/i2rt_yam/yam.urdf')).read_text())
    attach_frame(root, DEFAULT, YAM_WRIST_LINK, '0 0 0.1347', '0 0 -1.5707963268')
    return ET.tostring(root, encoding='unicode')


@pytest.mark.parametrize(
    'statics', [{CONTROL_FRAME: YAM_GRASP_SITE}, {CONTROL_FRAME: DEFAULT}], ids=['grasp_site', 'default']
)
def test_a_yam_recording_with_no_model_gets_the_bundled_one_and_reads_at_the_gripper(statics):
    read = YAM_ROBOT_TRANSFORM(_recording(**{eval_keys.POSE_SIGNALS: list(POSE_SIGNALS), **statics}))

    _assert_reads_at_the_gripper(read, YAM_EE_FRAME)
    assert read[roboarm_keys.MESHES] == bundled_yam_model()[roboarm_keys.MESHES]
    assert CONTROL_FRAME not in read


@pytest.mark.parametrize(
    'statics',
    [{CONTROL_FRAME: DEFAULT}, ee_frames_entry(dict.fromkeys(POSE_SIGNALS, EEFrame(site=DEFAULT)))],
    ids=['control_frame', 'ee_frames'],
)
def test_a_yam_recording_moves_from_the_default_site_of_the_model_it_carries(statics):
    recording = _recording(**{
        roboarm_keys.URDF: _yam_urdf_with_default(),
        eval_keys.POSE_SIGNALS: list(POSE_SIGNALS),
        **statics,
    })
    _assert_reads_at_the_gripper(YAM_ROBOT_TRANSFORM(recording), YAM_EE_FRAME)


def test_a_bimanual_yam_recording_reads_each_pose_of_both_arms_at_the_gripper():
    signals = [
        f'{channel}.{side}{suffix}'
        for channel, suffix in ((keys.ROBOT_STATE, keys.EE_POSE_SUFFIX), (keys.ROBOT_COMMAND, keys.POSE_SUFFIX))
        for side in (keys.LEFT_ARM, keys.RIGHT_ARM)
    ]
    poses = {signal: POSE * geom.Transform3D(np.array([0.0, 0.1 * i, 0.0])) for i, signal in enumerate(signals)}
    stated = ee_frames_entry(dict.fromkeys(signals, EEFrame(site=DEFAULT)))
    recording = _recording(
        poses, **{roboarm_keys.URDF: _yam_urdf_with_default(), eval_keys.POSE_SIGNALS: signals, **stated}
    )

    read = YAM_ROBOT_TRANSFORM(recording)

    _assert_reads_at_the_gripper(read, YAM_EE_FRAME, poses)
    assert set(ee_frames(read)) == set(signals)
