"""The robot transforms read every recording at ``GRIPPER_SITE``, from the model it was recorded on."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from positronic import geom, keys
from positronic.cfg.ds.internal import REAL_ROBOT_TRANSFORM, SIM_ROBOT_TRANSFORM
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
    EEFrame,
    attach_frame,
    bundled_franka_model,
    bundled_panda_model,
    ee_frames_entry,
)
from positronic.policy.training import ChangeEEFrame
from positronic.utils import package_assets_path

QUAT = geom.Rotation.Representation.QUAT
OFFSET = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.from_euler([0.0, 0.0, 0.3]))
POSE = geom.Transform3D(np.array([0.3, 0.1, 0.4]), geom.Rotation.from_euler([0.2, -0.1, 0.5]))
# rules-allow: hardcoded-keys — the keys and the site a recording wrote before ``gripper``, which no code writes now
CONTROL_FRAME, EE_FRAME, DEFAULT = 'control_frame', 'ee_frame', 'default'
POSE_SIGNALS = (keys.EE_POSE, keys.TARGET_EE_POSE)


def _recording(**statics):
    poses = DummySignal([1000, 2000], np.tile(POSE.as_vector(QUAT), (2, 1)))
    return EpisodeContainer(data={keys.EE_POSE: poses, keys.TARGET_EE_POSE: poses, **statics})


def _assert_reads_at_the_gripper(episode, gripper_to_recorded: geom.Transform3D) -> None:
    """Every pose signal of ``episode`` sits at ``GRIPPER_SITE``, moved off the frame the recording was made in."""
    want = POSE * gripper_to_recorded.inv
    for signal in POSE_SIGNALS:
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
    [(REAL_ROBOT_TRANSFORM, bundled_franka_model()), (SIM_ROBOT_TRANSFORM, bundled_panda_model())],
    ids=['real', 'sim'],
)
def test_a_recording_at_the_gripper_keeps_its_poses(transform, model):
    stated = ee_frames_entry(dict.fromkeys(POSE_SIGNALS, EEFrame(site=GRIPPER_SITE)))
    recording = _recording(**{roboarm_keys.URDF: model[roboarm_keys.URDF], **stated})

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
