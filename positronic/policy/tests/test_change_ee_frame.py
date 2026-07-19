import numpy as np

import positronic.drivers.roboarm.command as cmd_module
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm.ik import frame_transform
from positronic.drivers.roboarm.models import bundled_franka_model
from positronic.geom import Rotation, Transform3D
from positronic.policy.codec import ChangeEEFrame

QUAT = Rotation.Representation.QUAT
# RoboLab's DROID end-effector control frame: eef_frame = Robotiq_2F_85/base_link composed with a pure rotation
# EEF_OFFSET_ROT (wxyz) and zero translation (robolab/robots/droid.py). ``droid_eef`` reproduces it.
EEF_OFFSET_ROT = (0.5, -0.5, 0.5, -0.5)
URDF = bundled_franka_model()['urdf']
CONTROL_FRAME = 'end_effector'


def _pose(t, euler):
    return Transform3D(np.asarray(t, dtype=np.float64), Rotation.from_euler(euler))


def _quat_close(a, b, atol=1e-9):
    # Quaternion double cover: q and -q are the same rotation.
    return min(np.linalg.norm(a - b), np.linalg.norm(a + b)) < atol


def test_droid_eef_realizes_the_eef_offset_rotation():
    """The canonical->droid transform is the DROID ``EEF_OFFSET_ROT`` rotation plus the base offset, so our
    ``droid_eef`` site reproduces RoboLab's ``eef_frame = base_link ∘ EEF_OFFSET_ROT``."""
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    assert _quat_close(transform.rotation.as_quat, np.array(EEF_OFFSET_ROT))


def test_encode_maps_obs_to_policy_frame():
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {'robot_state.ee_pose': pose_c.as_vector(QUAT), 'urdf': URDF, 'control_frame': CONTROL_FRAME, 'grip': 0.5}

    encoded = ChangeEEFrame(to='droid_eef').encode(obs)

    np.testing.assert_allclose(encoded['robot_state.ee_pose'], (pose_c * transform).as_vector(QUAT), atol=1e-9)
    assert encoded['grip'] == 0.5, 'unrelated obs keys pass through'


def test_decode_maps_action_back_to_canonical():
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {'robot_state.ee_pose': pose_c.as_vector(QUAT), 'urdf': URDF, 'control_frame': CONTROL_FRAME}
    # The policy emits its command in the droid frame (canonical composed with the transform); decode must invert it.
    action = {'robot_command': cmd_module.CartesianPosition(pose=pose_c * transform), 'target_grip': 1.0}

    decoded = ChangeEEFrame(to='droid_eef')._decode_single(dict(action), context=obs)

    np.testing.assert_allclose(decoded['robot_command'].pose.as_vector(QUAT), pose_c.as_vector(QUAT), atol=1e-9)
    assert decoded['target_grip'] == 1.0


def test_decode_passes_non_cartesian_commands_through():
    obs = {'urdf': URDF, 'control_frame': CONTROL_FRAME}
    action = {'robot_command': cmd_module.JointPosition(positions=np.zeros(7)), 'target_grip': 0.0}
    decoded = ChangeEEFrame(to='droid_eef')._decode_single(dict(action), context=obs)
    assert isinstance(decoded['robot_command'], cmd_module.JointPosition)


def test_identity_when_target_equals_control_frame():
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {'robot_state.ee_pose': pose_c.as_vector(QUAT), 'urdf': URDF, 'control_frame': CONTROL_FRAME}
    encoded = ChangeEEFrame(to=CONTROL_FRAME).encode(obs)
    np.testing.assert_allclose(encoded['robot_state.ee_pose'], pose_c.as_vector(QUAT), atol=1e-9)


def test_training_encoder_maps_both_poses_forward():
    """At training both the observed and the commanded pose map forward ``* T`` (both are canonical->policy),
    the deliberate dual of the inference asymmetry (obs ``* T``, action ``* T``-inverse)."""
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    obs_pose = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    cmd_pose = _pose([0.2, 0.0, 0.5], [0.0, 0.1, -0.2])
    ts = [1000, 2000]
    episode = EpisodeContainer(
        data={
            'urdf': URDF,
            'control_frame': CONTROL_FRAME,
            'robot_state.ee_pose': DummySignal(ts, np.stack([obs_pose.as_vector(QUAT)] * 2)),
            'robot_command.pose': DummySignal(ts, np.stack([cmd_pose.as_vector(QUAT)] * 2)),
            'grip': DummySignal(ts, np.array([0.0, 1.0])),
        }
    )

    out = ChangeEEFrame(to='droid_eef').training_encoder(episode)

    np.testing.assert_allclose(out['robot_state.ee_pose'][0][0], (obs_pose * transform).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(out['robot_command.pose'][0][0], (cmd_pose * transform).as_vector(QUAT), atol=1e-9)
    assert out['control_frame'] == CONTROL_FRAME and 'grip' in out, 'statics and unrelated signals pass through'
