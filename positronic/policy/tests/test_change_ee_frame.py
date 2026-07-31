import numpy as np

import positronic.drivers.roboarm.command as cmd_module
from positronic import keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm.ik import frame_transform
from positronic.drivers.roboarm.models import bundled_franka_model
from positronic.geom import Rotation, Transform3D, quat_closest
from positronic.policy.codec import ChangeEEFrame
from positronic.policy.spec import from_spec

QUAT = Rotation.Representation.QUAT
# RoboLab's DROID end-effector control frame ``eef_frame`` = Robotiq_2F_85/base_link ∘ EEF_OFFSET_ROT with zero
# position (robolab/robots/droid.py). Measured off RoboLab's DROID USD, relative to the flange it is 18.17mm along
# Z and a +90deg Z rotation; ``droid_eef`` reproduces it.
URDF = bundled_franka_model()[keys.URDF]
CONTROL_FRAME = 'end_effector'


def _pose(t, euler):
    return Transform3D(np.asarray(t, dtype=np.float64), Rotation.from_euler(euler))


def test_droid_eef_matches_robolab_eef_frame():
    """``droid_eef`` relative to the flange is RoboLab's ``eef_frame``, measured from its DROID USD: 18.17mm along
    the flange Z and a +90deg Z rotation (``link8`` -> ``Robotiq_2F_85/base_link`` ∘ EEF_OFFSET_ROT)."""
    transform = frame_transform(URDF, 'link8', 'droid_eef')
    expected = Rotation.from_euler([0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.01817402261], atol=1e-9)
    assert quat_closest(transform.rotation, expected) == expected


def test_encode_maps_obs_to_policy_frame():
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose_c.as_vector(QUAT), keys.URDF: URDF, keys.CONTROL_FRAME: CONTROL_FRAME, 'grip': 0.5}

    encoded = ChangeEEFrame(to='droid_eef').encode(obs)

    np.testing.assert_allclose(encoded[keys.EE_POSE], (pose_c * transform).as_vector(QUAT), atol=1e-9)
    assert encoded['grip'] == 0.5, 'unrelated obs keys pass through'


def test_decode_maps_action_back_to_canonical():
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose_c.as_vector(QUAT), keys.URDF: URDF, keys.CONTROL_FRAME: CONTROL_FRAME}
    # The policy emits its command in the droid frame (canonical composed with the transform); decode must invert it.
    action = {'robot_command': cmd_module.CartesianPosition(pose=pose_c * transform), 'target_grip': 1.0}

    decoded = ChangeEEFrame(to='droid_eef')._decode_single(dict(action), context=obs)

    np.testing.assert_allclose(decoded['robot_command'].pose.as_vector(QUAT), pose_c.as_vector(QUAT), atol=1e-9)
    assert decoded['target_grip'] == 1.0


def test_decode_passes_non_cartesian_commands_through():
    obs = {keys.URDF: URDF, keys.CONTROL_FRAME: CONTROL_FRAME}
    action = {'robot_command': cmd_module.JointPosition(positions=np.zeros(7)), 'target_grip': 0.0}
    decoded = ChangeEEFrame(to='droid_eef')._decode_single(dict(action), context=obs)
    assert isinstance(decoded['robot_command'], cmd_module.JointPosition)


def test_identity_when_target_equals_control_frame():
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose_c.as_vector(QUAT), keys.URDF: URDF, keys.CONTROL_FRAME: CONTROL_FRAME}
    encoded = ChangeEEFrame(to=CONTROL_FRAME).encode(obs)
    np.testing.assert_allclose(encoded[keys.EE_POSE], pose_c.as_vector(QUAT), atol=1e-9)


def test_converts_every_pose_key_present_and_skips_the_rest():
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    a, b = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5]), _pose([0.1, 0.2, 0.3], [0.0, 0.1, -0.2])
    obs = {'a': a.as_vector(QUAT), 'b': b.as_vector(QUAT), keys.URDF: URDF, keys.CONTROL_FRAME: CONTROL_FRAME}

    encoded = ChangeEEFrame(to='droid_eef', keys=('a', 'b', 'absent')).encode(obs)

    np.testing.assert_allclose(encoded['a'], (a * transform).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(encoded['b'], (b * transform).as_vector(QUAT), atol=1e-9)
    assert 'absent' not in encoded


def test_one_key_carries_a_vector_one_way_and_a_command_the_other():
    """The same key list serves both directions: whatever shape a key's value has, its pose crosses the frame."""
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    obs = {'x': pose_c.as_vector(QUAT), keys.URDF: URDF, keys.CONTROL_FRAME: CONTROL_FRAME}
    codec = ChangeEEFrame(to='droid_eef', keys=('x',))

    encoded = codec.encode(obs)
    decoded = codec._decode_single({'x': cmd_module.CartesianPosition(pose=pose_c * transform)}, context=obs)

    np.testing.assert_allclose(encoded['x'], (pose_c * transform).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(decoded['x'].pose.as_vector(QUAT), pose_c.as_vector(QUAT), atol=1e-9)


def test_decode_leaves_a_joint_action_alone_without_the_model():
    """A joint target has no pose, so decoding one neither converts nor reaches for the robot model."""
    action = {'robot_command': cmd_module.JointPosition(positions=np.zeros(7))}
    assert ChangeEEFrame(to='droid_eef')._decode_single(action, context={}) is action


def test_reads_the_model_from_the_configured_keys():
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    pose = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose.as_vector(QUAT), 'model.urdf': URDF, 'model.frame': CONTROL_FRAME}

    codec = ChangeEEFrame(to='droid_eef', urdf_key='model.urdf', control_frame_key='model.frame')

    np.testing.assert_allclose(codec.encode(obs)[keys.EE_POSE], (pose * transform).as_vector(QUAT), atol=1e-9)


def test_encode_passes_through_when_no_pose_key_is_present():
    """A joint-only observation carries no pose, so the codec neither converts nor demands a robot model."""
    obs = {keys.JOINTS: np.zeros(7)}
    assert ChangeEEFrame(to='droid_eef').encode(obs) is obs


def test_advertises_the_frame_it_speaks():
    """The frame reaches episode meta both ways: through the codec at serving, its dual at conversion."""
    codec = ChangeEEFrame(to='droid_eef')
    assert codec.meta == {'ee_frame': 'droid_eef'}
    assert codec.training_encoder.meta == {'ee_frame': 'droid_eef'}


def test_survives_the_wire_spec_round_trip():
    """A server declares the conversion in its handshake, so the rig must rebuild an equivalent codec from the
    spec alone — the frame name is the server's to choose, the robot model the rig's."""
    codec = ChangeEEFrame(to='droid_eef')
    rebuilt = from_spec(codec.to_spec())
    assert isinstance(rebuilt, ChangeEEFrame)
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose_c.as_vector(QUAT), keys.URDF: URDF, keys.CONTROL_FRAME: CONTROL_FRAME}
    np.testing.assert_array_equal(rebuilt.encode(obs)[keys.EE_POSE], codec.encode(obs)[keys.EE_POSE])


def test_training_encoder_maps_both_poses_forward():
    """At training both the observed and the commanded pose map forward ``* T`` (both are canonical->policy),
    the deliberate dual of the inference asymmetry (obs ``* T``, action ``* T``-inverse)."""
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    obs_pose = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    cmd_pose = _pose([0.2, 0.0, 0.5], [0.0, 0.1, -0.2])
    ts = [1000, 2000]
    episode = EpisodeContainer(
        data={
            keys.URDF: URDF,
            keys.CONTROL_FRAME: CONTROL_FRAME,
            keys.EE_POSE: DummySignal(ts, np.stack([obs_pose.as_vector(QUAT)] * 2)),
            'robot_command.pose': DummySignal(ts, np.stack([cmd_pose.as_vector(QUAT)] * 2)),
            'grip': DummySignal(ts, np.array([0.0, 1.0])),
        }
    )

    out = ChangeEEFrame(to='droid_eef').training_encoder(episode)

    np.testing.assert_allclose(out[keys.EE_POSE][0][0], (obs_pose * transform).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(out['robot_command.pose'][0][0], (cmd_pose * transform).as_vector(QUAT), atol=1e-9)
    # ``control_frame`` is relabeled to the policy frame so downstream IK reads the transformed poses correctly.
    assert out[keys.CONTROL_FRAME] == 'droid_eef' and 'grip' in out, 'frame relabeled; unrelated signals pass through'


def test_training_encoder_skips_absent_command_pose():
    """A joint-only dataset has no ``robot_command.pose``; the training dual converts the observation pose and
    relabels the frame without registering (and dereferencing) the missing command pose."""
    obs_pose = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    ts = [1000, 2000]
    episode = EpisodeContainer(
        data={
            keys.URDF: URDF,
            keys.CONTROL_FRAME: CONTROL_FRAME,
            keys.EE_POSE: DummySignal(ts, np.stack([obs_pose.as_vector(QUAT)] * 2)),
            'robot_command.joints': DummySignal(ts, np.zeros((2, 7), dtype=np.float32)),
        }
    )

    out = ChangeEEFrame(to='droid_eef').training_encoder(episode)

    assert 'robot_command.pose' not in list(out), 'absent command pose must not be materialized'
    transform = frame_transform(URDF, CONTROL_FRAME, 'droid_eef')
    np.testing.assert_allclose(out[keys.EE_POSE][0][0], (obs_pose * transform).as_vector(QUAT), atol=1e-9)
    assert out[keys.CONTROL_FRAME] == 'droid_eef' and 'robot_command.joints' in out
