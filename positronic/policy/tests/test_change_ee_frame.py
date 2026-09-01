import numpy as np
import pytest

import positronic.drivers.roboarm.command as cmd_module
from positronic import keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm.ik import frame_transform
from positronic.drivers.roboarm.models import (
    CONTROL_FRAME,
    DEFAULT_FRAME,
    DROID_EEF_LINK,
    EE_FRAME,
    EE_LINK,
    FLANGE_LINK,
    URDF,
    bundled_franka_model,
)
from positronic.geom import Rotation, Transform3D, quat_closest
from positronic.policy.codec import ChangeEEFrame
from positronic.policy.spec import from_spec

QUAT = Rotation.Representation.QUAT
URDF = bundled_franka_model()[URDF]
TO_DROID = frame_transform(URDF, DEFAULT_FRAME, DROID_EEF_LINK)


def _pose(t, euler):
    return Transform3D(np.asarray(t, dtype=np.float64), Rotation.from_euler(euler))


def test_droid_eef_matches_robolab_eef_frame():
    transform = frame_transform(URDF, FLANGE_LINK, DROID_EEF_LINK)
    expected = Rotation.from_euler([0.0, 0.0, np.pi / 2])
    np.testing.assert_allclose(transform.translation, [0.0, 0.0, 0.01817402261], atol=1e-9)
    assert quat_closest(transform.rotation, expected) == expected


def test_encode_maps_obs_to_policy_frame():
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose_c.as_vector(QUAT), keys.GRIP: 0.5}

    encoded = ChangeEEFrame(TO_DROID).encode(obs)

    np.testing.assert_allclose(encoded[keys.EE_POSE], (pose_c * TO_DROID).as_vector(QUAT), atol=1e-9)
    assert encoded[keys.GRIP] == 0.5, 'unrelated obs keys pass through'


def test_decode_maps_action_back_to_canonical():
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    action = {keys.ROBOT_COMMAND: cmd_module.CartesianPosition(pose=pose_c * TO_DROID), 'target_grip': 1.0}

    decoded = ChangeEEFrame(TO_DROID)._decode_single(dict(action))

    np.testing.assert_allclose(decoded[keys.ROBOT_COMMAND].pose.as_vector(QUAT), pose_c.as_vector(QUAT), atol=1e-9)
    assert decoded['target_grip'] == 1.0


def test_decode_keeps_the_control_mode_a_command_pinned():
    """Re-expressing a pose does not change what law drives to it."""
    mode = cmd_module.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    pose = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    action = {
        keys.ROBOT_COMMAND: cmd_module.CartesianPosition(pose=pose, mode=mode),
        'other': cmd_module.CartesianDelta(delta=pose, mode=mode),
    }

    decoded = ChangeEEFrame(TO_DROID, keys=(keys.ROBOT_COMMAND, 'other'))._decode_single(action)

    assert decoded[keys.ROBOT_COMMAND].mode == mode
    assert decoded['other'].mode == mode


def test_decode_hands_a_delta_the_frame_it_was_meant_for():
    """A delta has no anchor to convert against, so it travels with its frame for the driver to apply."""
    delta = _pose([0.01, 0.0, -0.02], [0.0, 0.0, 0.1])
    action = {keys.ROBOT_COMMAND: cmd_module.CartesianDelta(delta=delta)}

    decoded = ChangeEEFrame(TO_DROID)._decode_single(action)[keys.ROBOT_COMMAND]

    np.testing.assert_allclose(decoded.delta.as_vector(QUAT), delta.as_vector(QUAT), atol=1e-12)
    np.testing.assert_allclose(decoded.frame.as_vector(QUAT), TO_DROID.as_vector(QUAT), atol=1e-9)


def test_a_delta_moves_the_arm_where_the_policy_meant():
    """The policy's own frame lands exactly where the policy asked, which is what the carried frame buys."""
    measured = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    delta = _pose([0.01, 0.0, -0.02], [0.0, 0.0, 0.1])

    action = {keys.ROBOT_COMMAND: cmd_module.CartesianDelta(delta)}
    decoded = ChangeEEFrame(TO_DROID)._decode_single(action)
    target = decoded[keys.ROBOT_COMMAND].apply(measured)

    before, after = measured * TO_DROID, target * TO_DROID
    np.testing.assert_allclose(after.translation, before.translation + delta.translation, atol=1e-12)
    np.testing.assert_allclose(after.rotation.as_quat, (delta.rotation * before.rotation).as_quat, atol=1e-12)
    ignoring_the_frame = cmd_module._compose_delta(measured, delta)
    assert not np.allclose(target.translation, ignoring_the_frame.translation)


def test_decode_passes_non_cartesian_commands_through():
    action = {keys.ROBOT_COMMAND: cmd_module.JointPosition(positions=np.zeros(7)), 'target_grip': 0.0}
    decoded = ChangeEEFrame(TO_DROID)._decode_single(dict(action))
    assert isinstance(decoded[keys.ROBOT_COMMAND], cmd_module.JointPosition)


def test_identity_transform_leaves_poses_alone():
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose_c.as_vector(QUAT)}
    encoded = ChangeEEFrame(Transform3D.identity).encode(obs)
    np.testing.assert_allclose(encoded[keys.EE_POSE], pose_c.as_vector(QUAT), atol=1e-9)


def test_converts_every_pose_key_present_and_skips_the_rest():
    a, b = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5]), _pose([0.1, 0.2, 0.3], [0.0, 0.1, -0.2])
    obs = {'a': a.as_vector(QUAT), 'b': b.as_vector(QUAT)}

    encoded = ChangeEEFrame(TO_DROID, keys=('a', 'b', 'absent')).encode(obs)

    np.testing.assert_allclose(encoded['a'], (a * TO_DROID).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(encoded['b'], (b * TO_DROID).as_vector(QUAT), atol=1e-9)
    assert 'absent' not in encoded


def test_one_key_carries_a_vector_one_way_and_a_command_the_other():
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    codec = ChangeEEFrame(TO_DROID, keys=('x',))

    encoded = codec.encode({'x': pose_c.as_vector(QUAT)})
    decoded = codec._decode_single({'x': cmd_module.CartesianPosition(pose=pose_c * TO_DROID)})

    np.testing.assert_allclose(encoded['x'], (pose_c * TO_DROID).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(decoded['x'].pose.as_vector(QUAT), pose_c.as_vector(QUAT), atol=1e-9)


def test_encode_passes_through_when_no_pose_key_is_present():
    obs = {keys.JOINTS: np.zeros(7)}
    assert ChangeEEFrame(TO_DROID).encode(obs) is obs


def test_advertises_the_frame_it_speaks():
    codec = ChangeEEFrame(TO_DROID)
    np.testing.assert_allclose(codec.meta[EE_FRAME], TO_DROID.as_vector(QUAT), atol=1e-12)
    assert codec.training_encoder.meta == codec.meta


def test_survives_the_wire_spec_round_trip():
    """The transform is the server's to choose; nothing about the rig's model crosses."""
    codec = ChangeEEFrame(TO_DROID)
    rebuilt = from_spec(codec.to_spec())
    assert isinstance(rebuilt, ChangeEEFrame)
    pose_c = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    obs = {keys.EE_POSE: pose_c.as_vector(QUAT)}
    np.testing.assert_array_equal(rebuilt.encode(obs)[keys.EE_POSE], codec.encode(obs)[keys.EE_POSE])


def test_training_encoder_maps_both_poses_forward():
    """Both poses map forward at training, the dual of the inference asymmetry (obs ``* T``, action ``* T⁻¹``)."""
    obs_pose = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    cmd_pose = _pose([0.2, 0.0, 0.5], [0.0, 0.1, -0.2])
    ts = [1000, 2000]
    episode = EpisodeContainer(
        data={
            URDF: URDF,
            CONTROL_FRAME: DEFAULT_FRAME,
            keys.EE_POSE: DummySignal(ts, np.stack([obs_pose.as_vector(QUAT)] * 2)),
            keys.TARGET_EE_POSE: DummySignal(ts, np.stack([cmd_pose.as_vector(QUAT)] * 2)),
            keys.GRIP: DummySignal(ts, np.array([0.0, 1.0])),
        }
    )

    out = ChangeEEFrame(TO_DROID).training_encoder(episode)

    np.testing.assert_allclose(out[keys.EE_POSE][0][0], (obs_pose * TO_DROID).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(out[keys.TARGET_EE_POSE][0][0], (cmd_pose * TO_DROID).as_vector(QUAT), atol=1e-9)
    np.testing.assert_allclose(out[EE_FRAME], TO_DROID.as_vector(QUAT), atol=1e-9)
    assert keys.GRIP in out, 'unrelated signals pass through'


def _episode(**statics):
    return EpisodeContainer(data={keys.EE_POSE: DummySignal([1000, 2000], np.zeros((2, 7))), **statics})


def test_training_encoder_rejects_poses_anchored_elsewhere():
    """A recording predating the contract names its own frame, and moving those poses is a silent 10cm."""
    with pytest.raises(ValueError, match=EE_LINK):
        ChangeEEFrame(TO_DROID).training_encoder(_episode(**{URDF: URDF, CONTROL_FRAME: EE_LINK}))


def test_training_encoder_accepts_a_rig_that_ships_no_model():
    """What frame the poses sit in is what the recording states; a model confirms that but is not the claim."""
    out = ChangeEEFrame(TO_DROID).training_encoder(_episode(**{CONTROL_FRAME: DEFAULT_FRAME}))
    np.testing.assert_allclose(out[EE_FRAME], TO_DROID.as_vector(QUAT), atol=1e-9)


def test_training_encoder_rejects_an_episode_another_codec_already_moved():
    """The transform names the policy frame from ``default``, so it has no meaning applied twice — the poses
    would land at the product while ``meta`` still declares one of the pair."""
    moved = _episode(**{CONTROL_FRAME: DEFAULT_FRAME, EE_FRAME: TO_DROID.as_vector(QUAT)})
    with pytest.raises(ValueError, match='already sit at'):
        ChangeEEFrame(TO_DROID).training_encoder(moved)


def test_training_encoder_skips_absent_command_pose():
    obs_pose = _pose([0.3, 0.1, 0.4], [0.2, -0.3, 0.5])
    ts = [1000, 2000]
    episode = EpisodeContainer(
        data={
            URDF: URDF,
            CONTROL_FRAME: DEFAULT_FRAME,
            keys.EE_POSE: DummySignal(ts, np.stack([obs_pose.as_vector(QUAT)] * 2)),
            keys.TARGET_JOINTS: DummySignal(ts, np.zeros((2, 7), dtype=np.float32)),
        }
    )

    out = ChangeEEFrame(TO_DROID).training_encoder(episode)

    assert keys.TARGET_EE_POSE not in list(out), 'absent command pose must not be materialized'
    np.testing.assert_allclose(out[keys.EE_POSE][0][0], (obs_pose * TO_DROID).as_vector(QUAT), atol=1e-9)
    assert keys.TARGET_JOINTS in out
