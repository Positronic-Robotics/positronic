import numpy as np
import pytest

from positronic import geom, keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm import models
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import ACTION
from positronic.vendors import gr00t
from positronic.vendors.gr00t.codecs import DroidCodec, droid, droid_three_cameras


@pytest.mark.parametrize('config', [droid, droid_three_cameras])
def test_pose_conversion_preserves_droid_convention_and_tool_frame_metadata(config):
    codec = config(image_mappings={})
    rng = np.random.default_rng(3)
    correction = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    for _ in range(20):
        pose = geom.Transform3D(rng.normal(size=3), geom.Rotation.from_quat(rng.normal(size=4)))
        inputs = {
            keys.EE_POSE: pose.as_vector(geom.Rotation.Representation.QUAT),
            keys.JOINTS: np.zeros(7),
            keys.GRIP: 0.2,
            keys.TASK: 'pick',
        }
        tool_pose = pose * models.DROID_EE_FRAME
        expected = np.concatenate([
            tool_pose.translation,
            (tool_pose.rotation.as_rotation_matrix @ correction)[:2].reshape(6),
        ]).astype(np.float32)
        encoded = codec.encode(inputs)[gr00t.STATE][gr00t.EE_POSE][0, 0]
        np.testing.assert_allclose(encoded, expected, atol=2e-7)

    expected_frame = models.DROID_EE_FRAME.as_vector(geom.Rotation.Representation.QUAT)
    np.testing.assert_array_equal(codec.meta[roboarm_keys.EE_FRAME], expected_frame)
    np.testing.assert_array_equal(codec.training_encoder.meta[roboarm_keys.EE_FRAME], expected_frame)


def test_droid_decodes_full_chunk_and_binarizes_grip():
    codec = droid()
    targets = np.arange(40 * 7, dtype=np.float32).reshape(40, 7) / 100
    output = [
        {gr00t.JOINT_POSITION: q, gr00t.GRIP: [0.5 if i % 2 else 0.51], gr00t.EE_POSE: np.zeros(9)}
        for i, q in enumerate(targets)
    ]
    decoded = codec.decode(output)
    assert len(decoded) == 40
    for i, item in enumerate(decoded):
        np.testing.assert_array_equal(item[keys.ROBOT_COMMAND].positions, targets[i])
        assert item[keys.ROBOT_COMMAND].mode == DROID_IMPEDANCE
        assert item[keys.TARGET_GRIP] == (0.0 if i % 2 else 1.0)
        assert 'timestamp' not in item


def test_training_cadence_is_preserved_without_timestamp_commands():
    codec = droid(training_fps=20)
    assert codec.training_encoder.meta[policy_keys.ACTION_FPS] == 20


def test_training_actions_align_recorded_samples():
    codec = droid(image_mappings={}, ee_frame=geom.Transform3D.identity)
    episode = EpisodeContainer({
        keys.EE_POSE: DummySignal([100], [[0, 0, 0, 1, 0, 0, 0]]),
        keys.JOINTS: DummySignal([100, 300], [np.zeros(7), np.ones(7)]),
        keys.GRIP: DummySignal([100, 200], [0.0, 1.0]),
    })
    encoded = codec.training_encoder(episode)
    action = encoded[ACTION]
    assert list(action.timestamps(RECORDED_TIME)) == [100, 200, 300]
    values = np.asarray(action.values())
    np.testing.assert_array_equal(values[:, :9], np.repeat(encoded[gr00t.EE_POSE].values(), 3, axis=0))
    np.testing.assert_array_equal(values[:, 9], [0, 1, 1])
    np.testing.assert_array_equal(values[:, 10:], [np.zeros(7), np.zeros(7), np.ones(7)])


def test_droid_packing_requires_xyz_rot6d_poses_for_training_and_inference():
    codec = DroidCodec(image_mappings={})
    pose = [0, 0, 0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match='reshape'):
        codec.encode({keys.EE_POSE: pose})
    training = codec.training_encoder(EpisodeContainer({keys.EE_POSE: DummySignal([0], [pose])}))
    with pytest.raises(ValueError, match='reshape'):
        training[gr00t.EE_POSE].values()[0]
