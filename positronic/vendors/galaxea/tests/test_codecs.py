import numpy as np
import pytest

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.vendors.galaxea import codecs, protocol


@pytest.fixture
def observation():
    return {
        keys.JOINTS: np.arange(7, dtype=np.float32) / 10,
        keys.GRIP: np.array([0.2], dtype=np.float32),
        keys.EXTERIOR_IMAGE: np.arange(18, dtype=np.uint8).reshape(2, 3, 3),
        keys.WRIST_IMAGE: np.full((4, 5, 3), 123, dtype=np.uint8),
        keys.TASK: 'pick up the towel',
    }


def test_observation_preserves_rgb_pixels_and_maps_state(observation):
    encoded = codecs.DroidCodec().encode(observation)
    np.testing.assert_array_equal(
        encoded[protocol.IMAGES][protocol.EXTERIOR_IMAGE], observation[keys.EXTERIOR_IMAGE].transpose(2, 0, 1)
    )
    np.testing.assert_array_equal(encoded[protocol.STATE][protocol.RIGHT_ARM], observation[keys.JOINTS])
    np.testing.assert_allclose(encoded[protocol.STATE][protocol.RIGHT_GRIPPER], [0.8])
    np.testing.assert_allclose(observation[keys.GRIP], [0.2])
    assert encoded[protocol.IMAGES][protocol.DUMMY_WRIST_RIGHT].shape == (3, 224, 224)
    assert not encoded[protocol.IMAGES][protocol.DUMMY_WRIST_RIGHT].any()
    assert encoded[protocol.TASK] == observation[keys.TASK]
    assert encoded[protocol.EMBODIMENT_TYPE] == 'Droid_Franka'
    assert encoded[protocol.FREQUENCY] == 15.0


@pytest.mark.parametrize('grip', [0.0, 0.2, 1.0])
def test_gripper_round_trip_keeps_canonical_endpoints(observation, grip):
    codec = codecs.DroidCodec()
    observation[keys.GRIP] = grip
    state = codec.encode(observation)[protocol.STATE]
    action = codec.decode(state)
    assert action[keys.TARGET_GRIP] == pytest.approx(grip)
    np.testing.assert_array_equal(action[keys.ROBOT_COMMAND].positions, observation[keys.JOINTS])


@pytest.mark.parametrize(('prediction', 'target'), [(-0.2, 1.0), (1.2, 0.0)])
def test_gripper_predictions_are_clipped_after_inversion(prediction, target):
    action = codecs.DroidCodec().decode({protocol.RIGHT_ARM: [0.0] * 7, protocol.RIGHT_GRIPPER: [prediction]})
    assert action[keys.TARGET_GRIP] == target


def test_missing_gripper_emits_only_arm_command():
    result = codecs.DroidCodec().decode({protocol.RIGHT_ARM: [0.0] * 7})
    assert set(result) == {keys.ROBOT_COMMAND}


def test_missing_arm_is_an_error():
    with pytest.raises(KeyError, match=protocol.RIGHT_ARM):
        codecs.DroidCodec().decode({protocol.RIGHT_GRIPPER: [0.5]})


@pytest.mark.parametrize('arm', [[0.0] * 6, [[0.0] * 7], [float('nan')] * 7, [float('inf')] * 7])
def test_malformed_arm_predictions_fail(arm):
    with pytest.raises(ValueError, match='finite'):
        codecs.DroidCodec().decode({protocol.RIGHT_ARM: arm})


@pytest.mark.parametrize('grip', [[float('nan')], [float('inf')], [0.0, 1.0]])
def test_invalid_gripper_is_not_treated_as_missing(grip):
    with pytest.raises(ValueError, match='finite'):
        codecs.DroidCodec().decode({protocol.RIGHT_ARM: [0.0] * 7, protocol.RIGHT_GRIPPER: grip})


def test_camera_keys_are_configurable(observation):
    observation['front'] = observation.pop(keys.EXTERIOR_IMAGE)
    observation['hand'] = observation.pop(keys.WRIST_IMAGE)
    encoded = codecs.DroidCodec(exterior_camera='front', wrist_camera='hand').encode(observation)
    assert encoded[protocol.IMAGES][protocol.WRIST_IMAGE].shape == (3, 4, 5)


@pytest.mark.parametrize('image', [np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2, 3)), np.zeros((0, 2, 3))])
def test_invalid_camera_input_fails(observation, image):
    observation[keys.WRIST_IMAGE] = image
    with pytest.raises(ValueError, match='RGB image'):
        codecs.DroidCodec().encode(observation)


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf')])
def test_invalid_frequency_fails(fps):
    with pytest.raises(ValueError, match='fps'):
        codecs.DroidCodec(fps=fps)


def test_entire_chunk_is_timed_and_uses_droid_control_mode(observation):
    codec = codecs.droid(codec=codecs.DroidCodec(fps=10))
    raw = [{protocol.RIGHT_ARM: [float(i)] * 7, protocol.RIGHT_GRIPPER: [i / 31]} for i in range(32)]
    trajectory = codec.decode(raw)
    assert len(trajectory) == 33
    for i, action in enumerate(trajectory[:-1]):
        assert action[keys.ACTION_TIMESTAMP] == pytest.approx(i / 10)
        np.testing.assert_array_equal(action[keys.ROBOT_COMMAND].positions, [i] * 7)
        assert action[keys.ROBOT_COMMAND].mode == DROID_IMPEDANCE
        assert action[keys.TARGET_GRIP] == pytest.approx(1 - i / 31, abs=1e-7)
    assert trajectory[-1] == {keys.ACTION_TIMESTAMP: 3.2}
    assert codec.encode(observation)[protocol.FREQUENCY] == 10
