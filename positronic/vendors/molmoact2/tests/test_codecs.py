import numpy as np
import pytest

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.policy.codecs import ACTION
from positronic.vendors import molmoact2
from positronic.vendors.molmoact2.codecs import droid, droid_3cam, yam_bimanual

TOP, LEFT, RIGHT = keys.EXTERIOR_IMAGE, 'image.wrist_left', 'image.wrist_right'


def _droid_obs() -> dict:
    return {
        keys.JOINTS: np.zeros(7),
        keys.GRIP: 0.0,
        keys.EXTERIOR_IMAGE: np.full((4, 4, 3), 1, dtype=np.uint8),
        keys.EXTERIOR_IMAGE_2: np.full((4, 4, 3), 2, dtype=np.uint8),
        keys.WRIST_IMAGE: np.full((4, 4, 3), 3, dtype=np.uint8),
        keys.TASK: 'pick up the bowl',
    }


def test_droid_3cam_sends_both_exterior_views_then_the_wrist():
    encoded = droid_3cam.instantiate().encode(_droid_obs())

    assert [int(img[0, 0, 0]) for img in encoded[molmoact2.IMAGES]] == [1, 2, 3]


def test_droid_sends_the_first_exterior_view_in_both_exterior_slots():
    encoded = droid.instantiate().encode(_droid_obs())

    assert [int(img[0, 0, 0]) for img in encoded[molmoact2.IMAGES]] == [1, 1, 3]


def test_droid_3cam_refuses_an_observation_without_a_second_exterior_view():
    obs = _droid_obs()
    del obs[keys.EXTERIOR_IMAGE_2]

    with pytest.raises(KeyError, match=keys.EXTERIOR_IMAGE_2):
        droid_3cam.instantiate().encode(obs)


def _obs() -> dict:
    return {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, dtype=np.float64),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.arange(6, 12, dtype=np.float64),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 1.0,
        TOP: np.full((4, 4, 3), 1, dtype=np.uint8),
        LEFT: np.full((4, 4, 3), 2, dtype=np.uint8),
        RIGHT: np.full((4, 4, 3), 3, dtype=np.uint8),
        keys.TASK: 'fold the towel',
    }


def test_observation_packs_left_arm_first_with_vendor_grip_widths():
    encoded = yam_bimanual.instantiate().encode(_obs())

    expected = np.concatenate([np.arange(6), [1.0], np.arange(6, 12), [0.0]]).astype(np.float32)
    np.testing.assert_array_equal(encoded[molmoact2.STATE], expected)
    assert [int(img[0, 0, 0]) for img in encoded[molmoact2.IMAGES]] == [1, 2, 3]
    assert encoded[molmoact2.TASK] == 'fold the towel'


def test_action_splits_into_per_arm_joint_commands_and_positronic_grips():
    vector = np.concatenate([np.arange(6), [1.0], np.arange(6, 12), [0.25]]).astype(np.float32)

    decoded = yam_bimanual.instantiate().decode({ACTION: vector})

    left, right = decoded['robot_command.left'], decoded['robot_command.right']
    assert isinstance(left, command.JointPosition) and isinstance(right, command.JointPosition)
    np.testing.assert_array_equal(left.positions, np.arange(6))
    np.testing.assert_array_equal(right.positions, np.arange(6, 12))
    assert decoded['target_grip.left'] == pytest.approx(0.0)
    assert decoded['target_grip.right'] == pytest.approx(0.75)
