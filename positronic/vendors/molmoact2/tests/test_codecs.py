import numpy as np
import pytest

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.policy.codec import ACTION
from positronic.vendors import molmoact2
from positronic.vendors.molmoact2.codecs import BimanualJointsAction, MolmoAct2BimanualObservationCodec

TOP, LEFT, RIGHT = keys.EXTERIOR_IMAGE, 'image.wrist_left', 'image.wrist_right'


def _obs() -> dict:
    return {
        'robot_state.left.q': np.arange(6, dtype=np.float64),
        'robot_state.right.q': np.arange(6, 12, dtype=np.float64),
        'grip.left': 0.0,
        'grip.right': 1.0,
        TOP: np.full((4, 4, 3), 1, dtype=np.uint8),
        LEFT: np.full((4, 4, 3), 2, dtype=np.uint8),
        RIGHT: np.full((4, 4, 3), 3, dtype=np.uint8),
        keys.TASK: 'fold the towel',
    }


def test_observation_packs_left_arm_first_with_vendor_grip_widths():
    encoded = MolmoAct2BimanualObservationCodec().encode(_obs())

    expected = np.concatenate([np.arange(6), [1.0], np.arange(6, 12), [0.0]]).astype(np.float32)
    np.testing.assert_array_equal(encoded[molmoact2.STATE], expected)
    assert [int(img[0, 0, 0]) for img in encoded[molmoact2.IMAGES]] == [1, 2, 3]
    assert encoded[molmoact2.TASK] == 'fold the towel'


def test_action_splits_into_per_arm_joint_commands_and_positronic_grips():
    vector = np.concatenate([np.arange(6), [1.0], np.arange(6, 12), [0.25]]).astype(np.float32)

    decoded = BimanualJointsAction().decode({ACTION: vector})

    left, right = decoded['robot_command.left'], decoded['robot_command.right']
    assert isinstance(left, command.JointPosition) and isinstance(right, command.JointPosition)
    np.testing.assert_array_equal(left.positions, np.arange(6))
    np.testing.assert_array_equal(right.positions, np.arange(6, 12))
    assert decoded['target_grip.left'] == pytest.approx(0.0)
    assert decoded['target_grip.right'] == pytest.approx(0.75)


def test_action_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match='14-D'):
        BimanualJointsAction().decode({ACTION: np.zeros(8, dtype=np.float32)})
