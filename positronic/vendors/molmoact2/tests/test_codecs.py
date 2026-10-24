import numpy as np

from positronic import keys
from positronic.vendors import molmoact2
from positronic.vendors.molmoact2.codecs import MolmoAct2BimanualObservationCodec

TOP, LEFT, RIGHT = keys.EXTERIOR_IMAGE, 'image.wrist_left', 'image.wrist_right'


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
    encoded = MolmoAct2BimanualObservationCodec().encode(_obs())

    expected = np.concatenate([np.arange(6), [1.0], np.arange(6, 12), [0.0]]).astype(np.float32)
    np.testing.assert_array_equal(encoded[molmoact2.STATE], expected)
    assert [int(img[0, 0, 0]) for img in encoded[molmoact2.IMAGES]] == [1, 2, 3]
    assert encoded[molmoact2.TASK] == 'fold the towel'
