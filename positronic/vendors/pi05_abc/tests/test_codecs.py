import numpy as np
import pytest

from positronic import keys
from positronic.vendors import pi05_abc
from positronic.vendors.pi05_abc import codecs


def _obs(views: dict[str, np.ndarray]) -> dict:
    return {
        keys.arm_channel(keys.ROBOT_STATE, keys.LEFT_ARM) + keys.JOINTS_SUFFIX: np.zeros(6, dtype=np.float32),
        keys.arm_channel(keys.ROBOT_STATE, keys.RIGHT_ARM) + keys.JOINTS_SUFFIX: np.zeros(6, dtype=np.float32),
        keys.arm_channel(keys.GRIP, keys.LEFT_ARM): 0.0,
        keys.arm_channel(keys.GRIP, keys.RIGHT_ARM): 0.0,
        **views,
        keys.TASK: codecs.TASK,
    }


def test_the_yambox_crop_sends_each_cropped_view_without_pad():
    # A 960x600 frame whose crop box is white and whose rest is black: a view that misses the box shows black.
    views = {}
    for key, box in codecs.YAMBOX_CROPS.items():
        frame = np.zeros((600, 960, 3), dtype=np.uint8)
        frame[round(box.top * 600) : round(box.bottom * 600), round(box.left * 960) : round(box.right * 960)] = 255
        views[key] = frame

    request = codecs.AbcObservationCodec(codecs.BOTTLES_PROMPTS, crops=codecs.YAMBOX_CROPS).encode(_obs(views))

    images = request[pi05_abc.IMAGES]
    assert {role: view.shape for role, view in images.items()} == dict.fromkeys(
        (pi05_abc.TOP_CAMERA, pi05_abc.LEFT_CAMERA, pi05_abc.RIGHT_CAMERA), (3, 168, 224)
    )
    assert all(view.min() == 255 for view in images.values())


def test_a_crop_box_for_an_unknown_camera_is_refused():
    with pytest.raises(ValueError, match='name no camera'):
        codecs.AbcObservationCodec(codecs.BOTTLES_PROMPTS, crops={'image.side': codecs.FULL_FRAME})
