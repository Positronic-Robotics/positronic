import numpy as np
import pytest

from positronic import keys
from positronic.vendors import flux3_action
from positronic.vendors.flux3_action import codecs


def _view(level: int) -> np.ndarray:
    return np.full((360, 640, 3), level, dtype=np.uint8)


@pytest.mark.parametrize(
    ('codec', 'second_exterior'), [(codecs.droid_3cam, 200), (codecs.droid, 100)], ids=['droid_3cam', 'droid']
)
def test_the_codec_names_which_camera_fills_the_second_exterior_slot(codec, second_exterior):
    encoded = codec.instantiate().encode({
        keys.WRIST_IMAGE: _view(50),
        keys.EXTERIOR_IMAGE: _view(100),
        keys.EXTERIOR_IMAGE_2: _view(200),
        keys.JOINTS: np.zeros(7),
        keys.GRIP: 0.0,
        keys.TASK: '',
    })
    assert np.all(encoded[flux3_action.WRIST_IMAGE] == 50)
    assert np.all(encoded[flux3_action.EXTERIOR_IMAGE_1] == 100)
    assert np.all(encoded[flux3_action.EXTERIOR_IMAGE_2] == second_exterior)
