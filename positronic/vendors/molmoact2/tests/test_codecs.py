import numpy as np
import pytest

from positronic import keys
from positronic.vendors import molmoact2
from positronic.vendors.molmoact2.codecs import MolmoAct2ObservationCodec


def _observation() -> dict:
    return {
        keys.JOINTS: np.arange(7, dtype=np.float32) / 10,
        keys.GRIP: np.array([0.2], dtype=np.float32),
        keys.EXTERIOR_IMAGE: np.full((4, 5, 3), 7, dtype=np.uint8),
        keys.WRIST_IMAGE: np.full((4, 5, 3), 9, dtype=np.uint8),
        keys.TASK: 'pick up the red cube',
    }


def test_observation_emits_the_ordered_views_and_the_8d_state():
    encoded = MolmoAct2ObservationCodec().encode(_observation())

    views = encoded[molmoact2.IMAGES]
    assert len(views) == 3
    # No second exterior camera is configured, so the first stands in for it.
    np.testing.assert_array_equal(views[0], views[1])
    np.testing.assert_array_equal(views[2], _observation()[keys.WRIST_IMAGE])
    assert encoded[molmoact2.STATE] == pytest.approx([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2])
    assert encoded[molmoact2.TASK] == 'pick up the red cube'


def test_an_image_that_is_not_hwc_rgb_is_refused():
    observation = _observation()
    observation[keys.WRIST_IMAGE] = np.zeros((4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match='must be HWC'):
        MolmoAct2ObservationCodec().encode(observation)


def test_observation_warms_at_the_size_the_model_tiles_to():
    obs = MolmoAct2ObservationCodec().warm_observation('stack the cubes')

    assert obs is not None
    assert obs[molmoact2.TASK] == 'stack the cubes'
    assert obs[molmoact2.STATE].shape == (8,)
    width, height = molmoact2.IMAGE_SIZE
    assert [view.shape for view in obs[molmoact2.IMAGES]] == [(height, width, 3)] * 3
