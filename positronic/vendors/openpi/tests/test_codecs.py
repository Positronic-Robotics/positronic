import numpy as np
import pytest

from positronic import keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.tests.utils import DummySignal
from positronic.vendors import openpi
from positronic.vendors.openpi import codecs


@pytest.fixture
def raw_observation() -> dict:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    return {
        keys.EE_POSE: np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        keys.JOINTS: np.zeros(7, dtype=np.float32),
        keys.GRIP: 0.0,
        keys.WRIST_IMAGE: frame,
        keys.EXTERIOR_IMAGE: frame,
        keys.TASK: 'pick up the cube',
    }


@pytest.mark.parametrize('name', ['ee_obs', 'ee_joints_obs', 'joints_obs', 'droid_obs', 'libero_obs'])
def test_warmup_observation_carries_every_field_a_codec_encodes(name, raw_observation):
    encoded = getattr(codecs, name).instantiate().encode(raw_observation)

    assert set(encoded) <= set(openpi.warm_observation())


def test_warmup_state_is_the_width_the_transform_that_does_not_pad_hands_over(raw_observation):
    """``LiberoInputs`` passes the state straight to the model, so this is the one width that has to be right."""
    encoded = codecs.libero_obs.instantiate().encode(raw_observation)

    assert openpi.warm_observation()[openpi.STATE].shape == encoded[openpi.STATE].shape


def test_training_state_aligns_recorded_samples():
    codec = codecs.ObservationCodec(state_features={keys.JOINTS: 2, keys.GRIP: 1})
    episode = EpisodeContainer({
        keys.JOINTS: DummySignal([100, 300], [[1, 2], [3, 4]]),
        keys.GRIP: DummySignal([100, 200], [0.0, 1.0]),
    })
    state = codec.training_encoder(episode)['observation.state']
    assert list(state.timestamps(RECORDED_TIME)) == [100, 200, 300]
    np.testing.assert_array_equal(state.values(), [[1, 2, 0], [1, 2, 1], [3, 4, 1]])
