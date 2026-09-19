import numpy as np
import pytest

from positronic import keys
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


def test_droid_observation_warms_at_the_widths_it_declares():
    codec = codecs.ObservationCodec(state_features={keys.EE_POSE: 7, keys.GRIP: 1}, image_size=(8, 6))

    obs = codec.warm_observation('pick up the red cube')

    assert obs is not None
    assert obs[openpi.PROMPT] == 'pick up the red cube'
    assert obs[openpi.STATE] == pytest.approx([0, 0, 0, 1, 0, 0, 0, 0])
    assert obs[openpi.IMAGE].shape == (6, 8, 3)


def test_libero_observation_warms_through_its_pose_maths():
    """Its state recodes the rotation, so a zero-filled pose would raise rather than warm."""
    obs = codecs.LiberoObservationCodec(image_size=(8, 6)).warm_observation('close the microwave')

    assert obs is not None
    assert obs[openpi.PROMPT] == 'close the microwave'
    assert obs[openpi.STATE].shape == (8,)
    assert obs[openpi.WRIST_IMAGE].shape == (6, 8, 3)
