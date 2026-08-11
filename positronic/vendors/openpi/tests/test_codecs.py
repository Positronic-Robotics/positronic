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
        'image.agentview': frame,
        keys.TASK: 'pick up the cube',
    }


# Each pair is a deployment's observation codec and the openpi config it is served under.
# rules-allow: hardcoded-keys — the pairing is the assertion; reading it from the code under test would pass
# whatever that code held.
_SERVED = [
    ('ee_obs', 'pi05_positronic_lowmem'),
    ('ee_joints_obs', 'pi05_positronic_lowmem'),
    ('joints_obs', 'pi05_positronic_lowmem'),
    ('droid_obs', 'pi05_droid'),
    ('libero_obs', 'pi05_libero'),
]


@pytest.mark.parametrize('codec_name, config_name', _SERVED)
def test_warmup_observation_carries_every_field_a_codec_encodes(codec_name, config_name, raw_observation):
    encoded = getattr(codecs, codec_name).instantiate().encode(raw_observation)

    assert set(encoded) <= set(openpi.warm_observation(config_name))


@pytest.mark.parametrize('codec_name', ['ee_obs', 'ee_joints_obs', 'joints_obs'])
def test_a_padding_config_warms_no_narrower_than_its_codec_encodes(codec_name, raw_observation):
    """The transform pads up to the action dimension, so the warm has only to reach the widest state served."""
    encoded = getattr(codecs, codec_name).instantiate().encode(raw_observation)

    warm = openpi.warm_observation('pi05_positronic_lowmem')

    assert warm[openpi.STATE].size >= encoded[openpi.STATE].size


def test_libero_warms_at_the_width_its_transform_hands_over(raw_observation):
    """``LiberoInputs`` does not pad, so this width is the model's own and has to match exactly."""
    encoded = codecs.libero_obs.instantiate().encode(raw_observation)

    warm = openpi.warm_observation('pi05_libero')

    assert warm[openpi.STATE].shape == encoded[openpi.STATE].shape
