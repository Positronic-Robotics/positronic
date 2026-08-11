import numpy as np
import pytest

from positronic import keys
from positronic.vendors.openpi import codecs, wire


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


@pytest.mark.parametrize('name', ['ee_obs', 'ee_joints_obs', 'joints_obs', 'droid_obs', 'libero_obs'])
def test_warmup_observation_carries_every_field_a_codec_encodes(name, raw_observation):
    encoded = getattr(codecs, name).instantiate().encode(raw_observation)

    assert set(encoded) <= set(wire.warm_observation())
