import numpy as np
import pytest

from positronic import keys
from positronic.cfg.eval.sim import libero as libero_cfg
from positronic.dataset.serializers import expand_suffixed
from positronic.eval import Embodiment
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.libero.adapter import LiberoAdapter
from positronic.vendors.openpi import codecs as openpi_codecs


def _policy_inputs(embodiment: Embodiment, adapter: EnvAdapter, raw_obs: dict) -> dict:
    """The observation a policy receives: the adapter's canonical signals through the embodiment's serializers,
    unfolded into wire keys the way ``Harness._build_obs`` assembles them, plus the task the context carries."""
    inputs = {keys.TASK: 'pick up the black bowl'}
    for name, value in adapter.observations(raw_obs).items():
        serializer = embodiment.observations[name].serializer
        inputs.update(expand_suffixed(name, serializer(value) if serializer is not None else value))
    return inputs


# A LIBERO ``step`` payload, the shape ``LiberoEnv._observe`` returns over the wire.
_LIBERO_RAW_OBS = {
    'eef_pos': np.array([0.1, 0.2, 0.3]),
    'eef_quat': np.array([0.0, 0.0, 0.0, 1.0]),
    'joint_pos': np.zeros(7),
    'joint_vel': np.zeros(7),
    'grip': 0.4,
    'agentview_image': np.zeros((256, 256, 3), dtype=np.uint8),
    'eye_in_hand_image': np.zeros((256, 256, 3), dtype=np.uint8),
    'sim_state': np.zeros(4),
}


@pytest.mark.parametrize('codec_cfg', [openpi_codecs.droid_obs, openpi_codecs.libero_obs], ids=['droid', 'libero'])
def test_libero_observation_encodes_through_droid_and_libero_codecs(codec_cfg):
    """Serve once, score anywhere: a LIBERO observation encodes through the DROID codec — built for another env —
    as readily as through LIBERO's own. What it scores is another matter; the viewpoints differ."""
    ev = libero_cfg.spatial.instantiate()
    inputs = _policy_inputs(ev.embodiment, LiberoAdapter(libero_cfg.spatial.kwargs['camera_dict']), _LIBERO_RAW_OBS)

    codec = codec_cfg.instantiate()
    assert set(codec.encode(inputs)) == set(codec.dummy_encoded())
