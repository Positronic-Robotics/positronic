import numpy as np
import pytest
from positronic_model_server import serialization
from positronic_model_server.rtc import ActionPrefix


def test_prefix_round_trip_preserves_sampling_and_native_targets():
    prefix = ActionPrefix(2, [4.5, 5.5], 1, [{'joints': np.arange(7)}, {'joints': np.ones(7)}])
    restored = ActionPrefix.from_observation(
        serialization.deserialise(serialization.serialise(prefix.to_observation()))
    )
    assert (restored.sequence, restored.positions, restored.frozen_steps) == (2, [4.5, 5.5], 1)
    for expected, actual in zip(prefix.actions, restored.actions, strict=True):
        np.testing.assert_array_equal(actual['joints'], expected['joints'])


@pytest.mark.parametrize(
    'changes',
    [
        {'sequence': True},
        {'sequence': -1},
        {'sequence': 0},
        {'positions': [float('nan')]},
        {'positions': [-1]},
        {'positions': [True]},
        {'positions': 'x'},
        {'frozen_steps': True},
        {'frozen_steps': 2},
        {'frozen_steps': -1},
        {'actions': []},
        {'actions': [None]},
    ],
)
def test_prefix_rejects_invalid_requests(changes):
    with pytest.raises(ValueError):
        ActionPrefix(**{'sequence': 1, 'positions': [0], 'frozen_steps': 1, 'actions': [{}], **changes})
