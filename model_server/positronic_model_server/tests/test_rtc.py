import numpy as np
import pytest
from positronic_model_server import serialization
from positronic_model_server.rtc import ESTIMATED_LATENCY_SEC, ActionPrefix, estimated_latency


def test_prefix_round_trip_preserves_sampling_and_native_targets():
    prefix = ActionPrefix(2, [4.5, 5.5], [{'joints': np.arange(7)}, {'joints': np.ones(7)}])
    restored = ActionPrefix.from_observation(
        serialization.deserialise(serialization.serialise(prefix.to_observation()))
    )
    assert (restored.sequence, restored.positions) == (2, [4.5, 5.5])
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
        {'actions': []},
        {'actions': [None]},
    ],
)
def test_prefix_rejects_invalid_requests(changes):
    with pytest.raises(ValueError):
        ActionPrefix(**{'sequence': 1, 'positions': [0], 'actions': [{}], **changes})


@pytest.mark.parametrize('value', [True, None, '0.2', -1, float('inf'), float('nan')])
def test_latency_rejects_invalid_estimates(value):
    with pytest.raises(ValueError, match='estimated_latency_sec'):
        estimated_latency({ESTIMATED_LATENCY_SEC: value})


@pytest.mark.parametrize('value', [0, 0.2, 2])
def test_latency_is_independent_of_the_action_prefix(value):
    request = serialization.deserialise(serialization.serialise({ESTIMATED_LATENCY_SEC: value}))
    assert estimated_latency(request) == value
