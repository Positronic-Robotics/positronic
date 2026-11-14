"""Native tensor conversion; run in Flux3's environment without weights or a GPU."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from .. import keys

torch = pytest.importorskip('torch', reason='Requires the native Flux3 environment')
pytest.importorskip('flux_action', reason='Requires the native Flux3 environment')


@pytest.fixture
def policy():
    native = import_module('..native', __package__)
    policy = native.Policy.__new__(native.Policy)
    policy._policy = Mock()
    policy._policy.config = SimpleNamespace(
        action_dim=8, chunk_size=32, camera_order=('images.wrist', 'images.left', 'images.right')
    )
    policy._policy.predict_action_chunk.return_value = torch.zeros((1, 32, 8))
    return policy


def observation():
    return {
        **{name: np.full((360, 640, 3), i * 40, np.uint8) for i, name in enumerate(keys.CAMERAS)},
        keys.JOINT_POSITION: np.arange(7, dtype=np.float32),
        keys.GRIPPER_POSITION: np.array([0.25], np.float32),
        keys.PROMPT: 'Pick up the cup.',
    }


@pytest.mark.parametrize('rtc', [False, True])
def test_native_batch_and_prefix_keep_physical_units(policy, rtc):
    obs = observation()
    obs[keys.WRIST].flags.writeable = False
    prefix = np.arange(24, dtype=np.float32).reshape(3, 8) if rtc else None
    result = policy.predict(obs, prefix, seed=42)
    args, kwargs = policy._policy.predict_action_chunk.call_args
    batch = args[0]
    np.testing.assert_array_equal(batch['state'], [[0, 1, 2, 3, 4, 5, 6, 0.25]])
    assert batch['task'] == ['Pick up the cup.']
    for i, name in enumerate(policy.config.camera_order):
        assert batch[name].shape == (1, 3, 360, 640)
        assert batch[name].dtype == torch.uint8
        assert torch.all(batch[name] == i * 40)
    if rtc:
        np.testing.assert_array_equal(kwargs['committed_actions'], prefix[None])
    else:
        assert kwargs['committed_actions'] is None
    assert policy.config.inference_seed == 42
    assert result.shape == (32, 8)
    assert result.dtype == np.float32


@pytest.mark.parametrize('result', [torch.zeros(1, 31, 8), torch.full((1, 32, 8), float('nan'))])
def test_bad_native_results_fail(policy, result):
    policy._policy.predict_action_chunk.return_value = result
    with pytest.raises(ValueError, match='invalid action chunk'):
        policy.predict(observation(), None, seed=0)


@pytest.mark.parametrize('image', [np.zeros((3, 360, 640), np.uint8), np.zeros((360, 640, 3), np.float32)])
def test_bad_camera_inputs_fail_before_model_call(policy, image):
    obs = observation()
    obs[keys.WRIST] = image
    with pytest.raises(ValueError, match='uint8 RGB'):
        policy.predict(obs, None, seed=0)
    policy._policy.predict_action_chunk.assert_not_called()


@pytest.mark.parametrize(
    'config',
    [
        {'camera_layout': 'single'},
        {'inference_profile': 'history'},
        {'action_dim': 6},
        {'rtc_max_prefix': 0},
        {'quantization': 'fp8r'},
        {'action_parameterization': 'joint_delta'},
    ],
)
def test_incompatible_checkpoint_fails_before_gpu_preparation(monkeypatch, config):
    native = import_module('..native', __package__)
    loaded = Mock(
        config=SimpleNamespace(**{
            'camera_layout': 'droid',
            'inference_profile': 'default',
            'action_dim': 8,
            'rtc_max_prefix': 15,
            'quantization': None,
            'action_parameterization': 'absolute',
            **config,
        })
    )
    monkeypatch.setattr(native.FluxActionPolicy, 'from_pretrained', Mock(return_value=loaded))
    prepare = Mock()
    monkeypatch.setattr(native.robolab, 'prepare_serving_policy', prepare)
    with pytest.raises(ValueError):
        native.Policy('checkpoint', 'revision', 'variant', rtc=True)
    prepare.assert_not_called()
