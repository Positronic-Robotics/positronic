"""Run in the GR00T environment; no checkpoint or GPU is needed."""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from .. import ACTION, EMBODIMENT, JOINT_POSITION, STATE, keys

torch = pytest.importorskip('torch', reason='Requires the native GR00T environment')


@pytest.fixture
def policy():
    pytest.importorskip('gr00t', reason='Requires the native GR00T environment')
    native = import_module('..native', __package__)
    processing = import_module('gr00t.data.state_action.state_action_processor')
    modalities = {
        EMBODIMENT: {
            STATE: {'delta_indices': [0], 'modality_keys': [JOINT_POSITION]},
            ACTION: {
                'delta_indices': [0, 1, 2],
                'modality_keys': [JOINT_POSITION],
                'action_configs': [{'rep': 'RELATIVE', 'type': 'NON_EEF', 'format': 'DEFAULT'}],
            },
        }
    }
    stats = {'min': [-1.0], 'max': [1.0], 'mean': [0.0], 'std': [1.0]}
    processor = processing.StateActionProcessor(
        modalities,
        {EMBODIMENT: {name: {JOINT_POSITION: stats} for name in (STATE, ACTION, 'relative_action')}},
        use_relative_action=True,
    )
    processor.eval()
    instance = native.Policy.__new__(native.Policy)
    instance.embodiment_tag = SimpleNamespace(value=EMBODIMENT)
    instance.processor = Mock(
        state_action_processor=processor,
        modality_configs=processor.modality_configs,
        max_action_horizon=4,
        max_action_dim=2,
    )
    return instance


@pytest.mark.parametrize('reference', [-3.0, 3.0])
def test_prefix_normalization_preserves_outliers_without_changing_state_processing(policy, reference):
    processor = policy.processor.state_action_processor
    state = {JOINT_POSITION: np.array([[reference]], dtype=np.float32)}
    relative = np.array([[-2.3], [0.1234567], [2.3]], dtype=np.float32)
    actions = {JOINT_POSITION: state[JOINT_POSITION] + relative}
    before = processor.apply_state(state, EMBODIMENT)

    encoded = policy._encode_action_prefix(actions, state).numpy()

    np.testing.assert_allclose(encoded[:3, :1], relative, atol=2e-7)
    assert not encoded[3:].any()
    assert not encoded[:, 1:].any()
    assert processor.clip_outliers is True
    np.testing.assert_array_equal(processor.apply_state(state, EMBODIMENT)[JOINT_POSITION], before[JOINT_POSITION])
    np.testing.assert_array_equal(before[JOINT_POSITION], [[np.sign(reference)]])
    np.testing.assert_array_equal(actions[JOINT_POSITION], state[JOINT_POSITION] + relative)


@pytest.mark.parametrize('frozen', [None, 0, 1, 2, 3])
def test_native_reply_keeps_frozen_targets_exact_and_preserves_predicted_tail(policy, frozen):
    state = {JOINT_POSITION: np.array([[[3.0]]], dtype=np.float32)}
    targets = np.array([[5.3], [3.1234567], [0.7]], dtype=np.float32)
    options = None if frozen is None else {ACTION: {JOINT_POSITION: targets.copy()}, keys.RTC.FROZEN_STEPS: frozen}
    processor = policy.processor.state_action_processor
    policy._unbatch_observation = Mock(return_value=[{STATE: {JOINT_POSITION: state[JOINT_POSITION][0]}}])
    policy._to_vla_step_data = Mock(return_value=SimpleNamespace(actions={}))
    policy.processor.return_value = {STATE: torch.ones((1, 1))}
    policy.collate_fn = lambda rows: {'inputs': {name: value[None] for name, value in rows[0].items()}}
    policy.model = Mock()
    prediction = torch.tensor([[[2.0], [0.125], [-2.0]]])
    policy.model.get_action.return_value = {'action_pred': prediction}
    policy.processor.decode_action.side_effect = lambda values, embodiment, state: processor.unapply_action(
        {JOINT_POSITION: values}, embodiment.value, state
    )
    expected = processor.unapply_action({JOINT_POSITION: prediction.numpy()}, EMBODIMENT, state)[JOINT_POSITION]

    result, _ = policy._get_action({STATE: state}, options)

    sent = policy.model.get_action.call_args.kwargs['inputs']
    assert processor.clip_outliers is True
    if options is None:
        assert ACTION not in sent
        np.testing.assert_array_equal(result[JOINT_POSITION], expected.astype(np.float32))
        return
    np.testing.assert_array_equal(result[JOINT_POSITION][0, :frozen], targets[:frozen])
    np.testing.assert_array_equal(result[JOINT_POSITION][:, frozen:], expected[:, frozen:].astype(np.float32))
    assert sent[ACTION].abs().max() > 1
    np.testing.assert_array_equal(options[ACTION][JOINT_POSITION], targets)
