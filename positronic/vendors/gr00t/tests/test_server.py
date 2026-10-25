import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from positronic_model_server.rtc import ActionPrefix
from positronic_model_server.spec import resolve_params

from positronic.policy import keys as policy_keys
from positronic.policy.codecs.metadata import MODEL_SETTINGS_PATH
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import keys, server
from positronic.vendors.gr00t.serving import settings as model_settings


@pytest.mark.parametrize(('step', 'expected'), [(None, '010000'), (5000, '005000')])
def test_checkpoint_selection_preserves_padded_directory_names(tmp_path, step, expected):
    for suffix in ('005000', '010000'):
        checkpoint = tmp_path / f'checkpoint-{suffix}'
        checkpoint.mkdir()
        (checkpoint / gr00t.CONFIG_FILENAME).write_text('{}')
    path, metadata = server.resolve_checkpoint(str(tmp_path), step)
    assert path == tmp_path / f'checkpoint-{expected}'
    assert metadata[policy_keys.CHECKPOINT_ID] == str(int(expected))
    _, direct_metadata = server.resolve_checkpoint(str(path))
    assert direct_metadata == metadata


def test_single_checkpoint_and_missing_steps(tmp_path):
    (tmp_path / gr00t.CONFIG_FILENAME).write_text('{}')
    path, _ = server.resolve_checkpoint(str(tmp_path))
    assert path == tmp_path
    with pytest.raises(ValueError, match='not a checkpoint'):
        server.resolve_checkpoint(str(tmp_path), 5)
    (tmp_path / gr00t.CONFIG_FILENAME).unlink()
    (tmp_path / 'checkpoint-5').mkdir()
    with pytest.raises(ValueError, match='Checkpoint not found'):
        server.resolve_checkpoint(str(tmp_path), 6)


def test_s3_selection_downloads_only_the_selected_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server.pos3, 'ls', lambda source, recursive: [f'{source}/checkpoint-005000/'])
    download = Mock(return_value=tmp_path)
    monkeypatch.setattr(server.pos3, 'download', download)
    path, metadata = server.resolve_checkpoint('s3://bucket/experiment', 5000)
    assert path == tmp_path
    assert metadata[policy_keys.CHECKPOINT_PATH] == 's3://bucket/experiment/checkpoint-005000'
    download.assert_called_once_with('s3://bucket/experiment/checkpoint-005000', exclude=[gr00t.OPTIMIZER_FILENAME])


def test_standalone_checkpoint_metadata_contract():
    assert model_settings.MODEL_SETTINGS_PATH == MODEL_SETTINGS_PATH
    for name in ('CHECKPOINT_ID', 'CHECKPOINT_PATH', 'EXPERIMENT_NAME', 'TYPE', 'JPEG_QUALITY', 'COMPRESS_IMAGES'):
        assert getattr(keys, name) == getattr(policy_keys, name)


@pytest.mark.parametrize(
    'field',
    [
        model_settings.IMAGE_SIZE,
        model_settings.EE_FRAMES,
        model_settings.ROTATION_OFFSET,
        model_settings.ACTION_FPS,
        model_settings.IMAGE_MAPPINGS,
    ],
)
def test_conflicting_checkpoint_settings_fail_before_native_loading(tmp_path, field):
    settings = model_settings.droid()
    recorded = model_settings.droid_three_cameras() if field == model_settings.IMAGE_MAPPINGS else settings.copy()
    if field != model_settings.IMAGE_MAPPINGS:
        recorded[field] = None
    destination = tmp_path / MODEL_SETTINGS_PATH
    destination.parent.mkdir()
    destination.write_text(json.dumps(recorded))
    (tmp_path / gr00t.CONFIG_FILENAME).write_text('{}')
    with pytest.raises(ValueError, match=field):
        server.load_model(str(tmp_path), settings)


def test_a_checkpoint_that_records_its_frame_as_one_vector_is_refused(tmp_path):
    settings = model_settings.droid()
    recorded = {key: value for key, value in settings.items() if key != model_settings.EE_FRAMES}
    # rules-allow: hardcoded-keys — the key a checkpoint recorded before ``ee_frames``, which no code writes now
    recorded['ee_frame'] = [0.0, 0.0, -0.1033999994, 0.9238795325107985, 0.0, 0.0, 0.3826834323662685]
    destination = tmp_path / MODEL_SETTINGS_PATH
    destination.parent.mkdir()
    destination.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match=model_settings.EE_FRAMES):
        model_settings.validate(settings, tmp_path)


def test_checkpoint_without_settings_and_deployment_only_differences(tmp_path):
    settings = model_settings.droid()
    model_settings.validate(settings, tmp_path)
    destination = tmp_path / MODEL_SETTINGS_PATH
    destination.parent.mkdir()
    destination.write_text(json.dumps(settings))
    settings[model_settings.IMAGE_MAPPINGS][gr00t.WRIST_IMAGE] = 'another-camera'
    settings[model_settings.OBSERVATION_KEYS][gr00t.TASK] = 'another-instruction'
    settings[model_settings.CONTROL_MODE] = {'type': 'joint_position'}
    model_settings.validate(settings, tmp_path)


@pytest.fixture
def native_policy():
    settings = model_settings.droid()
    modalities = {
        gr00t.VIDEO: SimpleNamespace(modality_keys=list(settings[model_settings.IMAGE_MAPPINGS]), delta_indices=[0]),
        gr00t.STATE: SimpleNamespace(modality_keys=list(gr00t.STATE_DIMS), delta_indices=[0]),
        gr00t.LANGUAGE: SimpleNamespace(modality_keys=[gr00t.TASK], delta_indices=[0]),
        gr00t.ACTION: SimpleNamespace(modality_keys=list(gr00t.STATE_DIMS), delta_indices=list(range(40))),
    }
    policy = Mock()
    policy.get_modality_config.return_value = modalities
    policy.get_action.return_value = (
        {name: np.zeros((1, 40, size), dtype=np.float32) for name, size in gr00t.STATE_DIMS.items()},
        {},
    )
    return policy


@pytest.mark.parametrize(
    ('modality', 'attribute', 'value'),
    [
        (gr00t.VIDEO, 'modality_keys', [gr00t.EXTERIOR_IMAGE, gr00t.EXTERIOR_IMAGE_2, gr00t.WRIST_IMAGE]),
        (gr00t.STATE, 'delta_indices', [-1, 0]),
        (gr00t.LANGUAGE, 'modality_keys', []),
        (gr00t.ACTION, 'delta_indices', [0, 2]),
        (gr00t.ACTION, 'modality_keys', [gr00t.EE_POSE, gr00t.GRIP]),
    ],
)
def test_incompatible_checkpoint_layout_fails_before_warmup(native_policy, modality, attribute, value):
    setattr(native_policy.get_modality_config.return_value[modality], attribute, value)
    with pytest.raises(ValueError, match='Checkpoint'):
        server.Gr00tModel(native_policy, model_settings.droid(), {})
    native_policy.get_action.assert_not_called()


def test_invalid_native_action_shape_fails_startup(native_policy):
    native_policy.get_action.return_value[0][gr00t.JOINT_POSITION] = np.zeros((1, 40, 6))
    with pytest.raises(ValueError, match='shape'):
        server.Gr00tModel(native_policy, model_settings.droid(), {})


def test_standalone_server_imports_without_positronic_or_native_model():
    script = """
import importlib.abc
import sys

class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'positronic', 'pimm', 'gr00t', 'torch'}:
            raise AssertionError(f'Standalone adapter imports {fullname}')

sys.meta_path.insert(0, ForbiddenImports())
sys.path.insert(0, sys.argv[1])
from serving import server
sys.argv = ['gr00t_serving.server', 'droid', '--help']
server.main()
"""
    subprocess.run([sys.executable, '-I', '-c', script, str(Path(server.__file__).parents[1])], check=True)


@pytest.fixture
def model(native_policy):
    return server.Gr00tModel(native_policy, model_settings.droid(), {})


def test_sessions_keep_their_own_parameters_and_share_native_weights(model):
    first = model.prepare_session(resolve_params(model.parameters, {keys.FPS: 10}))
    second = model.prepare_session(resolve_params(model.parameters, {keys.FPS: 20}))
    first_description = json.dumps(first.client_stack)
    assert first.client_stack != second.client_stack
    observation = {gr00t.LANGUAGE: {gr00t.TASK: [['pick']]}}
    result = first.infer(observation)
    assert result is model._policy.get_action.return_value
    model._policy.get_action.assert_called_with(observation)
    assert json.dumps(first.client_stack) == first_description
    assert model.parameters[keys.FPS] == 15
    model._policy.reset.assert_called_once()


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        (keys.FPS, 0),
        (keys.FPS, True),
        (keys.FPS, float('inf')),
        (keys.HORIZON_SEC, -1),
        (keys.JPEG_QUALITY, 101),
        (keys.JPEG_QUALITY, False),
        (keys.COMPRESS_IMAGES, 'yes'),
        (model_settings.IMAGE_MAPPINGS, {}),
    ],
)
def test_invalid_session_controls_fail_before_ready(model, name, value):
    with pytest.raises(ValueError):
        model.prepare_session({**model.parameters, name: value})


@pytest.fixture
def rtc_settings():
    return {keys.CALL_AFTER_SEC: 0.5, keys.BLEND_SEC: 0.5, keys.MAX_DELAY_SEC: 1.0, keys.RAMP_RATE: 5.0}


def test_rtc_reuses_native_pose_and_overrides_scheduled_joint_and_grip_targets(native_policy, rtc_settings):
    session = server.Gr00tSession(native_policy, {}, rtc_settings)
    previous = native_policy.get_action.return_value[0]
    for values in previous.values():
        values[:] = np.arange(40)[None, :, None]
    session.infer(ActionPrefix(0, [], 0, []).to_observation())
    targets = [
        {gr00t.JOINT_POSITION: np.full(7, 0.25), gr00t.GRIP: np.array([1.0])},
        {gr00t.JOINT_POSITION: np.full(7, 0.5), gr00t.GRIP: np.array([0.0])},
    ]
    session.infer(ActionPrefix(1, [5, 6], 1, targets).to_observation())
    options = native_policy.get_action.call_args.kwargs['options']
    assert options[keys.RTC_FROZEN_STEPS] == 1
    assert options[keys.RTC_OVERLAP_STEPS] == options[keys.ACTION_HORIZON] == 2
    assert options[keys.RTC_RAMP_RATE] == 5.0
    actions = options[gr00t.ACTION]
    np.testing.assert_array_equal(actions[gr00t.EE_POSE][:2, 0], [5, 6])
    np.testing.assert_array_equal(actions[gr00t.JOINT_POSITION][:2, 0], [0.25, 0.5])
    np.testing.assert_array_equal(actions[gr00t.GRIP][:2, 0], [1, 0])
    assert all(values.shape == (40, gr00t.STATE_DIMS[name]) for name, values in actions.items())
    np.testing.assert_array_equal(previous[gr00t.JOINT_POSITION][0, :2, 0], [0, 1])


def test_rtc_cache_is_session_local_and_copies_model_buffers(native_policy, rtc_settings):
    first = server.Gr00tSession(native_policy, {}, rtc_settings)
    second = server.Gr00tSession(native_policy, {}, rtc_settings)
    first.infer(ActionPrefix(0, [], 0, []).to_observation())
    native_policy.get_action.return_value[0][gr00t.EE_POSE][:] = 99
    targets = [{gr00t.JOINT_POSITION: np.zeros(7), gr00t.GRIP: np.zeros(1)}]
    request = ActionPrefix(1, [0], 0, targets).to_observation()
    with pytest.raises(ValueError, match='sequence'):
        second.infer(request)
    first.infer(request)
    assert not native_policy.get_action.call_args.kwargs['options'][gr00t.ACTION][gr00t.EE_POSE].any()
    first.close()
    assert first._previous == {}


@pytest.mark.parametrize('positions', [[0.5], [40], [1] * 41])
def test_rtc_rejects_unavailable_source_positions(native_policy, rtc_settings, positions):
    session = server.Gr00tSession(native_policy, {}, rtc_settings)
    session.infer(ActionPrefix(0, [], 0, []).to_observation())
    with pytest.raises(ValueError, match='positions'):
        session.infer(ActionPrefix(1, positions, 0, [{} for _ in positions]).to_observation())


def test_rtc_empty_overlap_uses_an_unconditioned_prediction(native_policy, rtc_settings):
    session = server.Gr00tSession(native_policy, {}, rtc_settings)
    for sequence in range(2):
        session.infer(ActionPrefix(sequence, [], 0, []).to_observation())
        assert native_policy.get_action.call_args.kwargs['options'] is None


def test_ordinary_session_refuses_rtc_instead_of_ignoring_it(native_policy):
    session = server.Gr00tSession(native_policy, {})
    with pytest.raises(ValueError, match='does not enable RTC'):
        session.infer(ActionPrefix(0, [], 0, []).to_observation())
    native_policy.get_action.assert_not_called()


def test_rtc_launch_warms_native_overlap_before_sessions_are_ready(native_policy, rtc_settings):
    model = server.Gr00tModel(native_policy, model_settings.droid(), {}, rtc=rtc_settings, horizon_sec=None)
    assert any(call.kwargs.get('options') for call in native_policy.get_action.call_args_list)
    session = model.prepare_session(resolve_params(model.parameters, {}))
    assert session._sequence == 0
    assert session._previous == {}


def test_ordinary_launch_refuses_enabling_an_unwarmed_rtc_path(model, rtc_settings):
    with pytest.raises(ValueError, match='warmed RTC'):
        model.prepare_session({**model.parameters, keys.RTC: rtc_settings})


def test_rtc_rejects_conflicting_execution_horizon(native_policy, rtc_settings):
    with pytest.raises(ValueError, match='horizon_sec=None'):
        server.Gr00tModel(native_policy, model_settings.droid(), {}, rtc=rtc_settings, horizon_sec=1.0)
