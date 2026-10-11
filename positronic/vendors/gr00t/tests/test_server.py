import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
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
        model_settings.EE_FRAME,
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
