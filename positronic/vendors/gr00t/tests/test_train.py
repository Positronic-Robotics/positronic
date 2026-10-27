import json
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from positronic import keys
from positronic.cfg.ds import transform
from positronic.dataset import Dataset
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, ee_frames_entry
from positronic.policy.codecs import ACTION, GR00T_MODALITY, GR00T_MODALITY_PATH, LEROBOT_FEATURES, Codec
from positronic.policy.codecs.metadata import MODEL_SETTINGS, MODEL_SETTINGS_PATH
from positronic.policy.keys import ACTION_FPS
from positronic.vendors.gr00t import recipes, train
from positronic.vendors.gr00t import serving as gr00t
from positronic.vendors.gr00t.serving import recipe
from positronic.vendors.gr00t.serving import settings as model_settings


@pytest.fixture
def raw_dataset():
    fields = {
        keys.EE_POSE: [[0, 0, 0, 1, 0, 0, 0]] * 2,
        keys.JOINTS: [np.zeros(7), np.ones(7)],
        keys.GRIP: [0.25, 0.75],
        keys.WRIST_IMAGE: [np.zeros((180, 320, 3), dtype=np.uint8)] * 2,
        keys.EXTERIOR_IMAGE: [np.ones((180, 320, 3), dtype=np.uint8)] * 2,
        keys.EXTERIOR_IMAGE_2: [np.full((180, 320, 3), 73, dtype=np.uint8)] * 2,
    }
    episode = EpisodeContainer(
        {
            **{key: DummySignal([0, 100_000_000], values) for key, values in fields.items()},
            keys.TASK: 'pick',
            **ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)}),
        },
        meta={'uid': 'recording'},
    )
    dataset = MagicMock(spec=Dataset)
    dataset.__len__.return_value = 1
    dataset.__getitem__.return_value = episode
    dataset.meta = {'source': 'recordings'}
    return dataset


@pytest.fixture
def prepared_dataset(raw_dataset):
    return transform(
        base=raw_dataset,
        transforms=[recipes.droid_three_cameras.override(**{f'settings.{model_settings.ACTION_FPS}': 20})],
    )


def test_recipe_preserves_dataset_features_labels_and_recording_metadata(prepared_dataset):
    dataset = prepared_dataset
    assert len(dataset) == 1
    assert dataset.meta['source'] == 'recordings'
    assert dataset.meta[ACTION_FPS] == 20
    features = dataset.meta[LEROBOT_FEATURES]
    assert features[ACTION] == {'shape': (17,), 'dtype': 'float32', 'names': ['actions']}
    for name, size in gr00t.STATE_DIMS.items():
        assert features[name] == {'shape': (size,), 'dtype': 'float32'}
    for name in (gr00t.EXTERIOR_IMAGE, gr00t.EXTERIOR_IMAGE_2, gr00t.WRIST_IMAGE):
        assert features[name] == {'shape': (180, 320, 3), 'dtype': 'video', 'names': ['height', 'width', 'channel']}
    assert dataset.meta[GR00T_MODALITY][ACTION] == {
        gr00t.EE_POSE: {gr00t.START: 0, gr00t.END: 9},
        gr00t.GRIP: {gr00t.START: 9, gr00t.END: 10},
        gr00t.JOINT_POSITION: {gr00t.START: 10, gr00t.END: 17},
    }
    episode = dataset[0]
    assert episode.meta == {'uid': 'recording'}
    assert episode[keys.TASK] == 'pick'
    action = episode[ACTION]
    assert list(action.timestamps(RECORDED_TIME)) == [0, 100_000_000]
    values = np.asarray(action.values())
    assert values.dtype == np.float32
    np.testing.assert_array_equal(values[:, 9], [0.25, 0.75])
    np.testing.assert_array_equal(values[:, 10:], [np.zeros(7), np.ones(7)])


def test_training_builds_and_materializes_without_inference_or_codec_training(monkeypatch, raw_dataset):
    def unexpected(*args):
        raise AssertionError('Training must not construct an inference pipeline or read Codec.training_encoder')

    monkeypatch.setattr(recipe, 'inference', unexpected)
    monkeypatch.setattr(Codec, 'training_encoder', property(unexpected))
    dataset = transform(base=raw_dataset, transforms=[recipes.droid_three_cameras])
    assert np.asarray(dataset[0][ACTION].values()).shape == (2, 17)


@pytest.mark.parametrize('image_name', [*gr00t.STATE_DIMS, ACTION, keys.TASK])
def test_training_rejects_image_names_that_overwrite_other_columns(image_name):
    settings = model_settings.droid()
    settings[model_settings.IMAGE_MAPPINGS] = {image_name: keys.WRIST_IMAGE}
    with pytest.raises(ValueError, match='Image names overlap training columns'):
        recipes.droid(settings=settings)


@pytest.mark.parametrize('resume', [False, True])
@pytest.mark.parametrize('has_settings', [False, True])
@pytest.mark.parametrize(
    'statistics', [{}, {'override_pretraining_statistics': True}, {'override_pretraining_statistics': False}]
)
def test_finetuning_forwards_dataset_cameras_resume_and_statistics_to_gr00t(
    tmp_path, monkeypatch, resume, has_settings, statistics, prepared_dataset
):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    cameras = [gr00t.EXTERIOR_IMAGE, gr00t.EXTERIOR_IMAGE_2, gr00t.WRIST_IMAGE]
    (dataset / GR00T_MODALITY_PATH).write_text(json.dumps(prepared_dataset.meta[GR00T_MODALITY]))
    if has_settings:
        (dataset / MODEL_SETTINGS_PATH).write_text(json.dumps(prepared_dataset.meta[MODEL_SETTINGS]))
    output = tmp_path / 'output'
    output.mkdir()
    if not resume or has_settings:
        (output / MODEL_SETTINGS_PATH).parent.mkdir()
        (output / MODEL_SETTINGS_PATH).write_text(
            json.dumps(prepared_dataset.meta[MODEL_SETTINGS] if resume else {'stale': True})
        )
    override = statistics.get('override_pretraining_statistics', False)
    normalization = {'base_model': gr00t.BASE_MODEL, 'override_pretraining_statistics': override}
    if resume:
        (output / train.NORMALIZATION_PATH).parent.mkdir(exist_ok=True)
        (output / train.NORMALIZATION_PATH).write_text(json.dumps(normalization))
    sync = Mock(return_value=output)
    run = Mock(side_effect=lambda *args, **kwargs: (output / 'checkpoint-2').mkdir())
    monkeypatch.setattr(train.pos3, 'download', lambda _: dataset)
    monkeypatch.setattr(train.pos3, 'sync', sync)
    monkeypatch.setattr(train.utils, 'save_run_metadata', Mock())
    monkeypatch.setattr(train.subprocess, 'run', run)
    train.main(
        input_path=str(dataset),
        output_path=str(output),
        exp_name='droid',
        resume=resume,
        num_train_steps=2,
        **statistics,
    )
    arguments = run.call_args.args[0]
    assert arguments[arguments.index('--base-model-path') + 1] == gr00t.BASE_MODEL
    offset = arguments.index('--video-keys') + 1
    assert arguments[offset : offset + 3] == cameras
    assert ('--resume-from-checkpoint' in arguments) == resume
    assert ('--override-pretraining-statistics' in arguments) == override
    assert ('--no-override-pretraining-statistics' in arguments) == (not override)
    assert sync.call_args.kwargs['delete_remote'] == (not resume)
    assert json.loads((output / train.NORMALIZATION_PATH).read_text()) == normalization
    assert run.call_args.kwargs['check'] is True
    assert (output / MODEL_SETTINGS_PATH).exists() == has_settings
    assert (output / 'checkpoint-2' / MODEL_SETTINGS_PATH).exists() == has_settings
    if has_settings:
        expected_settings = prepared_dataset.meta[MODEL_SETTINGS]
        assert json.loads((output / MODEL_SETTINGS_PATH).read_text()) == expected_settings
        assert json.loads((output / 'checkpoint-2' / MODEL_SETTINGS_PATH).read_text()) == expected_settings


@pytest.mark.parametrize('dataset_settings, experiment_settings', [(True, True), (True, False), (False, True)])
def test_resume_rejects_different_or_missing_settings(
    tmp_path, monkeypatch, prepared_dataset, dataset_settings, experiment_settings
):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / GR00T_MODALITY_PATH).write_text(json.dumps(prepared_dataset.meta[GR00T_MODALITY]))
    if dataset_settings:
        (dataset / MODEL_SETTINGS_PATH).write_text(json.dumps(prepared_dataset.meta[MODEL_SETTINGS]))
    output = tmp_path / 'output'
    (output / 'meta').mkdir(parents=True)
    if experiment_settings:
        (output / MODEL_SETTINGS_PATH).write_text(json.dumps(model_settings.droid()))
    checkpoint = output / 'checkpoint-1'
    checkpoint.mkdir()
    before = {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    run = Mock()
    save_metadata = Mock()
    monkeypatch.setattr(train.pos3, 'download', lambda _: dataset)
    monkeypatch.setattr(train.pos3, 'sync', lambda *args, **kwargs: output)
    monkeypatch.setattr(train.subprocess, 'run', run)
    monkeypatch.setattr(train.utils, 'save_run_metadata', save_metadata)
    with pytest.raises(ValueError, match='resumed training run'):
        train.main(input_path=str(dataset), output_path=str(output), exp_name='droid', resume=True)
    run.assert_not_called()
    save_metadata.assert_not_called()
    assert {path.relative_to(output): path.read_bytes() for path in output.rglob('*') if path.is_file()} == before
    assert list(checkpoint.iterdir()) == []


def _resume(tmp_path, monkeypatch, prepared_dataset, recorded, **kwargs):
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / GR00T_MODALITY_PATH).write_text(json.dumps(prepared_dataset.meta[GR00T_MODALITY]))
    output = tmp_path / 'output'
    (output / 'meta').mkdir(parents=True)
    if recorded is not None:
        (output / train.NORMALIZATION_PATH).write_text(json.dumps(recorded))
    run = Mock()
    monkeypatch.setattr(train.pos3, 'download', lambda _: dataset)
    monkeypatch.setattr(train.pos3, 'sync', lambda *args, **kwargs: output)
    monkeypatch.setattr(train.subprocess, 'run', run)
    monkeypatch.setattr(train.utils, 'save_run_metadata', Mock())
    train.main(input_path=str(dataset), output_path=str(output), exp_name='droid', resume=True, **kwargs)
    return output, run


_BASE = {'base_model': gr00t.BASE_MODEL, 'override_pretraining_statistics': False}


@pytest.mark.parametrize(
    'recorded, requested',
    [
        (None, {}),
        ({**_BASE, 'override_pretraining_statistics': True}, {}),
        (_BASE, {'override_pretraining_statistics': True}),
        (_BASE, {'base_model': 'nvidia/another-checkpoint'}),
    ],
)
def test_resume_rejects_another_normalization(tmp_path, monkeypatch, prepared_dataset, recorded, requested):
    with pytest.raises(ValueError, match='normalized with'):
        _resume(tmp_path, monkeypatch, prepared_dataset, recorded, **requested)
    record = tmp_path / 'output' / train.NORMALIZATION_PATH
    assert (json.loads(record.read_text()) if record.exists() else None) == recorded


def test_resume_of_a_run_with_no_recorded_normalization_keeps_the_dataset_statistics(
    tmp_path, monkeypatch, prepared_dataset
):
    output, run = _resume(tmp_path, monkeypatch, prepared_dataset, None, override_pretraining_statistics=True)
    assert '--override-pretraining-statistics' in run.call_args.args[0]
    assert json.loads((output / train.NORMALIZATION_PATH).read_text())['override_pretraining_statistics'] is True
