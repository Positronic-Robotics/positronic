import hashlib
import json
from pathlib import Path

import pytest

from positronic.vendors.droid_finetune import flux3_eval

ROOT_SETTINGS = {
    'sampler': 'cosmos_unipc',
    'num_inference_steps': 4,
    'sampler_shift': 5.0,
    'guidance_scale': 4.0,
    'guidance_scale_action': 1.0,
}


def _package(path: Path, config: dict, name: str = 'config.json') -> Path:
    path.mkdir(parents=True)
    (path / name).write_text(json.dumps(config))
    (path / flux3_eval.PACKAGE_WEIGHTS).write_bytes(b'')
    return path


def test_every_evaluation_samples_with_the_start_package_settings_and_the_one_seed(tmp_path):
    package = _package(tmp_path / 'droid', {**ROOT_SETTINGS, 'action_dim': 8})

    assert flux3_eval.sampling_settings(package, seed=3) == [
        'sampler="cosmos_unipc"',
        'num_inference_steps=4',
        'sampler_shift=5.0',
        'guidance_scale=4.0',
        'guidance_scale_action=1.0',
        'inference_seed=3',
    ]


def test_the_settings_come_from_the_config_that_from_pretrained_reads(tmp_path):
    package = _package(tmp_path / 'droid', {**ROOT_SETTINGS, 'guidance_scale': 1.0})
    (package / 'config.native.json').write_text(json.dumps(ROOT_SETTINGS))

    assert 'guidance_scale=4.0' in flux3_eval.sampling_settings(package, seed=0)


def test_evaluate_scores_every_window_of_the_tree_at_the_training_frame_size(tmp_path):
    command = flux3_eval.evaluate_command(
        Path('pkg'), Path('src'), Path('idx'), ['guidance_scale=4.0'], per_episode=8, seed=0, output=Path('out.json')
    )

    assert {'--split=all', '--frame-hw=360,640', '--windows-per-episode=8', '--setting=guidance_scale=4.0'} <= set(
        command
    )


def test_a_package_is_scored_as_it_is(tmp_path):
    package = _package(tmp_path / 'droid', ROOT_SETTINGS)

    assert flux3_eval.policy_package(package, tmp_path / 'exports', 'ema_0p10') == package


def test_an_incomplete_training_checkpoint_is_refused(tmp_path):
    (tmp_path / 'step-1000').mkdir()

    with pytest.raises(ValueError, match='complete training checkpoint'):
        flux3_eval.policy_package(tmp_path / 'step-1000', tmp_path / 'exports', 'ema_0p10')


def test_a_training_checkpoint_is_exported_once_per_profile(tmp_path, monkeypatch):
    step = tmp_path / 'step-1000'
    step.mkdir()
    (step / flux3_eval.CHECKPOINT_COMPLETE).write_text('')
    commands = []

    def export(command, check):
        commands.append(command)
        Path(command[command.index('export-checkpoint') + 2].removeprefix('--output=')).mkdir(parents=True)

    monkeypatch.setattr(flux3_eval.subprocess, 'run', export)
    first = flux3_eval.policy_package(step, tmp_path / 'exports', 'ema_0p10')
    second = flux3_eval.policy_package(step, tmp_path / 'exports', 'ema_0p10')

    assert first == second == tmp_path / 'exports' / f'step-1000-ema_0p10-{flux3_eval.path_key(step)}'
    assert first.is_dir() and len(commands) == 1
    assert '--dtype=bfloat16' in commands[0] and '--profile=ema_0p10' in commands[0]


def test_two_runs_with_the_same_step_name_get_separate_exports(tmp_path):
    assert flux3_eval.path_key(tmp_path / 'run-a' / 'step-1000') != flux3_eval.path_key(
        tmp_path / 'run-b' / 'step-1000'
    )


def test_the_report_carries_the_raw_action_error_of_flux_action_evaluate():
    flux_report = {
        'action_mse_raw': 0.2,
        'action_mse_raw_per_channel': [0.1] * 8,
        'windows': [
            {'episode_index': 0, 'start': 0, 'action_mse_raw': 0.1, 'action_mse_normalized': 9.0},
            {'episode_index': 1, 'start': 4, 'action_mse_raw': 0.3, 'action_mse_normalized': 9.0},
        ],
    }

    report = flux3_eval.report_from_flux(
        flux_report, checkpoint=Path('ckpt'), heldout_tree=Path('tree'), seed=0, per_episode=8, seconds=1.0
    )

    assert report.summary.action_mse == pytest.approx(0.2)
    assert [(w.episode_index, w.start, w.action_mse) for w in report.windows] == [(0, 0, 0.1), (1, 4, 0.3)]
    assert report.summary.framework == 'flux3' and report.summary.episodes == 2


def test_an_offline_package_reads_the_local_base_model_and_passes_its_manifest_check(tmp_path):
    hub = {'video_vae_id': 'org/base:video_vae.safetensors@rev', 'text_encoder_id': 'org/base:text_encoder@rev'}
    package = _package(tmp_path / 'droid', {**ROOT_SETTINGS, **hub}, name='config.native.json')
    (package / 'manifest.json').write_text(json.dumps({'sha256': {'config.native.json': 'old'}}))
    base = tmp_path / 'base'

    copy = flux3_eval.offline_package(package, tmp_path / 'offline', base)

    config = json.loads((copy / 'config.native.json').read_text())
    assert config['video_vae_id'] == str(base / 'video_vae.safetensors')
    assert config['text_encoder_id'] == str(base / 'text_encoder')
    digest = json.loads((copy / 'manifest.json').read_text())['sha256']['config.native.json']
    assert digest == hashlib.sha256((copy / 'config.native.json').read_bytes()).hexdigest()
    assert (copy / flux3_eval.PACKAGE_WEIGHTS).resolve() == (package / flux3_eval.PACKAGE_WEIGHTS).resolve()


def test_a_package_that_already_reads_the_local_base_model_is_scored_as_it_is(tmp_path):
    base = tmp_path / 'base'
    local = {'video_vae_id': str(base / 'video_vae.safetensors'), 'text_encoder_id': str(base / 'text_encoder')}
    package = _package(tmp_path / 'step-50', {**ROOT_SETTINGS, **local})

    assert flux3_eval.offline_package(package, tmp_path / 'offline', base) == package
