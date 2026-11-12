"""Score FLUX 3 Action weights on a held-out Cosmos3-DROID tree, through `flux-action evaluate`.

Usage (in the train-flux3 image, with the parent of this package on PYTHONPATH)
  python -m droid_finetune.flux3_eval --checkpoint=/opt/flux3/droid --heldout=<dir>/heldout --work-dir=<work> \
      --output=<dir>/start.json
  python -m droid_finetune.flux3_eval --checkpoint=<output_dir>/step-1000 --heldout=<dir>/heldout --work-dir=<work> \
      --output=<dir>/step-1000.json

`--checkpoint` is a policy package, or a complete training checkpoint, which is first exported in bfloat16 with
`--profile`. Every evaluation samples with the settings that `--settings-from` saves and with one noise seed, so the
start weights and each checkpoint see the same inputs and the same sampler. The work directory keeps the index of
the tree and the exports. It prints the summary of the report as one JSON line.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import heldout
from .report import Report, ScoredWindow, build_report, write_report

FLUX_ACTION = [sys.executable, '-m', 'flux_action.cli']
FLUX_ACTION_EAGER = [sys.executable, '-m', 'droid_finetune.flux3_eager']
# The settings that a DROID package saves for its sampler.
SAMPLING_SETTINGS = ('sampler', 'num_inference_steps', 'sampler_shift', 'guidance_scale', 'guidance_scale_action')
# The frame size of training and serving. `flux-action evaluate` defaults to 256x256.
FRAME_HW = '360,640'
CHECKPOINT_COMPLETE = 'COMPLETE'
PACKAGE_WEIGHTS = 'model.safetensors'


def package_config(package: Path) -> dict:
    """The config that `FluxActionPolicy.from_pretrained` reads."""
    native = package / 'config.native.json'
    return json.loads((native if native.is_file() else package / 'config.json').read_text())


def sampling_settings(package: Path, seed: int) -> list[str]:
    config = package_config(package)
    return [f'{key}={json.dumps(config.get(key))}' for key in SAMPLING_SETTINGS] + [f'inference_seed={seed}']


def index_command(source_root: Path, index_dir: Path) -> list[str]:
    return [
        *FLUX_ACTION,
        'index-droid',
        f'--source-root={source_root}',
        f'--filter={source_root / heldout.KEEP_RANGES_FILE}',
        f'--output-dir={index_dir}',
    ]


def export_command(checkpoint: Path, output: Path, profile: str) -> list[str]:
    return [
        *FLUX_ACTION,
        'export-checkpoint',
        f'--checkpoint={checkpoint}',
        f'--output={output}',
        f'--profile={profile}',
        '--dtype=bfloat16',
    ]


def evaluate_command(
    package: Path, source_root: Path, index_dir: Path, settings: list[str], per_episode: int, seed: int, output: Path
) -> list[str]:
    return [
        *FLUX_ACTION_EAGER,
        'evaluate',
        f'--checkpoint={package}',
        f'--source-root={source_root}',
        f'--index-dir={index_dir}',
        '--split=all',
        f'--windows-per-episode={per_episode}',
        f'--seed={seed}',
        f'--frame-hw={FRAME_HW}',
        *(f'--setting={setting}' for setting in settings),
        f'--output={output}',
    ]


def path_key(path: Path) -> str:
    """A short name for `path`, so that a work directory keeps one index per tree and one export per checkpoint."""
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]


def policy_package(checkpoint: Path, exports: Path, profile: str) -> Path:
    """`checkpoint` itself when it is a package, else the export of the training checkpoint, made once."""
    if (checkpoint / PACKAGE_WEIGHTS).is_file():
        return checkpoint
    if not (checkpoint / CHECKPOINT_COMPLETE).is_file():
        raise ValueError(f'{checkpoint} is neither a policy package nor a complete training checkpoint')
    package = exports / f'{checkpoint.name}-{profile}-{path_key(checkpoint)}'
    if not package.is_dir():
        partial = package.with_name(package.name + '.partial')
        shutil.rmtree(partial, ignore_errors=True)
        subprocess.run(export_command(checkpoint, partial, profile), check=True)
        partial.rename(package)
    return package


def offline_package(package: Path, packages: Path, base: Path) -> Path:
    """`package` with its video VAE and text encoder read from `base`, a local copy of the base model.

    A released package names them by Hugging Face id, and the base repository is gated. The copy links the weights,
    rewrites the config, and records the new config hash in its manifest, which `from_pretrained` checks.
    """
    config_name = 'config.native.json' if (package / 'config.native.json').is_file() else 'config.json'
    config = json.loads((package / config_name).read_text())
    local = {'video_vae_id': str(base / 'video_vae.safetensors'), 'text_encoder_id': str(base / 'text_encoder')}
    if all(config.get(key) == value for key, value in local.items()):
        return package
    copy = packages / f'{package.name}-{path_key(package)}'
    if copy.is_dir():
        return copy
    partial = copy.with_name(copy.name + '.partial')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for item in package.iterdir():
        if item.name not in (config_name, 'manifest.json'):
            (partial / item.name).symlink_to(item.resolve())
    (partial / config_name).write_text(json.dumps({**config, **local}, indent=1) + '\n')
    manifest = json.loads((package / 'manifest.json').read_text())
    manifest['sha256'][config_name] = hashlib.sha256((partial / config_name).read_bytes()).hexdigest()
    (partial / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    partial.rename(copy)
    return copy


def report_from_flux(
    flux_report: dict, *, checkpoint: Path, heldout_tree: Path, seed: int, per_episode: int, seconds: float
) -> Report:
    windows = [ScoredWindow(w['episode_index'], w['start'], w['action_mse_raw']) for w in flux_report['windows']]
    return build_report(
        windows,
        flux_report['action_mse_raw_per_channel'],
        framework='flux3',
        checkpoint=str(checkpoint),
        heldout=str(heldout_tree),
        seed=seed,
        windows_per_episode=per_episode,
        seconds=seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--heldout', type=Path, required=True, help=f'the tree that holds {heldout.DATASET_DIR}/')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings-from', type=Path, default=Path('/opt/flux3/droid'), help='a policy package')
    parser.add_argument('--profile', default='ema_0p10', help='model, ema_0p10 or ema_0p05')
    parser.add_argument('--windows-per-episode', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base', type=Path, default=Path('/opt/flux3/base'), help='a local copy of the base model')
    args = parser.parse_args()

    started = time.perf_counter()
    source_root = args.heldout / heldout.DATASET_DIR
    index_dir = args.work_dir / 'index' / path_key(source_root)
    if not (index_dir / 'manifest.json').is_file():
        subprocess.run(index_command(source_root, index_dir), check=True)
    package = policy_package(args.checkpoint, args.work_dir / 'exports', args.profile)
    package = offline_package(package, args.work_dir / 'offline', args.base)
    flux_output = args.work_dir / 'flux_reports' / args.output.name
    settings = sampling_settings(args.settings_from, args.seed)
    subprocess.run(
        evaluate_command(package, source_root, index_dir, settings, args.windows_per_episode, args.seed, flux_output),
        check=True,
    )
    report = report_from_flux(
        json.loads(flux_output.read_text()),
        checkpoint=args.checkpoint,
        heldout_tree=args.heldout,
        seed=args.seed,
        per_episode=args.windows_per_episode,
        seconds=time.perf_counter() - started,
    )
    write_report(report, args.output)


if __name__ == '__main__':
    main()
