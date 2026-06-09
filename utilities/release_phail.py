"""Release PhAIL datasets to public S3.

TODO: this script (and the phail-specific configs in `positronic.cfg.analysis`
it depends on — `phail_inference_release`, `phail_inference_prod_v1_0`,
`phail_teleop_release`, `phail_human_release`, `PROD_VARIANTS`,
`TRAINED_OBJECTS`, audit-corrected episode comments) belongs in phail-website.
positronic should remain a generic robotics library; phail-release ops belong
on the phail side.

Bakes source-of-truth fields and robot metadata into the output.
Display fields (model display name, UPH, completion, started) remain as
server-side transforms.

Usage:
    uv run --locked python utilities/release_phail.py all
    uv run --locked python utilities/release_phail.py training
    uv run --locked python utilities/release_phail.py inference
    uv run --locked python utilities/release_phail.py human
    uv run --locked python utilities/release_phail.py models
    uv run --locked python utilities/release_phail.py verify
"""

import dataclasses
import logging
from collections import defaultdict

import configuronic as cfn
import numpy as np
import pos3

from positronic.cfg import analysis as analysis_cfg
from positronic.dataset.dataset import Dataset
from positronic.dataset.utilities.migrate_remote import migrate_dataset
from positronic.utils import PUBLIC
from positronic.utils.checkpoints import get_latest_checkpoint
from positronic.utils.logging import init_logging

DEST_ROOT = 's3://positronic-public/phail'

# Same cache path as PUBLIC (local_name='positronic-public') but with credentials for writing.
PUBLIC_WRITE = dataclasses.replace(PUBLIC, public=False)

REQUIRED_FIELDS = ['model', 'eval.object', 'eval.successful_items', 'eval.total_items']


@dataclasses.dataclass(frozen=True)
class _ModelLayout:
    """Per-vendor checkpoint layout under `checkpoints_dir`.

    The vendor server resolves checkpoints as
    `<checkpoints_dir>/<inner_dir>/<prefix><N>/`. We must preserve that
    nested structure when copying so a server pointed at the dest finds it.
    """

    source: str
    inner_dir: str = ''
    prefix: str = ''


MODEL_SOURCES: dict[str, _ModelLayout] = {
    'openpi': _ModelLayout(source='s3://checkpoints/phail_unified/openpi/pi05_positronic_lowmem/270226-ee/'),
    'gr00t': _ModelLayout(source='s3://checkpoints/phail_unified/groot/270226-ee_rot6d_rel/', prefix='checkpoint-'),
    'smolvla': _ModelLayout(source='s3://checkpoints/phail_unified/smolvla/170316_ee/', inner_dir='checkpoints'),
    'act': _ModelLayout(source='s3://checkpoints/phail_unified/lerobot/270226-ee/', inner_dir='checkpoints'),
}


def _copy_latest_checkpoint(layout: _ModelLayout, dest: str):
    """Copy only the latest checkpoint subdir from `layout.source` to `dest`.

    Preserves the inner directory structure expected by the vendor server.
    """
    src_root = layout.source.rstrip('/')
    dest_root = dest.rstrip('/')
    src_checkpoints_dir = f'{src_root}/{layout.inner_dir}' if layout.inner_dir else src_root
    dest_checkpoints_dir = f'{dest_root}/{layout.inner_dir}' if layout.inner_dir else dest_root

    latest = get_latest_checkpoint(src_checkpoints_dir, prefix=layout.prefix)
    src_path = f'{src_checkpoints_dir}/{latest}'
    dest_path = f'{dest_checkpoints_dir}/{latest}'

    logging.info(f'Copying latest checkpoint {latest!r} from {src_path} to {dest_path}')
    local = pos3.download(src_path)
    pos3.upload(dest_path, local=local, sync_on_error=False, interval=None, profile=PUBLIC_WRITE)


def _check_dest_empty(dest: str, profile=None):
    """Raise if destination already contains data."""
    existing = pos3.ls(dest, profile=profile)
    if existing:
        raise SystemExit(
            f'Destination {dest} already contains {len(existing)} entries. Delete first or use a new path.'
        )


@cfn.config(dataset=analysis_cfg.phail_inference_prod_v1_0, force=False)
def verify_inference(dataset: Dataset, force: bool):
    """Verify inference episodes for consistency. Raises SystemExit on failure unless force=False."""
    issues = []

    uph_by_model: dict[str, list[float]] = defaultdict(list)
    for i, ep in enumerate(dataset):
        model = ep.get('model', '')
        items = ep.get('eval.successful_items', 0)
        duration = ep.get('eval.duration') or ep.duration_ns / 1e9
        if items and duration > 0:
            uph_by_model[model].append(items / (duration / 3600))

        for field in REQUIRED_FIELDS:
            if field not in ep:
                issues.append(f'  [{i}] {model}: missing required field {field!r}')

        outcome = ep.get('eval.outcome', '')
        successful = ep.get('eval.successful_items', 0)
        total = ep.get('eval.total_items', 0)
        if outcome == 'Success' and successful != total:
            issues.append(f'  [{i}] {model}: Success but {successful}/{total}')
        if total == 0:
            issues.append(f'  [{i}] {model}: total_items is 0')

    for model, uphs in uph_by_model.items():
        if len(uphs) < 3:
            continue
        arr = np.array(uphs)
        mean, std = arr.mean(), arr.std()
        if std == 0:
            continue
        for i, ep in enumerate(dataset):
            if ep.get('model', '') != model:
                continue
            ep_items = ep.get('eval.successful_items', 0)
            ep_duration = ep.get('eval.duration') or ep.duration_ns / 1e9
            if ep_items and ep_duration > 0:
                uph = ep_items / (ep_duration / 3600)
                if abs(uph - mean) > 3 * std:
                    issues.append(f'  [{i}] {model}: UPH={uph:.1f} is >3σ from mean={mean:.1f}±{std:.1f}')

    if issues:
        logging.warning(f'{len(issues)} issues found:')
        for issue in issues:
            logging.warning(issue)
        if force:
            raise SystemExit('Verification failed. Run without --force to proceed with warnings only.')


@cfn.config(dataset=analysis_cfg.phail_teleop_release, dest=f'{DEST_ROOT}/v1.0/dataset/teleoperation/')
def training(dataset: Dataset, dest: str):
    """Export fine-tuning dataset (DROID teleoperation with baked eval fields)."""
    _check_dest_empty(dest, profile=PUBLIC_WRITE)
    migrate_dataset(dataset, dest, profile=PUBLIC_WRITE)


@cfn.config(dataset=analysis_cfg.phail_inference_prod_v1_0, dest=f'{DEST_ROOT}/v1.0/dataset/rollouts/')
def inference(dataset: Dataset, dest: str):
    """Export prod-filtered inference runs."""
    _check_dest_empty(dest, profile=PUBLIC_WRITE)
    verify_inference(dataset=dataset, force=False)
    migrate_dataset(dataset, dest, profile=PUBLIC_WRITE)


@cfn.config(dataset=analysis_cfg.phail_human_release, dest=f'{DEST_ROOT}/v1.0/dataset/human/')
def human(dataset: Dataset, dest: str):
    """Export human baseline episodes."""
    _check_dest_empty(dest, profile=PUBLIC_WRITE)
    migrate_dataset(dataset, dest, profile=PUBLIC_WRITE)


@cfn.config(dest_root=DEST_ROOT, version='v1.0')
def models(dest_root: str, version: str):
    """Copy latest checkpoint per vendor to the public release layout.

    Preserves each vendor's expected inner structure so a server pointed at
    `s3://.../models/<vendor>/` resolves the checkpoint via its normal
    `get_latest_checkpoint` call.
    """
    for vendor, layout in MODEL_SOURCES.items():
        dest = f'{dest_root}/{version}/models/{vendor}/'
        _check_dest_empty(dest, profile=PUBLIC_WRITE)
        _copy_latest_checkpoint(layout, dest)


@cfn.config(
    training_ds=analysis_cfg.phail_teleop_release,
    inference_ds=analysis_cfg.phail_inference_prod_v1_0,
    human_ds=analysis_cfg.phail_human_release,
    dest_root=DEST_ROOT,
    version='v1.0',
    force=False,
)
def release_all(
    training_ds: Dataset, inference_ds: Dataset, human_ds: Dataset, dest_root: str, version: str, force: bool
):
    """Full release: training + inference + human + models."""
    dest = f'{dest_root}/{version}'
    _check_dest_empty(f'{dest}/dataset/teleoperation/', profile=PUBLIC_WRITE)
    _check_dest_empty(f'{dest}/dataset/rollouts/', profile=PUBLIC_WRITE)
    _check_dest_empty(f'{dest}/dataset/human/', profile=PUBLIC_WRITE)
    for vendor in MODEL_SOURCES:
        _check_dest_empty(f'{dest}/models/{vendor}/', profile=PUBLIC_WRITE)
    verify_inference(dataset=inference_ds, force=force)
    migrate_dataset(training_ds, f'{dest}/dataset/teleoperation/', profile=PUBLIC_WRITE)
    migrate_dataset(inference_ds, f'{dest}/dataset/rollouts/', profile=PUBLIC_WRITE)
    migrate_dataset(human_ds, f'{dest}/dataset/human/', profile=PUBLIC_WRITE)
    for vendor, layout in MODEL_SOURCES.items():
        _copy_latest_checkpoint(layout, f'{dest}/models/{vendor}/')


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli({
            'all': release_all,
            'training': training,
            'inference': inference,
            'human': human,
            'models': models,
            'verify': verify_inference,
        })
