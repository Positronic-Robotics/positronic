"""Release PhAIL datasets to public S3.

Bakes source-of-truth fields and robot metadata into the output.
Display fields (model display name, UPH, completion, started) remain as
server-side transforms.

Usage:
    uv run python utilities/release_phail.py all
    uv run python utilities/release_phail.py training
    uv run python utilities/release_phail.py inference
    uv run python utilities/release_phail.py human
    uv run python utilities/release_phail.py verify
"""

import logging
from collections import defaultdict

import configuronic as cfn
import numpy as np
import pos3

from positronic.cfg import eval as eval_cfg
from positronic.dataset.dataset import Dataset
from positronic.dataset.utilities.migrate_remote import migrate_dataset
from positronic.utils.logging import init_logging

DEST_ROOT = 's3://positronic-public/datasets/phail'

REQUIRED_FIELDS = ['model', 'eval.object', 'eval.successful_items', 'eval.total_items']


@cfn.config(dataset=eval_cfg.phail_inference_prod, force=True)
def verify_inference(dataset: Dataset, force: bool):
    """Verify inference episodes for consistency. Raises SystemExit on failure unless force=True."""
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
                    issues.append(f'  [{i}] {model}: UPH={uph:.1f} is >3\u03c3 from mean={mean:.1f}\u00b1{std:.1f}')

    if issues:
        logging.warning(f'{len(issues)} issues found:')
        for issue in issues:
            logging.warning(issue)
        if not force:
            raise SystemExit('Verification failed. Use --force to proceed anyway.')


@cfn.config(dataset=eval_cfg.phail_teleop_release, dest=f'{DEST_ROOT}/v1.0/training/')
def training(dataset: Dataset, dest: str):
    """Export fine-tuning dataset (DROID teleoperation with baked eval fields)."""
    migrate_dataset(dataset, dest)


@cfn.config(dataset=eval_cfg.phail_inference_prod, dest=f'{DEST_ROOT}/v1.0/inference/', force=False)
def inference(dataset: Dataset, dest: str, force: bool):
    """Export prod-filtered inference runs."""
    verify_inference(dataset, force)
    migrate_dataset(dataset, dest)


@cfn.config(dataset=eval_cfg.phail_human_release, dest=f'{DEST_ROOT}/v1.0/human/')
def human(dataset: Dataset, dest: str):
    """Export human baseline episodes."""
    migrate_dataset(dataset, dest)


@cfn.config(
    training_ds=eval_cfg.phail_teleop_release,
    inference_ds=eval_cfg.phail_inference_prod,
    human_ds=eval_cfg.phail_human_release,
    dest_root=DEST_ROOT,
    version='v1.0',
    force=False,
)
def release_all(
    training_ds: Dataset, inference_ds: Dataset, human_ds: Dataset, dest_root: str, version: str, force: bool
):
    """Full release: training + inference + human."""
    dest = f'{dest_root}/{version}'
    migrate_dataset(training_ds, f'{dest}/training/')
    verify_inference(dataset=inference_ds, force=force)
    migrate_dataset(inference_ds, f'{dest}/inference/')
    migrate_dataset(human_ds, f'{dest}/human/')


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli({
            'all': release_all,
            'training': training,
            'inference': inference,
            'human': human,
            'verify': verify_inference,
        })
