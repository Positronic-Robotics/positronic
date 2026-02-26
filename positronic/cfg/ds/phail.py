"""Public PhAIL datasets ready for training.

All datasets are hosted at s3://positronic-public/datasets/ and have transforms
baked in during migration (no runtime transforms).

Migration Process (for future reference):
------------------------------------------
To migrate a dataset from internal s3://raw/ to public s3://positronic-public/:

1. Start remote server serving the internal dataset:
   uv run python -m positronic.dataset.remote_server.server \\
     --dataset=@positronic.cfg.ds.internal.<INTERNAL_CONFIG> --port=8080

2. Run migration to local staging directory:
   uv run python -m positronic.dataset.utilities.migrate_remote \\
     --source_url=http://localhost:8080 \\
     --dest_path=~/staging/public-datasets/<DATASET_NAME>/

3. Stop the remote server (Ctrl+C)

4. Upload to S3:
   aws s3 sync ~/staging/public-datasets/<DATASET_NAME>/ \\
     s3://positronic-public/datasets/<DATASET_NAME>/ \\
     --endpoint-url=https://storage.eu-north1.nebius.cloud

See internal configs (cfg/ds/internal.py) for source dataset definitions.
"""

from datetime import datetime

import configuronic as cfn
import pos3

from positronic.dataset import Episode
from positronic.dataset.dataset import ConcatDataset, Dataset, FilterDataset
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import Derive, FromValue, Group, Identity
from positronic.server.positronic_server import ColumnConfig as C
from positronic.server.positronic_server import GroupTableConfig
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging

from . import PUBLIC, group, local, local_all, transform
from .internal import ALL_TASKS, RECOVERY_TASK

# DROID teleoperation data for PhAIL tasks (towels, spoons, scissors)
# Migrated from: @positronic.cfg.ds.internal.droid
# Size: 12GB, 352 episodes with task labels baked in static.json
phail = local_all.override(path='s3://positronic-public/datasets/phail/', profile=PUBLIC)

# Simulated cube stacking dataset
# Migrated from: @positronic.cfg.ds.internal.sim_stack
# Size: 499MB, 317 episodes with transforms baked in (ee_pose, robot_joints, task)
sim_stack_cubes = local.override(path='s3://positronic-public/datasets/sim-stack-cubes/', profile=PUBLIC)

# Simulated pick-and-place dataset
# Migrated from: @positronic.cfg.ds.internal.sim_pnp
# Size: 1.3GB, 214 episodes with transforms baked in
sim_pick_place = local.override(path='s3://positronic-public/datasets/sim-pick-place/', profile=PUBLIC)


@cfn.config()
def _duplicate_recovery(base: Dataset):
    """Duplicate recovery episodes for every real task."""

    def is_recovery(ep):
        return ep['task'] == RECOVERY_TASK

    non_recovery = FilterDataset(base, lambda ep: not is_recovery(ep))
    recovery = FilterDataset(base, is_recovery)
    datasets: list[Dataset] = [non_recovery]
    for task in ALL_TASKS:
        datasets.append(TransformedDataset(recovery, Group(Derive(task=FromValue(task)), Identity())))
    return ConcatDataset(*datasets)


phail_recovery = _duplicate_recovery.override(base=phail)

# Unified single-task variant: all episodes (including recovery) share one generic instruction.
# Recovery episodes appear once (not duplicated per task).
UNIFIED_TASK = 'Pick all the items one by one from transparent tote and place them into the large grey tote.'
phail_unified = transform.override(
    base=phail, transforms=[group.override(transforms=[Derive(task=FromValue(UNIFIED_TASK)), Identity()])]
)


# =============================================================================
# Server configuration for visualizing public PhAIL dataset
# No AWS credentials required - dataset is publicly accessible
# =============================================================================


@cfn.config()
def episodes_table():
    return {
        '__index__': C(label='#', format='%d'),
        '__duration__': C(label='Duration', format='%.0f sec'),
        'task': C(label='Task', filter=True),
        'started': C(label='Started', format='%Y-%m-%d %H:%M'),
    }


@cfn.config()
def group_by_task():
    def group_fn(episodes: list[Episode]):
        duration = sum(ep.duration_ns / 1e9 / 3600 for ep in episodes)
        return {'task': episodes[0]['task'], 'duration': duration, 'count': len(episodes)}

    format_table = {
        'task': C(label='Task'),
        'duration': C(label='Duration', format='%.2f hours'),
        'count': C(label='Count'),
    }

    return GroupTableConfig(group_keys='task', group_fn=group_fn, format_table=format_table)


phail_with_started = transform.override(
    base=phail,
    transforms=[
        group.override(
            transforms=[Identity(), Derive(started=lambda ep: datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9))]
        )
    ],
    extra_meta={'name': 'PhAIL Public Dataset'},
)

server = server_main.override(
    dataset=phail_with_started, ep_table_cfg=episodes_table, group_tables={'tasks': group_by_task}, home_page='tasks'
)


if __name__ == '__main__':
    with pos3.mirror():
        init_logging()
        cfn.cli(server)
