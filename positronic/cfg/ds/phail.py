"""Public PhAIL datasets for training and evaluation.

Versioned datasets hosted at s3://positronic-public/datasets/phail/<version>/.
All transforms (task labels, robot metadata, eval metrics) are baked in during
migration — no runtime transforms needed.

Release process: uv run python utilities/release_phail.py all
See utilities/release_phail.py for the full migration script.
"""

from datetime import datetime

import configuronic as cfn
import pos3

from positronic.dataset import Episode
from positronic.dataset.transforms.episode import Derive, FromValue, Identity
from positronic.server.positronic_server import ColumnConfig as C
from positronic.server.positronic_server import GroupTableConfig
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging

from . import PUBLIC, group, local, local_all, transform
from .internal import SIM_ROBOT_TRANSFORM

PHAIL_VERSION = 'v1.0'
_PHAIL_ROOT = f's3://positronic-public/datasets/phail/{PHAIL_VERSION}'

# DROID teleoperation data for PhAIL tasks (towels, spoons, scissors, batteries).
# Includes baked eval fields (model, status, item counts) so it doubles as the
# teleoperation baseline in eval_runs without duplicating data on S3.
phail = local_all.override(path=f'{_PHAIL_ROOT}/training/', profile=PUBLIC)

# Simulated cube stacking dataset
# Migrated from: @positronic.cfg.ds.internal.sim_stack
# Size: 499MB, 317 episodes with transforms baked in (ee_pose, robot_joints, task)
sim_stack_cubes = transform.override(
    base=local.override(path='s3://positronic-public/datasets/sim-stack-cubes/', profile=PUBLIC),
    transforms=[SIM_ROBOT_TRANSFORM],
)

# Simulated pick-and-place dataset
# Migrated from: @positronic.cfg.ds.internal.sim_pnp
# Size: 1.3GB, 214 episodes with transforms baked in
sim_pick_place = transform.override(
    base=local.override(path='s3://positronic-public/datasets/sim-pick-place/', profile=PUBLIC),
    transforms=[SIM_ROBOT_TRANSFORM],
)


# Evaluation runs (inference only). Servers concat with phail + human for the full leaderboard.
eval_runs = local_all.override(path=f'{_PHAIL_ROOT}/inference/', profile=PUBLIC)

# Human baseline: 40 episodes (10 per object, 8 items each, all success).
human = local_all.override(path=f'{_PHAIL_ROOT}/human/', profile=PUBLIC)


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
