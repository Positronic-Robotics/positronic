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
from positronic.dataset.transforms.episode import Derive, Identity
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging

from . import PUBLIC, group, local, local_all, transform

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


# =============================================================================
# Server configuration for visualizing public PhAIL dataset
# No AWS credentials required - dataset is publicly accessible
# =============================================================================


@cfn.config()
def episodes_table():
    return {
        '__index__': {'label': '#', 'format': '%d'},
        '__duration__': {'label': 'Duration', 'format': '%.0f sec'},
        'task': {'label': 'Task', 'filter': True},
        'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M'},
    }


@cfn.config()
def group_by_task():
    def group_fn(episodes: list[Episode]):
        duration = sum(ep.duration_ns / 1e9 / 3600 for ep in episodes)
        return {'task': episodes[0]['task'], 'duration': duration, 'count': len(episodes)}

    format_table = {
        'task': {'label': 'Task'},
        'duration': {'label': 'Duration', 'format': '%.2f hours'},
        'count': {'label': 'Count'},
    }

    return 'task', group_fn, format_table, {}


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
