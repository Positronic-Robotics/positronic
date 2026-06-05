"""PhAIL v1.0 release manifest.

Public S3 URLs for the v1.0 dataset release and model checkpoint paths.

`teleop_unified` is the single-task label variant that the released models
were trained on. Re-point this config at a different dataset to evaluate a
trained-on-unified policy against teleop reference data.
"""

import types
from datetime import datetime

import configuronic as cfn
import pos3

from positronic.cfg.ds import PUBLIC, group, local_all, transform
from positronic.dataset import Episode
from positronic.dataset.transforms.episode import Derive, FromValue, Identity
from positronic.server.positronic_server import ColumnConfig as C
from positronic.server.positronic_server import GroupTableConfig
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging

_ROOT = 's3://positronic-public/phail/v1.0'
# Model URLs carry the `PUBLIC@` profile selector so they resolve to anonymous access on
# their own — a no-creds reader can pass them straight to an inference server's
# `--checkpoints_dir`, which never sees the PUBLIC object the dataset configs pass explicitly.
_MODELS_ROOT = f'{_ROOT.replace("s3://", "s3://PUBLIC@")}/models'

ds = types.SimpleNamespace(
    teleop=local_all.override(path=f'{_ROOT}/dataset/teleoperation/', profile=PUBLIC),
    rollouts=local_all.override(path=f'{_ROOT}/dataset/rollouts/', profile=PUBLIC),
    human=local_all.override(path=f'{_ROOT}/dataset/human/', profile=PUBLIC),
)

models = types.SimpleNamespace(
    openpi=f'{_MODELS_ROOT}/openpi/',
    gr00t=f'{_MODELS_ROOT}/gr00t/',
    smolvla=f'{_MODELS_ROOT}/smolvla/',
    act=f'{_MODELS_ROOT}/act/',
)

UNIFIED_TASK = 'Pick all the items one by one from transparent tote and place them into the large grey tote.'
teleop_unified = transform.override(
    base=ds.teleop, transforms=[group.override(transforms=[Derive(task=FromValue(UNIFIED_TASK)), Identity()])]
)


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


# TODO: bad smell. This wrapper exists only to materialize `meta['created_ts_ns']`
# as a signal so the server table can display it. Teach `positronic_server`'s
# column resolver to fall back on `meta.*` keys (so `episodes_table` can reference
# `meta.created_ts_ns` directly), then delete this transform.
phail_with_started = transform.override(
    base=ds.teleop,
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
