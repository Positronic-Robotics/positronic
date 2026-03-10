"""Server configurations for positronic-server UI."""

from datetime import datetime

import configuronic as cfn
import pos3

from positronic.dataset import Episode
from positronic.dataset.transforms.episode import Derive, FromValue, Group, Identity, Rename
from positronic.server.positronic_server import ColumnConfig as C
from positronic.server.positronic_server import GroupTableConfig
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging

from . import ds
from . import eval as eval_cfg
from .ds import internal
from .eval import calculate_units


def uph(ep: Episode) -> float | None:
    items = ep['units']
    if items == 0:
        return None
    return items / (ep.duration_ns / 1e9 / 3600)


finetune_ds = ds.transform.override(
    base=ds.transform.override(
        base=internal.droid, transforms=[ds.group.override(transforms=[Identity(), Derive(units=calculate_units)])]
    ),
    transforms=[
        ds.group.override(
            transforms=[
                Identity(),
                Derive(started=lambda ep: datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9), uph=uph),
            ]
        )
    ],
    extra_meta={'name': 'PhAIL Finetuning Dataset'},
)


ft_eval_ds = ds.transform.override(
    base=ds.transform.override(
        base=finetune_ds,
        transforms=[
            Group(Identity(remove=['units']), Rename(**{'eval.successful_items': 'units', 'eval.total_items': 'units'}))
        ],
    ),
    transforms=[
        ds.group.override(
            transforms=[
                Identity(),
                Derive(
                    task_code=eval_cfg.task_code,
                    model=FromValue('Teleoperated by Human'),
                    units=eval_cfg.units,
                    uph=eval_cfg.uph,
                    checkpoint=FromValue(''),
                    success=FromValue(100),
                    started=eval_cfg.started,
                ),
            ]
        )
    ],
)


@cfn.config()
def finetune_episodes_table():
    return {
        '__index__': C(label='#', format='%d'),
        '__duration__': C(label='Duration', format='%.0f sec'),
        'task': C(label='Task', filter=True),
        'units': C(label='Units'),
        'uph': C(label='UPH', format='%.1f'),
        'started': C(label='Started', format='%Y-%m-%d %H:%M'),
    }


@cfn.config()
def finetune_group_by_task():
    def group_fn(episodes: list[Episode]):
        duration, units = 0, 0
        for ep in episodes:
            duration += ep.duration_ns / 1e9 / 3600
            units += ep['units']

        result = {'task': episodes[0]['task']}
        result.update({'duration': duration, 'count': len(episodes), 'uph': units / duration})
        return result

    format_table = {
        'task': C(label='Task'),
        'duration': C(label='Duration', format='%.2f hours'),
        'uph': C(label='UPH', format='%.1f'),
        'count': C(label='Count'),
    }

    return GroupTableConfig(group_keys='task', group_fn=group_fn, format_table=format_table)


finetune_server = server_main.override(
    dataset=finetune_ds, ep_table_cfg=finetune_episodes_table, group_tables={'tasks': finetune_group_by_task}
)

if __name__ == '__main__':
    with pos3.mirror():
        init_logging()
        cfn.cli(finetune_server)
