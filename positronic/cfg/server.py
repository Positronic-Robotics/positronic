"""Server configurations for positronic-server UI."""

from datetime import datetime
from enum import Enum

import configuronic as cfn
import pos3

from pimm.logging import init_logging
from positronic import keys
from positronic.dataset import Episode
from positronic.dataset.episode import META_CREATED_TS_NS
from positronic.dataset.transforms.episode import Derive, FromValue, Group, Identity, Rename
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.server.positronic_server import ColumnConfig as C
from positronic.server.positronic_server import GroupTableConfig, RendererConfig, SortConfig
from positronic.server.positronic_server import main as server_main

from . import analysis as analysis_cfg
from . import ds
from .analysis import calculate_units
from .ds import internal


@cfn.config()
def eval_table():
    """The episode table for an eval run: only what every eval writes, nothing task-specific.

    ``eval.success`` is absent on an episode that never reached its terminal, so it defaults to False;
    ``eval.terminated`` separates a task the policy failed from one whose budget ran out.
    """
    return {
        '__index__': C(label='#', format='%d'),
        '__duration__': C(label='Duration', format='%.2f sec'),
        keys.TASK: C(label='Task', filter=True),
        eval_keys.SUCCESS: C(
            label='Pass',
            default=False,
            renderer=RendererConfig(
                type='badge',
                options={True: {'label': 'Pass', 'variant': 'success'}, False: {'label': 'Fail', 'variant': 'danger'}},
            ),
        ),
        eval_keys.TERMINATED: C(label='Ended', default=False),
    }


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
                Derive(started=lambda ep: datetime.fromtimestamp(ep.meta[META_CREATED_TS_NS] / 1e9), uph=uph),
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
                    task_code=analysis_cfg.task_code,
                    model=FromValue('Teleoperated by Human'),
                    units=analysis_cfg.units,
                    uph=analysis_cfg.uph,
                    checkpoint=FromValue(''),
                    success=FromValue(100),
                    started=analysis_cfg.started,
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

        result = {'task': episodes[0][keys.TASK]}
        result.update({'duration': duration, 'count': len(episodes), 'uph': units / duration})
        return result

    format_table = {
        'task': C(label='Task'),
        'duration': C(label='Duration', format='%.2f hours'),
        'uph': C(label='UPH', format='%.1f'),
        'count': C(label='Count'),
    }

    return GroupTableConfig(group_keys='task', group_fn=group_fn, format_table=format_table)


# What a manual rollout writes into its episodes, spelled here because the writer is the platform repo's
# rollouts console and the two repositories share no module.
PROGRESS_STATE = 'progress.state'
POLICY_LABEL = f'{policy_keys.POLICY_META}.label'
OUTCOME = 'eval.outcome'
SUCCESSFUL_ITEMS = 'eval.successful_items'
TOTAL_ITEMS = 'eval.total_items'
SUCCESS = 'Success'
UNSCORED = 'Unscored'


class ProgressStage(Enum):
    """A rung of the operator's progress ladder, declared lowest first.

    `value` is the code a rollout records in `PROGRESS_STATE`, mirroring the platform repo's
    `rollouts_contract.progress.Stage`; `label` is what the table shows.
    """

    label: str

    FLOATING = ('floating', 'moving free')
    REACHING = ('reaching', 'reaching')
    CONTACT = ('contact', 'in contact')
    CONTROL = ('control', 'moving it')
    AT_TARGET = ('at-target', 'at the target')

    def __new__(cls, code: str, label: str):
        member = object.__new__(cls)
        member._value_ = code
        member.label = label
        return member

    @property
    def rank(self) -> int:
        return list(type(self)).index(self)


ROLLOUT_OUTCOME_BADGE = RendererConfig(
    type='badge',
    options={
        SUCCESS: {'label': SUCCESS, 'variant': 'success'},
        'Fail': {'label': 'Fail', 'variant': 'danger'},
        'Safety': {'label': 'Safety', 'variant': 'warning'},
        'Ran out of time': {'label': 'Ran out of time', 'variant': 'default'},
        UNSCORED: {'label': UNSCORED, 'variant': 'default'},
    },
)


def rollout_model(ep: Episode) -> str:
    """The endpoint the episode was served by; older recordings name it through their checkpoint path."""
    return ep[POLICY_LABEL] if POLICY_LABEL in ep else analysis_cfg.model(ep)


def rollout_outcome(ep: Episode) -> str:
    """What the operator scored, or that she has not scored it yet."""
    return ep[OUTCOME] if OUTCOME in ep else UNSCORED


def rollout_stage(ep: Episode) -> ProgressStage | None:
    """The highest rung the arm reached, or None on an episode that recorded no progress."""
    if PROGRESS_STATE not in ep:
        return None
    reached = {value for value, _ in ep[PROGRESS_STATE]}
    return max((stage for stage in ProgressStage if stage.value in reached), key=lambda s: s.rank, default=None)


def rollout_stage_label(ep: Episode) -> str | None:
    stage = rollout_stage(ep)
    return None if stage is None else stage.label


def rollout_stage_rank(ep: Episode) -> int | None:
    stage = rollout_stage(ep)
    return None if stage is None else stage.rank


def rollout_items(ep: Episode) -> str | None:
    if SUCCESSFUL_ITEMS in ep and TOTAL_ITEMS in ep:
        return f'{ep[SUCCESSFUL_ITEMS]}/{ep[TOTAL_ITEMS]}'
    return None


rollouts_ds = ds.transform.override(
    base=ds.local_all,
    transforms=[
        ds.group.override(
            transforms=[
                Identity(),
                Derive(
                    model=rollout_model,
                    outcome=rollout_outcome,
                    stage=rollout_stage_label,
                    stage_rank=rollout_stage_rank,
                    items=rollout_items,
                    started=analysis_cfg.started,
                ),
            ]
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)


@cfn.config()
def rollouts_episodes_table():
    return {
        '__index__': C(label='#', format='%d'),
        '__duration__': C(label='Duration', format='%.0f sec'),
        keys.TASK: C(label='Task', filter=True),
        'model': C(label='Model', filter=True),
        'outcome': C(label='Outcome', renderer=ROLLOUT_OUTCOME_BADGE, align='center'),
        'stage': C(label='Stage', filter=True, default='-'),
        'items': C(label='Items', default='-'),
        'started': C(label='Started', format='%Y-%m-%d %H:%M:%S'),
    }


@cfn.config()
def rollouts_by_model():
    def group_fn(episodes: list[Episode]):
        successes = sum(1 for ep in episodes if ep['outcome'] == SUCCESS)
        return {
            'model': episodes[0]['model'],
            'count': len(episodes),
            'successes': successes,
            'success_rate': 100 * successes / len(episodes),
            'at_target': sum(1 for ep in episodes if ep['stage_rank'] == ProgressStage.AT_TARGET.rank),
        }

    format_table = {
        'model': C(label='Model'),
        'count': C(label='Episodes'),
        'successes': C(label='Successes'),
        'success_rate': C(label='Success rate', format='%.0f%%'),
        'at_target': C(label='Reached target'),
    }

    return GroupTableConfig(
        group_keys='model',
        group_fn=group_fn,
        format_table=format_table,
        group_filter_keys={keys.TASK: 'Task'},
        default_sort=SortConfig(column='success_rate'),
    )


finetune_server = server_main.override(
    dataset=finetune_ds, ep_table_cfg=finetune_episodes_table, group_tables={'tasks': finetune_group_by_task}
)

# Manual rollout rounds:
#   uv run --locked python -m positronic.cfg.server rollouts --dataset.base.path=s3://inference/droid_three_way/020926/
rollouts_server = server_main.override(
    dataset=rollouts_ds, ep_table_cfg=rollouts_episodes_table, group_tables={'models': rollouts_by_model}
)

if __name__ == '__main__':
    with pos3.mirror():
        init_logging()
        cfn.cli({'finetune': finetune_server, 'rollouts': rollouts_server})
