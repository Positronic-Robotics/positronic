"""Server configurations for positronic-server UI."""

from datetime import datetime

import configuronic as cfn
import pos3
from eval_vocabulary.outcome import ABSENT, OUTCOME, SUCCESSFUL_ITEMS, TOTAL_ITEMS, Outcome, is_scored
from eval_vocabulary.progress import LADDER, STATE_SIGNAL, Stage

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
from positronic.server.rollouts import OUTCOME_BADGE, StageCell, stage_cell

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


# The endpoint that served an episode. What an attended rollout records beside it — the verdict, the
# item counts and the progress ladder — is `eval_vocabulary`'s, which the console writes and this reads.
POLICY_LABEL = f'{policy_keys.POLICY_META}.label'

# What this preset DERIVES onto each episode. The tables below address these again, so each one is
# spelled once and the producer and every consumer read the same name.
DERIVED_MODEL = 'model'
DERIVED_OUTCOME = 'outcome'
DERIVED_STAGE = 'stage'
DERIVED_ITEMS = 'items'
DERIVED_STARTED = 'started'


def rollout_model(ep: Episode) -> str:
    """The endpoint the episode was served by; older recordings name it through their checkpoint path."""
    return ep[POLICY_LABEL] if POLICY_LABEL in ep else analysis_cfg.model(ep)


def rollout_outcome(ep: Episode) -> str:
    """What the operator scored, or that she has not scored it yet.

    The word as the recording holds it: a console one word ahead of this vocabulary still reads,
    because `app.js` draws an unlisted word as itself on a neutral badge.
    """
    return ep[OUTCOME] if OUTCOME in ep else ABSENT


def highest_rollout_stage(ep: Episode) -> StageCell:
    """The highest rung the arm reached, as the cell `positronic.server.rollouts` defines."""
    marked = {value for value, _ in ep[STATE_SIGNAL]} if STATE_SIGNAL in ep else set()
    return stage_cell(marked)


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
                Derive(**{
                    DERIVED_MODEL: rollout_model,
                    DERIVED_OUTCOME: rollout_outcome,
                    DERIVED_STAGE: highest_rollout_stage,
                    DERIVED_ITEMS: rollout_items,
                    DERIVED_STARTED: analysis_cfg.started,
                }),
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
        DERIVED_MODEL: C(label='Model', filter=True),
        DERIVED_OUTCOME: C(label='Outcome', renderer=OUTCOME_BADGE, filter=True, align='center'),
        DERIVED_STAGE: C(label='Stage'),
        DERIVED_ITEMS: C(label='Items', default='-'),
        DERIVED_STARTED: C(label='Started', format='%Y-%m-%d %H:%M:%S'),
    }


@cfn.config()
def rollouts_by_model():
    def group_fn(episodes: list[Episode]):
        # An episode the operator discarded or never scored measures nothing, so it is listed and
        # counted and stays out of the rate. The report reads the same round the same way.
        scored = [ep for ep in episodes if is_scored(ep[DERIVED_OUTCOME])]
        successes = sum(1 for ep in scored if ep[DERIVED_OUTCOME] == Outcome.SUCCESS)
        at_target = LADDER.index(Stage.AT_TARGET)
        return {
            DERIVED_MODEL: episodes[0][DERIVED_MODEL],
            'count': len(episodes),
            'scored': len(scored),
            'successes': successes,
            'success_rate': 100 * successes / len(scored) if scored else None,
            'at_target': sum(1 for ep in episodes if ep[DERIVED_STAGE].rank == at_target),
        }

    format_table = {
        DERIVED_MODEL: C(label='Model'),
        'count': C(label='Episodes'),
        'scored': C(label='Scored'),
        'successes': C(label='Successes'),
        'success_rate': C(label='Success rate', format='%.0f%%', default='-'),
        'at_target': C(label='Reached target'),
    }

    return GroupTableConfig(
        group_keys=DERIVED_MODEL,
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
