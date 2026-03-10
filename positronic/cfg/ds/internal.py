"""
Internal PhAIL datasets with model-specific observation/action encodings.

These datasets remain on the private s3://raw/ bucket and include:
- Raw datasets from internal data collection
- Transformed variants (cubes_sim, pnp_sim with signal transforms)
- OpenPI-encoded variants (*_openpi_ft)
- GR00T-encoded variants (*_groot_ft)
- Full combined datasets for multi-task training
"""

import configuronic as cfn
import pos3

from positronic.cfg.eval import started
from positronic.dataset import Episode
from positronic.dataset.dataset import ConcatDataset, FilterDataset
from positronic.dataset.local_dataset import load_all_datasets
from positronic.dataset.transforms import TransformedDataset, agg_fraction_true, agg_max, agg_percentile
from positronic.dataset.transforms.episode import Concat, Derive, FromValue, Group, Identity, Rename
from positronic.dataset.transforms.quality import cmd_lag, cmd_velocity, idle_mask, jerk
from positronic.server.positronic_server import ColumnConfig as C
from positronic.server.positronic_server import GroupTableConfig, RendererConfig, SortConfig
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging

from . import concat_ds, group, local, local_all, transform

# Task constants
TOWELS_TASK = 'Pick all the towels one by one from transparent tote and place them into the large grey tote.'
SPOONS_TASK = 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.'
SCISSORS_TASK = 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.'
BATTERIES_TASK = 'Pick all the batteries one by one from transparent tote and place them into the large grey tote.'
RECOVERY_TASK = 'Recovery cases.'


ALL_TASKS = [TOWELS_TASK, SPOONS_TASK, SCISSORS_TASK, BATTERIES_TASK]


def _recovery_transforms(task: str):
    """Create transforms for recovery episodes with given task label."""
    return Group(Derive(task=FromValue(task), recovery=FromValue(True)), Identity())


@cfn.config(path='s3://raw/droid/', recovery_all=False, recovery_towels=False, duplicate_recovery=False)
def droid(path, recovery_all, recovery_towels, duplicate_recovery):
    """Internal DROID dataset with task label transforms.

    Args:
        path: S3 path to droid dataset
        recovery_all: Include general recovery episodes
        recovery_towels: Include towels-specific recovery episodes
        duplicate_recovery: If True, duplicate recovery episodes for each task (training strategy).
            If False, use single RECOVERY_TASK label (for public dataset).
    """
    root = pos3.download(path)

    datasets = [
        TransformedDataset(load_all_datasets(root / 'towels'), Group(Derive(task=FromValue(TOWELS_TASK)), Identity())),
        TransformedDataset(load_all_datasets(root / 'spoons'), Group(Derive(task=FromValue(SPOONS_TASK)), Identity())),
        TransformedDataset(
            # TODO: This is typo, it was done when the we started collecting the dataset, and now we are not fixing it.
            load_all_datasets(root / 'scisors'),
            Group(Derive(task=FromValue(SCISSORS_TASK)), Identity()),
        ),
        TransformedDataset(
            load_all_datasets(root / 'batteries'), Group(Derive(task=FromValue(BATTERIES_TASK)), Identity())
        ),
    ]
    if recovery_towels:
        datasets.append(
            TransformedDataset(load_all_datasets(root / 'recovery_towels/'), _recovery_transforms(TOWELS_TASK))
        )
    if recovery_all:
        recovery_ds = load_all_datasets(root / 'recovery/')
        if duplicate_recovery:
            for task in ALL_TASKS:
                datasets.append(TransformedDataset(recovery_ds, _recovery_transforms(task)))
        else:
            datasets.append(TransformedDataset(recovery_ds, _recovery_transforms(RECOVERY_TASK)))
    return ConcatDataset(*datasets)


# Signal transformations for sim datasets
old_to_new = Group(
    Derive(**{
        'controller_positions.right': Concat('right_controller_translation', 'right_controller_quaternion'),
        'robot_commands.pose': Concat('target_robot_position_translation', 'target_robot_position_quaternion'),
        'robot_state.ee_pose': Concat('robot_position_translation', 'robot_position_quaternion'),
        'task': FromValue('Pick up the green cube and place it on the red cube.'),
    }),
    Rename(**{
        'robot_state.q': 'robot_joints',
        'robot_state.dq': 'robot_joints_velocity',
        'image.wrist': 'image.handcam_left',
        'image.exterior': 'image.back_view',
    }),
    Identity(select=['grip', 'target_grip', 'mjSTATE_FULLPHYSICS', 'mjSTATE_INTEGRATION', 'mjSTATE_WARMSTART']),
)

sim_stack = transform.override(base=local.override(path='s3://raw/sim-cubes/luzan/'), transforms=[old_to_new])

sim_pnp = transform.override(
    base=local.override(path='s3://raw/sim_pnp/'),
    transforms=[
        Group(
            Derive(task=FromValue('Pick up objects from the red tote and place them in the green tote.')),
            Rename(**{'image.exterior': 'image.back_view'}),
            Identity(),
        )
    ],
)


# Episodes excluded from training: broken recordings and VR tracking glitches (cmd_vel_max > 5 m/s).
# TODO: Episodes with moderate tracking glitches (3-5 m/s) are kept for now. Future work: trim the
# glitch segments instead of removing entire episodes.
_DROID_EXCLUDE_PATHS = {
    'towels/151025/000000000000/000000000001',  # broken camera
    'spoons/161025/000000000000/000000000002',  # tracking glitch (14.9 m/s)
    'spoons/161025/000000000000/000000000022',  # tracking glitch (5.3 m/s)
    'spoons/250126/000000000000/000000000003',  # tracking glitch (7.5 m/s)
    'scisors/000000000000/000000000013',  # tracking glitch (22.8 m/s)
    'scisors/000000000000/000000000041',  # tracking glitch (6.4 m/s)
    'scisors/000000000000/000000000069',  # broken (0.8s, zero motion)
    'scisors/000000000000/000000000087',  # tracking glitch (7.2 m/s)
    'scisors/000000000000/000000000096',  # tracking glitch (5.6 m/s)
    'scisors/000000000000/000000000103',  # tracking glitch (8.9 m/s)
    'scisors/000000000000/000000000104',  # tracking glitch (5.4 m/s)
    'batteries/250126/000000000000/000000000000',  # tracking glitch (9.1 m/s)
    'batteries/250126/000000000000/000000000006',  # tracking glitch (16.9 m/s)
    'batteries/250126/000000000000/000000000007',  # tracking glitch (12.3 m/s)
    'batteries/250126/000000000000/000000000014',  # tracking glitch (11.8 m/s)
}


def _is_excluded(ep):
    path = ep.meta.get('path', '')
    return not any(path.endswith(suffix) for suffix in _DROID_EXCLUDE_PATHS)


@cfn.config()
def droid_clean(dataset):
    """DROID dataset with broken and glitched episodes removed."""
    return FilterDataset(dataset, _is_excluded)


droid_clean = droid_clean.override(dataset=droid)

droid_recovery = droid_clean.override(
    dataset=droid.override(recovery_all=True, recovery_towels=True, duplicate_recovery=False)
)
sim = concat_ds.override(datasets=[sim_stack, sim_pnp])
full_recovery = concat_ds.override(datasets=[droid_recovery, sim])
full = concat_ds.override(datasets=[droid, sim])


# =============================================================================
# PhAIL production evaluation — derived metrics and visualization server
# =============================================================================


TASK_CODES = {
    TOWELS_TASK: 'Towels',
    SPOONS_TASK: 'Wooden spoons',
    SCISSORS_TASK: 'Scissors',
    BATTERIES_TASK: 'Batteries',
}


def phail_task_code(ep: Episode) -> str:
    obj = ep.get('eval.object', '')
    if obj:
        return obj
    return TASK_CODES.get(ep.get('task', ''), ep.get('task', ''))


HOST_TO_MODEL = {'desktop': 'groot', 'vm-openpi': 'openpi', 'localhost': 'act'}


def phail_model(ep: Episode) -> str:
    policy_type = ep.get('inference.policy.type', '')
    if policy_type == 'remote':
        server_type = ep.get('inference.policy.server.type', '')
        if server_type:
            return server_type
        return HOST_TO_MODEL.get(ep.get('inference.policy.host', ''), 'unknown')
    return policy_type or 'unknown'


def outcome_category(ep: Episode) -> str:
    """Map eval outcomes to display categories: Success, Fail, Safety, System."""
    raw = ep.get('eval.outcome', '')
    if raw == 'Success':
        return 'Success'
    if raw in ('Fail', 'Stalled', 'Ran out of time'):
        return 'Fail'
    if raw == 'Safety':
        return 'Safety'
    return 'System'


def units_display(ep: Episode) -> str:
    items = ep.get('eval.successful_items')
    total = ep.get('eval.total_items')
    if items is not None and total is not None:
        return f'{items}/{total}'
    return '-'


def success_rate(ep: Episode) -> float:
    items = ep.get('eval.successful_items', 0)
    total = ep.get('eval.total_items', 0)
    if total == 0:
        return 0.0
    return 100 * items / total


def uph(ep: Episode) -> float | None:
    items = ep.get('eval.successful_items', 0)
    if items == 0:
        return None
    return items / (ep.duration_ns / 1e9 / 3600)


OUTCOME_RENDERER = RendererConfig(
    type='badge',
    options={
        'Success': {'label': 'Success', 'variant': 'success'},
        'Fail': {'label': 'Fail', 'variant': 'warning'},
        'Safety': {'label': 'Safety', 'variant': 'danger'},
        'System': {'label': 'System', 'variant': 'secondary'},
    },
)


phail_episodes = transform.override(
    base=local_all,
    transforms=[
        group.override(
            transforms=[
                Identity(),
                Derive(
                    task_code=phail_task_code,
                    model=phail_model,
                    outcome=outcome_category,
                    units_display=units_display,
                    uph=uph,
                    success_rate=success_rate,
                    started=started,
                ),
            ]
        )
    ],
)


@cfn.config()
def phail_episodes_table():
    return {
        '__index__': C(label='#', format='%d'),
        '__duration__': C(label='Duration', format='%.1f sec'),
        'task_code': C(label='Task', filter=True),
        'model': C(label='Model', filter=True),
        'outcome': C(label='Outcome', filter=True, renderer=OUTCOME_RENDERER),
        'units_display': C(label='Units'),
        'uph': C(label='UPH', format='%.1f', default='-'),
        'success_rate': C(label='Success', format='%.1f%%'),
        'started': C(label='Started', format='%Y-%m-%d %H:%M:%S'),
    }


@cfn.config()
def phail_leaderboard():
    def group_fn(episodes: list[Episode]):
        count = len(episodes)
        total_duration = sum(ep.duration_ns / 1e9 for ep in episodes)
        total_units = sum(ep.get('eval.successful_items', 0) for ep in episodes)
        total_possible = sum(ep.get('eval.total_items', 0) for ep in episodes)
        sr = 100 * total_units / total_possible if total_possible > 0 else 0
        failed_count = sum(1 for ep in episodes if ep['outcome'] != 'Success')

        return {
            'model': episodes[0]['model'],
            'task_code': episodes[0]['task_code'],
            'count': count,
            'UPH': total_units / (total_duration / 3600) if total_duration > 0 else 0,
            'success_rate': sr,
            'MTBF': total_duration / failed_count if failed_count > 0 else None,
        }

    format_table = {
        'model': C(label='Model'),
        'count': C(label='Runs', format='%d'),
        'UPH': C(label='UPH', format='%.1f'),
        'success_rate': C(label='Success', format='%.1f%%'),
        'MTBF': C(label='MTBF', format='%.1f sec', default='-'),
    }

    return GroupTableConfig(
        group_keys=('model',),
        group_fn=group_fn,
        format_table=format_table,
        group_filter_keys={'task_code': 'Task'},
        default_sort=SortConfig(column='UPH'),
    )


phail_server = server_main.override(
    dataset=phail_episodes,
    ep_table_cfg=phail_episodes_table,
    group_tables={'leaderboard': phail_leaderboard},
    home_page='leaderboard',
    port=5001,
)


# =============================================================================
# Quality signal debugging server for internal DROID dataset
# =============================================================================

# Per-frame quality signals (visible as time-series in Rerun viewer)
_quality_signals = Derive(
    quality_idle=idle_mask, quality_jerk=jerk, quality_cmd_lag=cmd_lag, quality_cmd_vel=cmd_velocity
)

# Scalar metrics (visible as columns in episode table).
# Chained after _quality_signals so lambdas can reference the quality signals.
_quality_scalars = Derive(
    idle_frac=lambda ep: agg_fraction_true(ep.signals['quality_idle']) * 100,
    cmd_lag_max=lambda ep: agg_max(ep.signals['quality_cmd_lag']),
    cmd_lag_p95=lambda ep: agg_percentile(ep.signals['quality_cmd_lag'], 95),
    jerk_p95=lambda ep: agg_percentile(ep.signals['quality_jerk'], 95),
    cmd_vel_max=lambda ep: agg_max(ep.signals['quality_cmd_vel']),
)


_droid_debug_table = {
    '__index__': C(label='#', format='%d'),
    '__duration__': C(label='Duration', format='%.0f sec'),
    'task': C(label='Task', filter=True),
    'idle_frac': C(label='Idle %', format='%.1f%%'),
    'cmd_lag_max': C(label='Lag Max', format='%.3f m'),
    'cmd_lag_p95': C(label='Lag p95', format='%.3f m'),
    'jerk_p95': C(label='Jerk p95', format='%.1f'),
    'cmd_vel_max': C(label='Cmd Vel Max', format='%.2f m/s'),
}

# Chain: add quality signals → compute scalar metrics
_quality_transform = Group(_quality_signals, Identity()) | Group(_quality_scalars, Identity())

droid_debug = server_main.override(
    dataset=transform.override(base=droid, transforms=[_quality_transform]), ep_table_cfg=_droid_debug_table
)


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli({'phail': phail_server, 'droid_debug': droid_debug})
