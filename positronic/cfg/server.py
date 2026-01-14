"""Server configurations for positronic-server UI."""

from datetime import datetime

import configuronic as cfn
import numpy as np
import pos3

from positronic.dataset import Episode
from positronic.dataset.transforms.episode import Derive, FromValue, Group, Identity, Rename
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging

from . import ds
from . import eval as eval_cfg
from .ds import internal

# Task constants
TOWELS_TASK = 'Pick all the towels one by one from transparent tote and place them into the large grey tote.'
SPOONS_TASK = 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.'
SCISSORS_TASK = 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.'


def calculate_units(episode: Episode) -> int:
    """Estimates the number of pick-and-place operations.

    This function is vibe-coded with Gemini 3 Pro (High). It works fine as a heuristic.
    """
    if episode['task'] == SCISSORS_TASK:
        return 10

    if 'target_grip' in episode.signals:
        grip_sig = episode.signals['target_grip']
    elif 'grip' in episode.signals:
        grip_sig = episode.signals['grip']
    else:
        return 0

    if 'robot_state.ee_pose' in episode.signals:
        pose_sig = episode.signals['robot_state.ee_pose']
    else:
        return 0

    # Sample signals at 10Hz to reduce noise and computation
    times = np.arange(episode.start_ts, episode.last_ts, int(1e8))
    if len(times) == 0:
        return 0

    grip_vals = np.array([v for v, _ in grip_sig.time[times]])
    pose_vals = np.array([v for v, _ in pose_sig.time[times]])
    x_vals = pose_vals[:, 0]
    y_vals = pose_vals[:, 1]
    z_vals = pose_vals[:, 2]

    # Heuristic: Binarize grip based on midpoint threshold
    threshold = (grip_vals.max() + grip_vals.min()) / 2

    units = 0
    state = 'OPEN'
    # Initial state
    if grip_vals[0] < threshold:
        state = 'CLOSED'

    min_z_holding = np.inf
    max_z_holding = -np.inf
    pick_x, pick_y = 0.0, 0.0

    lift_threshold = 0.05  # 5cm
    dist_threshold = 0.15  # 15cm

    for i in range(1, len(grip_vals)):
        val = grip_vals[i]
        x, y, z = x_vals[i], y_vals[i], z_vals[i]
        is_closed = val < threshold

        if state == 'OPEN':
            if is_closed:
                state = 'CLOSED'
                min_z_holding = z
                max_z_holding = z
                pick_x, pick_y = x, y
        elif state == 'CLOSED':
            max_z_holding = max(max_z_holding, z)
            min_z_holding = min(min_z_holding, z)
            if not is_closed:  # Drop event
                state = 'OPEN'
                amplitude = max_z_holding - min_z_holding

                dx = x - pick_x
                dy = y - pick_y
                dist = np.sqrt(dx * dx + dy * dy)

                if amplitude > lift_threshold and dist > dist_threshold:
                    units += 1

    return units


def uph(ep: Episode) -> float | None:
    items = ep['units']
    if items == 0:
        return None
    return items / (ep.duration_ns / 1e9 / 3600)


finetune_ds = ds.transform.override(
    base=ds.transform.override(
        base=internal.droid_ds, transforms=[ds.group.override(transforms=[Identity(), Derive(units=calculate_units)])]
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
        '__index__': {'label': '#', 'format': '%d'},
        '__duration__': {'label': 'Duration', 'format': '%.0f sec'},
        'task': {'label': 'Task', 'filter': True},
        'units': {'label': 'Units'},
        'uph': {'label': 'UPH', 'format': '%.1f'},
        'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M'},
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
        'task': {'label': 'Task'},
        'duration': {'label': 'Duration', 'format': '%.2f hours'},
        'uph': {'label': 'UPH', 'format': '%.1f'},
        'count': {'label': 'Count'},
    }

    return 'task', group_fn, format_table, {}


finetune_server = server_main.override(
    dataset=finetune_ds, ep_table_cfg=finetune_episodes_table, group_tables={'tasks': finetune_group_by_task}
)

if __name__ == '__main__':
    with pos3.mirror():
        init_logging()
        cfn.cli(finetune_server)
