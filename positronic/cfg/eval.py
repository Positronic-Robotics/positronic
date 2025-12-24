from datetime import datetime

import configuronic as cfn
import numpy as np

import positronic.cfg.dataset as base_cfg
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, Group, Identity


def task_code(ep: Episode) -> str:
    match ep['task']:
        case 'Pick all the towels one by one from transparent tote and place them into the large grey tote.':
            return 'Towels'
        case 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.':
            return 'Wooden spoons'
        case 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.':
            return 'Scissors'
        case _:
            return ''


def model(ep: Episode) -> str:
    match ep['inference.policy.type']:
        case 'act':
            return 'Action Chunking Trasnformer'
        case 'groot':
            return 'Nvidia Gr00t'
        case 'openpi':
            return 'Open PI 0.5'
        case _:
            return ''


def ckpt(ep: Episode) -> str | None:
    def _split_path(path: str) -> list[str]:
        return [p for p in path.strip('/').split('/') if p]

    try:
        match ep['inference.policy.type']:
            case 'act':
                raw_path = ep['inference.policy.checkpoint_path']
                # Path to ckpt id: full_ft_q/act/031225/checkpoints/300000/pretrained_model/ -> full_ft_q\031225\300000
                parts = _split_path(raw_path)
                chkpt_idxs = [i for i, p in enumerate(parts) if p == 'checkpoints']
                if chkpt_idxs:
                    idx = chkpt_idxs[-1]
                    dataset = parts[idx - 3] if idx >= 3 else None
                    experiment = parts[idx - 1] if idx >= 1 else None
                    step = parts[idx + 1] if idx + 1 < len(parts) else None
                    if dataset is not None and experiment is not None and step is not None:
                        return f'{dataset}\\{experiment}\\{step}'

                # Fallback: keep legacy behavior if path doesn't match the expected structure.
                return raw_path.split('/checkpoints/')[-1]
            case 'openpi':
                if 'inference.policy.checkpoint_path' in ep:
                    raw_path = ep['inference.policy.checkpoint_path']
                else:
                    raw_path = ep['inference.policy.server.directory']
                raw_path = raw_path.split('/checkpoints/')[-1]
                # Path to ckpt id: full_ft/openpi/pi05_positronic_lowmem/061025/119999 -> full_ft\061025\119999
                parts = _split_path(raw_path)
                if len(parts) >= 3:
                    return f'{parts[0]}\\{parts[-2]}\\{parts[-1]}'
                return raw_path
        return ''
    except Exception:
        return ''


def units(ep: Episode) -> str:
    return f'{ep["eval.successful_items"]}/{ep["eval.total_items"]}'


def uph(ep: Episode) -> float | None:
    items = ep['eval.successful_items']
    if items == 0:
        return None
    return items / (ep.duration_ns / 1e9 / 3600)


def started(ep: Episode) -> datetime:
    return datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9)


ds = base_cfg.transform.override(
    base=base_cfg.local,
    transforms=[
        base_cfg.group.override(
            transforms=[
                Identity(),
                Derive(
                    task_code=task_code,
                    model=model,
                    units=units,
                    uph=uph,
                    checkpoint=ckpt,
                    success=lambda ep: 100 * ep['eval.successful_items'] / ep['eval.total_items'],
                    started=started,
                ),
            ]
        )
    ],
)


@cfn.config()
def eval_table():
    return {
        '__index__': {'label': '#', 'format': '%d'},
        'task_code': {'label': 'Task', 'filter': True},
        'model': {'label': 'Model', 'filter': True},
        'checkpoint': {'label': 'Checkpoint'},
        'units': {'label': 'Units'},
        'uph': {'label': 'UPH', 'format': '%.1f'},
        'success': {'label': 'Success', 'format': '%.1f%%'},
        'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M:%S'},
        'eval.outcome': {
            'label': 'Status',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    # TODO: Currently the filter happens by original data, not the rendered value
                    'Success': {'label': 'Pass', 'variant': 'success'},
                    'Stalled': {'label': 'Fail', 'variant': 'warning'},
                    'Fail': {'label': 'Fail', 'variant': 'warning'},
                    'Ran out of time': {'label': 'Fail', 'variant': 'warning'},
                    'System': {'label': 'Fail', 'variant': 'warning'},
                    'Safety': {'label': 'Safety violation', 'variant': 'danger'},
                },
            },
        },
        '__duration__': {'label': 'Duration', 'format': '%.1f sec'},
    }


LABELS = {'model': 'Model', 'task_code': 'Task', 'checkpoint': 'Checkpoint'}


@cfn.config()
def grouped_table(group_keys: tuple[str, ...] | str):
    if isinstance(group_keys, str):
        group_keys = (group_keys,)

    def group_fn(episodes: list[Episode]):
        duration, suc_items, total_items, assists = 0, 0, 0, 0
        for ep in episodes:
            duration += ep['eval.duration']
            suc_items += ep['eval.successful_items']
            total_items += ep['eval.total_items']
            assists += ep['eval.outcome'] != 'Success'

        result = {key: episodes[0][key] for key in group_keys}
        result.update({
            'UPH': suc_items / (duration / 3600),
            'Success': 100 * suc_items / total_items,
            'MTBF/A': (duration / assists) if assists > 0 else None,
            'Assists': assists,
            'count': len(episodes),
        })
        return result

    format_table = {**{key: {'label': LABELS[key]} for key in group_keys}}
    format_table.update({
        'count': {'label': 'Count'},
        'UPH': {'format': '%.1f'},
        'Success': {'format': '%.2f%%'},
        'MTBF/A': {'format': '%.1f sec', 'default': '-'},
        'Assists': {'format': '%d'},
    })

    group_filter_keys = {key: LABELS[key] for key in group_keys}
    return group_keys, group_fn, format_table, group_filter_keys


# Set of group configurations for evaluation server:
# uv run positronic-server \
#   --dataset=@positronic.cfg.eval.ds \
#   --dataset.base.path /Users/vertix/.cache/positronic/s3/inference/real/191225/ \
#   --port=5001 \
#   --ep_table_cfg=@positronic.cfg.eval.eval_table \
#   --group_tables.models=@positronic.cfg.eval.model_table \
#   --group_tables.model_task=@positronic.cfg.eval.model_task_table \
#   --group_tables.model_ckpt_task=@positronic.cfg.eval.model_chkpt_task_table \
#   --group_tables.model_ckpt=@positronic.cfg.eval.model_chkpt_table

model_table = grouped_table.override(group_keys='model')
model_task_table = grouped_table.override(group_keys=('model', 'task_code'))
model_chkpt_table = grouped_table.override(group_keys=('model', 'checkpoint'))
model_chkpt_task_table = grouped_table.override(group_keys=('model', 'checkpoint', 'task_code'))


########################
# Simulator evaluation #
########################


def max_stacking_success(episode: Episode) -> float | None:
    if 'stacking_success' not in episode:
        return None
    success_signal = episode['stacking_success']
    if len(success_signal) == 0:
        return None
    return max(v for v, _ in success_signal)


def success(episode: Episode) -> bool:
    if 'stacking_success' not in episode:
        return False
    success_signal = episode['stacking_success']
    if len(success_signal) == 0:
        return False

    two_sec = 2 * 1e9
    end = success_signal.time[success_signal.last_ts - two_sec :]
    return len(end) > 0 and all(v == 1.0 for v, _ in end)


def success_time(episode: Episode) -> float | None:
    if 'stacking_success' not in episode:
        return None
    success_signal = episode['stacking_success']
    if len(success_signal) == 0 or success_signal[-1][0] != 1.0:
        return None

    # Compute moment when success_signal becomes and stays 1.0 at the end
    values = np.zeros(len(success_signal), dtype=np.float32)
    times = np.zeros(len(success_signal), dtype=np.int64)
    for i, (v, t) in enumerate(success_signal):
        values[i], times[i] = v, t

    # Find the last moment when value is not 1.0
    idx_not_1 = np.where(values != 1.0)[0]
    if len(idx_not_1) == 0:
        return 0
    last_not_1 = idx_not_1[-1]
    return (times[last_not_1 + 1].item() - episode.start_ts) / 1e9


def box_distance_progress(episode: Episode) -> float | None:
    if 'box_distance' not in episode:
        return None
    distance_signal = episode['box_distance']
    if len(distance_signal) == 0:
        return None

    mind = np.min(distance_signal.values())
    maxd = np.max(distance_signal.values())
    if maxd == mind:
        return None

    return (1 - mind / (maxd + 1e-6)).item() * 100


def ee_pose_movement(episode: Episode) -> float | None:
    signal_values = episode['robot_state.ee_pose'].values()
    result = 0.0
    prev_translation = signal_values[0][:3]
    for ee_pose in signal_values[1:]:
        translation = ee_pose[:3]
        result += np.linalg.norm(translation - prev_translation).item()
        prev_translation = translation
    return result


sim_episodes = base_cfg.transform.override(
    base=base_cfg.local,
    transforms=[
        Group(
            Identity(),
            Derive(
                checkpoint=ckpt,
                max_stacking_success=max_stacking_success,
                success=success,
                success_time=success_time,
                box_distance_progress=box_distance_progress,
                movement=ee_pose_movement,
            ),
        )
    ],
)


@cfn.config()
def sim_episodes_table():
    return {
        '__index__': {'label': '#', 'format': '%d'},
        '__duration__': {'label': 'Duration', 'format': '%.2f sec'},
        'checkpoint': {'label': 'CKPT'},
        'max_stacking_success': {'label': 'Max Success', 'format': '%.2f'},
        'success': {
            'label': 'Pass',
            'renderer': {
                'type': 'badge',
                'options': {
                    True: {'label': 'Pass', 'variant': 'success'},
                    False: {'label': 'Fail', 'variant': 'danger'},
                },
            },
        },
        'success_time': {'label': 'Time', 'format': '%.1f sec'},
        'box_distance_progress': {'label': 'Box progress', 'format': '%.1f%%'},
        'movement': {'label': 'Movement', 'format': '%.2f'},
    }


@cfn.config()
def sim_episodes_perf():
    def group_fn(episodes: list[Episode]):
        count = len(episodes)

        result = {
            'checkpoint': episodes[0]['checkpoint'],
            'max_stacking_success': np.mean([ep['max_stacking_success'] for ep in episodes]),
            'box_distance_progress': np.mean([ep.get('box_distance_progress', 0) for ep in episodes]),
            'movement': np.mean([ep['movement'] for ep in episodes]),
            'count': count,
        }
        return result

    format_table = {
        'checkpoint': {'label': 'CKPT'},
        'count': {'label': 'Count'},
        'max_stacking_success': {'label': 'Max Success', 'format': '%.2f'},
        'box_distance_progress': {'label': 'Box progress', 'format': '%.1f%%'},
        'movement': {'label': 'Movement', 'format': '%.2f'},
    }

    return 'checkpoint', group_fn, format_table, {}
