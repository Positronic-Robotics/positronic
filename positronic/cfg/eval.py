from datetime import datetime

import configuronic as cfn
import numpy as np
import pos3

import positronic.cfg.ds as base_cfg
from positronic.dataset.episode import Episode
from positronic.dataset.transforms.episode import Derive, Group, Identity
from positronic.server.positronic_server import main as server_main
from positronic.utils.logging import init_logging


def task_code(ep: Episode) -> str:
    match ep['task']:
        case 'Pick all the towels one by one from transparent tote and place them into the large grey tote.':
            return 'Towels'
        case 'Pick all the wooden spoons one by one from transparent tote and place them into the large grey tote.':
            return 'Wooden spoons'
        case 'Pick all the scissors one by one from transparent tote and place them into the large grey tote.':
            return 'Scissors'
        case 'Pick all the batteries one by one from transparent tote and place them into the large grey tote.':
            return 'Batteries'
        case _:
            return ''


def _model_label_from_path(model_type: str, checkpoint_path: str) -> str | None:
    """Extract a model label from a checkpoint path like .../checkpoints/sim_stack/groot/ee_rot6d/..."""
    if not checkpoint_path or '/checkpoints/' not in checkpoint_path:
        return None
    parts = [p for p in checkpoint_path.split('/checkpoints/')[-1].split('/') if p]
    if len(parts) >= 3:
        return f'{model_type}:{parts[-3]}'
    return None


def model(ep: Episode) -> str:
    policy_type = ep.get('inference.policy.type', '')

    if policy_type == 'remote':
        server_type = ep.get('inference.policy.server.type', '')
        path_label = _model_label_from_path(server_type, ep.get('inference.policy.server.checkpoint_path', ''))
        if path_label:
            return path_label
        return server_type or ''

    if policy_type:
        path_label = _model_label_from_path(policy_type, ep.get('inference.policy.checkpoint_path', ''))
        if path_label:
            return path_label
        return policy_type

    return ''


def _split_path(path: str) -> list[str]:
    return [p for p in path.strip('/').split('/') if p]


def _ckpt_act(ep: Episode) -> str:
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


def _ckpt_remote(ep: Episode) -> str:
    checkpoint_id = ep.get('inference.policy.server.checkpoint_id', '')
    if checkpoint_id:
        return str(checkpoint_id)
    raw_path = ep.get('inference.policy.server.checkpoint_path', '')
    if raw_path:
        step = _split_path(raw_path)[-1].removeprefix('checkpoint-')
        return step
    return ''


def ckpt(ep: Episode) -> str | None:
    try:
        match ep.get('inference.policy.type', ''):
            case 'act':
                return _ckpt_act(ep)
            case 'remote':
                return _ckpt_remote(ep)
        return ''
    except Exception:
        return ''


def started(ep: Episode) -> datetime:
    return datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9)


def units(ep: Episode) -> int | None:
    if 'eval.successful_items' in ep:
        return ep['eval.successful_items']
    if 'units' in ep:
        return ep['units']
    return None


def uph(ep: Episode) -> float | None:
    u = units(ep)
    if not u:
        return None
    return u / (ep.duration_ns / 1e9 / 3600)


########################
# Unified configs (real + sim)
########################


def is_sim_episode(ep: Episode) -> bool:
    return 'stacking_success' in ep


def unified_success_bool(ep: Episode) -> bool:
    if 'eval.outcome' in ep:
        return ep['eval.outcome'] == 'Success'
    if 'stacking_success' in ep:
        return success(ep)
    return False


def unified_units_display(ep: Episode) -> str:
    if 'eval.successful_items' in ep:
        return f'{ep["eval.successful_items"]}/{ep["eval.total_items"]}'
    if 'stacking_success' in ep:
        return str(1 if success(ep) else 0)
    return '-'


def unified_uph(ep: Episode) -> float | None:
    duration_hours = ep.duration_ns / 1e9 / 3600
    if 'eval.successful_items' in ep:
        items = ep['eval.successful_items']
        if items == 0:
            return None
        return items / duration_hours
    if 'stacking_success' in ep:
        if success(ep):
            return 1 / duration_hours
        return None
    return None


def unified_success_rate(ep: Episode) -> float:
    if 'eval.successful_items' in ep:
        return 100 * ep['eval.successful_items'] / ep['eval.total_items']
    if 'stacking_success' in ep:
        return 100.0 if success(ep) else 0.0
    return 0.0


episodes = base_cfg.transform.override(
    base=base_cfg.local,
    transforms=[
        Group(
            Identity(),
            Derive(
                task_code=task_code,
                model=model,
                checkpoint=ckpt,
                is_sim=is_sim_episode,
                success_bool=unified_success_bool,
                units_display=unified_units_display,
                uph=unified_uph,
                success_rate=unified_success_rate,
                started=started,
            ),
        )
    ],
)


@cfn.config()
def episodes_table():
    return {
        '__index__': {'label': '#', 'format': '%d'},
        '__duration__': {'label': 'Duration', 'format': '%.1f sec'},
        'task_code': {'label': 'Task', 'filter': True},
        'model': {'label': 'Model', 'filter': True},
        'checkpoint': {'label': 'Checkpoint', 'filter': True},
        'success_bool': {
            'label': 'Pass',
            'renderer': {
                'type': 'badge',
                'options': {
                    True: {'label': 'Pass', 'variant': 'success'},
                    False: {'label': 'Fail', 'variant': 'danger'},
                },
            },
        },
        'units_display': {'label': 'Units'},
        'uph': {'label': 'UPH', 'format': '%.1f', 'default': '-'},
        'success_rate': {'label': 'Success', 'format': '%.1f%%'},
        'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M:%S'},
    }


@cfn.config()
def checkpoint_table():
    def group_fn(episodes: list[Episode]):
        count = len(episodes)
        total_duration = sum(ep.duration_ns / 1e9 for ep in episodes)

        if 'stacking_success' in episodes[0]:
            successful_count = sum(1 for ep in episodes if success(ep))
            total_units = successful_count
            failed_count = count - successful_count
            success_rate = 100 * successful_count / count if count > 0 else 0
        else:
            total_units = sum(ep.get('eval.successful_items', 0) for ep in episodes)
            total_possible = sum(ep.get('eval.total_items', 0) for ep in episodes)
            success_rate = 100 * total_units / total_possible if total_possible > 0 else 0
            failed_count = sum(1 for ep in episodes if ep.get('eval.outcome') != 'Success')

        return {
            'checkpoint': episodes[0]['checkpoint'],
            'model': episodes[0]['model'],
            'count': count,
            'UPH': total_units / (total_duration / 3600) if total_duration > 0 else 0,
            'success_rate': success_rate,
            'MTBF': total_duration / failed_count if failed_count > 0 else None,
            'failures': failed_count,
        }

    format_table = {
        'model': {'label': 'Model'},
        'checkpoint': {'label': 'Checkpoint'},
        'count': {'label': 'Runs', 'format': '%d'},
        'UPH': {'label': 'UPH', 'format': '%.1f'},
        'success_rate': {'label': 'Success', 'format': '%.1f%%'},
        'MTBF': {'label': 'MTBF', 'format': '%.1f sec', 'default': '-'},
        'failures': {'label': 'Failures', 'format': '%d'},
    }

    group_filter_keys = {'checkpoint': 'Checkpoint', 'model': 'Model'}
    return ('model', 'checkpoint'), group_fn, format_table, group_filter_keys


########################
# Extended simulator evaluation #
########################


def max_stacking_success(episode: Episode) -> float | None:
    if 'stacking_success' not in episode:
        return None
    success_signal = episode['stacking_success']
    if len(success_signal) == 0:
        return None
    return max(v for v, _ in success_signal)


def success(episode: Episode) -> bool:
    """Check if stacking_success reached 1.0 and stayed there for at least 0.5 seconds."""
    if 'stacking_success' not in episode:
        return False
    success_signal = episode['stacking_success']
    if len(success_signal) == 0:
        return False

    threshold_ns = int(0.5 * 1e9)  # 0.5 seconds in nanoseconds

    # Find all stretches where value == 1.0
    in_success = False
    success_start_ts = None

    for value, timestamp in success_signal:
        if value == 1.0:
            if not in_success:
                in_success = True
                success_start_ts = timestamp
            else:
                # Check if we've been at 1.0 for at least threshold_ns
                if timestamp - success_start_ts >= threshold_ns:
                    return True
        else:
            in_success = False
            success_start_ts = None

    return False


def success_time(episode: Episode) -> float | None:
    """Return the time (seconds from episode start) when success was achieved (held 1.0 for 0.5s)."""
    if 'stacking_success' not in episode:
        return None
    success_signal = episode['stacking_success']
    if len(success_signal) == 0:
        return None

    threshold_ns = int(0.5 * 1e9)  # 0.5 seconds in nanoseconds
    in_success = False
    success_start_ts = None

    for value, timestamp in success_signal:
        if value == 1.0:
            if not in_success:
                in_success = True
                success_start_ts = timestamp
            else:
                # Check if we've been at 1.0 for at least threshold_ns
                if timestamp - success_start_ts >= threshold_ns:
                    # Return the time when threshold was reached
                    return (timestamp - episode.start_ts) / 1e9
        else:
            in_success = False
            success_start_ts = None

    return None


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
    if 'robot_state.ee_pose' not in episode:
        return None
    signal_values = episode['robot_state.ee_pose'].values()
    result = 0.0
    prev_translation = signal_values[0][:3]
    for ee_pose in signal_values[1:]:
        translation = ee_pose[:3]
        result += np.linalg.norm(translation - prev_translation).item()
        prev_translation = translation
    return result


def units_sim(episode: Episode) -> int:
    """Number of successful stacks (1 if success, 0 otherwise)."""
    return 1 if success(episode) else 0


def uph_sim(episode: Episode) -> float | None:
    """Units per hour for simulation (UPH = units / (duration / 3600))."""
    u = units_sim(episode)
    if u == 0:
        return None
    return u / (episode.duration_ns / 1e9 / 3600)


sim_episodes = base_cfg.transform.override(
    base=base_cfg.local,
    transforms=[
        Group(
            Identity(),
            Derive(
                model=model,
                checkpoint=ckpt,
                max_stacking_success=max_stacking_success,
                success=success,
                success_time=success_time,
                box_distance_progress=box_distance_progress,
                movement=ee_pose_movement,
                units=units_sim,
                uph=uph_sim,
            ),
        )
    ],
)


@cfn.config()
def sim_episodes_table():
    return {
        '__index__': {'label': '#', 'format': '%d'},
        '__duration__': {'label': 'Duration', 'format': '%.2f sec'},
        'checkpoint': {'label': 'CKPT', 'filter': True},
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
        'success_time': {'label': 'Success Time', 'format': '%.1f sec', 'default': '-'},
        'units': {'label': 'Units', 'format': '%d'},
        'uph': {'label': 'UPH', 'format': '%.1f', 'default': '-'},
        'max_stacking_success': {'label': 'Max Success', 'format': '%.2f'},
        'box_distance_progress': {'label': 'Box Progress', 'format': '%.1f%%', 'default': '-'},
        'movement': {'label': 'Movement', 'format': '%.2f'},
    }


@cfn.config()
def sim_checkpoint_table():
    """Grouped table by checkpoint with UPH and MTBF metrics."""

    def group_fn(episodes: list[Episode]):
        count = len(episodes)
        total_duration = sum(ep.duration_ns / 1e9 for ep in episodes)
        successful = [ep for ep in episodes if ep['success']]
        failed = [ep for ep in episodes if not ep['success']]

        successful_count = len(successful)
        failed_count = len(failed)

        # UPH: total units / (total duration in hours)
        total_units = sum(ep['units'] for ep in episodes)
        uph_value = total_units / (total_duration / 3600) if total_duration > 0 else 0

        # MTBF: total duration / number of failures
        mtbf_value = total_duration / failed_count if failed_count > 0 else None

        # Success rate
        success_rate = 100 * successful_count / count if count > 0 else 0

        # Average time to success (for successful episodes only)
        success_times = [ep['success_time'] for ep in successful if ep['success_time'] is not None]
        avg_success_time = np.mean(success_times) if success_times else None

        # Average max stacking success
        max_successes = [ep['max_stacking_success'] for ep in episodes if ep['max_stacking_success'] is not None]
        avg_max_success = np.mean(max_successes) if max_successes else None

        result = {
            'model': episodes[0]['model'],
            'checkpoint': episodes[0]['checkpoint'],
            'count': count,
            'UPH': uph_value,
            'success_rate': success_rate,
            'MTBF': mtbf_value,
            'avg_success_time': avg_success_time,
            'avg_max_success': avg_max_success,
            'failures': failed_count,
        }
        return result

    format_table = {
        'model': {'label': 'Model'},
        'checkpoint': {'label': 'Checkpoint'},
        'count': {'label': 'Runs', 'format': '%d'},
        'UPH': {'label': 'UPH', 'format': '%.1f'},
        'success_rate': {'label': 'Success', 'format': '%.1f%%'},
        'MTBF': {'label': 'MTBF', 'format': '%.1f sec', 'default': '-'},
        'avg_success_time': {'label': 'Avg Time', 'format': '%.1f sec', 'default': '-'},
        'avg_max_success': {'label': 'Avg Max', 'format': '%.2f', 'default': '-'},
        'failures': {'label': 'Failures', 'format': '%d'},
    }

    group_filter_keys = {'model': 'Model', 'checkpoint': 'Checkpoint'}
    return ('model', 'checkpoint'), group_fn, format_table, group_filter_keys


# ========================================================================================
# Pre-configured servers
# ========================================================================================
#
# Sim evaluation:
#   uv run python -m positronic.cfg.eval sim --dataset.base.path=s3://inference/sim_stack_validation/090226/
#
# Real (unified) evaluation:
#   uv run python -m positronic.cfg.eval real --dataset.base.path=s3://inference/real/191225/
# ========================================================================================

server = server_main.override(
    dataset=episodes,
    ep_table_cfg=episodes_table,
    group_tables={'checkpoints': checkpoint_table},
    home_page='checkpoints',
    port=5001,
)

sim_server = server_main.override(
    dataset=sim_episodes,
    ep_table_cfg=sim_episodes_table,
    group_tables={'checkpoints': sim_checkpoint_table},
    home_page='checkpoints',
    port=5001,
)

if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli({'sim': sim_server, 'real': server})
