from collections import defaultdict
from datetime import datetime
from functools import partial

import configuronic as cfn
import numpy as np
import pos3

import positronic.cfg.ds as base_cfg
from pimm.logging import init_logging
from positronic import keys
from positronic.cfg.ds import internal
from positronic.cfg.eval.real import tasks
from positronic.dataset.episode import META_CREATED_TS_NS, Episode
from positronic.dataset.transforms.episode import Derive, FromValue, Group, Identity
from positronic.offboard import keys as offboard_keys
from positronic.policy import keys as policy_keys
from positronic.server.positronic_server import ColumnConfig as C
from positronic.server.positronic_server import GroupTableConfig, RendererConfig, SortConfig
from positronic.server.positronic_server import main as server_main


def task_code(ep: Episode) -> str:
    if 'eval.object' in ep:
        return ep['eval.object']
    match ep[keys.TASK]:
        case tasks.TOWELS_TASK:
            return 'Towels'
        case tasks.SPOONS_TASK:
            return 'Wooden spoons'
        case tasks.SCISSORS_TASK:
            return 'Scissors'
        case tasks.BATTERIES_TASK:
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
    policy_type = ep.get(f'{policy_keys.POLICY_META}.{policy_keys.TYPE}', '')

    if policy_type == 'remote':
        server_type = ep.get(f'{policy_keys.SERVER_META}.{policy_keys.TYPE}', '')
        path_label = _model_label_from_path(
            server_type, ep.get(f'{policy_keys.SERVER_META}.{policy_keys.CHECKPOINT_PATH}', '')
        )
        if path_label:
            return path_label
        return server_type or ''

    if policy_type:
        path_label = _model_label_from_path(
            policy_type, ep.get(f'{policy_keys.POLICY_META}.{policy_keys.CHECKPOINT_PATH}', '')
        )
        if path_label:
            return path_label
        return policy_type

    return ''


def _split_path(path: str) -> list[str]:
    return [p for p in path.strip('/').split('/') if p]


def _ckpt_act(ep: Episode) -> str:
    raw_path = ep[f'{policy_keys.POLICY_META}.{policy_keys.CHECKPOINT_PATH}']
    parts = _split_path(raw_path)
    chkpt_idxs = [i for i, p in enumerate(parts) if p == 'checkpoints']
    if chkpt_idxs:
        idx = chkpt_idxs[-1]
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return raw_path


def _ckpt_remote(ep: Episode) -> str:
    checkpoint_id = ep.get(f'{policy_keys.SERVER_META}.{offboard_keys.CHECKPOINT_ID}', '')
    if checkpoint_id:
        return str(checkpoint_id)
    raw_path = ep.get(f'{policy_keys.SERVER_META}.{policy_keys.CHECKPOINT_PATH}', '')
    if raw_path:
        parts = _split_path(raw_path)
        if parts[-1] == 'pretrained_model' and len(parts) >= 2:
            return parts[-2]
        return parts[-1].removeprefix('checkpoint-')
    return ''


def ckpt(ep: Episode) -> str | None:
    try:
        match ep.get(f'{policy_keys.POLICY_META}.{policy_keys.TYPE}', ''):
            case 'act':
                return _ckpt_act(ep)
            case 'remote':
                return _ckpt_remote(ep)
        return ''
    except Exception:
        return ''


def started(ep: Episode) -> datetime:
    return datetime.fromtimestamp(ep.meta[META_CREATED_TS_NS] / 1e9)


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
    if 'eval.successful_items' in ep:
        items = ep['eval.successful_items']
        if items == 0:
            return None
        return items / (ep.duration_ns / 1e9 / 3600)
    if 'stacking_success' in ep:
        t = success_time(ep)
        if t is None:
            return None
        return 1 / (t / 3600)
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
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)


@cfn.config()
def episodes_table():
    return {
        '__index__': C(label='#', format='%d'),
        '__duration__': C(label='Duration', format='%.1f sec'),
        'task_code': C(label='Task', filter=True),
        'model': C(label='Model', filter=True),
        'checkpoint': C(label='Checkpoint', filter=True),
        'success_bool': C(
            label='Pass',
            renderer=RendererConfig(
                type='badge',
                options={True: {'label': 'Pass', 'variant': 'success'}, False: {'label': 'Fail', 'variant': 'danger'}},
            ),
        ),
        'units_display': C(label='Units'),
        'uph': C(label='UPH', format='%.1f', default='-'),
        'success_rate': C(label='Success', format='%.1f%%'),
        'started': C(label='Started', format='%Y-%m-%d %H:%M:%S'),
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
        'model': C(label='Model'),
        'checkpoint': C(label='Checkpoint'),
        'count': C(label='Runs', format='%d'),
        'UPH': C(label='UPH', format='%.1f'),
        'success_rate': C(label='Success', format='%.1f%%'),
        'MTBF': C(label='MTBF', format='%.1f sec', default='-'),
        'failures': C(label='Failures', format='%d'),
    }

    return GroupTableConfig(
        group_keys=('model', 'checkpoint'),
        group_fn=group_fn,
        format_table=format_table,
        group_filter_keys={'checkpoint': 'Checkpoint', 'model': 'Model'},
    )


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


def success(episode: Episode, score_threshold: float = 0.95) -> bool:
    """Check if stacking_success reached score_threshold and stayed there for at least 0.5 seconds."""
    if 'stacking_success' not in episode:
        return False
    success_signal = episode['stacking_success']
    if len(success_signal) == 0:
        return False

    threshold_ns = int(0.25 * 1e9)  # 0.25 seconds in nanoseconds

    in_success = False
    success_start_ts = None

    for value, timestamp in success_signal:
        if value >= score_threshold:
            if not in_success:
                in_success = True
                success_start_ts = timestamp
            elif timestamp - success_start_ts >= threshold_ns:
                return True
        else:
            in_success = False
            success_start_ts = None

    return False


def success_time(episode: Episode, score_threshold: float = 0.95) -> float | None:
    """Return the time (seconds from episode start) when success was achieved (held score_threshold for 0.25s)."""
    if 'stacking_success' not in episode:
        return None
    success_signal = episode['stacking_success']
    if len(success_signal) == 0:
        return None

    threshold_ns = int(0.25 * 1e9)  # 0.25 seconds in nanoseconds
    in_success = False
    success_start_ts = None

    for value, timestamp in success_signal:
        if value >= score_threshold:
            if not in_success:
                in_success = True
                success_start_ts = timestamp
            elif timestamp - success_start_ts >= threshold_ns:
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
    if keys.EE_POSE not in episode:
        return None
    signal_values = episode[keys.EE_POSE].values()
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
    """Units per hour based on success_time (not full episode duration)."""
    t = success_time(episode)
    if t is None:
        return None
    return 1 / (t / 3600)


stacking_episodes = base_cfg.transform.override(
    base=base_cfg.local_all,
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
        ),
        internal.SIM_ROBOT_TRANSFORM,
    ],
)


@cfn.config()
def stacking_episodes_table():
    return {
        '__index__': C(label='#', format='%d'),
        '__duration__': C(label='Duration', format='%.2f sec'),
        'checkpoint': C(label='CKPT', filter=True),
        'success': C(
            label='Pass',
            renderer=RendererConfig(
                type='badge',
                options={True: {'label': 'Pass', 'variant': 'success'}, False: {'label': 'Fail', 'variant': 'danger'}},
            ),
        ),
        'success_time': C(label='Success Time', format='%.1f sec', default='-'),
        'units': C(label='Units', format='%d'),
        'uph': C(label='UPH', format='%.1f', default='-'),
        'max_stacking_success': C(label='Max Success', format='%.2f'),
        'box_distance_progress': C(label='Box Progress', format='%.1f%%', default='-'),
        'movement': C(label='Movement', format='%.2f'),
    }


def _effective_duration(key: str, ep: Episode) -> float:
    t = ep.get(key)
    return t if t is not None else ep.duration_ns / 1e9


@cfn.config()
def stacking_checkpoint_table():
    """Grouped table by checkpoint with UPH and MTBF metrics."""

    def group_fn(episodes: list[Episode]):
        count = len(episodes)
        total_duration = sum(_effective_duration('success_time', ep) for ep in episodes)
        successful = [ep for ep in episodes if ep['success']]
        failed = [ep for ep in episodes if not ep['success']]

        successful_count = len(successful)
        failed_count = len(failed)

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
        'model': C(label='Model'),
        'checkpoint': C(label='Checkpoint'),
        'count': C(label='Runs', format='%d'),
        'UPH': C(label='UPH', format='%.1f'),
        'success_rate': C(label='Success', format='%.1f%%'),
        'MTBF': C(label='MTBF', format='%.1f sec', default='-'),
        'avg_success_time': C(label='Avg Time', format='%.1f sec', default='-'),
        'avg_max_success': C(label='Avg Max', format='%.2f', default='-'),
        'failures': C(label='Failures', format='%d'),
    }

    return GroupTableConfig(
        group_keys=('model', 'checkpoint'),
        group_fn=group_fn,
        format_table=format_table,
        group_filter_keys={'model': 'Model', 'checkpoint': 'Checkpoint'},
    )


# ========================================================================================
# Pick-and-place item counting
# ========================================================================================

FIXED_ITEM_COUNTS = {tasks.SCISSORS_TASK: 10, tasks.BATTERIES_TASK: 8}


def calculate_units(episode: Episode) -> int:  # noqa: C901
    """Estimates the number of pick-and-place operations. Vibe-coded heuristic."""
    if episode[keys.TASK] in FIXED_ITEM_COUNTS:
        return FIXED_ITEM_COUNTS[episode[keys.TASK]]

    if keys.TARGET_GRIP in episode.signals:
        grip_sig = episode.signals[keys.TARGET_GRIP]
    elif keys.GRIP in episode.signals:
        grip_sig = episode.signals[keys.GRIP]
    else:
        return 0

    if keys.EE_POSE not in episode.signals:
        return 0

    pose_sig = episode.signals[keys.EE_POSE]

    # Sample signals at 10Hz to reduce noise and computation
    times = np.arange(episode.start_ts, episode.last_ts, int(1e8))
    if len(times) == 0:
        return 0

    grip_vals = np.array([v for v, _ in grip_sig.time[times]])
    pose_vals = np.array([v for v, _ in pose_sig.time[times]])
    x_vals, y_vals, z_vals = pose_vals[:, 0], pose_vals[:, 1], pose_vals[:, 2]

    threshold = (grip_vals.max() + grip_vals.min()) / 2
    units = 0
    state = 'CLOSED' if grip_vals[0] > threshold else 'OPEN'
    min_z_holding = np.inf
    max_z_holding = -np.inf
    pick_x, pick_y = 0.0, 0.0

    for i in range(1, len(grip_vals)):
        val = grip_vals[i]
        x, y, z = x_vals[i], y_vals[i], z_vals[i]
        is_closed = val > threshold

        if state == 'OPEN':
            if is_closed:
                state = 'CLOSED'
                min_z_holding = z
                max_z_holding = z
                pick_x, pick_y = x, y
        elif state == 'CLOSED':
            max_z_holding = max(max_z_holding, z)
            min_z_holding = min(min_z_holding, z)
            if not is_closed:
                state = 'OPEN'
                amplitude = max_z_holding - min_z_holding
                dx, dy = x - pick_x, y - pick_y
                dist = np.sqrt(dx * dx + dy * dy)
                if amplitude > 0.05 and dist > 0.15:
                    units += 1

    return units


# ========================================================================================
# PhAIL benchmark (real robot bin-to-bin picking evaluation)
# ========================================================================================

HUMAN_MODEL = 'Human'
TELEOP_MODEL = 'Robot teleoperated by Human'

PHAIL_MODEL_DISPLAY = {
    'openpi': 'Physical Intelligence Open \u03c0\u2080.\u2085',
    'groot': 'NVIDIA GR00T N1.6',
    'act': 'Action Chunking Transformer',
    'smolvla': 'Hugging Face SmolVLA',
    'dreamzero': 'NVIDIA DreamZero',
    'human': HUMAN_MODEL,
    'teleop': TELEOP_MODEL,
}


PHAIL_OUTCOME_BADGE = RendererConfig(
    type='badge',
    options={
        'Pass': {'label': 'Pass', 'variant': 'success'},
        'Fail': {'label': 'Fail', 'variant': 'danger'},
        'Safety': {'label': 'Safety', 'variant': 'warning'},
    },
)


def phail_model(ep: Episode) -> str:
    return PHAIL_MODEL_DISPLAY.get(ep.get(f'{policy_keys.SERVER_META}.{policy_keys.TYPE}', ''), '')


def phail_status(ep: Episode) -> str:
    outcome = ep.get('eval.outcome', '')
    if outcome == 'Success':
        return 'Pass'
    if outcome == 'Safety':
        return 'Safety'
    return 'Fail'


def phail_completion(ep: Episode) -> float:
    s = ep.get('eval.successful_items', 0)
    t = ep.get('eval.total_items', 0)
    return 100 * s / t if t else 0.0


def phail_units(ep: Episode) -> str:
    return f'{ep.get("eval.successful_items", 0)}/{ep.get("eval.total_items", 0)}'


def phail_uph(ep: Episode) -> float | None:
    items = ep.get('eval.successful_items', 0)
    if not items:
        return None
    duration = ep.get('eval.duration') or ep.duration_ns / 1e9
    if not duration:
        return None
    return items / (duration / 3600)


def phail_variant(ep: Episode) -> str:
    exp = ep.get(f'{policy_keys.SERVER_META}.{policy_keys.EXPERIMENT_NAME}', '')
    ckpt = ep.get(f'{policy_keys.SERVER_META}.{offboard_keys.CHECKPOINT_ID}', '')
    if exp and ckpt:
        return f'{exp}:{ckpt}'
    if exp:
        return exp
    return ''


def _phail_task_label(ep: Episode) -> str:
    obj = task_code(ep)
    return f'Pick-and-place: {obj}' if obj else ''


_phail_derives = Derive(
    model=phail_model,
    variant=phail_variant,
    status=phail_status,
    equipment=FromValue('DROID'),
    units=phail_units,
    uph=phail_uph,
    completion=phail_completion,
    started=started,
)

phail_inference = base_cfg.transform.override(
    base=base_cfg.local_all.override(path='s3://inference/phail_final/'),
    transforms=[
        Group(
            Identity(
                remove=[
                    'robot_commands.reset',
                    'robot_command.reset',
                    'eval.object',
                    f'{policy_keys.POLICY_META}.{offboard_keys.PORT}',
                    f'{policy_keys.POLICY_META}.{offboard_keys.HOST}',
                    f'{policy_keys.SERVER_META}.{offboard_keys.CHECKPOINT_ID}',
                    f'{policy_keys.SERVER_META}.{policy_keys.CONFIG_NAME}',
                    f'{policy_keys.SERVER_META}.{policy_keys.EXPERIMENT_NAME}',
                    f'{policy_keys.SERVER_META}.{policy_keys.TYPE}',
                    f'{policy_keys.POLICY_META}.{policy_keys.TYPE}',
                ]
            ),
            # NOTE: _phail_derives reads inference.policy.server.type from the original episode,
            # before Identity(remove=...) strips it. Group applies all transforms to the same input.
            _phail_derives,
            Derive(**{'eval.object': _phail_task_label}),
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)


# Shared derives for baseline datasets (human and teleop) where all episodes are successful.
def _baseline_uph(ep: Episode, items: int) -> float | None:
    if not items:
        return None
    return items / (ep.duration_ns / 1e9 / 3600)


_PHAIL_BASELINE = {
    'status': FromValue('Pass'),
    'variant': FromValue(''),
    'equipment': FromValue('DROID'),
    'eval.object': _phail_task_label,
    'eval.outcome': FromValue('Success'),
    'completion': FromValue(100.0),
    'started': started,
}

# Human baseline: 40 episodes from s3://raw/human (10 per object, 8 items each, all success).
phail_human = base_cfg.transform.override(
    base=base_cfg.local_all.override(path='s3://raw/human'),
    transforms=[
        Group(
            Identity(),
            Derive(**{
                **_PHAIL_BASELINE,
                'model': FromValue(HUMAN_MODEL),
                'eval.successful_items': FromValue(8),
                'eval.total_items': FromValue(8),
                'units': FromValue('8/8'),
                'uph': partial(_baseline_uph, items=8),
            }),
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)

# DROID teleoperation data: robot controlled by human via VR controller.
# Two-step transform: first compute item counts from grip signals, then derive phail fields.
_teleop_with_items = base_cfg.transform.override(
    base=internal.droid_clean, transforms=[Group(Derive(item_count=calculate_units), Identity())]
)

phail_teleop = base_cfg.transform.override(
    base=_teleop_with_items,
    transforms=[
        Group(
            Identity(),
            Derive(**{
                **_PHAIL_BASELINE,
                'model': FromValue(TELEOP_MODEL),
                'eval.successful_items': lambda ep: ep['item_count'],
                'eval.total_items': lambda ep: ep['item_count'],
                'units': lambda ep: f'{ep["item_count"]}/{ep["item_count"]}',
                'uph': lambda ep: _baseline_uph(ep, ep['item_count']),
            }),
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)

phail_episodes = base_cfg.concat_ds.override(datasets=[phail_inference, phail_human, phail_teleop])


# =========================================================================================
# Release configs: source-of-truth fields + robot metadata only.
# Display fields (model display name, UPH, completion, started) are server-side transforms.
# =========================================================================================


def _raw_model(ep: Episode) -> str:
    return ep.get(f'{policy_keys.SERVER_META}.{policy_keys.TYPE}', '')


phail_inference_release = base_cfg.transform.override(
    base=base_cfg.concat_ds.override(
        datasets=[
            base_cfg.local_all.override(path='s3://inference/phail_final/'),
            base_cfg.local_all.override(path='s3://inference/phail_act_groot/'),
        ]
    ),
    transforms=[
        Group(
            Identity(
                remove=[
                    'robot_commands.reset',
                    'robot_command.reset',
                    f'{policy_keys.POLICY_META}.{offboard_keys.PORT}',
                    f'{policy_keys.POLICY_META}.{offboard_keys.HOST}',
                    f'{policy_keys.POLICY_META}.{policy_keys.TYPE}',
                ]
            ),
            Derive(model=_raw_model, variant=phail_variant),
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)

PROD_VARIANTS = {'groot': '270226-ee_rot6d_rel:150000'}

TRAINED_OBJECTS = ('Towels', 'Wooden spoons', 'Scissors', 'Batteries')


def _prod_predicate(ep):
    model = ep.get('model', '')
    if model not in PROD_VARIANTS:
        return True
    return ep.get('variant', '') == PROD_VARIANTS[model]


def _trained_objects_predicate(ep):
    return ep.get('eval.object') in TRAINED_OBJECTS


# AUDIT-CORRECTED EPISODES — manual edits to static.json on both private and
# public S3 (audit captured wrong eval.* fields; fixed in place rather than
# adding a transform). Re-running release_phail.py inference will OVERWRITE
# these corrections unless the same edit is re-applied. Keep this list:
#   - public 000000000000/000000000338 (private 100326/000000000000/000000000021):
#     eval.successful_items 0 → 2 (audit missed 2 successful placements; ACT/Scissors).
#   - public 000000000000/000000000285 (private 050326/000000000000/000000000015):
#     eval.object 'Towels' → 'Wooden spoons' (operator selected wrong task at record time;
#     content is wooden spoons; task field intentionally left as recorded).
phail_inference_prod_v1_0 = base_cfg.filter_ds.override(
    dataset=phail_inference_release, predicate=lambda ep: _prod_predicate(ep) and _trained_objects_predicate(ep)
)

# TELEOP HEURISTIC OVER-COUNTS — calculate_units (line ~486) returns
# FIXED_ITEM_COUNTS[BATTERIES_TASK]=8 for every Batteries teleop episode, but
# manual review found episodes that actually contain only 7 batteries. The
# heuristic does not look inside the video; it just trusts the task label.
# eval.successful_items / eval.total_items end up at 8/8 ('Success') instead
# of 7/7, so totals across the teleop split are inflated. Documented for now,
# not yet patched — see EPISODE_ITEM_COUNT_OVERRIDES if/when added.
#   - public training 000000000000/000000000380 (private raw/droid/batteries/310126/000000000000/000000000001):
#     manual count = 7 batteries, heuristic = 8.
#   - public training 000000000000/000000000408 (private raw/droid/batteries/310126/000000000000/000000000037):
#     manual count = 7 batteries, heuristic = 8.
phail_teleop_release = base_cfg.transform.override(
    base=_teleop_with_items,
    transforms=[
        Group(
            Identity(),
            Derive(**{
                'model': FromValue('teleop'),
                'variant': FromValue(''),
                'eval.object': task_code,
                'eval.outcome': FromValue('Success'),
                'eval.successful_items': lambda ep: ep['item_count'],
                'eval.total_items': lambda ep: ep['item_count'],
            }),
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)

phail_human_release = base_cfg.transform.override(
    base=base_cfg.local_all.override(path='s3://raw/human'),
    transforms=[
        Group(
            Identity(),
            Derive(**{
                'model': FromValue('human'),
                'variant': FromValue(''),
                'eval.object': task_code,
                'eval.outcome': FromValue('Success'),
                'eval.successful_items': FromValue(8),
                'eval.total_items': FromValue(8),
            }),
        ),
        internal.REAL_ROBOT_TRANSFORM,
    ],
)


@cfn.config()
def phail_episodes_table():
    return {
        '__index__': C(label='#', format='%d'),
        'model': C(label='Model', filter=True),
        'variant': C(label='Variant', filter=True),
        'eval.object': C(label='Task', filter=True),
        'started': C(label='Started', format='%Y-%m-%d %H:%M'),
        'units': C(label='Units', align='right'),
        'uph': C(label='UPH', subtitle='Units Per Hour', format='%.1f', default='-', align='right'),
        'completion': C(label='Done %', subtitle='Completed / Total Operations', format='%.1f%%', align='right'),
        'status': C(label='Status', renderer=PHAIL_OUTCOME_BADGE, align='center'),
    }


@cfn.config()
def phail_leaderboard():
    def group_fn(episodes: list[Episode]):
        count = len(episodes)
        total_duration = sum(_effective_duration('eval.duration', ep) for ep in episodes)
        failed_count = sum(1 for ep in episodes if ep['status'] != 'Pass')

        # Per-object aggregation: compute UPH and completion per object type, then average equally.
        by_object: dict[str, list[Episode]] = defaultdict(list)
        for ep in episodes:
            by_object[ep['eval.object']].append(ep)

        object_uphs = []
        object_completions = []
        for obj_eps in by_object.values():
            obj_duration = sum(_effective_duration('eval.duration', ep) for ep in obj_eps)
            obj_items = sum(ep['eval.successful_items'] for ep in obj_eps)
            obj_possible = sum(ep['eval.total_items'] for ep in obj_eps)
            if obj_duration > 0:
                object_uphs.append(obj_items / (obj_duration / 3600))
            if obj_possible > 0:
                object_completions.append(100 * obj_items / obj_possible)

        return {
            'model': episodes[0]['model'],
            'variant': episodes[0].get('variant', ''),
            'count': count,
            'UPH': sum(object_uphs) / len(object_uphs) if object_uphs else None,
            'completion': sum(object_completions) / len(object_completions) if object_completions else None,
            'MTBF': total_duration / failed_count / 60 if failed_count > 0 else None,
        }

    format_table = {
        'model': C(label='Model', filter=True),
        'variant': C(label='Variant'),
        'count': C(label='Runs', format='%d', align='right', sortable=False),
        'UPH': C(label='UPH', subtitle='Units Per Hour', format='%.1f', align='right'),
        'completion': C(label='Done %', subtitle='Completed / Total Operations', format='%.1f%%', align='right'),
        'MTBF': C(
            label='MTBF/A', subtitle='Mean Time Between Failures/Assists', format='%.1f min', default='-', align='right'
        ),
    }

    return GroupTableConfig(
        group_keys=('model', 'variant'),
        group_fn=group_fn,
        format_table=format_table,
        group_filter_keys={'equipment': 'Equipment', 'eval.object': 'Task'},
        default_sort=SortConfig(column='UPH'),
    )


# ========================================================================================
# Pre-configured servers
# ========================================================================================
#
# Cube-stacking evaluation:
#   uv run --locked python -m positronic.cfg.analysis stacking --dataset.base.path=s3://inference/sim_stack_validation/090226/
#
# Real (unified) evaluation:
#   uv run --locked python -m positronic.cfg.analysis real --dataset.base.path=s3://inference/real/191225/
#
# PhAIL benchmark:
#   uv run --locked python -m positronic.cfg.analysis phail --dataset.datasets.0.base.path=s3://inference/phail_final/
# ========================================================================================

server = server_main.override(
    dataset=episodes,
    ep_table_cfg=episodes_table,
    group_tables={'checkpoints': checkpoint_table},
    home_page='checkpoints',
    port=5001,
)

stacking_server = server_main.override(
    dataset=stacking_episodes,
    ep_table_cfg=stacking_episodes_table,
    group_tables={'checkpoints': stacking_checkpoint_table},
    home_page='checkpoints',
    port=5001,
)

phail_server = server_main.override(
    dataset=phail_episodes,
    ep_table_cfg=phail_episodes_table,
    group_tables={'leaderboard': phail_leaderboard},
    home_page='leaderboard',
    port=5001,
)

if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli({'stacking': stacking_server, 'real': server, 'phail': phail_server})
