# With these configurations, you can call to_lerobot in the following way:
# ```bash
# uv run python -m positronic.training.to_lerobot convert \
#     --output_dir=/tmp/lr_test/ \
#     --dataset=@pint.cfg.ds.droid_openpi_ft \
#     --dataset.base.root=../datasets/droid/
# ```

from datetime import datetime

import configuronic as cfn
import numpy as np
import pos3

from positronic.dataset import Episode, Signal
from positronic.dataset.local_dataset import load_all_datasets
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import Concat, Derive, FromValue, Group, Identity, Rename

from . import dataset, policy


@cfn.config(path='s3://raw/droid/')
def droid_ds(path):
    root = pos3.download(path)

    towels = load_all_datasets(root / 'towels')
    towels = TransformedDataset(
        towels,
        Group(
            Derive(
                task=FromValue(
                    'Pick all the towels one by one from transparent tote and place them into the large grey tote.'
                )
            ),
            Identity(),
        ),
    )
    spoons = load_all_datasets(root / 'spoons')
    spoons = TransformedDataset(
        spoons,
        Group(
            Derive(
                task=FromValue(
                    'Pick all the wooden spoons one by one from transparent tote '
                    'and place them into the large grey tote.'
                )
            ),
            Identity(),
        ),
    )
    scissors = load_all_datasets(root / 'scisors')
    scissors = TransformedDataset(
        scissors,
        Group(
            Derive(
                task=FromValue(
                    'Pick all the scissors one by one from transparent tote and place them into the large grey tote.'
                )
            ),
            Identity(),
        ),
    )
    return towels + spoons + scissors


old_to_new = dataset.group.override(
    transforms=[
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
        Identity('grip', 'target_grip', 'mjSTATE_FULLPHYSICS', 'mjSTATE_INTEGRATION', 'mjSTATE_WARMSTART'),
    ]
)

legacy_sim_raw = dataset.local.override(path='s3://raw/sim-cubes/luzan/')
legacy_sim = dataset.transform.override(base=legacy_sim_raw, transforms=[old_to_new])

pnp_sim_raw = dataset.local.override(path='s3://raw/sim_pnp/')
pnp_sim = dataset.transform.override(
    base=pnp_sim_raw,
    transforms=[
        dataset.group.override(
            transforms=[
                Derive(task=FromValue('Pick up objects from the red tote and place them in the green tote.')),
                Rename(**{'image.exterior': 'image.back_view'}),
                Identity(),
            ]
        )
    ],
)

droid_openpi_ft = dataset.encoded.override(
    base=droid_ds, observation=policy.observation.eepose, action=policy.action.absolute_position
)
sim_stack_openpi_ft = dataset.encoded.override(
    base=legacy_sim, observation=policy.observation.eepose, action=policy.action.absolute_position
)
sim_pnp_openpi_ft = dataset.encoded.override(
    base=pnp_sim, observation=policy.observation.eepose, action=policy.action.absolute_position
)
full_openpi_ft = dataset.encoded.override(
    base=dataset.concat_ds.override(datasets=[droid_ds, legacy_sim, pnp_sim]),
    observation=policy.observation.eepose,
    action=policy.action.absolute_position,
)

droid_groot_ft = dataset.encoded.override(
    base=droid_ds, observation=policy.observation.groot_ee_absolute, action=policy.action.absolute_position
)
sim_stack_groot_ft = dataset.encoded.override(
    base=legacy_sim, observation=policy.observation.groot_ee_absolute, action=policy.action.absolute_position
)
sim_pnp_groot_ft = dataset.encoded.override(
    base=pnp_sim, observation=policy.observation.groot_ee_absolute, action=policy.action.absolute_position
)
full_groot_ft = dataset.encoded.override(
    base=dataset.concat_ds.override(datasets=[droid_ds, legacy_sim, pnp_sim]),
    observation=policy.observation.groot_ee_absolute,
    action=policy.action.absolute_position,
)


def max_stacking_success(episode: Episode) -> float:
    if 'stacking_success' not in episode:
        return 0
    success_signal: Signal[float] = episode['stacking_success']
    if len(success_signal) == 0:
        return 0
    return max(v for v, _ in success_signal)


def success(episode: Episode) -> bool:
    if 'stacking_success' not in episode:
        return False
    success_signal: Signal[float] = episode['stacking_success']
    if len(success_signal) == 0:
        return False

    two_sec = 2 * 1e9
    end: Signal[float] = success_signal.time[success_signal.last_ts - two_sec :]
    return len(end) > 0 and all(v == 1.0 for v, _ in end)


def success_time(episode: Episode) -> float:
    if 'stacking_success' not in episode:
        return np.inf
    success_signal: Signal[float] = episode['stacking_success']
    if len(success_signal) == 0 or success_signal[-1][0] != 1.0:
        return np.inf

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


@cfn.config()
def sim_metrics():
    return Group(
        Derive(success=success, success_time=success_time, max_stacking_success=max_stacking_success), Identity()
    )


act_latest = policy.policy.act_absolute.override(**{
    'base.checkpoints_dir': 's3://checkpoints/full_ft/act/021225/',
    'base.n_action_steps': 15,
})
act_q_latest = policy.policy.act_absolute.override(**{
    'base.checkpoints_dir': 's3://checkpoints/full_ft_q/act/031225/',
    'observation': policy.observation.eepose_q,
    'base.n_action_steps': 15,
})
openpi = policy.policy.openpi_positronic.copy()
openpi_q = openpi.override(observation=policy.observation.openpi_eeq)

groot = policy.policy.groot_ee.copy()
groot_q = policy.policy.groot_ee_q.copy()

sample = policy.policy.sample.copy()


def calculate_units(episode: Episode) -> int:
    """Estimates the number of pick-and-place operations.

    This function is vibe-coded with Gemini 3 Pro (High). It works fine as a heuristic.
    """
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


ft_ds = dataset.transform.override(
    base=dataset.transform.override(
        base=droid_ds, transforms=[dataset.group.override(transforms=[Identity(), Derive(units=calculate_units)])]
    ),
    transforms=[
        dataset.group.override(
            transforms=[
                Identity(),
                Derive(started=lambda ep: datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9)),
                Derive(uph=uph),
            ]
        )
    ],
    extra_meta={'name': 'PhAIL Finetuning Dataset'},
)


@cfn.config()
def ft_ep_table():
    return {
        '__index__': {'label': '#', 'format': '%d'},
        '__duration__': {'label': 'Duration', 'format': '%.2f sec'},
        'task': {'label': 'Task', 'filter': True},
        'units': {'label': 'Units'},
        'uph': {'label': 'UPH', 'format': '%.1f'},
        'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M'},
    }


@cfn.config()
def ft_by_task():
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
