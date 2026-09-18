"""Record observations and camera hashes from a MolmoSpaces hold rollout for parity.py.

Runs in the MolmoSpaces interpreter. It shares the DROID configuration with the server
and reads robot state directly from MolmoSpaces.
"""

import argparse
import hashlib
from pathlib import Path

# env.py installs the CGL stub before loading MolmoSpaces; import it before any other molmo_spaces import.
import env  # noqa: E402
import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import parity_record  # noqa: E402 -- the record's field names, on PYTHONPATH beside this file

from molmo_spaces.evaluation.benchmark_schema import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    load_all_episodes,
)
from molmo_spaces.evaluation.eval_main import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    determine_task_horizon,
)
from molmo_spaces.tasks.json_eval_task_sampler import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    JsonEvalTaskSampler,
)

# Closed finger position from upstream pi_policy.py, kept independent of mapping for the parity check.
_GRIPPER_QPOS_CLOSED = 0.824033


def _observe(robot_view, env_obs: dict, camera_names: list[str]) -> dict:
    """Joint state, grasp-site world pose, gripper closure and camera images from MolmoSpaces."""
    arm = robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
    eef_world = np.asarray(arm.leaf_frame_to_world, dtype=np.float64)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(eef_world[:3, :3].reshape(9)))
    qpos = env_obs[mapping.MOLMO_OBS_QPOS][mapping.MOLMO_GRIPPER_GROUP]
    grip = np.clip(qpos[0] / _GRIPPER_QPOS_CLOSED, 0.0, 1.0)
    return {
        mapping.OBS_JOINT_POS: np.asarray(arm.joint_pos, dtype=np.float32),
        mapping.OBS_JOINT_VEL: np.asarray(arm.joint_vel, dtype=np.float32),
        mapping.OBS_EEF_POS: eef_world[:3, 3].astype(np.float32),
        mapping.OBS_EEF_QUAT: quat.astype(np.float32),
        mapping.OBS_GRIP: np.float32(grip),
        **{name: np.ascontiguousarray(env_obs[name]) for name in camera_names},
    }


def _run(benchmark_dir: Path, episode_index: int, seed: int, max_steps: int, out_path: Path) -> None:
    episodes = load_all_episodes(benchmark_dir)
    episode = episodes[episode_index]
    cfg = env._DroidEvalConfig()
    cfg.seed = seed
    native_horizon = determine_task_horizon([episode], None, cfg.policy_dt_ms)
    cfg.task_horizon = native_horizon
    sampler = JsonEvalTaskSampler(cfg, episode)
    task = sampler.sample_task(house_index=episode.house_index)
    robot_view = task.env.current_robot.robot_view

    obs, _info = task.reset()
    camera_names = [k for k, v in obs[0].items() if mapping.is_rgb_frame(v)]
    fields: dict[str, list] = {
        k: []
        for k in (
            mapping.OBS_JOINT_POS,
            mapping.OBS_JOINT_VEL,
            mapping.OBS_EEF_POS,
            mapping.OBS_EEF_QUAT,
            mapping.OBS_GRIP,
        )
    }
    cam_hashes: dict[str, list[str]] = {name: [] for name in camera_names}

    def record(env_obs: dict) -> None:
        payload = _observe(robot_view, env_obs, camera_names)
        for key in fields:
            fields[key].append(payload[key])
        for name in camera_names:
            cam_hashes[name].append(hashlib.sha256(payload[name].tobytes()).hexdigest())

    record(obs[0])
    step, success = 0, False
    while not bool(task.is_done()) and step < max_steps:
        measured_q = np.asarray(robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).joint_pos, dtype=np.float32)
        action = {mapping.MOLMO_ARM_GROUP: measured_q, mapping.MOLMO_GRIPPER_GROUP: np.array([0.0], dtype=np.float32)}
        obs, _reward, _term, _trunc, _infos = task.step(action)
        step += 1
        record(obs[0])
        success = bool(task.judge_success())
        if success:
            break
    sampler.close()

    recorded: dict = {key: np.stack(values) for key, values in fields.items()}
    recorded.update({f'{parity_record.CAM_HASH_PREFIX}{name}': np.array(cam_hashes[name]) for name in camera_names})
    recorded[parity_record.CAMERA_NAMES] = np.array(camera_names)
    recorded[parity_record.HORIZON_STEPS] = native_horizon
    recorded[parity_record.TERMINATION_STEP] = step
    recorded[parity_record.FINAL_SUCCESS] = success
    # NumPy's typing treats this dict as a possible allow_pickle argument.
    np.savez(out_path, **recorded)  # pyright: ignore[reportArgumentType]


def main() -> None:
    parser = argparse.ArgumentParser(description='Native-drive MolmoSpaces reference for the parity test.')
    parser.add_argument(parity_record.OPT_BENCHMARK_DIR, required=True)
    parser.add_argument(parity_record.OPT_EPISODE_INDEX, type=int, default=0)
    parser.add_argument(parity_record.OPT_SEED, type=int, required=True)
    parser.add_argument(parity_record.OPT_MAX_STEPS, type=int, required=True)
    parser.add_argument(parity_record.OPT_OUT, required=True, help='npz path for the recorded native rollout')
    args = parser.parse_args()
    _run(Path(args.benchmark_dir), args.episode_index, args.seed, args.max_steps, Path(args.out))


if __name__ == '__main__':
    main()
