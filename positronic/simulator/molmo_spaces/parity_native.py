"""Native-drive reference for the MolmoSpaces parity test — MolmoSpaces' own stack, no positronic.

Runs in MolmoSpaces' venv, flat off ``PYTHONPATH`` like ``env.py`` (positronic-free: ``molmo_spaces`` + this
package's ``mapping``/``env`` modules). It drives one benchmark episode through MolmoSpaces' native rollout —
``JsonEvalTaskSampler`` -> ``reset``/``step``/``is_done``/``judge_success``, the sequence ``JsonEvalRunner`` runs —
holding the arm every step, and records the per-step raw sim state, per-camera frame hashes, and where the native
horizon terminates the episode. ``parity.py`` drives the *same* episode through the positronic env-server path and
asserts byte-identical outcomes against this reference.

The reference derives what it compares from MolmoSpaces, not from the integration: the horizon from upstream's
own ``determine_task_horizon``, and each observation field read off the robot view here — the gripper closure
through upstream's own normalisation (``policy/learned_policy/pi_policy.py``). So an integration that resolves
the horizon or maps an observation differently from MolmoSpaces shows up as a parity failure rather than being
reproduced on both sides.

One thing is deliberately shared and cannot be otherwise: ``env._DroidPickEvalConfig``, the eval config both
rollouts build the task from. It is the task definition — two rollouts built from different configs are not
comparable at all — and MolmoSpaces has no per-benchmark config to derive it from, the config being the
operator's own input. ``import env`` also runs the GL backend selection and the CGL stub before any
``molmo_spaces`` import; that decides whether the process renders at all, never what it records.

Needs ``MLSPACES_ASSETS_DIR`` + a GL backend, like ``e2e.py``. Invoked by ``parity.py``; not run by hand.
"""

# ``env`` and ``molmo_spaces`` resolve only inside MolmoSpaces' own venv (the flat ``env`` module off PYTHONPATH,
# and the molmo stack), which pyright checks against positronic's deps and cannot see. This module imports no
# positronic packages, so missing-import errors here are exclusively those foreign imports — suppress just that
# category file-wide; every other type check stays active.
# pyright: reportMissingImports=false

import argparse
import hashlib
from pathlib import Path

# env.py (imported flat off PYTHONPATH, like mapping/server) sets MUJOCO_GL and installs the CGL stub at import,
# GL-safely pulling in the molmo_spaces stack — so import it before any other molmo_spaces import.
import env  # noqa: E402
import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from molmo_spaces.evaluation.benchmark_schema import load_all_episodes  # noqa: E402
from molmo_spaces.evaluation.eval_main import determine_task_horizon  # noqa: E402
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler  # noqa: E402

# The Robotiq finger qpos the DROID observation's closure is normalised against, as MolmoSpaces' own policies
# read it (``np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)``, pi_policy.py:126).
_GRIPPER_QPOS_CLOSED = 0.824033


def _is_rgb_frame(value) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8


def _observe(robot_view, env_obs: dict, camera_names: list[str]) -> dict:
    """One frame's compared values, read off MolmoSpaces directly: measured joints, the grasp-site world pose,
    the gripper closure, and each camera's frame."""
    arm = robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
    eef_world = np.asarray(arm.leaf_frame_to_world, dtype=np.float64)
    quat = np.zeros(4)  # wxyz
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(eef_world[:3, :3].reshape(9)))  # pyright: ignore[reportAttributeAccessIssue]
    grip = np.clip(env_obs['qpos'][mapping.MOLMO_GRIPPER_GROUP][0] / _GRIPPER_QPOS_CLOSED, 0.0, 1.0)
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
    cfg = env._DroidPickEvalConfig()
    cfg.seed = seed
    native_horizon = determine_task_horizon([episode], None, cfg.policy_dt_ms)
    cfg.task_horizon = native_horizon
    sampler = JsonEvalTaskSampler(cfg, episode)
    task = sampler.sample_task(house_index=episode.house_index)
    robot_view = task.env.current_robot.robot_view

    obs, _info = task.reset()
    camera_names = [k for k, v in obs[0].items() if _is_rgb_frame(v)]
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
    # The native rollout: hold the measured joints (the gripper open) and let the sim run until its own is_done —
    # is_terminal or horizon expiry. A hold never succeeds, so this drives the horizon case. max_steps bounds a sim
    # that never terminates (a wrong horizon); the caller asserts termination lands below it.
    while not bool(task.is_done()) and step < max_steps:
        measured_q = np.asarray(robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).joint_pos, dtype=np.float32)
        action = {mapping.MOLMO_ARM_GROUP: measured_q, mapping.MOLMO_GRIPPER_GROUP: np.array([0.0], dtype=np.float32)}
        obs, _reward, _term, _trunc, _infos = task.step(action)
        step += 1
        record(obs[0])
        success = bool(task.judge_success())
        if success:  # end-on-success, matching env.py's step (a hold never reaches it)
            break
    sampler.close()

    np.savez(
        out_path,
        joint_pos=np.stack(fields[mapping.OBS_JOINT_POS]),
        joint_vel=np.stack(fields[mapping.OBS_JOINT_VEL]),
        eef_pos=np.stack(fields[mapping.OBS_EEF_POS]),
        eef_quat=np.stack(fields[mapping.OBS_EEF_QUAT]),
        grip=np.array(fields[mapping.OBS_GRIP], dtype=np.float32),
        camera_names=np.array(camera_names),
        native_horizon=native_horizon,
        horizon_sec=native_horizon * (cfg.policy_dt_ms / 1000.0),  # the sim-seconds env.py reports at reset
        termination_step=step,
        final_success=success,
        # numpy's savez **kwds stub reads a dict-unpack as possibly supplying ``allow_pickle`` (as in make_fixture.py).
        **{f'cam_hash__{name}': np.array(cam_hashes[name]) for name in camera_names},  # pyright: ignore[reportArgumentType]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Native-drive MolmoSpaces reference for the parity test.')
    parser.add_argument('--benchmark_dir', required=True)
    parser.add_argument('--episode_index', type=int, default=0)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max_steps', type=int, required=True)
    parser.add_argument('--out', required=True, help='npz path for the recorded native rollout')
    args = parser.parse_args()
    _run(Path(args.benchmark_dir), args.episode_index, args.seed, args.max_steps, Path(args.out))


if __name__ == '__main__':
    main()
