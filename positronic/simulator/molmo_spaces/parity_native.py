"""Native-drive reference for the MolmoSpaces parity test — MolmoSpaces' own stack, no positronic.

Runs in MolmoSpaces' venv, flat off ``PYTHONPATH`` like ``env.py`` (positronic-free: ``molmo_spaces`` + this
package's ``mapping``/``env`` modules). It drives one benchmark episode through MolmoSpaces' native rollout —
``JsonEvalTaskSampler`` -> ``reset``/``step``/``is_done``/``judge_success``, the sequence ``JsonEvalRunner`` runs —
holding the arm every step, and records the per-step raw sim state, per-camera frame hashes, and where the native
horizon terminates the episode. ``parity.py`` drives the *same* episode through the positronic env-server path and
asserts byte-identical outcomes against this reference.

The horizon comes from ``mapping.resolve_task_horizon_steps`` (the benchmark's own ``task_horizon_sec``), the same
resolver ``env.py`` uses — its correctness is covered by a unit test, not re-derived here (MolmoSpaces' own
``determine_task_horizon`` reads the wrong field and raises on the shipped benchmarks, so it is not a usable
reference). Everything except the rollout drive is shared with ``env.py`` (the eval config, the horizon resolver,
the observation extraction), so the comparison isolates the sim rollout, not the observation mapping.

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
# GL-safely pulling in the molmo_spaces stack — so import it before any other molmo_spaces import. It supplies the
# shared observation extraction (observe_payload), the DROID eval config, and (via ``env.mapping``) the horizon
# resolver env.py uses. Reaching into env's private helpers is deliberate: the reference must use env.py's exact
# config + resolver so the comparison isolates the rollout.
import env  # noqa: E402
import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import numpy as np


def _run(benchmark_dir: Path, episode_index: int, seed: int, max_steps: int, out_path: Path) -> None:
    episodes = env.load_all_episodes(benchmark_dir)
    episode = episodes[episode_index]
    cfg = env._DroidPickEvalConfig()
    cfg.seed = seed
    native_horizon = env.mapping.resolve_task_horizon_steps(episode, cfg.policy_dt_ms)
    cfg.task_horizon = native_horizon
    sampler = env.JsonEvalTaskSampler(cfg, episode)
    task = sampler.sample_task(house_index=episode.house_index)
    robot_view = task.env.current_robot.robot_view

    obs, _info = task.reset()
    camera_names = [k for k, v in obs[0].items() if env._is_rgb_frame(v)]
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
        payload = env.observe_payload(robot_view, env_obs, camera_names)
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
