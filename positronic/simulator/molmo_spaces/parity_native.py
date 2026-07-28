"""Native-drive reference for the MolmoSpaces parity test — MolmoSpaces' own stack, no positronic.

Runs in MolmoSpaces' venv, flat off ``PYTHONPATH`` like ``env.py`` (positronic-free: ``molmo_spaces`` + this
package's ``mapping``/``env`` modules). It drives one benchmark episode through MolmoSpaces' native rollout —
``JsonEvalTaskSampler`` -> ``reset``/``step``/``is_done``/``judge_success``, the sequence ``JsonEvalRunner`` runs —
holding the arm every step, and records the per-step raw sim state, per-camera frame hashes, and where the native
horizon terminates the episode. ``parity.py`` drives the *same* episode through the positronic env-server path and
asserts byte-identical outcomes against this reference.

The horizon is resolved by MolmoSpaces' own ``determine_task_horizon`` and cross-checked here against ``env.py``'s
``_resolve_task_horizon``, so the reference proves the two agree without any sim run. Everything except the horizon
resolution and the rollout drive is shared with ``env.py`` (the eval config and the observation extraction), so the
comparison isolates the sim rollout and its horizon, not the observation mapping.

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
# shared observation extraction (observe_payload), the DROID eval config, and its own horizon resolver, which this
# reference cross-checks against native. Reaching into env's private helpers is deliberate: the reference must use
# the exact config + resolver env.py uses.
import env  # noqa: E402
import numpy as np

from molmo_spaces.evaluation.eval_main import determine_task_horizon  # noqa: E402


def _run(benchmark_dir: Path, episode_index: int, seed: int, max_steps: int, out_path: Path) -> None:
    episodes = env.load_all_episodes(benchmark_dir)
    episode = episodes[episode_index]
    cfg = env._DroidPickEvalConfig()
    cfg.seed = seed
    native_horizon = determine_task_horizon([episode], None, cfg.policy_dt_ms)
    resolved = env._resolve_task_horizon(episode, cfg.policy_dt_ms)
    if native_horizon != resolved:
        raise AssertionError(f'env.py resolved horizon {resolved} != native determine_task_horizon {native_horizon}')
    cfg.task_horizon = native_horizon
    sampler = env.JsonEvalTaskSampler(cfg, episode)
    task = sampler.sample_task(house_index=episode.house_index)
    robot_view = task.env.current_robot.robot_view

    obs, _info = task.reset()
    camera_names = [k for k, v in obs[0].items() if env._is_rgb_frame(v)]
    fields: dict[str, list] = {k: [] for k in ('joint_pos', 'joint_vel', 'eef_pos', 'eef_quat', 'grip')}
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
        measured_q = np.asarray(robot_view.get_move_group('arm').joint_pos, dtype=np.float32)
        action = {'arm': measured_q, 'gripper': np.array([0.0], dtype=np.float32)}
        obs, _reward, _term, _trunc, _infos = task.step(action)
        step += 1
        record(obs[0])
        success = bool(task.judge_success())
        if success:  # end-on-success, matching env.py's step (a hold never reaches it)
            break
    sampler.close()

    np.savez(
        out_path,
        joint_pos=np.stack(fields['joint_pos']),
        joint_vel=np.stack(fields['joint_vel']),
        eef_pos=np.stack(fields['eef_pos']),
        eef_quat=np.stack(fields['eef_quat']),
        grip=np.array(fields['grip'], dtype=np.float32),
        camera_names=np.array(camera_names),
        native_horizon=native_horizon,
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
