"""End-to-end check that the MolmoSpaces env server works over the socket + the adapter maps its payload.

The mapping/adapter unit tests exercise the transforms in-process; this drives the **real boundary**: the
launcher spawns the env-server subprocess in MolmoSpaces' own venv, and a client resets + steps it over the
actual socket, then feeds the wire payload through ``MolmoAdapter`` — validating the launcher, the wire
protocol, ``env.py``'s task drive, and the observation mapping together. A hold command holds the arm, so a
healthy server keeps the joints steady and reports frames of the right shape; a wrong obs key, quaternion
order, or a broken wire codec fails the mapping.

Needs the MolmoSpaces asset packs (``MLSPACES_ASSETS_DIR``) and a GL backend (``MUJOCO_GL``; a GPU-less box uses
mesa software EGL — ``EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1``). Run on a box with those::

    MLSPACES_ASSETS_DIR=... MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
        uv run --locked python -m positronic.simulator.molmo_spaces.tests.e2e --benchmark_dir <dir>
"""

import argparse
from pathlib import Path

import numpy as np

from positronic import keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import DEFAULT_CAMERA_DICT, MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces


def _check_sim_state(adapter: MolmoAdapter, raw_obs: dict) -> np.ndarray:
    """The privileged full MuJoCo state must survive the wire and reach the recorder as a finite qpos+qvel vector."""
    sim_state = adapter.privileged(raw_obs)[mapping.OBS_SIM_STATE]
    assert isinstance(sim_state, np.ndarray) and sim_state.ndim == 1 and sim_state.size > 0, (
        f'privileged sim_state malformed: {type(sim_state)} shape={getattr(sim_state, "shape", None)}'
    )
    assert np.isfinite(sim_state).all(), 'privileged sim_state carries non-finite values'
    return sim_state


def run(
    benchmark_dir: Path,
    *,
    episodes: int = 1,
    steps: int = 5,
    camera_dict: dict[str, str] | None = None,
    task_horizon_steps: int | None = None,
) -> None:
    """Reset + step the first ``episodes`` benchmark episodes over the socket, mapping each frame with the adapter."""
    camera_dict = camera_dict or DEFAULT_CAMERA_DICT
    adapter = MolmoAdapter(camera_dict)
    with serve_molmo_spaces(benchmark_dir, task_horizon_steps=task_horizon_steps) as (host, port):
        conn = EnvConnection(host, port)
        try:
            for i in range(episodes):
                frame = conn.reset({mapping.TOKEN_EPISODE_INDEX: i, mapping.TOKEN_SEED: None})
                obs = adapter.observations(frame[protocol.FRAME_OBS])
                assert keys.ROBOT_STATE in obs and keys.GRIP in obs, f'missing contract keys: {sorted(obs)}'
                assert all(logical in obs for logical in camera_dict), f'missing cameras: {sorted(obs)}'
                q = obs[keys.ROBOT_STATE].q
                assert q.shape == (7,), f'unexpected joint shape {q.shape}'
                sim_state = _check_sim_state(adapter, frame[protocol.FRAME_OBS])
                print(
                    f'  episode {i}: reset ok — task={frame[protocol.FRAME_META][mapping.META_TASK]!r} '
                    f'grip={obs[keys.GRIP]:.3f} '
                    f'q0={q[0]:.4f} sim_state={sim_state.size}d'
                )
                out = {protocol.FRAME_DONE: False}
                for _ in range(steps):
                    hold = {protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.HOLD}, protocol.ACTION_GRIP: 0.0}
                    out = conn.step(hold)
                    adapter.observations(out[protocol.FRAME_OBS])  # the mapping round-trips on step frames too
                    _check_sim_state(adapter, out[protocol.FRAME_OBS])
                print(f'  episode {i}: {steps} steps ok (done={out[protocol.FRAME_DONE]})')
        finally:
            conn.close()
    print('E2E PASSED')


def main() -> None:
    parser = argparse.ArgumentParser(description='Drive the MolmoSpaces env server over the socket.')
    parser.add_argument('--benchmark_dir', required=True, help='dir containing benchmark.json')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument(
        '--task_horizon_steps', type=int, default=None, help='override the benchmark horizon (steps per episode)'
    )
    args = parser.parse_args()
    run(Path(args.benchmark_dir), episodes=args.episodes, steps=args.steps, task_horizon_steps=args.task_horizon_steps)


if __name__ == '__main__':
    main()
