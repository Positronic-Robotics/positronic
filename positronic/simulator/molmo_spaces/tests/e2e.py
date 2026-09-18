"""Smoke-test the MolmoSpaces socket server and observation adapter.

Needs the MolmoSpaces asset packs (``MLSPACES_ASSETS_DIR``) and a GL backend (``MUJOCO_GL``; a GPU-less box uses
mesa software EGL — ``EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1``). Run on a box with those::

    MLSPACES_ASSETS_DIR=... MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
        uv run --locked python -m positronic.simulator.molmo_spaces.tests.e2e \
            --benchmark <suite/scene_dataset/task_config/benchmark>
"""

import argparse

import numpy as np

from positronic import keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.molmo_spaces.adapter import CAMERAS, MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces


def _check_sim_state(adapter: MolmoAdapter, raw_obs: dict) -> np.ndarray:
    """Check that the privileged simulation state is a finite, nonempty vector."""
    sim_state = adapter.privileged(raw_obs)[mapping.OBS_SIM_STATE]
    assert isinstance(sim_state, np.ndarray) and sim_state.ndim == 1 and sim_state.size > 0, (
        f'privileged sim_state malformed: {type(sim_state)} shape={getattr(sim_state, "shape", None)}'
    )
    assert np.isfinite(sim_state).all(), 'privileged sim_state carries non-finite values'
    return sim_state


def run(bench: mapping.BenchmarkPath | None, *, episodes: int = 1, steps: int = 5) -> None:
    """Check the first ``episodes`` matching records over the socket, resetting and stepping each one."""
    adapter = MolmoAdapter()
    with serve_molmo_spaces() as (host, port):
        conn = EnvConnection(host, port)
        try:
            for record in conn.tasks(bench._asdict() if bench is not None else {})[:episodes]:
                i = record[mapping.TOKEN_EPISODE_INDEX]
                benchmark = mapping.BenchmarkPath(**{d: record[d] for d in mapping.BenchmarkPath._fields})
                token = {**benchmark._asdict(), mapping.TOKEN_EPISODE_INDEX: i}
                frame = protocol.one_slot(conn.reset({**token, mapping.TOKEN_SEED: None}))
                obs = adapter.observations(frame[protocol.FRAME_OBS])
                assert keys.ROBOT_STATE in obs and keys.GRIP in obs, f'missing contract keys: {sorted(obs)}'
                assert all(logical in obs for logical in CAMERAS), f'missing cameras: {sorted(obs)}'
                q = obs[keys.ROBOT_STATE].q
                assert q.shape == (7,), f'unexpected joint shape {q.shape}'
                sim_state = _check_sim_state(adapter, frame[protocol.FRAME_OBS])
                print(
                    f'  episode {i} of {benchmark.relative}: reset ok — '
                    f'task={frame[protocol.FRAME_META][mapping.META_TASK]!r} grip={obs[keys.GRIP]:.3f} '
                    f'q0={q[0]:.4f} sim_state={sim_state.size}d'
                )
                out = {protocol.FRAME_DONE: False}
                for _ in range(steps):
                    hold = {protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.HOLD}, protocol.ACTION_GRIP: 0.0}
                    out = protocol.one_slot(conn.step([hold]))
                    adapter.observations(out[protocol.FRAME_OBS])
                    _check_sim_state(adapter, out[protocol.FRAME_OBS])
                print(f'  episode {i}: {steps} steps ok (done={out[protocol.FRAME_DONE]})')
        finally:
            conn.close()
    print('E2E PASSED')


def main() -> None:
    parser = argparse.ArgumentParser(description='Drive the MolmoSpaces env server over the socket.')
    parser.add_argument(
        '--benchmark',
        type=mapping.BenchmarkPath.parse,
        default=None,
        help='suite/scene_dataset/task_config/benchmark under the asset packs; absent, the first one found',
    )
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    run(args.benchmark, episodes=args.episodes, steps=args.steps)


if __name__ == '__main__':
    main()
