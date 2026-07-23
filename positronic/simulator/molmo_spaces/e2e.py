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
        uv run --locked python -m positronic.simulator.molmo_spaces.e2e --benchmark_dir <dir>
"""

import argparse

from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.molmo_spaces.adapter import MolmoAdapter
from positronic.simulator.molmo_spaces.launcher import serve_molmo_spaces

_CAMERA_DICT = {'image.wrist': 'wrist_camera', 'image.exterior': 'exo_camera_1'}


def run(benchmark_dir: str, *, episodes: int = 1, steps: int = 5, camera_dict: dict[str, str] | None = None) -> None:
    """Reset + step the first ``episodes`` benchmark episodes over the socket, mapping each frame with the adapter."""
    camera_dict = camera_dict or _CAMERA_DICT
    adapter = MolmoAdapter(camera_dict)
    with serve_molmo_spaces(benchmark_dir) as (host, port):
        conn = EnvConnection(host, port)
        try:
            for i in range(episodes):
                frame = conn.reset({'episode_index': i, 'seed': None})
                obs = adapter.observations(frame['obs'])
                assert 'robot_state' in obs and 'grip' in obs, f'missing contract keys: {sorted(obs)}'
                assert all(logical in obs for logical in camera_dict), f'missing cameras: {sorted(obs)}'
                q = obs['robot_state'].q
                assert q.shape == (7,), f'unexpected joint shape {q.shape}'
                print(f'  episode {i}: reset ok — task={frame["meta"]["task"]!r} grip={obs["grip"]:.3f} q0={q[0]:.4f}')
                out = {'done': False}
                for _ in range(steps):
                    out = conn.step({'command': {'type': 'hold'}, 'grip': 0.0})
                    adapter.observations(out['obs'])  # the mapping round-trips on step frames too
                print(f'  episode {i}: {steps} steps ok (done={out["done"]})')
        finally:
            conn.close()
    print('E2E PASSED')


def main() -> None:
    parser = argparse.ArgumentParser(description='Drive the MolmoSpaces env server over the socket.')
    parser.add_argument('--benchmark_dir', required=True, help='dir containing benchmark.json')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    run(args.benchmark_dir, episodes=args.episodes, steps=args.steps)


if __name__ == '__main__':
    main()
