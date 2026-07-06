"""End-to-end check that the RoboLab env server works: replay recorded demos over the socket.

``validate.py`` checks the command transform's *algebra* in-process. This drives the **real boundary**:
positronic launches the env-server subprocess (RoboLab's own Isaac Lab interpreter) and replays a recorded
demo's raw joint-position actions through the actual socket, asserting the task reports success before the
log runs out. The ``joint_pos`` path is a bit-identical passthrough to the 8-dim ``[q1..q7, gripper]`` action
RoboLab's own leaderboard replay feeds its jointpos env, so a success proves the exact-state reset restores
the recorded scene and the wire command reaches the articulation unmangled.

The demos come from a tiny committed ``.npz`` fixture — each demo's action log + initial scene state, not the
multi-GB recording; ``make_fixture.py`` extracts it once from a RoboLab ``data.hdf5``. Run on a RoboLab-capable
box (the env server bootstraps the Isaac Lab stack via ``uv run --project`` on the pinned checkout)::

    uv run --locked python -m positronic.simulator.robolab.e2e \
        --fixture positronic/simulator/robolab/tests/rubiks_cube_and_banana.npz
"""

import argparse

import numpy as np

from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.robolab.launcher import serve_robolab

# Keep re-sending the last action after the log ends, matching RoboLab's own replay tail: the final placement
# can settle (and the success term fire) a few control steps after the last recorded command.
_HOLD_TAIL_STEPS = 10


def _load_fixture(path: str) -> list[tuple[np.ndarray, dict]]:
    """The ``(actions, initial_state)`` episodes ``make_fixture.py`` packed into a fixture."""
    data = np.load(path)
    episodes = []
    for i in range(sum(k.startswith('actions_') for k in data.files)):
        prefix = f'initial_state_{i}/'
        state: dict = {}
        for key in data.files:
            if key.startswith(prefix):
                node = state
                *parents, leaf = key[len(prefix) :].split('/')
                for part in parents:
                    node = node.setdefault(part, {})
                node[leaf] = data[key]
        episodes.append((data[f'actions_{i}'], state))
    return episodes


def _replay_episode(conn: EnvConnection, actions: np.ndarray, initial_state: dict, task: str) -> bool:
    # Exact-state reset: the token carries the task plus the demo's own recorded initial scene state.
    conn.reset({'task': task, 'instruction_type': 'default', 'state': initial_state})
    for action in [*actions, *([actions[-1]] * _HOLD_TAIL_STEPS)]:
        out = conn.step({'command': {'type': 'joint_pos', 'q': action[:7]}, 'grip': float(action[7])})
        if out['done']:
            return bool(out['success'])
    return False


def run_replay(fixture_path: str, *, task: str) -> float:
    """Replay every episode in ``fixture_path`` through the env server; return the success rate."""
    episodes = _load_fixture(fixture_path)
    successes = 0
    with serve_robolab() as (host, port):
        conn = EnvConnection(host, port)
        try:
            for i, (actions, initial_state) in enumerate(episodes):
                ok = _replay_episode(conn, actions, initial_state, task)
                successes += int(ok)
                print(f'  episode {i}: {"success" if ok else "FAIL"} ({len(actions)} steps)')
        finally:
            conn.close()
    return successes / len(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay RoboLab demos through the env server over the socket.')
    parser.add_argument('--fixture', required=True, help='the .npz fixture from make_fixture.py')
    parser.add_argument('--task', default='RubiksCubeAndBananaTask', help='RoboLab task the fixture was recorded on')
    parser.add_argument('--min-success', type=float, default=0.8, help='replay success rate below this fails the run')
    args = parser.parse_args()

    rate = run_replay(args.fixture, task=args.task)
    print(f'replay success rate: {rate:.2f}')
    assert rate >= args.min_success, f'success rate {rate:.2f} below {args.min_success} — env server likely broken'
    print('E2E REPLAY PASSED')


if __name__ == '__main__':
    main()
