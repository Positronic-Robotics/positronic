"""End-to-end check that the LIBERO env server works: replay recorded demo episodes over the socket.

``validate.py`` checks the command transform's *algebra* in-process. This drives the **real boundary**:
positronic launches the env-server subprocess (its own 3.10 interpreter) and replays LIBERO demo episodes through
the actual socket as ``CartesianPosition`` waypoints, asserting the task reaches ``done``.

It is a genuine (non-circular) oracle because each waypoint is anchored on the eef pose read **from the
observation that came back over the wire** — the same pose a policy sees, decoded with the adapter's
``from_quat_xyzw`` convention — not on robosuite's controller state. The server's ``_arm_action`` recovers the
OSC delta from its own controller pose; the round-trip only closes (and the demo only stays on its recorded
trajectory) if the observed pose matches the controller pose in the same frame and quaternion order. A wrong
quaternion order, pose frame, eef site, or action scale makes the demo diverge and fail. ``_compose_pose`` is
also the forward conversion a real policy adapter must perform (normalized OSC delta -> absolute pose), so this
exercises that path too.

The episodes come from a tiny committed ``.npz`` fixture — a few demos' action sequences + initial states, a few
KB, not the multi-GB benchmark; ``make_fixture.py`` extracts it once from a demo HDF5 on a LIBERO box. Run on a
LIBERO box (the env server bootstraps its 3.10 deps via ``uv run --no-project``)::

    uv run --locked python -m positronic.simulator.libero.e2e \
        --fixture positronic/simulator/libero/tests/libero_spatial_task0.npz
"""

import argparse

import numpy as np

from positronic import geom
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.libero.launcher import serve_libero

_ROTMAT = geom.Rotation.Representation.ROTATION_MATRIX
_SETTLE_STEPS = 10  # let objects fall and settle after the scene loads, matching the openvla/openpi replay ritual
# OSC_POSE maps a normalized [-1, 1] action to this per-step pose-delta range (robosuite osc_pose.json). The demo
# actions are normalized, so un-normalizing by this is the forward conversion a policy adapter applies.
_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])


def _compose_pose(obs: dict, delta: np.ndarray) -> np.ndarray:
    """The absolute pose the normalized OSC ``delta`` targets, anchored on the observed eef pose.

    Mirrors robosuite ``set_goal``: world-frame translation ``ee_pos + Δpos`` and orientation ``R(Δrot) @ ee_ori``
    (left-multiply), but reads ``ee_pos``/``ee_ori`` from the wire observation rather than the controller."""
    physical = np.asarray(delta, dtype=float) * _OUTPUT_MAX
    cur_rot = np.asarray(geom.Rotation.from_quat_xyzw(obs['eef_quat']).to(_ROTMAT)).reshape(3, 3)
    delta_rot = np.asarray(geom.Rotation.from_rotvec(physical[3:]).to(_ROTMAT)).reshape(3, 3)
    target_pos = np.asarray(obs['eef_pos']) + physical[:3]
    return np.concatenate([target_pos, (delta_rot @ cur_rot).reshape(9)])


def _replay_episode(conn: EnvConnection, actions: np.ndarray, init_state: np.ndarray) -> bool:
    obs = conn.reset(init_state)['obs']  # exact-state reset: start from the demo's own recorded scene
    for _ in range(_SETTLE_STEPS):
        obs = conn.step({'command': {'type': 'hold'}, 'grip': 0.0})['obs']
    success = False
    for action in actions:
        pose = _compose_pose(obs, action[:6])
        grip = (float(action[6]) + 1.0) / 2.0  # robosuite gripper [-1, 1] -> positronic [0, 1]
        out = conn.step({'command': {'type': 'cartesian', 'pose': pose}, 'grip': grip})
        obs = out['obs']
        success = success or out['done']
    return success


def _load_fixture(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``(actions, init_state)`` episodes ``make_fixture.py`` packed into a fixture."""
    data = np.load(path)
    n = sum(k.startswith('actions_') for k in data.files)
    return [(data[f'actions_{i}'], data[f'init_state_{i}']) for i in range(n)]


def run_replay(
    fixture_path: str, *, suite: str = 'libero_spatial', task_id: int = 0, camera_resolution: int = 128
) -> float:
    """Replay every episode in ``fixture_path`` through the env server; return the success rate."""
    episodes = _load_fixture(fixture_path)
    successes = 0
    with serve_libero(suite, task_id, camera_resolution, 'ee') as (host, port):
        conn = EnvConnection(host, port)
        try:
            for i, (actions, init_state) in enumerate(episodes):
                ok = _replay_episode(conn, actions, init_state)
                successes += int(ok)
                print(f'  episode {i}: {"success" if ok else "FAIL"} ({len(actions)} steps)')
        finally:
            conn.close()
    return successes / len(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay LIBERO demo episodes through the env server over the socket.')
    parser.add_argument('--fixture', required=True, help='the .npz fixture from make_fixture.py')
    parser.add_argument('--suite', default='libero_spatial', help='LIBERO task suite the fixture was extracted from')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--camera-resolution', type=int, default=128)
    parser.add_argument('--min-success', type=float, default=0.8, help='replay success rate below this fails the run')
    args = parser.parse_args()

    rate = run_replay(args.fixture, suite=args.suite, task_id=args.task_id, camera_resolution=args.camera_resolution)
    print(f'replay success rate: {rate:.2f}')
    assert rate >= args.min_success, f'success rate {rate:.2f} below {args.min_success} — env server likely broken'
    print('E2E REPLAY PASSED')


if __name__ == '__main__':
    main()
