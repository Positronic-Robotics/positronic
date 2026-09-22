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

``--command-mode compare`` replays the same actions through ``ChunkedSchedule`` in five-action chunks and
an outer ``DeltaToAbsolute``, then through the direct-delta path. It compares every joint and end-effector
sample and the task outcome. Bounds are 0.1 mrad for joints and orientation, and 0.01 mm for position:
pose serialization and the OSC controller's float32 rotation conversion prevent exact trajectory equality.
This fixture checks fresh observations at the controller cadence; real sensor delay needs separate evaluation.

The episodes come from a tiny committed ``.npz`` fixture — a few demos' action sequences + initial states, a few
KB, not the multi-GB benchmark; ``make_fixture.py`` extracts it once from a demo HDF5 on a LIBERO box. Run on a
LIBERO box (the env server bootstraps its 3.10 deps via ``uv run --no-project``)::

    uv run --locked python -m positronic.simulator.libero.e2e \
        --fixture positronic/simulator/libero/tests/libero_spatial_task0.npz --command-mode compare
"""

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

import pimm
from positronic import geom, keys
from positronic.dataset.serializers import Serializers, expand_suffixed
from positronic.policy.action import DeltaToAbsolute
from positronic.policy.base import Step
from positronic.policy.codec import ACTION
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.sequential import Sequential
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.libero.adapter import LiberoAdapter
from positronic.simulator.libero.launcher import serve_libero
from positronic.vendors.openpi.codecs import PoseDeltaAction

_ROTMAT = geom.Rotation.Representation.ROTATION_MATRIX
_SETTLE_STEPS = 10  # let objects fall and settle after the scene loads, matching the openvla/openpi replay ritual
# OSC_POSE maps a normalized [-1, 1] action to this per-step pose-delta range (robosuite osc_pose.json). The demo
# actions are normalized, so un-normalizing by this is the forward conversion a policy adapter applies.
_OUTPUT_MAX = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
_REPLAY_STEP = 'replay_step'


class ReplayMode(StrEnum):
    CARTESIAN = 'cartesian'
    CARTESIAN_DELTA = 'cartesian_delta'
    POLICY = 'policy'
    COMPARE = 'compare'


@contextmanager
def replay_policy(actions: np.ndarray, chunk_size: int = 5):
    """A per-tick callable replaying recorded model outputs through the client stack and env adapter.

    Inference is uncharged, so a completion is consumed before stepping physics. Each chunk still
    passes through a worker and the scheduler; its deltas are converted only as they come due.
    """
    index = 0
    now_ns = 0
    runtime = Executor(lambda: now_ns, simulated=True, charge_inference_time=False)
    adapter = LiberoAdapter(camera_dict={})
    decoder = PoseDeltaAction()

    def infer(obs):
        start = obs[_REPLAY_STEP]
        return decoder.decode([{ACTION: action} for action in actions[start : start + chunk_size]])

    run = runtime.start(Sequential(DeltaToAbsolute(), ChunkedSchedule(20)), infer)

    def advance(raw_obs):
        nonlocal index, now_ns
        now_ns = index * 50_000_000
        state = adapter.observations(raw_obs)[keys.ROBOT_STATE]
        obs = dict(expand_suffixed(keys.ROBOT_STATE, Serializers.robot_state(state)))
        obs[_REPLAY_STEP] = index
        runtime.start_tick()
        step = run.send(obs)
        assert isinstance(step, Step)
        commands = dict(step.commands)
        while runtime.has_pending:
            result = runtime.wait(timeout_sec=5)
            if result.status is not WaitStatus.ANSWERS_READY:
                raise TimeoutError('Recorded inference did not complete')
            step = run.send(obs)
            assert isinstance(step, Step)
            commands.update(step.commands)
        index += 1
        return adapter.action({
            name: pimm.Message(commands[name], now_ns) if name in commands else None
            for name in (keys.ROBOT_COMMAND, keys.TARGET_GRIP)
        })

    try:
        yield advance
    finally:
        runtime.close()
        run.close()


def _physical_delta(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Un-normalize an OSC ``delta`` ([-1, 1]) into the world-frame per-step pose delta ``(Δpos, Δrot matrix)``."""
    physical = np.asarray(delta, dtype=float) * _OUTPUT_MAX
    delta_rot = np.asarray(geom.Rotation.from_rotvec(physical[3:]).to(_ROTMAT)).reshape(3, 3)
    return physical[:3], delta_rot


def _compose_pose(obs: dict, delta: np.ndarray) -> np.ndarray:
    """The absolute pose the normalized OSC ``delta`` targets, anchored on the observed eef pose.

    Mirrors robosuite ``set_goal``: world-frame translation ``ee_pos + Δpos`` and orientation ``R(Δrot) @ ee_ori``
    (left-multiply), but reads ``ee_pos``/``ee_ori`` from the wire observation rather than the controller."""
    delta_pos, delta_rot = _physical_delta(delta)
    cur_rot = np.asarray(geom.Rotation.from_quat_xyzw(obs['eef_quat']).to(_ROTMAT)).reshape(3, 3)
    return np.concatenate([np.asarray(obs['eef_pos']) + delta_pos, (delta_rot @ cur_rot).reshape(9)])


def _step_command(obs: dict, delta: np.ndarray, command_mode: ReplayMode) -> dict:
    """The per-step wire command for the active replay path: ``cartesian`` ships an absolute pose the env
    re-derives a delta from; ``cartesian_delta`` ships the world-frame delta straight to the OSC controller."""
    match command_mode:
        case ReplayMode.CARTESIAN:
            return {'type': 'cartesian', 'pose': _compose_pose(obs, delta)}
        case ReplayMode.CARTESIAN_DELTA:
            delta_pos, delta_rot = _physical_delta(delta)
            return {'type': 'cartesian_delta', 'delta': np.concatenate([delta_pos, delta_rot.reshape(9)])}
        case _:
            raise ValueError(f'unknown command mode {command_mode!r}')


@dataclass
class ReplayResult:
    success: bool
    joints: np.ndarray
    poses: np.ndarray

    def assert_matches(self, reference: 'ReplayResult') -> None:
        """Bound trajectory differences from pose serialization and OSC's float32 rotation conversion."""
        joint_error = np.max(np.abs(self.joints - reference.joints))
        position_error = np.linalg.norm(self.poses[:, :3] - reference.poses[:, :3], axis=1)
        rotations = self.poses[:, 3:] / np.linalg.norm(self.poses[:, 3:], axis=1, keepdims=True)
        other_rotations = reference.poses[:, 3:] / np.linalg.norm(reference.poses[:, 3:], axis=1, keepdims=True)
        dots = np.abs(np.sum(rotations * other_rotations, axis=1))
        rotation_error = 2 * np.arccos(np.clip(dots, 0, 1))
        print(
            f'  max errors: joint {joint_error:.3g} rad, position {position_error.max():.3g} m, '
            f'rotation {rotation_error.max():.3g} rad'
        )
        np.testing.assert_allclose(self.joints, reference.joints, rtol=0, atol=1e-4)
        assert position_error.max() <= 1e-5, 'end-effector trajectories differ by more than 0.01 mm'
        assert rotation_error.max() <= 1e-4, 'orientation trajectories differ by more than 0.1 mrad'
        assert self.success == reference.success, 'policy and direct delta replay disagree on success'


def _replay_episode(
    conn: EnvConnection, actions: np.ndarray, init_state: np.ndarray, scene: dict, *, command_mode: ReplayMode
) -> ReplayResult:
    # Exact-state reset: the token carries the task spec plus the demo's own recorded full state to restore.
    obs = conn.reset({**scene, 'state': init_state})['obs']
    for _ in range(_SETTLE_STEPS):
        obs = conn.step(protocol.single_arm_action({protocol.COMMAND_TYPE: protocol.HOLD}, 0.0))[protocol.FRAME_OBS]
    success = False
    joints, poses = [], []
    with replay_policy(actions) if command_mode == ReplayMode.POLICY else nullcontext(None) as policy:
        for action in actions:
            if policy is not None:
                wire = policy(obs)
            else:
                grip = (float(action[6]) + 1.0) / 2.0  # robosuite [-1, 1] -> positronic [0, 1]
                wire = protocol.single_arm_action(_step_command(obs, action[:6], command_mode), grip)
            out = conn.step(wire)
            obs = out['obs']
            success = success or out['done']
            joints.append(obs['joint_pos'])
            poses.append(np.concatenate([obs['eef_pos'], obs['eef_quat']]))
    return ReplayResult(success, np.asarray(joints), np.asarray(poses))


def _load_fixture(path: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """The ``(actions, init_state)`` episodes ``make_fixture.py`` packed into a fixture."""
    data = np.load(path)
    n = sum(k.startswith('actions_') for k in data.files)
    return [(data[f'actions_{i}'], data[f'init_state_{i}']) for i in range(n)]


def run_replay(
    fixture_path: str,
    *,
    suite: str = 'libero_spatial',
    task_id: int = 0,
    camera_resolution: int = 128,
    command_mode: ReplayMode = ReplayMode.CARTESIAN,
) -> float:
    """Replay every episode in ``fixture_path`` through the env server; return the success rate."""
    episodes = _load_fixture(fixture_path)
    # The task spec rides every reset token now; the demo replay drives the ``ee``/OSC_POSE controller.
    scene = {'suite': suite, 'task_id': task_id, 'camera_resolution': camera_resolution, 'control_mode': 'ee'}
    successes = 0
    with serve_libero() as (host, port):
        conn = EnvConnection(host, port)
        try:
            for i, (actions, init_state) in enumerate(episodes):
                result = _replay_episode(
                    conn,
                    actions,
                    init_state,
                    scene,
                    command_mode=ReplayMode.POLICY if command_mode == ReplayMode.COMPARE else command_mode,
                )
                if command_mode == ReplayMode.COMPARE:
                    reference = _replay_episode(
                        conn, actions, init_state, scene, command_mode=ReplayMode.CARTESIAN_DELTA
                    )
                    result.assert_matches(reference)
                successes += int(result.success)
                print(f'  episode {i}: {"success" if result.success else "FAIL"} ({len(actions)} steps)')
        finally:
            conn.close()
    return successes / len(episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description='Replay LIBERO demo episodes through the env server over the socket.')
    parser.add_argument('--fixture', required=True, help='the .npz fixture from make_fixture.py')
    parser.add_argument('--suite', default='libero_spatial', help='LIBERO task suite the fixture was extracted from')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--camera-resolution', type=int, default=128)
    parser.add_argument('--command-mode', type=ReplayMode, choices=list(ReplayMode), default=ReplayMode.CARTESIAN)
    parser.add_argument('--min-success', type=float, default=0.8, help='replay success rate below this fails the run')
    args = parser.parse_args()

    rate = run_replay(
        args.fixture,
        suite=args.suite,
        task_id=args.task_id,
        camera_resolution=args.camera_resolution,
        command_mode=args.command_mode,
    )
    print(f'replay success rate: {rate:.2f}')
    assert rate >= args.min_success, f'success rate {rate:.2f} below {args.min_success} — env server likely broken'
    print('E2E REPLAY PASSED')


if __name__ == '__main__':
    main()
