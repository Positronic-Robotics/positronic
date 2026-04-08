"""Golden integration test: harness + codec pipeline with reactive state.

A proportional controller reads robot EE pose and returns action chunks toward a target.
A fake robot applies each CartesianPosition command to its state, so the trajectory evolves
across inference calls. Captures all emitted (command, grip) tuples and compares against
a golden reference recorded on main (pre-timestamp changes).

Set GOLDEN=1 to regenerate the reference file:
    GOLDEN=1 uv run pytest positronic/policy/tests/test_golden_inference.py -p no:cacheprovider -o "addopts="
"""

import json
import os
from functools import partial
from pathlib import Path

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.drivers import roboarm
from positronic.drivers.roboarm.command import CartesianPosition, to_wire
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Policy, Session
from positronic.policy.codec import ActionTiming
from positronic.policy.harness import ChunkedSchedule, Directive, Harness
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler

GOLDEN_FILE = Path(__file__).parent / 'golden_inference.json'

TARGET_POS = np.array([0.5, 0.0, 0.45], dtype=np.float32)
INITIAL_POS = np.array([0.3, 0.0, 0.4], dtype=np.float32)
INITIAL_JOINTS = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)


class _ReactiveSession(Session):
    def __call__(self, obs):
        current_pos = np.asarray(obs['robot_state.ee_pose'][:3], dtype=np.float32)
        delta = TARGET_POS - current_pos
        actions = []
        for i in range(10):
            t = (i + 1) / 10.0
            step_pos = current_pos + delta * 0.05 * t
            pose = Transform3D(translation=step_pos, rotation=Rotation.identity)
            actions.append({
                'robot_command': to_wire(CartesianPosition(pose=pose)),
                'target_grip': round(0.5 + i * 0.01, 4),
            })
        return actions


class ReactivePolicy(Policy):
    def new_session(self, context=None):
        return _ReactiveSession()


class SimulatedRobotState:
    """Fake robot that applies CartesianPosition commands to its EE pose."""

    def __init__(self):
        self._pos = INITIAL_POS.copy()
        self.q = INITIAL_JOINTS.copy()
        self.dq = np.zeros(7, dtype=np.float32)
        self.status = roboarm.RobotStatus.AVAILABLE

    @property
    def ee_pose(self):
        return Transform3D(translation=self._pos.copy(), rotation=Rotation.identity)

    def apply_command(self, cmd):
        if isinstance(cmd, CartesianPosition):
            self._pos = np.asarray(cmd.pose.translation, dtype=np.float32)


def _emit_sensors(frame_em, robot_em, grip_em, robot_state):
    frame_adapter = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.uint8)
    frame_adapter.array[:] = 42
    frame_em.emit(frame_adapter)
    robot_em.emit(robot_state)
    grip_em.emit(0.25)


class _ApplyAndRecord(RecordingEmitter):
    def __init__(self, robot_state):
        super().__init__()
        self._robot_state = robot_state

    def emit(self, data, ts=-1):
        for _t, cmd in data:
            super().emit(cmd, ts)
            self._robot_state.apply_command(cmd)


def _pair_all(world, harness, robot_state):
    cmd_recorder = _ApplyAndRecord(robot_state)
    grip_recorder = RecordingEmitter()

    harness.robot_commands._bind(cmd_recorder)
    harness.target_grip._bind(grip_recorder)

    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)

    return {
        'frame_em': world.pair(harness.frames['image.cam']),
        'robot_em': world.pair(harness.robot_state),
        'grip_em': world.pair(harness.gripper_state),
        'directive_em': world.pair(harness.directive),
        'meta_em': world.pair(harness.robot_meta_in),
        'cmd_recorder': cmd_recorder,
        'grip_recorder': grip_recorder,
    }


def _run_pipeline():
    """Run the full harness + codec pipeline with state feedback, return emitted commands."""
    clock = MockClock()
    robot_state = SimulatedRobotState()

    policy = ReactivePolicy()
    codec = ActionTiming(fps=15.0, horizon_sec=0.5)
    wrapped = codec.wrap(policy)

    with pimm.World(clock=clock) as world:
        harness = Harness(ChunkedSchedule().wrap(wrapped))
        p = _pair_all(world, harness, robot_state)

        # Build a script that emits sensors before each inference cycle.
        # The robot state updates from commands, so each inference sees fresh observations.
        script = [(partial(p['directive_em'].emit, Directive.RUN(task='golden-test')), 0.0)]
        for _ in range(6):
            script.append((partial(_emit_sensors, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01))
            script.append((None, 0.5))
        script.append((None, 0.1))

        driver = ManualDriver(script)
        scheduler = world.start([harness, driver])
        drive_scheduler(scheduler, clock=clock, steps=500)

    cmd_list = p['cmd_recorder'].emitted  # individual commands (unpacked by _ApplyAndRecord)
    # Grip recorder receives trajectory lists — flatten them
    grip_list = []
    for _, traj in p['grip_recorder'].emitted:
        for _t, grip in traj:
            grip_list.append(grip)

    commands = []
    for (_cmd_ts, cmd_data), grip_data in zip(cmd_list, grip_list, strict=True):
        if isinstance(cmd_data, CartesianPosition):
            commands.append({
                'type': 'cartesian_pos',
                'translation': cmd_data.pose.translation.tolist(),
                'rotation': cmd_data.pose.rotation.as_quat.tolist(),
                'grip': float(grip_data),
            })
        else:
            commands.append({'type': type(cmd_data).__name__, 'grip': float(grip_data)})

    return commands


def test_golden_inference():
    """Verify harness + codec pipeline against golden reference."""
    commands = _run_pipeline()
    assert len(commands) > 0, 'Pipeline produced no commands'

    if os.environ.get('GOLDEN'):
        GOLDEN_FILE.write_text(json.dumps(commands, indent=2) + '\n')
        pytest.skip(f'Golden reference saved to {GOLDEN_FILE} ({len(commands)} commands)')

    assert GOLDEN_FILE.exists(), f'Golden file not found: {GOLDEN_FILE}. Run with GOLDEN=1 to generate.'
    golden = json.loads(GOLDEN_FILE.read_text())

    assert len(commands) == len(golden), f'Command count mismatch: {len(commands)} vs {len(golden)} (golden)'

    for i, (actual, expected) in enumerate(zip(commands, golden, strict=True)):
        assert actual['type'] == expected['type'], f'Command {i}: type {actual["type"]} != {expected["type"]}'
        assert actual['grip'] == pytest.approx(expected['grip'], abs=1e-6), f'Command {i}: grip mismatch'
        if 'translation' in actual and 'translation' in expected:
            np.testing.assert_allclose(
                actual['translation'], expected['translation'], atol=1e-6, err_msg=f'Command {i}: translation'
            )
