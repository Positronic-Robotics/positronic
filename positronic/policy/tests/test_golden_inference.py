"""Golden integration test: verifies the harness + codec pipeline produces identical robot commands.

A reactive deterministic policy reads robot state and returns action chunks. The harness
processes them through the codec chain (ActionTiming with timestamps + horizon). We capture
every emitted (clock_time, robot_command, grip) tuple and compare against a golden reference.

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
from positronic.policy.base import Policy
from positronic.policy.codec import ActionTiming
from positronic.policy.harness import Directive, Harness
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler

GOLDEN_FILE = Path(__file__).parent / 'golden_inference.json'

TARGET_POS = np.array([0.4, 0.3, 0.5], dtype=np.float32)
INITIAL_POS = np.array([0.3, 0.0, 0.4], dtype=np.float32)
INITIAL_JOINTS = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)


class ReactivePolicy(Policy):
    """Moves toward TARGET_POS proportional to distance from current EE pose.

    Returns 10-action chunks so ActionTiming stamps timestamps and ActionHorizon truncates.
    Deterministic: same observation always produces the same actions.
    """

    def select_action(self, obs):
        current_pos = np.asarray(obs['robot_state.ee_pose'][:3], dtype=np.float32)
        delta = TARGET_POS - current_pos
        actions = []
        for i in range(10):
            t = (i + 1) / 10.0
            step_pos = current_pos + delta * 0.1 * t
            pose = Transform3D(translation=step_pos, rotation=Rotation.identity)
            actions.append({
                'robot_command': to_wire(CartesianPosition(pose=pose)),
                'target_grip': round(0.5 + i * 0.01, 4),
            })
        return actions

    def reset(self, context=None):
        pass


class FakeRobotState:
    def __init__(self):
        self.ee_pose = Transform3D(translation=INITIAL_POS.copy(), rotation=Rotation.identity)
        self.q = INITIAL_JOINTS.copy()
        self.dq = np.zeros(7, dtype=np.float32)
        self.status = roboarm.RobotStatus.AVAILABLE


def _emit_sensors(frame_em, robot_em, grip_em, robot_state):
    frame_adapter = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.uint8)
    frame_adapter.array[:] = 42
    frame_em.emit(frame_adapter)
    robot_em.emit(robot_state)
    grip_em.emit(0.25)


def _pair_all(world, harness):
    cmd_recorder = RecordingEmitter()
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
    """Run the full harness + codec pipeline, return sequence of emitted commands."""
    clock = MockClock()
    robot_state = FakeRobotState()

    policy = ReactivePolicy()
    codec = ActionTiming(fps=15.0, horizon_sec=0.5)
    wrapped = codec.wrap(policy)

    with pimm.World(clock=clock) as world:
        harness = Harness(wrapped)
        p = _pair_all(world, harness)

        driver = ManualDriver([
            (partial(p['directive_em'].emit, Directive.RUN(task='golden-test')), 0.0),
            (partial(_emit_sensors, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
            (None, 0.5),
            # Re-emit sensors to trigger second inference after first chunk drains
            (partial(_emit_sensors, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
            (None, 0.5),
            # Third inference
            (partial(_emit_sensors, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
            (None, 0.5),
        ])

        scheduler = world.start([harness, driver])
        drive_scheduler(scheduler, clock=clock, steps=200)

    # Extract emitted commands as serializable data
    commands = []
    cmd_list = p['cmd_recorder'].emitted
    grip_list = p['grip_recorder'].emitted

    for (_cmd_ts, cmd_data), (_grip_ts, grip_data) in zip(cmd_list, grip_list, strict=True):
        if isinstance(cmd_data, CartesianPosition):
            commands.append({
                'type': 'cartesian_pos',
                'translation': cmd_data.pose.translation.tolist(),
                'rotation': cmd_data.pose.rotation.as_quat.tolist(),
                'grip': float(grip_data),
            })
        elif isinstance(cmd_data, roboarm.command.Reset):
            commands.append({'type': 'reset', 'grip': float(grip_data)})

    return commands


def _save_golden(commands):
    GOLDEN_FILE.write_text(json.dumps(commands, indent=2) + '\n')


def _load_golden():
    return json.loads(GOLDEN_FILE.read_text())


def test_golden_inference():
    """Verify harness + codec pipeline against golden reference."""
    commands = _run_pipeline()
    assert len(commands) > 0, 'Pipeline produced no commands'

    if os.environ.get('GOLDEN'):
        _save_golden(commands)
        pytest.skip(f'Golden reference saved to {GOLDEN_FILE} ({len(commands)} commands)')

    assert GOLDEN_FILE.exists(), f'Golden file not found: {GOLDEN_FILE}. Run with --golden to generate.'
    golden = _load_golden()

    assert len(commands) == len(golden), f'Command count mismatch: {len(commands)} vs {len(golden)} (golden)'

    for i, (actual, expected) in enumerate(zip(commands, golden, strict=True)):
        assert actual['type'] == expected['type'], f'Command {i}: type mismatch'
        assert actual['grip'] == pytest.approx(expected['grip'], abs=1e-6), f'Command {i}: grip mismatch'
        if actual['type'] == 'cartesian_pos':
            np.testing.assert_allclose(
                actual['translation'], expected['translation'], atol=1e-6, err_msg=f'Command {i}: translation'
            )
            np.testing.assert_allclose(
                actual['rotation'], expected['rotation'], atol=1e-6, err_msg=f'Command {i}: rotation'
            )
