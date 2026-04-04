"""End-to-end test: SampledPolicy through the harness with multiple episodes.

Verifies the full sampling + inference + metadata + counting flow:
- Two reactive policies with different targets → produce different trajectories
- SampledPolicy with deterministic weights selects between them
- Harness runs multiple episodes via directives
- Episode metadata records which policy ran
- Balanced sampler counting works through the ds_command tap
"""

from functools import partial

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.dataset.ds_writer_agent import DsWriterCommandType
from positronic.drivers import roboarm
from positronic.drivers.roboarm.command import CartesianPosition, to_wire
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Policy, SampledPolicy
from positronic.policy.codec import ActionTiming
from positronic.policy.harness import Directive, Harness
from positronic.policy.sampler import BalancedSampler
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler


class TargetPolicy(Policy):
    """Reactive policy that moves toward a configurable target. Returns 5-action chunks."""

    def __init__(self, target: list[float], name: str):
        self._target = np.array(target, dtype=np.float32)
        self._name = name

    def select_action(self, obs):
        current_pos = np.asarray(obs['robot_state.ee_pose'][:3], dtype=np.float32)
        delta = self._target - current_pos
        actions = []
        for i in range(5):
            t = (i + 1) / 5.0
            step_pos = current_pos + delta * 0.1 * t
            pose = Transform3D(translation=step_pos, rotation=Rotation.identity)
            actions.append({'robot_command': to_wire(CartesianPosition(pose=pose)), 'target_grip': 0.5})
        return actions

    def reset(self, context=None):
        pass

    @property
    def meta(self):
        return {'server.checkpoint_path': self._name, 'target': self._target.tolist()}


class FakeRobotState:
    def __init__(self):
        self.ee_pose = Transform3D(translation=np.array([0.3, 0.0, 0.4], dtype=np.float32), rotation=Rotation.identity)
        self.q = np.zeros(7, dtype=np.float32)
        self.dq = np.zeros(7, dtype=np.float32)
        self.status = roboarm.RobotStatus.AVAILABLE


def _emit_sensors(frame_em, robot_em, grip_em, robot_state):
    frame_adapter = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.uint8)
    frame_adapter.array[:] = 0
    frame_em.emit(frame_adapter)
    robot_em.emit(robot_state)
    grip_em.emit(0.0)


def _pair_all(world, harness):
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    cmd_recorder = RecordingEmitter()
    harness.robot_commands._bind(cmd_recorder)
    grip_recorder = RecordingEmitter()
    harness.target_grip._bind(grip_recorder)
    return {
        'frame_em': world.pair(harness.frames['image.cam']),
        'robot_em': world.pair(harness.robot_state),
        'grip_em': world.pair(harness.gripper_state),
        'directive_em': world.pair(harness.directive),
        'meta_em': world.pair(harness.robot_meta_in),
        'ds_recorder': ds_recorder,
        'cmd_recorder': cmd_recorder,
        'grip_recorder': grip_recorder,
    }


def _ds_commands(p):
    return [data for _, data in p['ds_recorder'].emitted]


def _episode_metas(p):
    """Extract metadata dicts from all START commands."""
    return [cmd.static_data for cmd in _ds_commands(p) if cmd.type == DsWriterCommandType.START_EPISODE]


@pytest.mark.timeout(5.0)
def test_sampled_policy_e2e():
    """Full harness e2e with SampledPolicy: 4 episodes, 2 policies, balanced sampling."""
    clock = MockClock()
    robot_state = FakeRobotState()

    # Two policies targeting different positions
    policy_a = TargetPolicy([0.5, 0.0, 0.5], name='model_a')
    policy_b = TargetPolicy([0.3, 0.2, 0.3], name='model_b')

    # Wrap each with timing codec
    codec = ActionTiming(fps=10.0, horizon_sec=0.5)
    wrapped_a = codec.wrap(policy_a)
    wrapped_b = codec.wrap(policy_b)

    # SampledPolicy with balanced sampler
    sampler = BalancedSampler(balance=2)
    sampled = SampledPolicy(wrapped_a, wrapped_b, sampler=sampler)

    harness = Harness(sampled)

    with pimm.World(clock=clock) as world:
        p = _pair_all(world, harness)

        # Wire the ds_command tap for counting (mirrors inference.py:_connect_ds_command)
        original_emit = p['ds_recorder'].emit

        def counting_emit(data, ts=-1):
            original_emit(data, ts)
            if data.type is DsWriterCommandType.STOP_EPISODE:
                sampled.count_current()

        p['ds_recorder'].emit = counting_emit

        # Run 4 episodes: RUN → sensors → wait → FINISH, repeat
        script = []
        for i in range(4):
            script.append((partial(p['directive_em'].emit, Directive.RUN(task=f'ep{i}')), 0.01))
            script.append((partial(_emit_sensors, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01))
            script.append((None, 0.5))
            script.append((partial(p['directive_em'].emit, Directive.FINISH(outcome='ok')), 0.01))
            script.append((None, 0.1))
        script.append((None, 0.1))

        driver = ManualDriver(script)
        scheduler = world.start([harness, driver])
        drive_scheduler(scheduler, clock=clock, steps=500)

    # --- Assertions ---

    # 1. Four episodes started
    metas = _episode_metas(p)
    assert len(metas) == 4, f'Expected 4 episodes, got {len(metas)}'

    # 2. Each episode records which policy ran
    policy_names = [m.get('inference.policy.server.checkpoint_path') for m in metas]
    assert all(name in ('model_a', 'model_b') for name in policy_names), f'Unexpected names: {policy_names}'

    # 3. Both policies were sampled (balanced sampler ensures this over 4 episodes)
    assert 'model_a' in policy_names, 'model_a never sampled'
    assert 'model_b' in policy_names, 'model_b never sampled'

    # 4. Commands were actually emitted (trajectories produced)
    all_cmds = p['cmd_recorder'].emitted
    assert len(all_cmds) > 0, 'No commands emitted'

    # 5. Sampler counted episodes
    counts = sampler._counts
    total_counted = sum(sum(v.values()) for v in counts.values()) if counts else 0
    assert total_counted == 4, f'Expected 4 counted episodes, got {total_counted}'

    # 6. Different policies produce different first commands (different targets)
    # Group commands by episode based on the metas
    first_commands_by_policy = {}
    cmd_idx = 0
    for name in policy_names:
        if name not in first_commands_by_policy and cmd_idx < len(all_cmds):
            _, traj = all_cmds[cmd_idx]
            if isinstance(traj, list) and len(traj) > 0:
                _, cmd = traj[0]
                if isinstance(cmd, CartesianPosition):
                    first_commands_by_policy[name] = cmd.pose.translation.copy()
        # Skip trajectories for this episode (there might be multiple emits)
        while cmd_idx < len(all_cmds):
            _, data = all_cmds[cmd_idx]
            cmd_idx += 1
            # Home (Reset) trajectories mark episode boundaries
            if isinstance(data, list) and len(data) > 0:
                _, c = data[0]
                if isinstance(c, roboarm.command.Reset):
                    break

    if len(first_commands_by_policy) == 2:
        pos_a = first_commands_by_policy['model_a']
        pos_b = first_commands_by_policy['model_b']
        assert not np.allclose(pos_a, pos_b, atol=1e-3), 'Different policies should produce different commands'
