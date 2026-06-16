from functools import partial

import numpy as np
import pytest

import pimm
from positronic.dataset.ds_writer_agent import DsWriterCommand, DsWriterCommandType
from positronic.dataset.serializers import Serializers
from positronic.drivers import roboarm
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianPosition, Recover, Reset, from_wire, to_wire
from positronic.eval import Command, Embodiment, Observation, Task
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Policy, Session
from positronic.policy.codec import ActionTimestamp
from positronic.policy.harness import Directive, DirectiveType, Harness
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler


def make_embodiment(descriptor: str = '', cameras=('image.cam',)) -> Embodiment:
    """Minimal Franka-shaped embodiment for harness unit tests.

    The sources/dests are no-ops: these tests pair the harness ports directly
    (never via ``wire_embodiment``), so only the spec — names, serializers,
    home values, descriptor — is read by the Harness.
    """
    observations = {
        'robot_state': Observation(pimm.NoOpEmitter(), Serializers.robot_state),
        'grip': Observation(pimm.NoOpEmitter(), None),
    }
    for cam in cameras:
        observations[cam] = Observation(pimm.NoOpEmitter(), Serializers.camera_images)
    commands = {
        'robot_command': Command(pimm.NoOpReceiver(), Reset(), Serializers.robot_command),
        'target_grip': Command(pimm.NoOpReceiver(), 0.0, None),
    }
    return Embodiment(descriptor, observations, commands, {}, pimm.NoOpEmitter())


class _SpySession(Session):
    def __init__(self, policy):
        self._policy = policy

    def __call__(self, obs):
        self._policy.last_obs = obs
        return [{'robot_command': self._policy.command, 'target_grip': self._policy.target_grip, 'timestamp': 0.0}]


class SpyPolicy(Policy):
    def __init__(self, command: roboarm.command.CommandType | None = None, target_grip: float = 0.33) -> None:
        if command is None:
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
        self.command = command
        self.target_grip = float(target_grip)
        self.last_obs: dict[str, object] | None = None
        self.reset_calls: int = 0
        self.last_reset_context = None

    def new_session(self, context=None):
        self.reset_calls += 1
        self.last_reset_context = context
        return _SpySession(self)


class _StubSession(Session):
    def __init__(self, policy):
        self._policy = policy
        self._meta = dict(policy._meta)

    def __call__(self, obs):
        self._policy.last_obs = obs
        self._policy.observations.append(obs)
        return [{'robot_command': self._policy.command, 'target_grip': self._policy.target_grip, 'timestamp': 0.0}]

    @property
    def meta(self):
        return self._meta


class StubPolicy(Policy):
    """Reusable policy stub for tests."""

    def __init__(
        self,
        command: roboarm.command.CommandType | None = None,
        target_grip: float = 0.33,
        meta: dict[str, object] | None = None,
    ) -> None:
        if command is None:
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
        self.command = command
        self.target_grip = float(target_grip)
        self.last_obs: dict[str, object] | None = None
        self.observations: list[dict[str, object]] = []
        self.reset_calls = 0
        self.last_reset_context = None
        self._meta: dict[str, object] = meta or {}

    @property
    def meta(self) -> dict[str, object]:
        return self._meta

    def new_session(self, context=None):
        self.reset_calls += 1
        self.last_reset_context = context
        return _StubSession(self)


class _ChunkSession(Session):
    def __init__(self, policy):
        self._policy = policy

    def __call__(self, obs):
        self._policy.counter += 1
        dt = 0.005
        return [
            {
                'robot_command': self._policy.command,
                'target_grip': self._policy.counter * 100.0 + i,
                'timestamp': i * dt,
            }
            for i in range(10)
        ]


class ChunkPolicy(StubPolicy):
    """Policy that returns chunks of 10 actions with grip values encoding the chunk number."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def new_session(self, context=None):
        self.reset_calls += 1
        self.last_reset_context = context
        return _ChunkSession(self)


class FakeRobotState:
    def __init__(self, translation: np.ndarray, joints: np.ndarray, status: RobotStatus) -> None:
        self.ee_pose = Transform3D(translation=translation, rotation=Rotation.identity)
        self.q = joints
        self.dq = np.zeros_like(joints)
        self.status = status


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


def make_robot_state(translation, joints, status=RobotStatus.AVAILABLE) -> FakeRobotState:
    translation = np.asarray(translation, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    return FakeRobotState(translation, joints, status)


def emit_ready_payload(frame_emitter, robot_emitter, grip_emitter, robot_state):
    frame_adapter = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.uint8)
    frame_adapter.array[:] = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_emitter.emit(frame_adapter)
    robot_emitter.emit(robot_state)
    grip_emitter.emit(0.25)


def _pair_all(world, harness):
    """Pair all harness signals and return a dict of test handles."""
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    return {
        'frame_em': world.pair(harness.observations['image.cam']),
        'robot_em': world.pair(harness.observations['robot_state']),
        'grip_em': world.pair(harness.observations['grip']),
        'directive_em': world.pair(harness.directive),
        'command_rx': world.pair(harness.commands['robot_command']),
        'grip_rx': world.pair(harness.commands['target_grip']),
        'meta_em': world.pair(harness.robot_meta_in),
        'ds_recorder': ds_recorder,
    }


def _ds_commands(p) -> list[DsWriterCommand]:
    return [data for _, data in p['ds_recorder'].emitted]


def _ds_types(p) -> list[DsWriterCommandType]:
    return [cmd.type for cmd in _ds_commands(p)]


def _last_command(p):
    """Extract the last robot command from the trajectory signal."""
    msg = p['command_rx'].read()
    if msg is None or msg.data is None:
        return None
    traj = msg.data  # list[tuple[float, CommandType]]
    return traj[-1][1] if traj else None


def _last_grip(p):
    """Extract the last grip value from the grip trajectory signal."""
    msg = p['grip_rx'].read()
    if msg is None or msg.data is None:
        return None
    traj = msg.data  # list[tuple[float, float]]
    return traj[-1][1] if traj else None


def _all_grips(p):
    """Extract all grip values from the grip trajectory signal."""
    msg = p['grip_rx'].read()
    if msg is None or msg.data is None:
        return []
    return [g for _, g in msg.data]


def _emitted_commands(recorder):
    """All robot commands across a recorder's non-empty emitted trajectories."""
    return [cmd for _ts, traj in recorder.emitted if traj for _t, cmd in traj]


def _emitted_grips(recorder):
    """All grip targets across a recorder's non-empty emitted trajectories."""
    return [g for _ts, traj in recorder.emitted if traj for _t, g in traj]


@pytest.mark.timeout(3.0)
def test_harness_emits_cartesian_move(world):
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    policy = SpyPolicy(command=CartesianPosition(pose=pose), target_grip=0.33)
    harness = Harness(policy, make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands['robot_command']._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='stack-blocks')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    assert policy.last_obs is not None
    obs = policy.last_obs
    assert 'image.cam' in obs
    expected_pose = np.concatenate([robot_state.ee_pose.translation, robot_state.ee_pose.rotation.as_quat])
    np.testing.assert_allclose(obs['robot_state.ee_pose'], expected_pose)
    np.testing.assert_allclose(obs['robot_state.q'], robot_state.q)
    np.testing.assert_allclose(obs['robot_state.dq'], np.zeros_like(robot_state.q))
    assert obs['grip'] == pytest.approx(0.25)
    assert obs['task'] == 'stack-blocks'
    assert obs['descriptor'] == ''  # no descriptor passed -> empty string reaches the policy
    # Recording == canonical policy I/O: the policy sees the same ``robot_state`` serializer
    # the dataset records (``.error`` int, not the raw ``RobotStatus``). wall/obs
    # timestamps carry volatile values, so lock the stable key set.
    assert set(obs) - {'wall_time_ns', 'obs_time_ns'} == {
        'image.cam',
        'robot_state.q',
        'robot_state.dq',
        'robot_state.ee_pose',
        'robot_state.error',
        'grip',
        'task',
        'descriptor',
    }

    # Last non-empty command (a trailing ``[]`` cancel is emitted on shutdown).
    cmds = _emitted_commands(cmd_recorder)
    assert cmds, 'no robot command emitted'
    cmd = cmds[-1]
    assert isinstance(cmd, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(cmd.pose.translation, pose.translation)
    np.testing.assert_allclose(cmd.pose.rotation.as_quat, pose.rotation.as_quat)

    grips = _emitted_grips(grip_recorder)
    assert grips and grips[-1] == pytest.approx(0.33)


@pytest.mark.timeout(3.0)
def test_harness_passes_descriptor_to_policy(world):
    """The embodiment descriptor reaches the policy on every call (stateless policy)."""
    policy = SpyPolicy()
    harness = Harness(policy, make_embodiment(descriptor='mujoco.franka'))
    harness.commands['robot_command']._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    assert policy.last_obs is not None
    assert policy.last_obs['descriptor'] == 'mujoco.franka'


@pytest.mark.timeout(3.0)
def test_harness_waits_for_complete_inputs(world):
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    policy = SpyPolicy(command=CartesianPosition(pose=pose), target_grip=0.33)
    harness = Harness(policy, make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands['robot_command']._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])
    directive_em = world.pair(harness.directive)

    assert 'image.cam' in harness.observations

    robot_state = make_robot_state([0.2, 0.0, -0.1], [0.7, 0.1, -0.2])

    def assert_no_outputs():
        assert not cmd_recorder.emitted
        assert not grip_recorder.emitted
        assert policy.last_obs is None

    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='dummy-task')), 0.01),
        (partial(robot_em.emit, robot_state), 0.01),
        (partial(grip_em.emit, 0.25), 0.01),
        (assert_no_outputs, 0.01),  # still missing a frame
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.01),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=30)

    assert policy.last_obs is not None

    cmds = _emitted_commands(cmd_recorder)
    assert cmds, 'no robot command emitted'
    cmd = cmds[-1]
    assert isinstance(cmd, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(cmd.pose.translation, pose.translation)

    grips = _emitted_grips(grip_recorder)
    assert grips and grips[-1] == pytest.approx(0.33)


@pytest.mark.timeout(3.0)
def test_run_emits_ds_start_with_meta(world):
    policy = StubPolicy(meta={'type': 'stub', 'checkpoint': 'v1'})
    harness = Harness(policy, make_embodiment(), static_meta={'joint_signal': 'robot_state.q'})
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['meta_em'].emit, {'urdf': '<robot/>', 'joint_names': ['j1']}), 0.0),
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.01),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=15)

    starts = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.START_EPISODE]
    assert len(starts) == 1
    meta = starts[0].static_data
    assert meta['joint_signal'] == 'robot_state.q'
    assert meta['urdf'] == '<robot/>'
    assert meta['joint_names'] == ['j1']
    assert meta['inference.policy.type'] == 'stub'
    assert meta['inference.policy.checkpoint'] == 'v1'
    assert meta['task'] == 'test'


@pytest.mark.timeout(3.0)
def test_episode_meta_includes_policy_static_meta(world):
    """Static fields exposed only via ``Policy.meta`` (empty ``Session.meta``) must
    still reach episode metadata once the policy is wrapped."""

    class _StaticMetaSession(Session):
        def __init__(self, command):
            self._command = command

        def __call__(self, obs):
            return [{'robot_command': self._command, 'target_grip': 0.0, 'timestamp': 0.0}]

    class _StaticMetaPolicy(Policy):
        def __init__(self):
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            self._command = CartesianPosition(pose=pose)

        def new_session(self, context=None):
            return _StaticMetaSession(self._command)  # Session.meta defaults to {}

        @property
        def meta(self):
            return {'checkpoint': 'v1', 'type': 'static'}

    harness = Harness(_StaticMetaPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (None, 0.02),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=15)

    starts = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.START_EPISODE]
    assert len(starts) == 1
    meta = starts[0].static_data
    assert meta['inference.policy.checkpoint'] == 'v1'
    assert meta['inference.policy.type'] == 'static'


@pytest.mark.timeout(3.0)
def test_finish_emits_ds_stop_with_data_and_homes(world):
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.0),
        (partial(p['directive_em'].emit, Directive.FINISH(outcome='Success', notes='good')), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['outcome'] == 'Success'
    assert stops[0].static_data['notes'] == 'good'

    assert isinstance(_last_command(p), Reset)


@pytest.mark.timeout(3.0)
def test_trial_timeout_self_terminates(world):
    """A self-driven trial ends at ``task.timeout``: terminated=False, robot homed."""
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment(), task=Task(instruction='test', timeout=0.05), trials=[{}])
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=200)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is False
    assert isinstance(_last_command(p), Reset)


@pytest.mark.timeout(3.0)
def test_trial_seed_reaches_task_reset_and_meta(world):
    """Each RUN hands its ``eval.seed`` to the task's scene reset; the seed and the
    eval-identity block land in episode meta."""
    policy = StubPolicy()
    seeds = []
    trials = [{'eval.seed': 7 + i} for i in range(2)]
    task = Task(instruction='stack', timeout=0.05, reset=seeds.append)
    harness = Harness(policy, make_embodiment(), task=task, trials=trials)
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=400)

    assert seeds == [7, 8]
    starts = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.START_EPISODE]
    assert [s.static_data['eval.seed'] for s in starts] == [7, 8]
    assert all(s.static_data['eval.universe'] == 'real' for s in starts)
    assert all(s.static_data['eval.embodiment'] == '' for s in starts)
    assert all(s.static_data['eval.timeout'] == 0.05 for s in starts)


@pytest.mark.timeout(3.0)
def test_trial_plan_self_drives(world):
    """With a trial plan the harness runs unattended: no driver, one episode per entry."""
    policy = StubPolicy()
    trials = [{'eval.trial_index': i} for i in range(2)]
    harness = Harness(policy, make_embodiment(), task=Task(instruction='stack', timeout=0.05), trials=trials)
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=400)

    starts = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.START_EPISODE]
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert [s.static_data['eval.trial_index'] for s in starts] == [0, 1]
    assert all(s.static_data['task'] == 'stack' for s in starts)
    assert len(stops) == 2
    assert all(s.static_data['eval.terminated'] is False for s in stops)
    assert policy.reset_calls == 2


@pytest.mark.timeout(3.0)
def test_timeout_crossed_during_latency_sleep_drops_chunk(world):
    """A chunk whose latency sleep crosses the deadline is dropped, never emitted."""
    policy = StubPolicy()
    # The 0.2s latency sleep crosses the 0.05s deadline before the chunk is emitted.
    harness = Harness(
        policy, make_embodiment(), task=Task(instruction='test', timeout=0.05), trials=[{'inference_latency': 0.2}]
    )
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    ds_recorder = RecordingEmitter()
    harness.commands['robot_command']._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(ds_recorder)

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([(partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01), (None, 0.3)])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    stops = [data for _, data in ds_recorder.emitted if data.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is False
    # The post-deadline chunk must not reach the drivers: the only non-empty emissions are the homing
    # Reset / home grip from the timeout FINISH.
    assert all(isinstance(c, Reset) for c in _emitted_commands(cmd_recorder))
    assert _emitted_grips(grip_recorder) == [0.0]


@pytest.mark.timeout(3.0)
def test_home_aborts_recording_and_homes(world):
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.0),
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (partial(p['directive_em'].emit, Directive.HOME()), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    assert DsWriterCommandType.ABORT_EPISODE in _ds_types(p)

    assert isinstance(_last_command(p), Reset)

    assert policy.reset_calls == 1  # only from RUN


@pytest.mark.timeout(3.0)
def test_run_while_running_auto_finalizes(world):
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='ep1')), 0.0),
        (partial(p['directive_em'].emit, Directive.RUN(task='ep2')), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    types = _ds_types(p)
    # START(ep1), STOP (auto-finalize), START(ep2), STOP (shutdown cleanup)
    assert types.count(DsWriterCommandType.START_EPISODE) == 2
    assert types.count(DsWriterCommandType.STOP_EPISODE) == 2
    assert policy.reset_calls == 2


@pytest.mark.timeout(3.0)
def test_run_calls_policy_reset_with_context(world):
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([(partial(p['directive_em'].emit, Directive.RUN(task='test-task')), 0.0), (None, 0.01)])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=5)

    assert policy.reset_calls == 1
    assert policy.last_reset_context == {'task': 'test-task'}


@pytest.mark.timeout(3.0)
def test_finish_cancels_buffered_trajectory_before_stop_episode(world):
    """FINISH must cancel the recording's trajectory tail *before* `STOP_EPISODE`.

    `STOP_EPISODE` calls `flush()` on `TrajectoryOverrideSerializer`, which
    commits whatever is still buffered. The harness must emit `[]` on
    `robot_command`/`target_grip` first, so the serializer drops its tail and
    canceled waypoints are not recorded.
    """

    class _LabeledRecorder(pimm.SignalEmitter):
        def __init__(self, label, events):
            self._label = label
            self._events = events

        def emit(self, data, ts: int = -1):
            self._events.append((self._label, data))

    events: list[tuple[str, object]] = []
    policy = ChunkPolicy()
    wrapped = ActionTimestamp(fps=5.0).wrap(policy)  # 1.8 s chunk — won't drain before FINISH
    harness = Harness(wrapped, make_embodiment())
    harness.commands['robot_command']._bind(_LabeledRecorder('robot_command', events))
    harness.commands['target_grip']._bind(_LabeledRecorder('target_grip', events))
    harness.ds_command._bind(_LabeledRecorder('ds_command', events))

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    script = [
        (partial(directive_em.emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
        (partial(directive_em.emit, Directive.FINISH()), 0.0),
        (None, 0.1),
    ]
    scheduler = world.start([harness, ManualDriver(script)])
    drive_scheduler(scheduler, steps=200)

    cancels = [i for i, (lbl, data) in enumerate(events) if lbl == 'robot_command' and data == []]
    stops = [
        i
        for i, (lbl, data) in enumerate(events)
        if lbl == 'ds_command' and getattr(data, 'type', None) is DsWriterCommandType.STOP_EPISODE
    ]
    assert cancels, 'FINISH did not emit a cancel on robot_command'
    assert stops, 'FINISH did not emit STOP_EPISODE'
    assert cancels[0] < stops[0], (
        f'cancel ({cancels[0]}) must precede STOP_EPISODE ({stops[0]}); otherwise flush() commits canceled waypoints'
    )


@pytest.mark.timeout(3.0)
def test_empty_chunk_cancels_both_robot_and_grip(world):
    """A session returning ``[]`` must cancel *both* driver buffers.

    Empty action chunk is the session-level cancel signal (per the
    ``Session.__call__`` contract). If only ``robot_command`` gets ``[]`` while
    ``target_grip`` is skipped, the gripper ``TrajectoryPlayer`` keeps draining
    stale waypoints — a partial cancel that's worse than no cancel.
    """

    class _EmptyChunkSession(Session):
        def __call__(self, obs):
            return []

    class EmptyChunkPolicy(Policy):
        def new_session(self, context=None):
            return _EmptyChunkSession()

    harness = Harness(EmptyChunkPolicy(), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands['robot_command']._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    script = [
        (partial(directive_em.emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
    ]
    scheduler = world.start([harness, ManualDriver(script)])
    drive_scheduler(scheduler, steps=200)

    cmd_emits = [data for _ts, data in cmd_recorder.emitted]
    grip_emits = [data for _ts, data in grip_recorder.emitted]
    assert [] in cmd_emits, 'empty chunk did not cancel robot_command buffer'
    assert [] in grip_emits, 'empty chunk did not cancel target_grip buffer'


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_home(world):
    """Verify that HOME resets trajectory state so next RUN gets a fresh chunk."""
    policy = ChunkPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=5)

    grips = _all_grips(p)
    assert grips[0] >= 100.0, f'Expected chunk 1, got {grips}'

    p['directive_em'].emit(Directive.HOME())
    drive_scheduler(scheduler, steps=2)

    assert _last_grip(p) == 0.0, 'Expected 0.0 (Home)'

    p['directive_em'].emit(Directive.RUN(task='test'))
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=4)

    grips = _all_grips(p)
    assert grips[0] >= 200.0, f'Expected chunk 2 (>= 200.0), got {grips}. Trajectory clearing failed!'


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_run(world):
    """Verify that RUN resets trajectory state so a fresh chunk is emitted."""
    policy = ChunkPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=5)

    grips = _all_grips(p)
    assert grips[0] >= 100.0

    p['directive_em'].emit(Directive.RUN(task='test-restart'))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=4)

    grips = _all_grips(p)
    assert grips[0] >= 200.0, f'Expected chunk 2 (>= 200.0), got {grips}. Trajectory clearing on RUN failed!'


@pytest.mark.timeout(3.0)
def test_harness_recovers_from_error(world):
    """ERROR emits Recover trajectory, skips policy; AVAILABLE resumes with fresh chunk."""
    policy = ChunkPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    state_ok = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    state_err = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.ERROR)

    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, steps=1)
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, steps=3)
    grips = _all_grips(p)
    assert grips[0] >= 100.0

    obs_before = len(policy.observations)
    p['robot_em'].emit(state_err)
    drive_scheduler(scheduler, steps=2)
    assert isinstance(_last_command(p), Recover)
    assert len(policy.observations) == obs_before

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, steps=3)
    grips = _all_grips(p)
    assert grips[0] >= 200.0


def test_directive_preserves_payload():
    assert Directive.RUN(task='test').payload == {'task': 'test'}
    assert Directive.FINISH(outcome='Success').payload == {'outcome': 'Success'}
    assert Directive.FINISH().payload == {}
    assert Directive.HOME().payload == 'home'
    assert Directive.HOME('zeros').payload == 'zeros'


def test_directive_types():
    assert DirectiveType.RUN.value == 'run'
    assert DirectiveType.FINISH.value == 'finish'
    assert DirectiveType.HOME.value == 'home'


def test_recover_command_wire_roundtrip():
    wire = to_wire(Recover())
    assert wire == {'type': 'recover'}
    assert isinstance(from_wire(wire), Recover)


@pytest.mark.parametrize('status, expected_error', [(RobotStatus.AVAILABLE, 0), (RobotStatus.ERROR, 1)])
def test_robot_state_serializer_records_error(status, expected_error):
    state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=status)
    assert Serializers.robot_state(state)['.error'] == expected_error


def test_robot_state_serializer_drops_resetting():
    state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.RESETTING)
    assert Serializers.robot_state(state) is None


@pytest.mark.timeout(3.0)
def test_recovery_cancels_gripper_buffer(world):
    """Entering recovery must cancel the gripper buffer, not just the arm.

    The Recover-only chunk carries no ``target_grip``; without an explicit cancel
    the gripper ``TrajectoryPlayer`` keeps draining the interrupted chunk's grip
    waypoints while the robot recovers.
    """
    harness = Harness(ChunkPolicy(), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands['robot_command']._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])
    directive_em = world.pair(harness.directive)

    state_ok = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    state_err = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.ERROR)

    scheduler = world.start([harness])
    directive_em.emit(Directive.RUN(task='t'))
    drive_scheduler(scheduler, steps=1)
    emit_ready_payload(frame_em, robot_em, grip_em, state_ok)
    drive_scheduler(scheduler, steps=3)

    cmd_before = len(cmd_recorder.emitted)
    grip_before = len(grip_recorder.emitted)
    robot_em.emit(state_err)
    drive_scheduler(scheduler, steps=2)

    new_cmds = [data for _ts, data in cmd_recorder.emitted[cmd_before:]]
    new_grips = [data for _ts, data in grip_recorder.emitted[grip_before:]]
    assert any(isinstance(t, list) and t and isinstance(t[-1][1], Recover) for t in new_cmds), (
        'recovery did not emit a Recover on robot_command'
    )
    assert [] in new_grips, 'recovery did not cancel the gripper buffer'


@pytest.mark.timeout(3.0)
def test_shutdown_cancels_trajectory_before_stop(world):
    """Shutdown while recording must cancel buffered trajectories before STOP_EPISODE.

    ``STOP_EPISODE`` flushes ``TrajectoryOverrideSerializer``; without a prior
    cancel it would commit the unexecuted tail of an in-flight chunk (the
    FINISH/RUN paths already cancel first).
    """
    events: list[tuple[str, object]] = []

    class _LabeledRecorder(pimm.SignalEmitter):
        def __init__(self, label):
            self._label = label

        def emit(self, data, ts: int = -1):
            events.append((self._label, data))

    wrapped = ActionTimestamp(fps=5.0).wrap(ChunkPolicy())  # 1.8 s chunk — won't drain before shutdown
    harness = Harness(wrapped, make_embodiment())
    harness.commands['robot_command']._bind(_LabeledRecorder('robot_command'))
    harness.commands['target_grip']._bind(_LabeledRecorder('target_grip'))
    harness.ds_command._bind(_LabeledRecorder('ds_command'))

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations['grip'])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # RUN + a complete obs buffers a chunk; the driver then ends, which makes the
    # world signal shutdown while still recording — exercising the run() finalizer.
    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    cancels = [i for i, (lbl, data) in enumerate(events) if lbl == 'robot_command' and data == []]
    stops = [
        i
        for i, (lbl, data) in enumerate(events)
        if lbl == 'ds_command' and getattr(data, 'type', None) is DsWriterCommandType.STOP_EPISODE
    ]
    assert cancels, 'shutdown did not cancel robot_command'
    assert stops, 'shutdown did not emit STOP_EPISODE'
    assert cancels[0] < stops[0], 'cancel must precede STOP_EPISODE on shutdown'
