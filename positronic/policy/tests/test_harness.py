from functools import partial

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.dataset.ds_writer_agent import DsWriterCommand, DsWriterCommandType, Serializers
from positronic.drivers import roboarm
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianPosition, Recover, Reset, from_wire, to_wire
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Policy, Session
from positronic.policy.harness import ChunkedSchedule, Directive, DirectiveType, ErrorRecovery, Harness
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler


class _SpySession(Session):
    def __init__(self, policy):
        self._policy = policy

    def __call__(self, obs):
        self._policy.last_obs = obs
        return {'robot_command': to_wire(self._policy.command), 'target_grip': self._policy.target_grip}


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
        return {'robot_command': to_wire(self._policy.command), 'target_grip': self._policy.target_grip}

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
        return [
            {'robot_command': to_wire(self._policy.command), 'target_grip': self._policy.counter * 100.0 + i}
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
def clock():
    return MockClock()


@pytest.fixture
def world(clock):
    with pimm.World(clock=clock) as w:
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
        'frame_em': world.pair(harness.frames['image.cam']),
        'robot_em': world.pair(harness.robot_state),
        'grip_em': world.pair(harness.gripper_state),
        'directive_em': world.pair(harness.directive),
        'command_rx': world.pair(harness.robot_commands),
        'grip_rx': world.pair(harness.target_grip),
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


@pytest.mark.timeout(3.0)
def test_harness_emits_cartesian_move(world, clock):
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    policy = SpyPolicy(command=CartesianPosition(pose=pose), target_grip=0.33)
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='stack-blocks')), 0.0),
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=20)

    assert policy.last_obs is not None
    obs = policy.last_obs
    assert 'image.cam' in obs
    expected_pose = np.concatenate([robot_state.ee_pose.translation, robot_state.ee_pose.rotation.as_quat])
    np.testing.assert_allclose(obs['robot_state.ee_pose'], expected_pose)
    np.testing.assert_allclose(obs['robot_state.q'], robot_state.q)
    np.testing.assert_allclose(obs['robot_state.dq'], np.zeros_like(robot_state.q))
    assert obs['grip'] == pytest.approx(0.25)
    assert obs['task'] == 'stack-blocks'

    cmd = _last_command(p)
    assert isinstance(cmd, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(cmd.pose.translation, pose.translation)
    np.testing.assert_allclose(cmd.pose.rotation.as_quat, pose.rotation.as_quat)

    assert _last_grip(p) == pytest.approx(0.33)


@pytest.mark.timeout(3.0)
def test_harness_waits_for_complete_inputs(world, clock):
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    policy = SpyPolicy(command=CartesianPosition(pose=pose), target_grip=0.33)
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    assert len(harness.frames) == 1

    robot_state = make_robot_state([0.2, 0.0, -0.1], [0.7, 0.1, -0.2])

    def assert_no_outputs():
        assert p['command_rx'].read() is None
        assert p['grip_rx'].read() is None
        assert policy.last_obs is None

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='dummy-task')), 0.01),
        (partial(p['robot_em'].emit, robot_state), 0.01),
        (partial(p['grip_em'].emit, 0.25), 0.01),
        (assert_no_outputs, 0.01),  # still missing a frame
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (None, 0.01),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=30)

    assert policy.last_obs is not None

    cmd = _last_command(p)
    assert isinstance(cmd, roboarm.command.CartesianPosition)
    np.testing.assert_allclose(cmd.pose.translation, pose.translation)

    assert _last_grip(p) == pytest.approx(0.33)


@pytest.mark.timeout(3.0)
def test_run_emits_ds_start_with_meta(world, clock):
    policy = StubPolicy(meta={'type': 'stub', 'checkpoint': 'v1'})
    harness = Harness(ChunkedSchedule().wrap(policy), static_meta={'joint_signal': 'robot_state.q'})
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['meta_em'].emit, {'urdf': '<robot/>', 'joint_names': ['j1']}), 0.0),
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.01),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=15)

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
def test_stop_emits_ds_suspend(world, clock):
    policy = StubPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.0),
        (partial(p['directive_em'].emit, Directive.STOP()), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=15)

    assert DsWriterCommandType.SUSPEND_EPISODE in _ds_types(p)


@pytest.mark.timeout(3.0)
def test_finish_emits_ds_stop_with_data_and_homes(world, clock):
    policy = StubPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.0),
        (partial(p['directive_em'].emit, Directive.STOP()), 0.02),
        (partial(p['directive_em'].emit, Directive.FINISH(outcome='Success', notes='good')), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=20)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['outcome'] == 'Success'
    assert stops[0].static_data['notes'] == 'good'

    assert isinstance(_last_command(p), Reset)


@pytest.mark.timeout(3.0)
def test_home_aborts_recording_and_homes(world, clock):
    policy = StubPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.0),
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (partial(p['directive_em'].emit, Directive.HOME()), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=20)

    assert DsWriterCommandType.ABORT_EPISODE in _ds_types(p)

    assert isinstance(_last_command(p), Reset)

    assert policy.reset_calls == 1  # only from RUN


@pytest.mark.timeout(3.0)
def test_run_from_paused_auto_finalizes(world, clock):
    policy = StubPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='ep1')), 0.0),
        (partial(p['directive_em'].emit, Directive.STOP()), 0.02),
        (partial(p['directive_em'].emit, Directive.RUN(task='ep2')), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=20)

    types = _ds_types(p)
    # START(ep1), SUSPEND, STOP (auto-finalize), START(ep2), STOP (cleanup)
    assert types.count(DsWriterCommandType.START_EPISODE) == 2
    assert types.count(DsWriterCommandType.STOP_EPISODE) == 2  # auto-finalize + cleanup
    assert types.count(DsWriterCommandType.SUSPEND_EPISODE) == 1
    assert policy.reset_calls == 2


@pytest.mark.timeout(3.0)
def test_run_calls_policy_reset_with_context(world, clock):
    policy = StubPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    driver = ManualDriver([(partial(p['directive_em'].emit, Directive.RUN(task='test-task')), 0.0), (None, 0.01)])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, clock=clock, steps=5)

    assert policy.reset_calls == 1
    assert policy.last_reset_context == {'task': 'test-task'}


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_home(world, clock):
    """Verify that HOME resets trajectory state so next RUN gets a fresh chunk."""
    policy = ChunkPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, clock=clock, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, clock=clock, steps=5)

    grips = _all_grips(p)
    assert grips[0] >= 100.0, f'Expected chunk 1, got {grips}'

    p['directive_em'].emit(Directive.HOME())
    drive_scheduler(scheduler, clock=clock, steps=2)

    assert _last_grip(p) == 0.0, 'Expected 0.0 (Home)'

    p['directive_em'].emit(Directive.RUN(task='test'))
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, clock=clock, steps=4)

    grips = _all_grips(p)
    assert grips[0] >= 200.0, f'Expected chunk 2 (>= 200.0), got {grips}. Trajectory clearing failed!'


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_run(world, clock):
    """Verify that RUN resets trajectory state so a fresh chunk is emitted."""
    policy = ChunkPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy))
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, clock=clock, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, clock=clock, steps=5)

    grips = _all_grips(p)
    assert grips[0] >= 100.0

    p['directive_em'].emit(Directive.RUN(task='test-restart'))
    drive_scheduler(scheduler, clock=clock, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, clock=clock, steps=4)

    grips = _all_grips(p)
    assert grips[0] >= 200.0, f'Expected chunk 2 (>= 200.0), got {grips}. Trajectory clearing on RUN failed!'


@pytest.mark.timeout(3.0)
def test_harness_recovers_from_error(world, clock):
    """ERROR emits Recover trajectory, skips policy; AVAILABLE resumes with fresh chunk."""
    policy = ChunkPolicy()
    harness = Harness(ErrorRecovery().wrap(ChunkedSchedule().wrap(policy)))
    p = _pair_all(world, harness)

    state_ok = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    state_err = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.ERROR)

    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, clock=clock, steps=1)
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, clock=clock, steps=3)
    grips = _all_grips(p)
    assert grips[0] >= 100.0

    obs_before = len(policy.observations)
    p['robot_em'].emit(state_err)
    drive_scheduler(scheduler, clock=clock, steps=2)
    assert isinstance(_last_command(p), Recover)
    assert len(policy.observations) == obs_before

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, clock=clock, steps=3)
    grips = _all_grips(p)
    assert grips[0] >= 200.0


def test_directive_preserves_payload():
    assert Directive.RUN(task='test').payload == {'task': 'test'}
    assert Directive.STOP().payload is None
    assert Directive.FINISH(outcome='Success').payload == {'outcome': 'Success'}
    assert Directive.FINISH().payload == {}
    assert Directive.HOME().payload == 'home'
    assert Directive.HOME('zeros').payload == 'zeros'


def test_directive_types():
    assert DirectiveType.RUN.value == 'run'
    assert DirectiveType.STOP.value == 'stop'
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
