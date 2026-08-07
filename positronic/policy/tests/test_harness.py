from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import pimm
from positronic import keys, telemetry, telemetry_keys, wire
from positronic.dataset.ds_writer_agent import DsWriterCommand, DsWriterCommandType
from positronic.dataset.serializers import Serializers
from positronic.drivers import roboarm
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import (
    CartesianDelta,
    CartesianPosition,
    JointDelta,
    JointPosition,
    Reset,
    TrajectoryPlayer,
    _compose_delta,
    from_wire,
    reduce,
    to_wire,
)
from positronic.drivers.roboarm.models import DEFAULT_FRAME, EE_LINK, bundled_franka_model
from positronic.eval import Command, Embodiment, Observation, Task
from positronic.geom import Rotation, Transform3D
from positronic.offboard.client import InferenceSession
from positronic.policy.base import Policy, Session
from positronic.policy.codec import ActionTimestamp
from positronic.policy.harness import FINISH_HOME_GRACE_NS, Directive, DirectiveType, Harness
from positronic.policy.remote import RemoteSession
from positronic.policy.wrappers import ChunkedSchedule
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler


@contextmanager
def _eval_pass(run_id: str):
    """The eval CLI's pass span, which the harness's episode spans parent to: a span its owner holds open and
    anchors, rather than entering as the OTel-current span."""
    span = telemetry.start_span(telemetry_keys.SPAN_EVAL_PASS, **{telemetry.ATTR_RUN_ID: run_id})
    telemetry.push_anchor(span)
    try:
        yield
    finally:
        span.end()
        telemetry.pop_anchor(span)


def make_embodiment(descriptor: str = '', cameras=('image.cam',), static_meta=None) -> Embodiment:
    """Minimal Franka-shaped embodiment for harness unit tests.

    The sources/dests are no-ops: these tests pair the harness ports directly
    (never via ``wire_embodiment``), so only the spec — names, serializers,
    home values, descriptor — is read by the Harness.
    """
    observations = {
        'robot_state': Observation(pimm.NoOpEmitter(), Serializers.robot_state),
        keys.GRIP: Observation(pimm.NoOpEmitter(), None),
    }
    for cam in cameras:
        observations[cam] = Observation(pimm.NoOpEmitter(), Serializers.camera_images)
    commands = {
        keys.ROBOT_COMMAND: Command(pimm.NoOpReceiver(), Reset(), Serializers.robot_command),
        'target_grip': Command(pimm.NoOpReceiver(), 0.0, None),
    }
    return Embodiment(descriptor, observations, commands, static_meta or {}, pimm.NoOpEmitter())


class _SpySession(Session):
    def __init__(self, policy):
        self._policy = policy

    def __call__(self, obs):
        self._policy.last_obs = obs
        return [{keys.ROBOT_COMMAND: self._policy.command, 'target_grip': self._policy.target_grip, 'timestamp': 0.0}]


class SpyPolicy(Policy):
    def __init__(self, command: roboarm.command.CommandType | None = None, target_grip: float = 0.33) -> None:
        if command is None:
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
        self.command = command
        self.target_grip = float(target_grip)
        self.last_obs: dict[str, Any] | None = None
        self.reset_calls: int = 0
        self.last_reset_context = None

    def new_session(self, context=None, now=None):
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
        return [{keys.ROBOT_COMMAND: self._policy.command, 'target_grip': self._policy.target_grip, 'timestamp': 0.0}]

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
        self.last_obs: dict[str, Any] | None = None
        self.observations: list[dict[str, object]] = []
        self.reset_calls = 0
        self.last_reset_context = None
        self._meta: dict[str, object] = meta or {}

    @property
    def meta(self) -> dict[str, object]:
        return self._meta

    def new_session(self, context=None, now=None):
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
                keys.ROBOT_COMMAND: self._policy.command,
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

    def new_session(self, context=None, now=None):
        self.reset_calls += 1
        self.last_reset_context = context
        return _ChunkSession(self)


class _FakeInferenceSession(InferenceSession):
    """A stub ``InferenceSession`` returning a canned action, so a ``RemoteSession`` over it round-trips
    ``RemoteSession.__call__`` — the real inference boundary that records the ``policy.infer`` span."""

    def __init__(self, action: list[dict[str, Any]]) -> None:
        self._action = action

    def infer(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        return self._action

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


class RemoteStubPolicy(Policy):
    """A stub policy served through a real ``RemoteSession`` over a fake inference session, so its inference
    round-trips ``RemoteSession.__call__`` and records the ``policy.infer`` span independent of any wrapper."""

    def __init__(self, command: roboarm.command.CommandType | None = None, target_grip: float = 0.33) -> None:
        if command is None:
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
        self.command = command
        self.target_grip = float(target_grip)

    def new_session(self, context=None, now=None) -> RemoteSession:
        action = [{'robot_command': self.command, 'target_grip': self.target_grip, 'timestamp': 0.0}]
        return RemoteSession(_FakeInferenceSession(action))


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
        'grip_em': world.pair(harness.observations[keys.GRIP]),
        'directive_em': world.pair(harness.directive),
        'command_rx': world.pair(harness.commands[keys.ROBOT_COMMAND]),
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
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
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
    np.testing.assert_allclose(obs[keys.EE_POSE], expected_pose)
    np.testing.assert_allclose(obs[keys.JOINTS], robot_state.q)
    np.testing.assert_allclose(obs[keys.JOINT_VEL], np.zeros_like(robot_state.q))
    assert obs[keys.GRIP] == pytest.approx(0.25)
    assert obs[keys.TASK] == 'stack-blocks'
    assert obs['descriptor'] == ''  # no descriptor passed -> empty string reaches the policy
    # Recording == canonical policy I/O: the policy sees the same ``robot_state`` serializer
    # the dataset records. wall/obs timestamps carry volatile values, so lock the stable key set.
    assert set(obs) - {'wall_time_ns', 'obs_time_ns'} == {
        'image.cam',
        keys.JOINTS,
        keys.JOINT_VEL,
        keys.EE_POSE,
        keys.GRIP,
        keys.TASK,
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
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
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
def test_robot_model_stays_out_of_the_observation(world):
    """A codec carries its frame as a transform, so the model never has to leave the rig."""
    policy = SpyPolicy()
    model = bundled_franka_model()
    statics = {keys.URDF: model[keys.URDF], keys.CONTROL_FRAME: model[keys.CONTROL_FRAME]}
    harness = Harness(policy, make_embodiment(static_meta=statics))
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
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
    assert keys.URDF not in policy.last_obs and keys.CONTROL_FRAME not in policy.last_obs


@pytest.mark.timeout(3.0)
def _run_with_model(world, model, static_meta=None):
    """Drive one episode with ``model`` published on ``robot_meta_in``, or baked into embodiment statics."""
    harness = Harness(SpyPolicy(), make_embodiment(static_meta=static_meta))
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())
    directive_em = world.pair(harness.directive)
    meta_em = world.pair(harness.robot_meta_in)

    steps = [(partial(directive_em.emit, Directive.RUN(task='t')), 0.0)]
    if model is not None:
        steps.append((partial(meta_em.emit, model), 0.01))
    drive_scheduler(world.start([harness, ManualDriver([*steps, (None, 0.05)])]), steps=20)


@pytest.mark.timeout(3.0)
def test_rejects_a_control_frame_that_is_not_the_default(world):
    """A rig reporting at another of its own frames shifts every codec transform by the offset between them."""
    statics = {keys.URDF: bundled_franka_model()[keys.URDF], keys.CONTROL_FRAME: EE_LINK}
    with pytest.raises(ValueError, match=EE_LINK):
        _run_with_model(world, None, static_meta=statics)


@pytest.mark.timeout(3.0)
def test_rejects_a_default_frame_the_model_does_not_declare(world):
    """Every frame transform is measured from this one, so a name the model lacks must not run."""
    statics = {keys.URDF: '<robot name="r"><link name="base"/></robot>', keys.CONTROL_FRAME: DEFAULT_FRAME}
    with pytest.raises(ValueError, match=DEFAULT_FRAME):
        _run_with_model(world, None, static_meta=statics)


@pytest.mark.timeout(3.0)
def test_rejects_a_control_frame_a_late_model_declares(world):
    """A remote env publishes its model a turn after the reset that produced it, so the check runs on the
    live metadata rather than on whatever was known when the episode opened."""
    model = {keys.URDF: bundled_franka_model()[keys.URDF], keys.CONTROL_FRAME: EE_LINK}
    with pytest.raises(ValueError, match=EE_LINK):
        _run_with_model(world, model)


@pytest.mark.timeout(3.0)
def test_harness_waits_for_complete_inputs(world):
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    policy = SpyPolicy(command=CartesianPosition(pose=pose), target_grip=0.33)
    harness = Harness(policy, make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
    directive_em = world.pair(harness.directive)

    assert 'image.cam' in harness.observations

    robot_state = make_robot_state([0.2, 0.0, -0.1], [0.7, 0.1, -0.2])

    def assert_no_inference():
        # The startup home may have emitted (Reset / grip 0.0); the policy must not have run on partial inputs.
        assert policy.last_obs is None
        assert all(isinstance(c, Reset) for c in _emitted_commands(cmd_recorder))

    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='dummy-task')), 0.01),
        (partial(robot_em.emit, robot_state), 0.01),
        (partial(grip_em.emit, 0.25), 0.01),
        (assert_no_inference, 0.01),  # still missing a frame
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
def test_episode_meta_stamped_at_finalize(world):
    policy = StubPolicy(meta={'type': 'stub', 'checkpoint': 'v1'})
    harness = Harness(policy, make_embodiment(), static_meta={'joint_signal': keys.JOINTS})
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['meta_em'].emit, {keys.URDF: '<robot/>', keys.JOINT_NAMES: ['j1']}), 0.0),
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.01),
        (partial(p['directive_em'].emit, Directive.FINISH()), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=25)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    meta = stops[0].static_data
    assert meta['joint_signal'] == keys.JOINTS
    assert meta[keys.URDF] == '<robot/>'
    assert meta[keys.JOINT_NAMES] == ['j1']
    assert meta['inference.policy.type'] == 'stub'
    assert meta['inference.policy.checkpoint'] == 'v1'
    assert meta[keys.TASK] == 'test'


@pytest.mark.timeout(3.0)
def test_episode_meta_includes_policy_static_meta(world):
    """Static fields exposed only via ``Policy.meta`` (empty ``Session.meta``) must
    still reach episode metadata once the policy is wrapped."""

    class _StaticMetaSession(Session):
        def __init__(self, command):
            self._command = command

        def __call__(self, obs):
            return [{keys.ROBOT_COMMAND: self._command, 'target_grip': 0.0, 'timestamp': 0.0}]

    class _StaticMetaPolicy(Policy):
        def __init__(self):
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            self._command = CartesianPosition(pose=pose)

        def new_session(self, context=None, now=None):
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
        (partial(p['directive_em'].emit, Directive.FINISH()), 0.02),
        (None, 0.02),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=25)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    meta = stops[0].static_data
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
def test_attended_task_run_respects_timeout(world):
    """A task's ``timeout`` bounds an attended (directive-driven) run too: RUN arrives but no FINISH, yet
    the trial still self-terminates at the deadline. The deadline is armed whenever a task is supplied, not
    only on the self-driven ``trials`` path."""
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment(), task=Task(instruction='test', timeout=0.05))
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, steps=200)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is False
    assert isinstance(_last_command(p), Reset)


@pytest.mark.timeout(3.0)
def test_attended_task_run_respects_done(world):
    """The privileged ``done`` ends an attended run too: a fresh terminal within budget terminates the
    episode even though no FINISH arrives. ``done`` is honored whenever a task supplies it, attended or
    self-driven."""
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment(), task=Task(instruction='test', timeout=100.0))
    p = _pair_all(world, harness)
    done_em = world.pair(harness.done)

    scheduler = world.start([harness])
    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, steps=5)
    assert not [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]

    done_em.emit({'eval.success': True})
    drive_scheduler(scheduler, steps=10)
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is True
    assert stops[0].static_data['eval.success'] is True


@pytest.mark.timeout(3.0)
def test_trial_stop_signal_terminates(world):
    """Delivering the privileged ``done`` ends a trial early: terminated=True, payload recorded, homed."""
    policy = StubPolicy()
    # Timeout far in the future so the stop-signal, not the clock, ends the trial.
    harness = Harness(policy, make_embodiment(), task=Task(instruction='test', timeout=100.0), trials=[{}])
    p = _pair_all(world, harness)
    done_em = world.pair(harness.done)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=5)
    # Trial is live and unbounded by the clock: nothing committed yet.
    assert not [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]

    done_em.emit({'eval.success': True})
    drive_scheduler(scheduler, steps=10)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is True
    assert stops[0].static_data['eval.success'] is True  # the delivered payload lands in static data
    assert isinstance(_last_command(p), Reset)


@pytest.mark.timeout(3.0)
def test_stale_done_does_not_terminate_next_trial(world):
    """``done`` latches (last-writer-wins): trial 0's terminal would re-fire on trial 1, whose later
    deadline still sits after the stale timestamp. Only a freshly delivered ``done`` terminates, so the
    latched value is ignored — no producer ``reset`` clears it here (``reset`` is ``None``, as on a real
    embodiment). A falsy payload never terminates; trial 1 runs until its own fresh terminal lands."""
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment(), task=Task(instruction='t', timeout=100.0), trials=[{}, {}])
    p = _pair_all(world, harness)
    done_em = world.pair(harness.done)

    def stop_count():
        return len([c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE])

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=5)
    done_em.emit({})  # falsy: does not terminate
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 0

    done_em.emit({'eval.success': True})  # fresh truthy: ends trial 0
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 1

    # Trial 1 auto-started. The terminal is still latched but no longer fresh, so it must NOT re-fire.
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 1

    done_em.emit({'eval.success': True})  # a fresh delivery ends trial 1
    drive_scheduler(scheduler, steps=10)
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 2
    assert all(s.static_data['eval.terminated'] is True for s in stops)


class _FrameIndexDevice(pimm.ControlSystem):
    """Publishes a rising frame index on ``state``. ``reset`` arms frame-0; the run loop publishes it (with
    fresh ``meta``) on its next turn — in sequence, before any step — then steps and publishes the next each
    tick. A reader whose first frame is >= 1 saw the device step before it read."""

    def __init__(self):
        self.state = pimm.ControlSystemEmitter(self)
        self.meta = pimm.ControlSystemEmitter(self)
        self.cmd = pimm.ControlSystemReceiver(self)
        self._frame = 0
        self._reset_pending = False

    def reset(self, _context):
        self._frame = 0
        self._reset_pending = True

    def run(self, should_stop, clock):
        while not should_stop.value:
            yield pimm.Sleep(0.01)
            if self._reset_pending:
                self._reset_pending = False
                self.meta.emit({})  # fresh scene meta, recorded into the episode at finalize
                self.state.emit(float(self._frame))  # frame-0
            else:
                self._frame += 1
                self.state.emit(float(self._frame))


@pytest.mark.timeout(3.0)
def test_policy_first_obs_is_frame0(world):
    """The first inference reads the post-reset frame-0, never a stepped frame. The harness arms the device's
    reset and steps last; running after the harness, the device publishes frame-0 that round and the harness
    reads it the next round — so the policy's first observation is frame 0, before the device steps. Guards
    the [harness, device] ordering and the in-sequence reset."""
    device = _FrameIndexDevice()
    embodiment = Embodiment(
        descriptor='',
        observations={'frame': Observation(device.state, None)},
        commands={keys.ROBOT_COMMAND: Command(device.cmd, Reset(), None)},
        static_meta={},
        meta_source=device.meta,
        control_systems=(device,),
        simulated=True,
    )
    task = Task(instruction='t', timeout=100.0, reset=device.reset)
    policy = StubPolicy()
    harness = Harness(policy, embodiment, task=task, trials=[{}])
    wire.wire_embodiment(world, harness, embodiment, None)

    scheduler = world.start([harness, device])
    drive_scheduler(scheduler, steps=20)

    assert policy.observations, 'policy was never called'
    assert policy.observations[0]['frame'] == 0.0  # frame-0, not a stepped frame
    assert any(o['frame'] >= 1.0 for o in policy.observations)  # the device did step (so the guard can fail)


@pytest.mark.timeout(3.0)
def test_task_done_terminates_through_wire_embodiment(world):
    """A Task's ``done`` source reaches ``harness.done`` through ``wire_embodiment`` and ends the
    trial, recording its payload — the production wiring path, not a direct port pairing."""

    class _Device(pimm.ControlSystem):
        def __init__(self):
            self.state = pimm.ControlSystemEmitter(self)
            self.cmd = pimm.ControlSystemReceiver(self)
            self.done = pimm.ControlSystemEmitter(self)

        def run(self, should_stop, clock):
            n = 0
            while not should_stop.value:
                self.state.emit(0.0)
                n += 1
                if n == 5:
                    self.done.emit({'eval.success': True})
                yield pimm.Sleep(0.01)

    device = _Device()
    embodiment = Embodiment(
        descriptor='',
        observations={'x': Observation(device.state, None)},
        commands={'x': Command(device.cmd, 0.0, None)},
        static_meta={},
        meta_source=None,
    )
    task = Task(instruction='t', timeout=100.0, done=device.done)
    # Termination is independent of the policy wrappers; the minimal embodiment has no
    # ``robot_state``, so run the stub policy bare.
    harness = Harness(StubPolicy(), embodiment, task=task, trials=[{}])
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    wire.wire_embodiment(world, harness, embodiment, None, done=task.done)

    scheduler = world.start([harness, device])
    drive_scheduler(scheduler, steps=60)

    stops = [d for _, d in ds_recorder.emitted if d.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is True
    assert stops[0].static_data['eval.success'] is True


@pytest.mark.timeout(3.0)
def test_done_after_deadline_is_a_timeout(world):
    """The deadline is hard: a ``done`` delivered past it (here during the latency sleep) records as a
    timeout — ``eval.terminated`` False, payload dropped — not a late stop-signal success."""
    policy = StubPolicy()
    harness = Harness(
        policy, make_embodiment(), task=Task(instruction='t', timeout=0.05), trials=[{'inference_latency': 0.2}]
    )
    p = _pair_all(world, harness)
    done_em = world.pair(harness.done)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # Obs starts inference + the 0.2s latency sleep; the 0.05s deadline lapses during it, and done is
    # delivered at ~0.1s — past the deadline but before the harness next polls. The timeout must win.
    driver = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.1),
        (partial(done_em.emit, {'eval.success': True}), 0.3),
        (None, 0.0),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is False
    assert 'eval.success' not in stops[0].static_data


@pytest.mark.timeout(3.0)
def test_trial_seed_reaches_task_reset_and_meta(world):
    """Each RUN hands its ``eval.seed`` to the task's scene reset; the seed and the
    eval-identity block land in episode meta."""
    policy = StubPolicy()
    seeds = []
    trials = [{'eval.seed': 7 + i} for i in range(2)]

    def reset(context):
        seeds.append(context.get('eval.seed'))
        p['meta_em'].emit({})  # the producer publishes fresh scene meta, recorded into the episode at finalize

    task = Task(instruction='stack', timeout=0.05, reset=reset)
    harness = Harness(policy, make_embodiment(), task=task, trials=trials)
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=400)

    assert seeds == [7, 8]
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert [s.static_data['eval.seed'] for s in stops] == [7, 8]
    assert all(s.static_data['eval.universe'] == 'real' for s in stops)
    assert all(s.static_data['eval.embodiment'] == '' for s in stops)
    assert all(s.static_data['eval.timeout'] == 0.05 for s in stops)


@pytest.mark.timeout(3.0)
def test_trial_plan_self_drives(world):
    """With a trial plan the harness runs unattended: no driver, one episode per entry."""
    policy = StubPolicy()
    trials = [{'eval.trial_index': i} for i in range(2)]
    harness = Harness(policy, make_embodiment(), task=Task(instruction='stack', timeout=0.05), trials=trials)
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=400)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert [s.static_data['eval.trial_index'] for s in stops] == [0, 1]
    assert all(s.static_data[keys.TASK] == 'stack' for s in stops)
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
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(ds_recorder)

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([(partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01), (None, 0.3)])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    stops = [data for _, data in ds_recorder.emitted if data.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['eval.terminated'] is False
    # The post-deadline chunk must not reach the drivers: the only non-empty emissions are the homing
    # Reset / home grip from the startup home and the timeout FINISH.
    assert all(isinstance(c, Reset) for c in _emitted_commands(cmd_recorder))
    assert _emitted_grips(grip_recorder) == [0.0, 0.0]


@pytest.mark.timeout(3.0)
def test_abort_discards_recording_and_homes(world):
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(p['directive_em'].emit, Directive.RUN(task='test')), 0.0),
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (partial(p['directive_em'].emit, Directive.ABORT()), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    assert DsWriterCommandType.ABORT_EPISODE in _ds_types(p)

    assert isinstance(_last_command(p), Reset)

    assert policy.reset_calls == 1  # only from RUN


@pytest.mark.timeout(3.0)
def test_run_while_running_is_ignored(world):
    """A RUN mid-trial is ignored — the operator must finish the live trial before starting a new one."""
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
    # ep2's RUN is ignored; ep1 stays live and is finalized once at shutdown.
    assert types.count(DsWriterCommandType.START_EPISODE) == 1
    assert types.count(DsWriterCommandType.STOP_EPISODE) == 1
    assert policy.reset_calls == 1


@pytest.mark.timeout(3.0)
def test_run_calls_policy_reset_with_context(world):
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([(partial(p['directive_em'].emit, Directive.RUN(task='test-task')), 0.0), (None, 0.01)])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=5)

    assert policy.reset_calls == 1
    assert policy.last_reset_context == {keys.TASK: 'test-task'}


@pytest.mark.timeout(3.0)
def test_task_instruction_reaches_session_context_after_reset(world):
    """A task eval resets the scene before opening the session, so an instruction resolvable only on reset
    (as a remote env reports its task) still reaches ``new_session`` — task-grouped sampling/counting needs it."""
    policy = StubPolicy()
    scene = {}

    def reset(_context):
        scene['task'] = 'resolved-on-reset'  # the env reports its task only here

    task = Task(instruction=lambda: scene['task'], timeout=0.05, reset=reset)
    harness = Harness(policy, make_embodiment(), task=task, trials=[{}])
    _pair_all(world, harness)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=200)

    assert policy.last_reset_context[keys.TASK] == 'resolved-on-reset'


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
    harness.commands[keys.ROBOT_COMMAND]._bind(_LabeledRecorder(keys.ROBOT_COMMAND, events))
    harness.commands['target_grip']._bind(_LabeledRecorder('target_grip', events))
    harness.ds_command._bind(_LabeledRecorder('ds_command', events))

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
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

    cancels = [i for i, (lbl, data) in enumerate(events) if lbl == keys.ROBOT_COMMAND and data == []]
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
        def new_session(self, context=None, now=None):
            return _EmptyChunkSession()

    harness = Harness(EmptyChunkPolicy(), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
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

    p['directive_em'].emit(Directive.ABORT())
    drive_scheduler(scheduler, steps=2)

    assert _last_grip(p) == 0.0, 'Expected 0.0 (Abort homes)'

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
def test_harness_skips_inference_on_error(world):
    """An errored arm serializes to None, so the harness feeds nothing to the policy until the arm reports
    AVAILABLE again, then resumes with a fresh chunk."""
    policy = ChunkPolicy()
    harness = Harness(ChunkedSchedule().wrap(policy), make_embodiment())
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
    assert len(policy.observations) == obs_before  # the errored state is never fed to the policy

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, steps=3)
    grips = _all_grips(p)
    assert grips[0] >= 200.0


def test_directive_preserves_payload():
    assert Directive.RUN(task='test').payload == {keys.TASK: 'test'}
    assert Directive.FINISH(outcome='Success').payload == {'outcome': 'Success'}
    assert Directive.FINISH().payload == {}
    assert Directive.ABORT().payload is None


def test_directive_types():
    assert DirectiveType.RUN.value == 'run'
    assert DirectiveType.FINISH.value == 'finish'
    assert DirectiveType.ABORT.value == 'abort'


def test_cartesian_delta_wire_roundtrip():
    delta = Transform3D(np.array([0.01, -0.02, 0.03]), Rotation.from_rotvec(np.array([0.0, 0.1, 0.0])))
    frame = Transform3D(np.array([0.0, 0.0, 0.1]), Rotation.from_rotvec(np.array([0.0, 0.0, 0.5])))
    wire = to_wire(CartesianDelta(delta=delta, frame=frame))
    assert wire['type'] == 'cartesian_delta'
    out = from_wire(wire)
    assert isinstance(out, CartesianDelta)
    np.testing.assert_allclose(out.delta.translation, delta.translation)
    np.testing.assert_allclose(out.delta.rotation.as_quat, delta.rotation.as_quat, atol=1e-9)
    np.testing.assert_allclose(out.frame.translation, frame.translation)
    np.testing.assert_allclose(out.frame.rotation.as_quat, frame.rotation.as_quat, atol=1e-9)


def test_cartesian_delta_without_a_frame_is_rejected():
    """A delta means nothing without the frame it is expressed in, so the payload has to carry one."""
    delta = Transform3D(np.array([0.01, -0.02, 0.03]), Rotation.from_rotvec(np.array([0.0, 0.1, 0.0])))
    wire = to_wire(CartesianDelta(delta=delta))
    del wire['frame']
    with pytest.raises(KeyError):
        from_wire(wire)


def test_cartesian_delta_applies_in_world_frame():
    current = Transform3D(np.array([0.5, 0.1, 0.3]), Rotation.from_rotvec(np.array([0.2, 0.1, 0.4])))
    delta = Transform3D(np.array([0.02, -0.01, 0.05]), Rotation.from_rotvec(np.array([0.1, 0.0, 0.0])))
    target = CartesianDelta(delta).apply(current)
    # World frame: translation adds directly (not rotated by current, as Transform3D.__mul__ would) and the
    # rotation left-multiplies.
    np.testing.assert_allclose(target.translation, current.translation + delta.translation)
    np.testing.assert_allclose(target.rotation.as_quat, (delta.rotation * current.rotation).as_quat, atol=1e-12)
    assert not np.allclose(target.translation, (current * delta).translation)  # guards against body-frame compose


def test_reduce_accumulates_due_cartesian_deltas():
    # Rotations about different axes so the world-frame compose is non-commutative -- this pins the fold order
    # (apply d0 then d1), not just that a fold happened.
    d0 = Transform3D(np.array([0.01, 0.0, 0.0]), Rotation.from_rotvec(np.array([0.3, 0.0, 0.0])))
    d1 = Transform3D(np.array([0.02, 0.01, 0.0]), Rotation.from_rotvec(np.array([0.0, 0.0, 0.2])))
    out = reduce([(10, CartesianDelta(d0)), (20, CartesianDelta(d1))])
    assert isinstance(out, CartesianDelta)
    expected = _compose_delta(d0, d1)  # two due deltas catch up as their world-frame compose, not last-wins
    np.testing.assert_allclose(out.delta.translation, expected.translation)
    np.testing.assert_allclose(out.delta.rotation.as_quat, expected.rotation.as_quat, atol=1e-12)
    assert not np.allclose(out.delta.rotation.as_quat, _compose_delta(d1, d0).rotation.as_quat)


def test_reduce_sums_due_joint_deltas():
    out = reduce([(10, JointDelta(np.array([0.1, -0.2, 0.3]))), (20, JointDelta(np.array([0.0, 0.2, -0.1])))])
    assert isinstance(out, JointDelta)
    np.testing.assert_allclose(out.velocities, [0.1, 0.0, 0.2])


def test_reduce_absolute_run_keeps_last():
    p0 = CartesianPosition(Transform3D(np.array([0.1, 0.0, 0.0]), Rotation.from_rotvec(np.zeros(3))))
    p1 = JointPosition(np.array([0.2, 0.0, 0.0]))
    assert reduce([(10, p0), (20, p1)]) is p1


def test_reduce_raises_on_absolute_delta_mix():
    cart_pos = CartesianPosition(Transform3D(np.zeros(3), Rotation.from_rotvec(np.zeros(3))))
    cart_delta = CartesianDelta(Transform3D(np.array([0.01, 0.0, 0.0]), Rotation.from_rotvec(np.zeros(3))))
    joint_pos = JointPosition(np.zeros(3))
    joint_delta = JointDelta(np.array([0.1, 0.0, 0.0]))
    with pytest.raises(ValueError):
        reduce([(10, cart_pos), (20, cart_delta)])
    with pytest.raises(ValueError):
        reduce([(10, cart_delta), (20, cart_pos)])
    with pytest.raises(ValueError):  # JointPosition then JointDelta: the delta has no faithful anchor to fold onto
        reduce([(10, joint_pos), (20, joint_delta)])


def test_reduce_raises_on_mixed_delta_spaces():
    cart_delta = CartesianDelta(Transform3D(np.array([0.01, 0.0, 0.0]), Rotation.from_rotvec(np.zeros(3))))
    joint_delta = JointDelta(np.array([0.1, 0.0, 0.0]))
    with pytest.raises(ValueError):
        reduce([(10, cart_delta), (20, joint_delta)])
    with pytest.raises(ValueError):
        reduce([(10, joint_delta), (20, cart_delta)])


def test_trajectory_player_accumulates_missed_deltas():
    d0 = Transform3D(np.array([0.01, 0.0, 0.0]), Rotation.from_rotvec(np.zeros(3)))
    d1 = Transform3D(np.array([0.02, 0.0, 0.0]), Rotation.from_rotvec(np.zeros(3)))
    player = TrajectoryPlayer(reduce=reduce)
    player.set([(10, CartesianDelta(d0)), (20, CartesianDelta(d1))])
    out = player.advance(20)  # both waypoints due in one tick -> summed, not dropped to the last
    assert isinstance(out, CartesianDelta)
    np.testing.assert_allclose(out.delta.translation, [0.03, 0.0, 0.0])
    assert player.advance(30) is None


@pytest.mark.parametrize('status', [RobotStatus.RESETTING, RobotStatus.ERROR])
def test_robot_state_serializer_drops_not_ready(status):
    state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=status)
    assert Serializers.robot_state(state) is None


def test_robot_state_serializer_available_has_no_error_key():
    state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    assert set(Serializers.robot_state(state)) == {'.q', '.dq', '.ee_pose'}


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
    harness.commands[keys.ROBOT_COMMAND]._bind(_LabeledRecorder(keys.ROBOT_COMMAND))
    harness.commands['target_grip']._bind(_LabeledRecorder('target_grip'))
    harness.ds_command._bind(_LabeledRecorder('ds_command'))

    frame_em = world.pair(harness.observations['image.cam'])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
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

    cancels = [i for i, (lbl, data) in enumerate(events) if lbl == keys.ROBOT_COMMAND and data == []]
    stops = [
        i
        for i, (lbl, data) in enumerate(events)
        if lbl == 'ds_command' and getattr(data, 'type', None) is DsWriterCommandType.STOP_EPISODE
    ]
    assert cancels, 'shutdown did not cancel robot_command'
    assert stops, 'shutdown did not emit STOP_EPISODE'
    assert cancels[0] < stops[0], 'cancel must precede STOP_EPISODE on shutdown'


@pytest.mark.timeout(5.0)
class _ManualClock:
    """A self-advancing clock for driving ``Harness.run`` directly, without a world."""

    def __init__(self):
        self.t = 0.0

    def now(self) -> float:
        self.t += 0.001
        return self.t

    def now_ns(self) -> int:
        return int(self.now() * 1e9)


@pytest.mark.timeout(3.0)
def test_stop_mid_episode_keeps_episode_open_for_recorder_flush(tmp_path):
    """A stop arriving mid-episode winds down through the same close order as ``_end_episode``: the harness
    yields a turn between queueing the recorder's STOP and closing the episode span, so the recorder's
    shutdown-flush ``record.io`` span parents to the episode, not the pass. Driven straight through the
    generator protocol: the yield after the queued STOP is the recorder's flush slot."""
    policy = StubPolicy()
    task = Task(instruction='stack', timeout=10.0, reset=lambda context: None)  # never ends within the drive
    harness = Harness(policy, make_embodiment(), task=task, trials=[{'eval.trial_index': 0}])
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    stop = SimpleNamespace(value=False)
    clock = _ManualClock()

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-stop'), _eval_pass('run-stop'):
        gen = harness.run(cast(pimm.SignalReceiver, stop), cast(pimm.Clock, clock))
        for _ in range(20):
            next(gen)
            if any(d.type == DsWriterCommandType.START_EPISODE for _, d in ds_recorder.emitted):
                break
        else:
            pytest.fail('the self-driven trial never started')

        stop.value = True
        try:
            next(gen)  # the post-STOP yield: the turn the recorder commits the queued STOP on
        except StopIteration:
            pytest.fail('harness must yield a recorder turn between queueing STOP and ending the episode span')
        assert any(d.type == DsWriterCommandType.STOP_EPISODE for _, d in ds_recorder.emitted)
        with telemetry.span(telemetry_keys.SPAN_RECORD_IO):  # the recorder's shutdown flush, emitted in that turn
            pass
        with pytest.raises(StopIteration):
            while True:
                next(gen)

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episodes = [s for s in spans if s.name == telemetry_keys.SPAN_EPISODE]
    assert len(episodes) == 1
    episode = episodes[0]
    assert telemetry_keys.ATTR_EPISODE_STEPS in episode.attrs  # sealed via end_episode, neither leaked open nor aborted
    flushes = [s for s in spans if s.name == telemetry_keys.SPAN_RECORD_IO]
    assert flushes and all(s.parent_id == episode.span_id for s in flushes)


def test_timing_spans_recorded_with_taxonomy(world, tmp_path):
    """Under ``telemetry.bind`` a self-driven episode writes the span taxonomy to the harness file: the
    episode parents to the pass, and reset + policy.infer parent to the episode, with the episode carrying its
    index, step count, and virtual duration. Read back from the file so the OTLP encoding is exercised. The
    ``policy.infer`` span is recorded at the remote inference boundary, so the terminal is a ``RemoteStubPolicy``
    (a real ``RemoteSession`` over a fake inference session)."""
    policy = ChunkedSchedule().wrap(RemoteStubPolicy())
    task = Task(instruction='stack', timeout=0.05, reset=lambda context: None)
    harness = Harness(policy, make_embodiment(), task=task, trials=[{'eval.trial_index': 0}])
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # A latched observation set makes every step's inference fire (the harness reads the latest value).
    producer = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.0)
    ])

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-taxonomy'), _eval_pass('run-taxonomy'):
        scheduler = world.start([harness, producer])
        drive_scheduler(scheduler, steps=400)

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    by_name: dict[str, list] = {}
    for rec in spans:
        by_name.setdefault(rec.name, []).append(rec)

    assert len(by_name[telemetry_keys.SPAN_EVAL_PASS]) == 1
    assert len(by_name[telemetry_keys.SPAN_EPISODE]) == 1
    assert by_name[telemetry_keys.SPAN_RESET]  # the task's scene reset was timed
    assert by_name[telemetry_keys.SPAN_POLICY_INFER]  # at least one real inference round-trip

    pass_span = by_name[telemetry_keys.SPAN_EVAL_PASS][0]
    episode = by_name[telemetry_keys.SPAN_EPISODE][0]
    assert pass_span.parent_id is None
    assert episode.parent_id == pass_span.span_id
    assert all(r.parent_id == episode.span_id for r in by_name[telemetry_keys.SPAN_RESET])
    assert all(r.parent_id == episode.span_id for r in by_name[telemetry_keys.SPAN_POLICY_INFER])

    assert episode.attrs[telemetry_keys.ATTR_EPISODE_INDEX] == 0
    assert episode.attrs[telemetry_keys.ATTR_EPISODE_STEPS] == len(by_name[telemetry_keys.SPAN_POLICY_INFER])
    assert episode.attrs[telemetry_keys.ATTR_EPISODE_VIRTUAL_S] >= 0.0


@pytest.mark.timeout(3.0)
def test_aborted_episode_span_marked_aborted(world, tmp_path):
    """An ABORT discards the rollout, and its episode span is stamped ``episode.aborted`` — the reduce drops
    it rather than charging a partial rollout's wall to a real episode."""
    harness = Harness(ChunkPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-abort'):
        scheduler = world.start([harness])
        p['directive_em'].emit(Directive.RUN(task='test'))
        drive_scheduler(scheduler, steps=1)
        emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
        drive_scheduler(scheduler, steps=5)
        p['directive_em'].emit(Directive.ABORT())
        drive_scheduler(scheduler, steps=3)

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episodes = [s for s in spans if s.name == telemetry_keys.SPAN_EPISODE]
    assert len(episodes) == 1
    assert episodes[0].attrs[telemetry_keys.ATTR_EPISODE_ABORTED] is True


@pytest.mark.timeout(3.0)
def test_failed_pass_seals_open_episode_span(tmp_path):
    """A ``task.reset`` raising after the episode span was opened must seal that span before the
    provider flushes on exit. Ending it is what exports it at all: an unended span never leaves the batch
    processor, so its finished ``reset`` child orphans (unknown parent) and the report loses that phase and
    charges the episode's whole wall to ``between_episodes``. Sealed and marked ``episode.partial`` — with its
    step count and virtual duration stamped, like a clean end — the span exports parented to the (failed) pass,
    so the reduce keeps it and its phases attribute."""

    def boom(context):
        raise RuntimeError('reset boom')

    policy = StubPolicy()
    task = Task(instruction='stack', timeout=10.0, reset=boom)
    harness = Harness(policy, make_embodiment(), task=task, trials=[{'eval.trial_index': 0}])
    harness.ds_command._bind(RecordingEmitter())
    stop = SimpleNamespace(value=False)
    clock = _ManualClock()

    with pytest.raises(RuntimeError, match='reset boom'):
        with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-fail'), _eval_pass('run-fail'):
            for _ in harness.run(cast(pimm.SignalReceiver, stop), cast(pimm.Clock, clock)):
                pass

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episodes = [s for s in spans if s.name == telemetry_keys.SPAN_EPISODE]
    assert len(episodes) == 1  # the open span was sealed and exported, not lost with the failure
    episode = episodes[0]
    assert (
        episode.attrs.get(telemetry_keys.ATTR_EPISODE_PARTIAL) is True
    )  # flagged so the reduce sees it did not complete
    assert (
        episode.attrs[telemetry_keys.ATTR_EPISODE_STEPS] == 0
    )  # stamped like a clean end — no step ran before the failure
    # The rollout never started — reset raised before its virtual anchor was stamped — so its virtual duration
    # is zero, not the garbage ``clock.now() - 0`` a never-stamped anchor would otherwise yield.
    assert episode.attrs[telemetry_keys.ATTR_EPISODE_VIRTUAL_S] == 0.0
    passes = [s for s in spans if s.name == telemetry_keys.SPAN_EVAL_PASS]
    assert len(passes) == 1
    assert episode.parent_id == passes[0].span_id  # parented to the pass, so the reduce does not drop it as an orphan
    resets = [s for s in spans if s.name == telemetry_keys.SPAN_RESET]
    assert resets and all(r.parent_id == episode.span_id for r in resets)  # finished child attributes to its phase


@pytest.mark.timeout(3.0)
def test_episode_virtual_duration_starts_at_the_first_observation(world, tmp_path):
    """A simulated producer's reset only arms frame zero, which it publishes on a later turn; the rounds in
    between advance the virtual clock without stepping the environment. The rollout's virtual duration
    measures from the first cycle that has an observation, so that gap stays reset work instead of inflating
    the real-time factor the report derives from it."""
    harness = Harness(ChunkPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-anchor'), _eval_pass('run-anchor'):
        scheduler = world.start([harness])
        p['directive_em'].emit(Directive.RUN(task='test'))
        drive_scheduler(scheduler, steps=1)
        gap_start = world.clock.now()
        drive_scheduler(scheduler, steps=40)  # no observation yet: the producer has not published frame zero
        gap_s = world.clock.now() - gap_start
        emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
        drive_scheduler(scheduler, steps=5)
        p['directive_em'].emit(Directive.FINISH())
        drive_scheduler(scheduler, steps=3)

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episodes = [s for s in spans if s.name == telemetry_keys.SPAN_EPISODE]
    assert len(episodes) == 1
    virtual_s = episodes[0].attrs[telemetry_keys.ATTR_EPISODE_VIRTUAL_S]
    assert virtual_s > 0.0  # the observed cycles are measured
    # Five observed rounds against forty unobserved ones: anchoring when the reset returned would swallow the
    # whole gap into the rollout's virtual duration.
    assert virtual_s < gap_s


@pytest.mark.timeout(3.0)
def test_a_later_episode_waits_for_its_own_first_observation(world, tmp_path):
    """An observation channel latches its last value, so after the first episode every channel already holds
    one. A rollout still anchors on a value delivered after its own reset — anchoring on the latched frame
    would charge the wait for frame zero to the rollout and infer on the previous episode's last scene."""
    harness = Harness(ChunkPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    def episode(gap_steps: int) -> float:
        p['directive_em'].emit(Directive.RUN(task='test'))
        drive_scheduler(scheduler, steps=1)
        gap_start = world.clock.now()
        drive_scheduler(scheduler, steps=gap_steps)  # the producer has not published this episode's frame zero
        gap_s = world.clock.now() - gap_start
        emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
        drive_scheduler(scheduler, steps=5)
        p['directive_em'].emit(Directive.FINISH())
        drive_scheduler(scheduler, steps=3)
        return gap_s

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-anchor-2'), _eval_pass('run-anchor-2'):
        scheduler = world.start([harness])
        episode(gap_steps=2)
        gap_s = episode(gap_steps=40)

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episodes = [s for s in spans if s.name == telemetry_keys.SPAN_EPISODE]
    assert len(episodes) == 2
    assert episodes[1].attrs[telemetry_keys.ATTR_EPISODE_VIRTUAL_S] < gap_s


# --- ending a run from outside it: one wind-down, reached for either reason --------------------


def _grace_s() -> float:
    return FINISH_HOME_GRACE_NS / 1e9


@pytest.mark.timeout(10.0)
def test_a_finish_request_ends_an_idle_run(world):
    """The whole point of the request: a run with no operator surface to post at still ends the way a
    plan running out ends — the loop returns, so the World unwinds, the recorder commits and the
    mirror uploads, none of which a signal does."""
    harness = Harness(SpyPolicy(), make_embodiment())
    _pair_all(world, harness)
    finish_em = world.pair(harness.finish_requested)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=3)  # the pre-first-episode home
    finish_em.emit(True)
    drive_scheduler(scheduler, steps=2000)

    with pytest.raises(StopIteration):
        next(scheduler)


@pytest.mark.timeout(10.0)
def test_a_finish_gives_the_arm_its_home_travel_before_the_world_stops(world):
    """`_home` publishes targets and returns, so the arm is still moving when the loop breaks. The
    World unwinding under it parks the brakes wherever it got to, which is what the wind-down's wait
    exists to prevent — measured on the clock, since that is what the arm travels against."""
    harness = Harness(SpyPolicy(), make_embodiment())
    _pair_all(world, harness)
    finish_em = world.pair(harness.finish_requested)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=3)
    homed_at = harness._homed_at_ns
    finish_em.emit(True)
    drive_scheduler(scheduler, steps=2000)

    assert world.clock.now_ns() - homed_at >= FINISH_HOME_GRACE_NS


@pytest.mark.timeout(10.0)
def test_a_simulated_run_does_not_wait_for_a_motion_it_never_makes(world):
    """A sim has no travel to finish, and its clock advances only when something asks it to — so a
    wait measured on it is one nothing ever satisfies."""
    embodiment = replace(make_embodiment(), simulated=True)
    harness = Harness(SpyPolicy(), embodiment)
    _pair_all(world, harness)
    finish_em = world.pair(harness.finish_requested)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=3)
    finish_em.emit(True)
    before = world.clock.now()
    drive_scheduler(scheduler, steps=2000)

    with pytest.raises(StopIteration):
        next(scheduler)
    assert world.clock.now() - before < _grace_s()


@pytest.mark.timeout(10.0)
def test_a_finish_after_a_manual_command_homes_the_arm_before_stopping(world):
    """`_apply_manual` publishes a pose and nothing homes after it — an episode's own `_end_episode`
    does, and a jog happens between episodes. So the arm sits where the operator put it, and a stop
    that did not re-home would end the run there and start the next one from a pose nothing recorded."""
    harness = Harness(SpyPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    manual_em = world.pair(harness.manual_command)
    finish_em = world.pair(harness.finish_requested)
    home_pose = make_embodiment().home[keys.ROBOT_COMMAND]

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=3)
    manual_em.emit({keys.ROBOT_COMMAND: CartesianPosition(pose=Transform3D())})
    drive_scheduler(scheduler, steps=3)
    assert harness._moved_since_home  # the jog landed and nothing homed after it

    finish_em.emit(True)
    drive_scheduler(scheduler, steps=2000)

    # The last thing commanded is the home pose, not the operator's jog.
    last = p['command_rx'].read().data[-1][1]
    assert last == home_pose
    assert not harness._moved_since_home


@pytest.mark.timeout(10.0)
def test_a_pending_finish_stops_a_trial_plan_instead_of_starting_another_episode(world):
    """A request taken AFTER the plan advances starts one more episode and is then blocked by it —
    and since a trial ends on its own budget, on a long plan that is the difference between ending
    the run now and not being able to end it at all."""
    trials = [{'seed': i} for i in range(50)]
    plan = iter(trials)
    harness = Harness(SpyPolicy(), make_embodiment(), task=Task(instruction='test', timeout=100.0), trials=plan)
    p = _pair_all(world, harness)
    finish_em = world.pair(harness.finish_requested)

    scheduler = world.start([harness])
    finish_em.emit(True)
    drive_scheduler(scheduler, steps=2000)

    with pytest.raises(StopIteration):
        next(scheduler)
    assert not harness._running
    # The plan never advanced: no episode was opened, so nothing of it reached the dataset.
    assert DsWriterCommandType.START_EPISODE not in [c.type for c in _ds_commands(p)]
    assert next(plan) == {'seed': 0}  # the whole plan is still ahead of it


@pytest.mark.timeout(10.0)
def test_a_queued_directive_is_handled_before_a_pending_finish(world):
    """The directive read runs every round, ahead of the idle decision, so an operator press that
    arrived in the same round is acted on rather than dropped by a stop taken beside it."""
    harness = Harness(SpyPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    finish_em = world.pair(harness.finish_requested)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=3)
    finish_em.emit(True)
    p['directive_em'].emit(Directive.RUN(task='t'))
    drive_scheduler(scheduler, steps=2)

    assert harness._running  # her episode started; the finish waits for it


@pytest.mark.timeout(10.0)
def test_a_finish_request_does_not_truncate_the_episode_in_progress(world):
    """The property that makes this a finish rather than a kill. A request arriving mid-episode must
    not close the recording where it stands: the loop's exit path finalizes whatever is open, so a
    half-episode would land in the dataset indistinguishable from one the policy actually failed."""
    harness = Harness(SpyPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    finish_em = world.pair(harness.finish_requested)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    scheduler = world.start([harness])
    p['directive_em'].emit(Directive.RUN(task='t'))
    drive_scheduler(scheduler, steps=2)
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=5)

    finish_em.emit(True)
    drive_scheduler(scheduler, steps=2000)

    assert harness._running
    assert DsWriterCommandType.STOP_EPISODE not in [c.type for c in _ds_commands(p)]

    # The operator's own FINISH closes the episode; only then does the run end.
    p['directive_em'].emit(Directive.FINISH())
    drive_scheduler(scheduler, steps=2000)
    assert DsWriterCommandType.STOP_EPISODE in [c.type for c in _ds_commands(p)]
    with pytest.raises(StopIteration):
        next(scheduler)


@pytest.mark.timeout(10.0)
def test_an_unrequested_run_never_ends_itself(world):
    """The receiver defaults to False, so every run nobody asked to end is untouched by this path."""
    harness = Harness(SpyPolicy(), make_embodiment())
    _pair_all(world, harness)

    scheduler = world.start([harness])
    drive_scheduler(scheduler, steps=2000)

    next(scheduler)  # still going
