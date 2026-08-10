import time
from contextlib import contextmanager
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
from positronic.drivers.roboarm.command import CartesianDelta, CartesianPosition, Reset, from_wire, to_wire
from positronic.drivers.roboarm.models import DEFAULT_FRAME, EE_LINK, bundled_franka_model
from positronic.eval import Command, Embodiment, Observation, Task
from positronic.geom import Rotation, Transform3D
from positronic.offboard.client import InferenceSession
from positronic.policy.base import DelegatingSession, Policy, SchedulingWrapper, Session
from positronic.policy.codec import ActionTimestamp
from positronic.policy.harness import Directive, DirectiveType, Harness, TrajectoryPlayer, _assert_anchored
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


CAM = 'image.cam'


def make_embodiment(descriptor: str = '', cameras=(CAM,), static_meta=None, simulated=False) -> Embodiment:
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
    return Embodiment(descriptor, observations, commands, static_meta or {}, pimm.NoOpEmitter(), simulated=simulated)


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

    def new_session(self, context=None, *, now=None, gate=None):
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

    def new_session(self, context=None, *, now=None, gate=None):
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

    def new_session(self, context=None, *, now=None, gate=None):
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

    def new_session(self, context=None, *, now=None, gate=None) -> RemoteSession:
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


class _Pacer(pimm.ControlSystem):
    """Stands in for the simulator: the sole time-master, sleeping one control period every turn."""

    def __init__(self, period: float = 0.005):
        self._period = period

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        while not should_stop.value:
            yield pimm.Sleep(self._period)


def _pair_all(world, harness):
    """Pair all harness signals and return a dict of test handles."""
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    return {
        'frame_em': world.pair(harness.observations[CAM]),
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
    """The latest robot command the harness put on the channel."""
    msg = p['command_rx'].read()
    assert msg is not None, 'no robot command was emitted'
    return msg.data


def _last_grip(p):
    """The latest grip target the harness put on the channel."""
    msg = p['grip_rx'].read()
    assert msg is not None, 'no grip target was emitted'
    return msg.data


def _emitted_commands(recorder):
    """Every robot command a recorder saw, in emission order."""
    return [cmd for _ts, cmd in recorder.emitted]


def _emitted_grips(recorder):
    """Every grip target a recorder saw, in emission order."""
    return [grip for _ts, grip in recorder.emitted]


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

    frame_em = world.pair(harness.observations[CAM])
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
    assert CAM in obs
    expected_pose = np.concatenate([robot_state.ee_pose.translation, robot_state.ee_pose.rotation.as_quat])
    np.testing.assert_allclose(obs[keys.EE_POSE], expected_pose)
    np.testing.assert_allclose(obs[keys.JOINTS], robot_state.q)
    np.testing.assert_allclose(obs[keys.JOINT_VEL], np.zeros_like(robot_state.q))
    assert obs[keys.GRIP] == pytest.approx(0.25)
    assert obs[keys.TASK] == 'stack-blocks'
    assert obs['descriptor'] == ''  # no descriptor passed -> empty string reaches the policy
    # Recording == canonical policy I/O: the policy sees the same ``robot_state`` serializer
    # the dataset records. wall/obs timestamps carry volatile values, so lock the stable key set.
    assert set(obs) - {keys.WALL_TIME_NS, keys.OBS_TIME_NS} == {
        CAM,
        keys.JOINTS,
        keys.JOINT_VEL,
        keys.EE_POSE,
        keys.GRIP,
        keys.TASK,
        'descriptor',
    }

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

    frame_em = world.pair(harness.observations[CAM])
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

    frame_em = world.pair(harness.observations[CAM])
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

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
    directive_em = world.pair(harness.directive)

    assert CAM in harness.observations

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
    harness = Harness(policy, make_embodiment(), static_meta={keys.JOINT_SIGNALS: [keys.JOINTS]})
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
    assert meta[keys.JOINT_SIGNALS] == [keys.JOINTS]
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

        def new_session(self, context=None, *, now=None, gate=None):
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
    assert stops[0].static_data[keys.EVAL_TERMINATED] is False
    assert keys.EVAL_SUCCESS not in stops[0].static_data
    assert isinstance(_last_command(p), Reset)


@pytest.mark.timeout(3.0)
def test_trial_budget_starts_at_the_first_usable_observation(world):
    """The 0.05 budget is measured from the episode's first usable observation, not from the reset.

    The driver holds every channel silent for 0.02, so the episode opens with nothing to infer on. The
    delivery at 0.02 is rejected by the camera serializer — the state a real embodiment is in while the
    arm resets — so the first usable observation is the one at 0.04, and a third at 0.06 carries grip 0.75
    as a marker. Measured from the reset the budget ends at 0.05 and the policy never sees the marker;
    measured from the first usable observation it ends at 0.09 and the policy does.

    The rejected delivery matters on its own: it clears the channels out of ``_awaiting_obs`` without
    yielding an observation, so a budget anchored on that set alone would never move off the reset.
    """
    policy = SpyPolicy()
    embodiment = make_embodiment()
    usable = iter([None])  # the first camera sample is not ready; every later one is
    embodiment.observations[CAM] = Observation(
        pimm.NoOpEmitter(), lambda frames: next(usable, Serializers.camera_images(frames))
    )
    harness = Harness(policy, embodiment, task=Task(instruction='test', timeout=0.05), trials=[{}])
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.2, 0.0, -0.1], [0.7, 0.1, -0.2])
    payload = partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state)

    def marked():
        payload()
        p['grip_em'].emit(0.75)

    driver = ManualDriver([(None, 0.02), (payload, 0.02), (payload, 0.02), (marked, 0.2)])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=400)

    assert policy.last_obs is not None, 'the trial expired before its first observation landed'
    assert policy.last_obs[keys.GRIP] == 0.75, 'the budget expired early — measured from reset, not the first obs'


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
    assert stops[0].static_data[keys.EVAL_TERMINATED] is False
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

    done_em.emit({keys.EVAL_SUCCESS: True})
    drive_scheduler(scheduler, steps=10)
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is True
    assert stops[0].static_data[keys.EVAL_SUCCESS] is True


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

    done_em.emit({keys.EVAL_SUCCESS: True})
    drive_scheduler(scheduler, steps=10)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is True
    assert stops[0].static_data[keys.EVAL_SUCCESS] is True  # the delivered payload lands in static data
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

    done_em.emit({keys.EVAL_SUCCESS: True})  # fresh truthy: ends trial 0
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 1

    # Trial 1 auto-started. The terminal is still latched but no longer fresh, so it must NOT re-fire.
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 1

    done_em.emit({keys.EVAL_SUCCESS: True})  # a fresh delivery ends trial 1
    drive_scheduler(scheduler, steps=10)
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 2
    assert all(s.static_data[keys.EVAL_TERMINATED] is True for s in stops)


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
                    self.done.emit({keys.EVAL_SUCCESS: True})
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
    assert stops[0].static_data[keys.EVAL_TERMINATED] is True
    assert stops[0].static_data[keys.EVAL_SUCCESS] is True


@pytest.mark.timeout(3.0)
def test_done_after_deadline_is_a_timeout(world):
    """The deadline is hard: a ``done`` delivered past it records as a timeout — ``eval.terminated`` False,
    payload dropped — not a late stop-signal success."""
    policy = StubPolicy()
    harness = Harness(policy, make_embodiment(), task=Task(instruction='t', timeout=0.05), trials=[{}])
    p = _pair_all(world, harness)
    done_em = world.pair(harness.done)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # The 0.05s deadline lapses first; done lands at ~0.1s, after the trial has already timed out.
    driver = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.1),
        (partial(done_em.emit, {keys.EVAL_SUCCESS: True}), 0.3),
        (None, 0.0),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is False
    assert keys.EVAL_SUCCESS not in stops[0].static_data


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
    assert all(s.static_data[keys.EVAL_TERMINATED] is False for s in stops)
    assert policy.reset_calls == 2


@pytest.mark.timeout(3.0)
def test_timeout_during_inference_drops_the_chunk(world):
    """A trial whose deadline lapses while the model is still owed its latency ends with the call in flight:
    the trajectory it eventually returns is discarded, never emitted past the advertised termination point."""
    policy = StubPolicy()
    harness = Harness(
        ChunkedSchedule().wrap(policy),
        make_embodiment(simulated=True),
        task=Task(instruction='test', timeout=0.05),
        trials=[{keys.INFERENCE_LATENCY: 0.2}],  # the gate holds the answer well past the deadline
    )
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    ds_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(ds_recorder)

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([(partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01), (None, 0.3)])

    scheduler = world.start([harness, driver, _Pacer()])
    drive_scheduler(scheduler, steps=2000)

    stops = [data for _, data in ds_recorder.emitted if data.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is False
    # The only commands are the homing Reset / home grip from the startup home and the timeout FINISH.
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


class _LabeledRecorder(pimm.SignalEmitter):
    """Records emissions from several channels into one shared list, so their order is comparable."""

    def __init__(self, label, events):
        self._label = label
        self._events = events

    def emit(self, data, ts: int = -1):
        self._events.append((self._label, data))


@pytest.mark.timeout(3.0)
def test_finish_stops_playing_the_live_chunk(world):
    """FINISH drops the schedule the harness is playing: the chunk's remaining waypoints never reach the
    devices, and the only command after the recorder's STOP is the home the close emits."""
    policy = ChunkPolicy()
    wrapped = ActionTimestamp(fps=5.0).wrap(policy)  # 1.8 s chunk — won't drain before FINISH
    harness = Harness(wrapped, make_embodiment())
    events: list[tuple[str, object]] = []
    harness.commands[keys.ROBOT_COMMAND]._bind(_LabeledRecorder(keys.ROBOT_COMMAND, events))
    harness.commands['target_grip']._bind(_LabeledRecorder('target_grip', events))
    harness.ds_command._bind(_LabeledRecorder('ds_command', events))

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    script = [
        (partial(directive_em.emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
        (partial(directive_em.emit, Directive.FINISH()), 0.0),
        (None, 0.5),
    ]
    scheduler = world.start([harness, ManualDriver(script)])
    drive_scheduler(scheduler, steps=400)

    stops = [i for i, (_, data) in enumerate(events) if getattr(data, 'type', None) is DsWriterCommandType.STOP_EPISODE]
    assert stops, 'FINISH did not emit STOP_EPISODE'
    grips_after = [data for lbl, data in events[stops[0] :] if lbl == 'target_grip']
    assert grips_after == [0.0], f'the cancelled chunk kept playing past FINISH: {grips_after}'


@pytest.mark.timeout(3.0)
def test_empty_trajectory_leaves_every_channel_holding(world):
    """A trajectory with no waypoints schedules nothing on any channel, so every device holds where the
    startup home left it rather than one channel draining on while another stops."""

    class _EmptyChunkSession(Session):
        def __call__(self, obs):
            return []

    class EmptyChunkPolicy(Policy):
        def new_session(self, context=None, *, now=None, gate=None):
            return _EmptyChunkSession()

    harness = Harness(EmptyChunkPolicy(), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
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

    assert all(isinstance(c, Reset) for c in _emitted_commands(cmd_recorder))  # only the startup home
    assert _emitted_grips(grip_recorder) == [0.0]


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_abort(world):
    policy = ChunkPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=5)

    assert _last_grip(p) >= 100.0, 'Expected chunk 1'

    p['directive_em'].emit(Directive.ABORT())
    drive_scheduler(scheduler, steps=2)

    assert _last_grip(p) == 0.0, 'Expected 0.0 (Abort homes)'

    p['directive_em'].emit(Directive.RUN(task='test'))
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=4)

    assert _last_grip(p) >= 200.0, 'Expected chunk 2; trajectory clearing failed'


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_run(world):
    policy = ChunkPolicy()
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['directive_em'].emit(Directive.RUN(task='test'))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=5)

    assert _last_grip(p) >= 100.0

    p['directive_em'].emit(Directive.RUN(task='test-restart'))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=4)

    assert _last_grip(p) >= 200.0, 'Expected chunk 2; trajectory clearing on RUN failed'


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
    assert _last_grip(p) >= 100.0

    obs_before = len(policy.observations)
    p['robot_em'].emit(state_err)
    drive_scheduler(scheduler, steps=2)
    assert len(policy.observations) == obs_before  # the errored state is never fed to the policy

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, steps=20)  # long enough for the first chunk to play out and the next to land
    assert _last_grip(p) >= 200.0


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


def test_trajectory_player_collapses_several_due_waypoints_to_the_last():
    player = TrajectoryPlayer()
    player.set([(10, 'a'), (20, 'b'), (30, 'c')])
    assert player.next_due() == 10
    assert player.advance(5) is None
    assert player.advance(25) == 'b'  # a late round overtakes 'a'; the trailing setpoint is the live one
    assert player.next_due() == 30
    assert player.advance(30) == 'c'
    assert player.next_due() is None
    assert player.advance(40) is None


def test_cartesian_delta_applies_in_world_frame():
    current = Transform3D(np.array([0.5, 0.1, 0.3]), Rotation.from_rotvec(np.array([0.2, 0.1, 0.4])))
    delta = Transform3D(np.array([0.02, -0.01, 0.05]), Rotation.from_rotvec(np.array([0.1, 0.0, 0.0])))
    target = CartesianDelta(delta).apply(current)
    # World frame: translation adds directly (not rotated by current, as Transform3D.__mul__ would) and the
    # rotation left-multiplies.
    np.testing.assert_allclose(target.translation, current.translation + delta.translation)
    np.testing.assert_allclose(target.rotation.as_quat, (delta.rotation * current.rotation).as_quat, atol=1e-12)
    assert not np.allclose(target.translation, (current * delta).translation)  # guards against body-frame compose


@pytest.mark.parametrize('status', [RobotStatus.RESETTING, RobotStatus.ERROR])
def test_robot_state_serializer_drops_not_ready(status):
    state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=status)
    assert Serializers.robot_state(state) is None


def test_robot_state_serializer_available_has_no_error_key():
    state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    assert set(Serializers.robot_state(state)) == {'.q', '.dq', '.ee_pose'}


@pytest.mark.timeout(3.0)
def test_shutdown_stops_playing_the_live_chunk(world):
    """Shutdown while recording drops the schedule too: the unplayed tail of the live chunk never reaches
    the devices after the recorder's STOP."""
    events: list[tuple[str, object]] = []
    wrapped = ActionTimestamp(fps=5.0).wrap(ChunkPolicy())  # 1.8 s chunk — won't drain before shutdown
    harness = Harness(wrapped, make_embodiment())
    harness.commands[keys.ROBOT_COMMAND]._bind(_LabeledRecorder(keys.ROBOT_COMMAND, events))
    harness.commands['target_grip']._bind(_LabeledRecorder('target_grip', events))
    harness.ds_command._bind(_LabeledRecorder('ds_command', events))

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # RUN + a complete obs schedules a chunk; the driver then ends, which makes the
    # world signal shutdown while still recording — exercising the run() finalizer.
    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    stops = [i for i, (_, data) in enumerate(events) if getattr(data, 'type', None) is DsWriterCommandType.STOP_EPISODE]
    assert stops, 'shutdown did not emit STOP_EPISODE'
    assert not [lbl for lbl, _ in events[stops[0] :] if lbl == 'target_grip']


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


def test_unanchored_chunk_is_refused():
    """A stack that never anchored leaves chunk-relative stamps, which read as decades before now."""
    with pytest.raises(ValueError, match='not anchoring'):
        _assert_anchored([{'timestamp': 0.0}], now=1.7e9)


def test_doubly_anchored_chunk_is_refused():
    """Two schedulers each add the clock, putting the chunk a lifetime ahead."""
    with pytest.raises(ValueError, match='not anchoring'):
        _assert_anchored([{'timestamp': 3.4e9}], now=1.7e9)


def test_anchored_chunk_passes():
    """A real chunk spans seconds around now, and a late action sits just behind it."""
    _assert_anchored([{'timestamp': 1.7e9 - 0.2}, {'timestamp': 1.7e9 + 1.5}], now=1.7e9)


class _SlowSession(Session):
    """A session whose inference costs ``wall_sec`` of real time and returns a fixed-length chunk."""

    def __init__(self, wall_sec: float, span_sec: float, steps: int):
        self._wall_sec = wall_sec
        self._span_sec = span_sec
        self._steps = steps

    def __call__(self, obs):
        time.sleep(self._wall_sec)
        dt = self._span_sec / self._steps
        pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
        return [
            {keys.ROBOT_COMMAND: CartesianPosition(pose=pose), 'target_grip': float(i), 'timestamp': i * dt}
            for i in range(self._steps)
        ]


class SlowPolicy(Policy):
    def __init__(self, wall_sec: float = 0.0, span_sec: float = 0.2, steps: int = 10):
        self._wall_sec = wall_sec
        self._span_sec = span_sec
        self._steps = steps

    def new_session(self, context=None, *, now=None, gate=None):
        return _SlowSession(self._wall_sec, self._span_sec, self._steps)


class _ReplanEarly(SchedulingWrapper):
    """Infers on the first observation and again halfway through the chunk it returned.

    The re-query-before-exhaustion shape (RTC, temporal ensembling) that the substrate exists for: unlike
    ``ChunkedSchedule`` it leaves waypoints to play while a call is in flight.
    """

    class _Session(DelegatingSession):
        def __init__(self, inner: Session, now):
            super().__init__(inner)
            self._now = now
            self._replan_at: float | None = None

        def __call__(self, obs):
            if self._replan_at is not None and self._now() < self._replan_at:
                return None
            result = self._inner(obs)
            assert result is not None, 'the inner policy of this test wrapper always returns a chunk'
            now = self._now()
            result = [{**action, 'timestamp': now + action['timestamp']} for action in result]
            self._replan_at = now + (result[-1]['timestamp'] - now) / 2
            return result

    def wrap_session(self, inner: Session, context, now):
        return _ReplanEarly._Session(inner, now)


class _TimedRecorder(pimm.SignalEmitter):
    """Records each emission against the world clock, so a test can read when a command went out."""

    def __init__(self, clock: pimm.Clock):
        self._clock = clock
        self.emitted: list[tuple[float, Any]] = []

    def emit(self, data, ts: int = -1):
        self.emitted.append((self._clock.now(), data))


def _run_sim_episode(world, policy, wrapper, *, latency, steps=4000, run_sec=1.5) -> list[tuple[float, Any]]:
    """One sim trial under ``latency``; returns the grip commands with the world time each went out at."""
    harness = Harness(wrapper.wrap(policy), make_embodiment(simulated=True))
    grip_recorder = _TimedRecorder(world.clock)
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='t', inference_latency=latency)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.001),
        (None, run_sec),
    ])
    drive_scheduler(world.start([harness, driver, _Pacer()]), steps=steps)
    return grip_recorder.emitted[1:]  # drop the startup home


@pytest.mark.timeout(20.0)
def test_default_latency_pauses_the_world_for_the_call(world):
    """Sim's default charges nothing: the world does not advance while the model runs, so the chunk is
    anchored at the observation's own instant however long the call really took."""
    played = _run_sim_episode(world, SlowPolicy(wall_sec=0.05), ChunkedSchedule(), latency=False)

    assert played, 'no command was played'
    assert played[0][0] < 0.01, f'the world advanced during the call: first command at {played[0][0]}s'


@pytest.mark.timeout(20.0)
@pytest.mark.parametrize('wall_sec', [0.0, 0.05])
def test_declared_latency_ignores_what_the_call_really_took(world, wall_sec):
    """The reproducible mode: the wrapper is released a fixed delay after the call started, so the played
    trace is the same against a fast server and a slow one."""
    played = _run_sim_episode(world, SlowPolicy(wall_sec=wall_sec), ChunkedSchedule(), latency=0.3)

    assert played, 'no command was played'
    assert played[0][0] == pytest.approx(0.3, abs=0.02), f'first command at {played[0][0]}s, expected the 0.3s delay'


@pytest.mark.timeout(20.0)
def test_measured_latency_charges_the_calls_own_wall_duration(world):
    """``inference_latency=True`` charges the world what the model really took, so a slow server is scored
    as slow — at the cost of a trace that inherits the machine's noise."""
    played = _run_sim_episode(world, SlowPolicy(wall_sec=0.2), ChunkedSchedule(), latency=True)

    assert played, 'no command was played'
    assert played[0][0] >= 0.2, f'first command at {played[0][0]}s, under the 0.2s the call took'


@pytest.mark.timeout(20.0)
def test_harness_keeps_playing_while_a_call_is_in_flight(world):
    """A wrapper that replans before its chunk is exhausted leaves waypoints due during inference, and the
    harness emits them on time instead of standing still until the model answers."""
    played = _run_sim_episode(world, SlowPolicy(span_sec=0.4, steps=20), _ReplanEarly(), latency=0.15)

    # The second call starts halfway through the first chunk (0.2s in) and is owed 0.15s; the waypoints due
    # in that window have to keep going out.
    during = [t for t, _ in played if 0.2 <= t < 0.35]
    assert len(during) >= 3, f'the harness stopped playing during inference: {[t for t, _ in played]}'


@pytest.mark.timeout(3.0)
def test_installed_trajectory_clears_the_channels_it_omits(world):
    """A trajectory naming only one channel replaces the whole schedule: the omitted channel stops being
    played rather than draining the previous trajectory's tail."""

    class _GripThenArm(Session):
        """First a two-channel chunk, then an arm-only one that must silence the gripper."""

        def __init__(self):
            self._calls = 0

        def __call__(self, obs):
            self._calls += 1
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
            if self._calls == 1:
                return [{keys.ROBOT_COMMAND: command, 'target_grip': 0.5, 'timestamp': i * 0.01} for i in range(10)]
            return [{keys.ROBOT_COMMAND: command, 'timestamp': i * 0.01} for i in range(10)]

    class _GripThenArmPolicy(Policy):
        def new_session(self, context=None, *, now=None, gate=None):
            return _GripThenArm()

    harness = Harness(ChunkedSchedule().wrap(_GripThenArmPolicy()), make_embodiment())
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='t')), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.001),
        (None, 0.5),
    ])
    drive_scheduler(world.start([harness, driver]), steps=1000)

    grips = _emitted_grips(grip_recorder)
    assert grips[0] == 0.0  # the startup home
    assert set(grips[1:]) == {0.5}, f'the second chunk kept the gripper playing: {grips}'


@pytest.mark.timeout(3.0)
def test_home_and_manual_commands_are_emitted_as_plain_values(world):
    """Homing and operator commands bypass the schedule: they are the command, not a plan to play."""
    harness = Harness(StubPolicy(), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())
    manual_em = world.pair(harness.manual_command)

    pose = Transform3D(translation=np.array([0.1, 0.1, 0.1], dtype=np.float32), rotation=Rotation.identity)
    manual = CartesianPosition(pose=pose)
    driver = ManualDriver([(partial(manual_em.emit, {keys.ROBOT_COMMAND: manual}), 0.01), (None, 0.02)])
    drive_scheduler(world.start([harness, driver]), steps=50)

    assert _emitted_commands(cmd_recorder) == [Reset(), manual]
    assert _emitted_grips(grip_recorder) == [0.0]


@pytest.mark.timeout(20.0)
def test_abort_discards_a_call_that_is_still_in_flight(world):
    """An ABORT while the gate is still holding the model's answer throws that answer away: the trajectory
    it carries never reaches the devices."""
    harness = Harness(ChunkedSchedule().wrap(SlowPolicy()), make_embodiment(simulated=True))
    cmd_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations['robot_state'])
    grip_em = world.pair(harness.observations[keys.GRIP])
    directive_em = world.pair(harness.directive)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(directive_em.emit, Directive.RUN(task='t', inference_latency=1.0)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.001),
        (None, 0.05),  # well inside the 1.0s the gate owes the call
        (partial(directive_em.emit, Directive.ABORT()), 0.0),
        (None, 0.05),
    ])
    drive_scheduler(world.start([harness, driver, _Pacer()]), steps=2000)

    assert all(isinstance(c, Reset) for c in _emitted_commands(cmd_recorder))
