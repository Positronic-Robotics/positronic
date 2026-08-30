import threading
import time
from collections import deque
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import pimm
from pimm.tests.testing import Passive, wire_call
from positronic import keys, telemetry, telemetry_keys, wire
from positronic.dataset.ds_writer_agent import DsWriterCommand, DsWriterCommandType
from positronic.dataset.serializers import Serializers
from positronic.drivers import roboarm
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianDelta, CartesianPosition, JointPosition, from_wire, to_wire
from positronic.drivers.roboarm.models import DEFAULT_FRAME, EE_LINK, bundled_franka_model
from positronic.drivers.roboarm.tests.fakes import make_robot_state
from positronic.eval import Command, Embodiment, Observation, Task
from positronic.geom import Rotation, Transform3D
from positronic.offboard.client import InferenceSession
from positronic.policy.base import INFER, Answer, DelegatingPolicy, DelegatingSession, Fn, Layer, Policy, Session
from positronic.policy.codec import ActionTimestamp
from positronic.policy.harness import MAX_ROUND_SEC, MIN_ROUND_SEC, WAIT_PERIOD_SEC, Harness, _EpisodeInference
from positronic.policy.layers import ChunkPlayer, StopOnFault
from positronic.policy.remote import round_trip
from positronic.tests.testing_coutils import ManualDriver, RecordingEmitter, drive_scheduler

WAIT_PERIOD_NS = round(WAIT_PERIOD_SEC * 1e9)


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
# What an operator's finish puts on ``done``; the harness stamps ``eval.terminated`` over it.
OPERATOR_DONE = {keys.EVAL_ENDED_BY: keys.ENDED_BY_OPERATOR}


def make_embodiment(
    descriptor: str = '', cameras=(CAM,), static_meta=None, simulated=False, prepare_handlers=None
) -> Embodiment:
    """Minimal Franka-shaped embodiment for harness unit tests.

    The sources/dests are unbound: these tests pair the harness ports directly
    (never via ``wire_embodiment``), so only the spec — names, serializers,
    descriptor — is read by the Harness.
    """
    observations = {
        keys.ROBOT_STATE: Observation(pimm.ControlSystemEmitter(Passive()), Serializers.robot_state),
        keys.GRIP: Observation(pimm.ControlSystemEmitter(Passive()), None),
    }
    for cam in cameras:
        observations[cam] = Observation(pimm.ControlSystemEmitter(Passive()), Serializers.camera_images)
    commands = {
        keys.ROBOT_COMMAND: Command(pimm.ControlSystemReceiver(Passive()), Serializers.robot_command),
        'target_grip': Command(pimm.ControlSystemReceiver(Passive()), None),
    }
    return Embodiment(
        descriptor=descriptor,
        observations=observations,
        commands=commands,
        prepare_handlers=prepare_handlers or {},
        static_meta=static_meta or {},
        meta_source=pimm.ControlSystemEmitter(Passive()),
        simulated=simulated,
    )


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

    @contextmanager
    def episode(self, context=None):
        self.reset_calls += 1
        self.last_reset_context = context
        yield {INFER: self._infer}

    def _infer(self, obs):
        self.last_obs = obs
        return [{keys.ROBOT_COMMAND: self.command, 'target_grip': self.target_grip, 'timestamp': 0.0}]


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
        self.observations: list[dict[str, Any]] = []
        self.reset_calls = 0
        self.last_reset_context: dict[str, Any] | None = None
        self._meta: dict[str, object] = meta or {}

    @property
    def meta(self) -> dict[str, object]:
        return self._meta

    @contextmanager
    def episode(self, context=None):
        self.reset_calls += 1
        self.last_reset_context = context
        yield {INFER: self._infer}

    def _infer(self, obs):
        self.last_obs = obs
        self.observations.append(obs)
        return [{keys.ROBOT_COMMAND: self.command, 'target_grip': self.target_grip, 'timestamp': 0.0}]


class ChunkPolicy(StubPolicy):
    """Policy that returns chunks of 10 actions with grip values encoding the chunk number."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    def _infer(self, obs):
        self.counter += 1
        dt = 0.005
        return [
            {keys.ROBOT_COMMAND: self.command, 'target_grip': self.counter * 100.0 + i, 'timestamp': i * dt}
            for i in range(10)
        ]


class _FakeInferenceSession(InferenceSession):
    """A stub ``InferenceSession`` returning a canned action after ``wall_sec`` of real time, so the wire
    round trip over it is the real inference boundary that records the ``policy.infer`` span."""

    def __init__(self, action: list[dict[str, Any]], wall_sec: float = 0.0) -> None:
        self._action = action
        self._wall_sec = wall_sec

    def infer(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        time.sleep(self._wall_sec)
        return self._action

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


class ServedPolicy(Policy):
    """A policy whose model runs in a served function: a real wire round trip over the ``InferenceSession``
    it is given, so its inference records the ``policy.infer`` span independent of any layer.

    A model that costs wall time, hangs or raises belongs in that session's ``infer``.
    """

    def __init__(self, session: InferenceSession) -> None:
        self._session = session

    @contextmanager
    def episode(self, context=None):
        try:
            yield {INFER: partial(round_trip, self._session, False)}
        finally:
            self._session.close()


def _boom(obs):
    raise RuntimeError('inference boom')


def slow_chunk(span_sec: float = 0.2, steps: int = 10) -> list[dict[str, Any]]:
    """A chunk of ``steps`` waypoints over ``span_sec``, which a scheduler plays for that long before it
    asks again."""
    dt = span_sec / steps
    pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
    return [
        {keys.ROBOT_COMMAND: CartesianPosition(pose=pose), keys.TARGET_GRIP: float(i), keys.ACTION_TIMESTAMP: i * dt}
        for i in range(steps)
    ]


class RemoteStubPolicy(ServedPolicy):
    """A stub policy answering a canned chunk after ``wall_sec`` of real time."""

    def __init__(
        self,
        command: roboarm.command.CommandType | None = None,
        target_grip: float = 0.33,
        wall_sec: float = 0.0,
        chunk: list[dict[str, Any]] | None = None,
    ) -> None:
        if command is None:
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
        action = chunk or [
            {keys.ROBOT_COMMAND: command, keys.TARGET_GRIP: float(target_grip), keys.ACTION_TIMESTAMP: 0.0}
        ]
        super().__init__(_FakeInferenceSession(action, wall_sec))


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


def emit_ready_payload(frame_emitter, robot_emitter, grip_emitter, robot_state):
    frame_adapter = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.dtype(np.uint8))
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


class _Scene(pimm.ControlSystem):
    """The scene a trial runs in, drawn by ``draw`` when the harness asks and answered ``draw_s`` later."""

    def __init__(self, draw, draw_s: float = 0.0):
        self.env_reset = pimm.calls.ControlSystemHandler[Any, None](self)
        self._draw = draw
        self._draw_s = draw_s

    def run(self, should_stop, clock):
        while not should_stop.value:
            for call in self.env_reset.incoming():
                with pimm.calls.raise_to(call):
                    self._draw(call.request)
                    if self._draw_s:
                        yield pimm.Sleep(self._draw_s)
                    call.set_result(None)
            yield pimm.Sleep(0.001)


_NEVER_STOPS: pimm.SignalReceiver[bool] = pimm.NoOpReceiver()


def _pair_all(world, harness):
    """Pair all harness signals and return a dict of test handles."""
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    deadline_recorder = RecordingEmitter()
    harness.deadline_ns._bind(deadline_recorder)
    return {
        'frame_em': world.pair(harness.observations[CAM]),
        'robot_em': world.pair(harness.observations[keys.ROBOT_STATE]),
        'grip_em': world.pair(harness.observations[keys.GRIP]),
        'perform_task': world.pair(harness.perform_task),
        'done_em': world.pair(harness.done),
        'command_rx': world.pair(harness.commands[keys.ROBOT_COMMAND]),
        'grip_rx': world.pair(harness.commands['target_grip']),
        'meta_em': world.pair(harness.robot_meta_in),
        'ds_recorder': ds_recorder,
        'deadline_recorder': deadline_recorder,
    }


def _ds_commands(p) -> list[DsWriterCommand]:
    return [data for _, data in p['ds_recorder'].emitted]


def _deadlines(p) -> list[int | None]:
    """Every deadline the harness published, in the order it published them."""
    return [data for _, data in p['deadline_recorder'].emitted]


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
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='stack-blocks', timeout_sec=None)), 0.0),
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
    assert obs[keys.DESCRIPTOR] == ''  # no descriptor passed -> empty string reaches the policy
    # Recording == canonical policy I/O: the policy sees the same ``robot_state`` serializer
    # the dataset records. wall/obs timestamps carry volatile values, so lock the stable key set.
    assert set(obs) - {keys.WALL_TIME_NS, keys.OBS_TIME_NS} == {
        CAM,
        keys.JOINTS,
        keys.JOINT_VEL,
        keys.EE_POSE,
        keys.GRIP,
        keys.ROBOT_STATUS,
        keys.TASK,
        keys.DESCRIPTOR,
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
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment(descriptor='mujoco.franka'))
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.05),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    assert policy.last_obs is not None
    assert policy.last_obs[keys.DESCRIPTOR] == 'mujoco.franka'


@pytest.mark.timeout(3.0)
def test_robot_model_stays_out_of_the_observation(world):
    """A codec carries its frame as a transform, so the model never has to leave the rig."""
    policy = SpyPolicy()
    model = bundled_franka_model()
    statics = {keys.URDF: model[keys.URDF], keys.CONTROL_FRAME: model[keys.CONTROL_FRAME]}
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment(static_meta=statics))
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations['grip'])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
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
    harness = Harness(ChunkPlayer().wrap(SpyPolicy()), make_embodiment(static_meta=static_meta))
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands['target_grip']._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())
    perform_task = world.pair(harness.perform_task)
    meta_em = world.pair(harness.robot_meta_in)

    steps = [(partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0)]
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
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    assert CAM in harness.observations

    robot_state = make_robot_state([0.2, 0.0, -0.1], [0.7, 0.1, -0.2])

    def assert_no_inference():
        assert policy.last_obs is None
        assert not _emitted_commands(cmd_recorder)

    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='dummy-task', timeout_sec=None)), 0.01),
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
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment(), static_meta={keys.JOINT_SIGNALS: [keys.JOINTS]})
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['meta_em'].emit, {keys.URDF: '<robot/>', keys.JOINT_NAMES: ['j1']}), 0.0),
        (partial(p['perform_task'], Task(instruction_source='test', timeout_sec=None)), 0.01),
        (partial(p['done_em'].emit, OPERATOR_DONE), 0.02),
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

    class _StaticMetaPolicy(Policy):
        def __init__(self):
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            self._command = CartesianPosition(pose=pose)

        @contextmanager
        def episode(self, context=None):
            yield {INFER: lambda obs: [{keys.ROBOT_COMMAND: self._command, 'target_grip': 0.0, 'timestamp': 0.0}]}

        @property
        def meta(self):
            return {'checkpoint': 'v1', 'type': 'static'}

    harness = Harness(ChunkPlayer().wrap(_StaticMetaPolicy()), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(p['perform_task'], Task(instruction_source='t', timeout_sec=None)), 0.0),
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (partial(p['done_em'].emit, OPERATOR_DONE), 0.02),
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
def test_finish_emits_ds_stop_with_data(world):
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['perform_task'], Task(instruction_source='test', timeout_sec=None)), 0.0),
        (partial(p['done_em'].emit, {'outcome': 'Success', 'notes': 'good'}), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data['outcome'] == 'Success'
    assert stops[0].static_data['notes'] == 'good'


@pytest.mark.timeout(3.0)
def test_the_call_is_answered_with_the_terminal_the_episode_ended_on(world):
    """A task with no timeout ends on ``done`` alone, and its caller gets what it ended on."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    answer = p['perform_task'](Task(instruction_source='test', timeout_sec=None))
    drive_scheduler(scheduler, steps=5)
    assert not answer.done()

    p['done_em'].emit({keys.EVAL_SUCCESS: True})
    drive_scheduler(scheduler, steps=10)

    assert answer.result() == {keys.EVAL_SUCCESS: True, keys.EVAL_TERMINATED: True}


@pytest.mark.timeout(3.0)
def test_the_world_stopping_under_a_live_episode_fails_the_call(world):
    """The caller hears that its episode will never answer, rather than holding a handle that never completes."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    p = _pair_all(world, harness)

    scheduler = world.start([harness, ManualDriver([(None, 0.02)])])
    answer = p['perform_task'](Task(instruction_source='test', timeout_sec=None))
    drive_scheduler(scheduler, steps=20)

    with pytest.raises(RuntimeError):
        answer.result()


@pytest.mark.timeout(10.0)
def test_an_uncharged_wait_ends_when_the_world_comes_down(world):
    """An uncharged trial waits its function out, so a model that never answers would hold the loop for ever.
    The wait ends on ``should_stop`` as well."""
    never_answers = threading.Event()

    class _HangingPolicy(Policy):
        @contextmanager
        def episode(self, context=None):
            yield {INFER: lambda obs: never_answers.wait()}

    inference = _EpisodeInference(ChunkPlayer().wrap(_HangingPolicy()), {}, charges_wall_time=False, clock=world.clock)
    try:
        inference({})  # starts the function, which never answers
        world.request_stop()
        inference.wait(world.should_stop_reader())
    finally:
        never_answers.set()


@pytest.mark.timeout(3.0)
def test_a_session_that_raises_fails_the_call_that_asked_for_the_episode(world):
    """The session is called on the loop thread, so its failure reaches whoever asked for the episode rather
    than the log."""

    class _RaisingPolicy(Policy):
        @contextmanager
        def episode(self, context=None):
            yield {INFER: _boom}

    harness = Harness(ChunkPlayer().wrap(_RaisingPolicy()), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    answer = p['perform_task'](Task(instruction_source='test', timeout_sec=None))
    with pytest.raises(RuntimeError, match='inference boom'):
        drive_scheduler(scheduler, steps=40)

    with pytest.raises(RuntimeError, match='inference boom'):
        answer.result()


@pytest.mark.timeout(3.0)
def test_trial_ends_at_its_timeout(world):
    """Nothing ever lands on ``done``, yet the trial still ends at ``task.timeout_sec``: terminated=False."""
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    p['perform_task'](Task(instruction_source='test', timeout_sec=0.05))
    drive_scheduler(scheduler, steps=200)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is False
    assert keys.EVAL_SUCCESS not in stops[0].static_data


@pytest.mark.timeout(3.0)
def test_trial_budget_starts_when_the_rig_is_ready(world):
    """The 0.05 budget is measured from the end of the prepare, not from the ask: the 0.2 the scene takes to
    draw is not the trial's to spend."""
    scene = _Scene(lambda _: None, draw_s=0.2)
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment(prepare_handlers={keys.SCENE: scene.env_reset}))
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.SCENE], scene.env_reset)

    scheduler = world.start([harness, scene, _Pacer()])
    opened = world.clock.now()
    answer = p['perform_task'](Task(instruction_source='test', timeout_sec=0.05, prepare_args={keys.SCENE: {}}))
    drive_scheduler(scheduler, steps=2000)

    assert answer.done(), 'the trial never ended'
    assert world.clock.now() - opened >= 0.25, 'the budget was spent on the draw'


@pytest.mark.timeout(3.0)
def test_trial_stop_signal_terminates(world):
    """Delivering the privileged ``done`` ends a trial early: terminated=True, payload recorded."""
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    # Timeout far in the future so the stop-signal, not the clock, ends the trial.
    p['perform_task'](Task(instruction_source='test', timeout_sec=100.0))
    drive_scheduler(scheduler, steps=5)
    # Trial is live and unbounded by the clock: nothing committed yet.
    assert not [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]

    p['done_em'].emit({keys.EVAL_SUCCESS: True})
    drive_scheduler(scheduler, steps=10)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is True
    assert stops[0].static_data[keys.EVAL_SUCCESS] is True  # the delivered payload lands in static data


@pytest.mark.timeout(3.0)
def test_stale_done_does_not_terminate_next_trial(world):
    """``done`` latches (last-writer-wins): trial 0's terminal would re-fire on trial 1, whose later
    deadline still sits after the stale timestamp. Only a freshly delivered ``done`` terminates, so the
    latched value is ignored — no producer ``reset`` clears it here (``reset`` is ``None``, as on a real
    embodiment). A falsy payload never terminates; trial 1 runs until its own fresh terminal lands."""
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)
    task = Task(instruction_source='t', timeout_sec=100.0)

    def stop_count():
        return len([c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE])

    scheduler = world.start([harness])
    p['perform_task'](task)
    drive_scheduler(scheduler, steps=5)
    p['done_em'].emit({})  # falsy: does not terminate
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 0

    p['done_em'].emit({keys.EVAL_SUCCESS: True})  # fresh truthy: ends trial 0
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 1

    # The terminal is still latched on trial 1 but no longer fresh, so it must NOT re-fire.
    p['perform_task'](task)
    drive_scheduler(scheduler, steps=10)
    assert stop_count() == 1

    p['done_em'].emit({keys.EVAL_SUCCESS: True})  # a fresh delivery ends trial 1
    drive_scheduler(scheduler, steps=10)
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 2
    assert all(s.static_data[keys.EVAL_TERMINATED] is True for s in stops)


class _FrameIndexDevice(pimm.ControlSystem):
    """Publishes a rising frame index on ``state``. ``reset`` publishes frame 0 (with fresh ``meta``) as it
    answers, then each turn steps and publishes the next. A reader whose first frame is >= 1 saw the device
    step before it read."""

    def __init__(self):
        self.state = pimm.ControlSystemEmitter(self)
        self.meta = pimm.ControlSystemEmitter(self)
        self.cmd = pimm.ControlSystemReceiver(self)
        self.env_reset = pimm.calls.ControlSystemHandler[Any, None](self)
        self._frame = 0

    def run(self, should_stop, clock):
        while not should_stop.value:
            yield pimm.Sleep(0.01)
            if (call := next(self.env_reset.incoming(), None)) is not None:
                self._frame = 0
                self.meta.emit({})  # fresh scene meta, recorded into the episode at finalize
                self.state.emit(float(self._frame))
                call.set_result(None)
            else:
                self._frame += 1
                self.state.emit(float(self._frame))


@pytest.mark.timeout(3.0)
def test_the_policy_opens_on_the_frame_the_reset_published(world):
    """The first inference reads the frame the reset published, never a stepped one: the episode opens and
    infers in the same round, so the device gets no turn to step in between."""
    device = _FrameIndexDevice()
    embodiment = Embodiment(
        descriptor='',
        observations={'frame': Observation(device.state, None)},
        commands={keys.ROBOT_COMMAND: Command(device.cmd, None)},
        prepare_handlers={keys.SCENE: device.env_reset},
        static_meta={},
        meta_source=device.meta,
        control_systems=(device,),
        simulated=True,
    )
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), embodiment)
    perform_task = world.pair(harness.perform_task)
    wire.wire_embodiment(world, harness, embodiment, None)

    scheduler = world.start([harness, device])
    perform_task(Task(instruction_source='t', timeout_sec=100.0, prepare_args={keys.SCENE: {}}))
    drive_scheduler(scheduler, steps=20)

    assert policy.observations, 'policy was never called'
    assert policy.observations[0]['frame'] == 0.0  # the reset's frame, not a stepped one
    assert any(o['frame'] >= 1.0 for o in policy.observations)  # the device did step (so the guard can fail)


@pytest.mark.timeout(3.0)
def test_task_done_terminates_through_wire_embodiment(world):
    """An eval's ``done`` source reaches ``harness.done`` through ``wire_embodiment`` and ends the
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
        commands={'x': Command(device.cmd, None)},
        prepare_handlers={},
        static_meta={},
        meta_source=None,
    )
    # Termination is independent of the policy layers; the minimal embodiment has no
    # ``robot_state``, so run the stub policy bare.
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), embodiment)
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    perform_task = world.pair(harness.perform_task)
    wire.wire_embodiment(world, harness, embodiment, None, done=device.done)

    scheduler = world.start([harness, device])
    perform_task(Task(instruction_source='t', timeout_sec=100.0))
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
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # The 0.05s deadline lapses first; done lands at ~0.1s, after the trial has already timed out.
    driver = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.1),
        (partial(p['done_em'].emit, {keys.EVAL_SUCCESS: True}), 0.3),
        (None, 0.0),
    ])
    scheduler = world.start([harness, driver])
    p['perform_task'](Task(instruction_source='t', timeout_sec=0.05))
    drive_scheduler(scheduler, steps=200)

    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is False
    assert keys.EVAL_SUCCESS not in stops[0].static_data


@pytest.mark.timeout(3.0)
def test_a_handler_the_trial_does_not_name_is_left_alone(world):
    """A rig readies more than any one trial wants, so what a trial leaves unnamed it leaves as it stands."""
    drawn, moved = [], []
    scene, arm = _Scene(drawn.append), _Scene(moved.append)
    handlers = {keys.SCENE: scene.env_reset, keys.ARM: arm.env_reset}
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment(prepare_handlers=handlers))
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.SCENE], scene.env_reset)
    wire_call(world, harness.prepare[keys.ARM], arm.env_reset)

    scheduler = world.start([harness, scene, arm])
    p['perform_task'](Task(instruction_source='stack', timeout_sec=0.05, prepare_args={keys.SCENE: {}}))
    drive_scheduler(scheduler, steps=50)

    assert drawn == [{}]
    assert moved == []


@pytest.mark.timeout(3.0)
@pytest.mark.parametrize('simulated', [False, True], ids=['real', 'sim'])
def test_every_rig_is_put_back_where_the_trial_placed_it(world, simulated):
    """A powered arm holds the policy's last setpoint through the gap to the next trial, so what the trial
    placed is placed again once the recording stops — the same joints it opened on, never a fresh draw. A
    sim rig is asked no differently."""
    placed = []
    arm = _Scene(placed.append)
    handlers = {keys.ARM: arm.env_reset}
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment(simulated=simulated, prepare_handlers=handlers))
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.ARM], arm.env_reset)

    start = JointPosition(np.arange(7, dtype=np.float64))
    scheduler = world.start([harness, arm, _Pacer()])
    answer = p['perform_task'](Task(instruction_source='stack', timeout_sec=0.05, prepare_args={keys.ARM: start}))
    drive_scheduler(scheduler, steps=2000)

    assert answer.done(), 'the episode never ended, so there was no close to be put back by'
    assert len(placed) == 2, 'the trial opened on its start pose and closes back at it'
    # The same object every time: a trial closes on the joints it opened on, never a fresh draw.
    assert all(asked is start for asked in placed)


class _PlacesOnce(pimm.ControlSystem):
    """A device that answers the first ask and keeps every one after it in hand."""

    def __init__(self):
        self.env_reset = pimm.calls.ControlSystemHandler[Any, None](self)
        self.asks = 0

    def run(self, should_stop, clock):
        while not should_stop.value:
            for call in self.env_reset.incoming():
                self.asks += 1
                if self.asks == 1:
                    call.set_result(None)
            yield pimm.Sleep(0.001)


@pytest.mark.timeout(3.0)
def test_a_trial_does_not_end_until_the_rig_is_back_where_it_started(world):
    """The terminal waits on the return move. Handed back sooner, the next trial's scene draw goes ahead of a
    move still travelling and rebuilds the model under it, leaving nothing but its timeout to end it."""
    arm = _PlacesOnce()
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment(prepare_handlers={keys.ARM: arm.env_reset}))
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.ARM], arm.env_reset)

    scheduler = world.start([harness, arm, _Pacer()])
    task = Task(instruction_source='stack', timeout_sec=0.05, prepare_args={keys.ARM: JointPosition(np.zeros(7))})
    answer = p['perform_task'](task)
    drive_scheduler(scheduler, steps=2000)

    assert arm.asks == 2, 'the rig was never asked to go back'
    assert not answer.done(), 'the terminal landed while the return move was still in hand'


def test_the_deadline_is_published_once_the_rig_is_ready(world):
    """An idle harness publishes nothing: ``deadline_ns`` states the instant the harness will stop at, and
    between episodes there is none to state. The first one goes out when the episode's prepare has
    answered — this embodiment readies nothing, so that is the round the task is taken."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([(None, 100.0)])  # outlives the pumping, so the world stays up between phases
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=20)
    assert _deadlines(p) == []

    asked_at_ns = world.clock.now_ns()
    p['perform_task'](Task(instruction_source='t', timeout_sec=5.0))
    drive_scheduler(scheduler, steps=20)
    # The world is already past zero here, so the published instant and a bare ``timeout_sec`` are
    # different numbers, which is what this pins.
    assert _deadlines(p) == [pytest.approx(asked_at_ns + 5e9, abs=2 * WAIT_PERIOD_NS)]
    assert asked_at_ns > 2 * WAIT_PERIOD_NS, 'the two would be indistinguishable at a clock still near zero'


@pytest.mark.timeout(3.0)
def test_no_deadline_is_published_while_the_rig_is_still_readying(world):
    """A rig that takes its time readying gets no deadline until it answers, so the reset never comes out
    of the budget — and a display shows no countdown rather than one measured from before the episode."""

    class _SlowScene(pimm.ControlSystem):
        """A prepare handler that answers only once ``release`` is set."""

        def __init__(self):
            self.env_reset = pimm.calls.ControlSystemHandler[Any, None](self)
            self.release = False

        def run(self, should_stop, clock):
            held = []
            while not should_stop.value:
                held.extend(self.env_reset.incoming())
                if self.release:
                    for call in held:
                        call.set_result(None)
                    held = []
                yield pimm.Sleep(0.001)

    scene = _SlowScene()
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment(prepare_handlers={keys.SCENE: scene.env_reset}))
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.SCENE], scene.env_reset)

    scheduler = world.start([harness, scene])
    p['perform_task'](Task(instruction_source='t', timeout_sec=5.0, prepare_args={keys.SCENE: {}}))
    drive_scheduler(scheduler, steps=100)
    assert _deadlines(p) == [], 'a deadline went out while the rig was still readying'

    scene.release = True
    drive_scheduler(scheduler, steps=100)
    assert _deadlines(p) and _deadlines(p)[0] is not None, 'the readied rig never got its deadline'


@pytest.mark.timeout(3.0)
def test_the_deadline_clears_when_the_episode_ends(world):
    """``None`` at the close is what tells a display the countdown is over, and it is published only
    there: cleared mid-episode it would stop a countdown the harness is still enforcing."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.2),
        (partial(p['done_em'].emit, {keys.EVAL_SUCCESS: True}), 100.0),
    ])
    scheduler = world.start([harness, driver])
    p['perform_task'](Task(instruction_source='t', timeout_sec=5.0))
    drive_scheduler(scheduler, steps=200)

    assert [c.type for c in _ds_commands(p)].count(DsWriterCommandType.STOP_EPISODE) == 1, 'the episode never ended'
    assert _deadlines(p)[-1] is None
    assert _deadlines(p).count(None) == 1, 'a deadline was withdrawn while the episode was still running'


@pytest.mark.timeout(3.0)
def test_an_episode_with_no_timeout_publishes_no_deadline(world):
    """A task with no ``timeout_sec`` publishes ``None``, which corrects a reader still holding the
    previous episode's deadline. Silence would leave it counting down against one that has lapsed."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 100.0)
    ])
    scheduler = world.start([harness, driver])
    p['perform_task'](Task(instruction_source='t', timeout_sec=None))
    drive_scheduler(scheduler, steps=200)

    assert _deadlines(p) == [None]


@pytest.mark.timeout(3.0)
def test_a_world_stopping_mid_episode_withdraws_the_deadline(world):
    """A run that ends with an episode still live withdraws its deadline like any other close: a receiver
    latches what it last got, and there is no later episode to correct it."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    # The script runs out while the episode is live; a control system returning is what stops the world.
    driver = ManualDriver([(partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.1)])
    scheduler = world.start([harness, driver])
    # A budget far longer than the run, so expiry cannot be what withdraws the deadline.
    p['perform_task'](Task(instruction_source='t', timeout_sec=100.0))
    drive_scheduler(scheduler, steps=200)

    assert _deadlines(p)[0] is not None, 'no deadline was ever armed, so nothing here was under test'
    assert _deadlines(p)[-1] is None


@pytest.mark.timeout(3.0)
def test_an_episode_abandoned_by_a_raise_withdraws_the_deadline(world):
    """An episode a raise abandons withdraws its deadline like any other close."""

    class _BoomPolicy(Policy):
        @contextmanager
        def episode(self, context=None):
            yield {INFER: _boom}

    harness = Harness(ChunkPlayer().wrap(_BoomPolicy()), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 100.0)
    ])
    scheduler = world.start([harness, driver])
    p['perform_task'](Task(instruction_source='t', timeout_sec=100.0))
    with pytest.raises(RuntimeError, match='inference boom'):
        drive_scheduler(scheduler, steps=200)

    assert _deadlines(p)[0] is not None, 'no deadline was ever armed, so nothing here was under test'
    assert _deadlines(p)[-1] is None


@pytest.mark.timeout(3.0)
def test_a_trial_asking_to_ready_what_the_rig_has_not_got_fails_loudly(world):
    """Only the handlers a task names are asked, so a name matching none of them would go unasked and the
    trial would open on a rig nothing readied."""
    scene = _Scene(lambda _params: None)
    embodiment = make_embodiment(descriptor='yam', prepare_handlers={keys.SCENE: scene.env_reset})
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), embodiment)
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.SCENE], scene.env_reset)

    scheduler = world.start([harness, scene])
    answer = p['perform_task'](
        Task(instruction_source='stack', timeout_sec=0.05, prepare_args={keys.ARM: JointPosition(np.zeros(7))})
    )

    named = r"\['arm'\] is not something yam readies; it readies \['scene'\]"
    with pytest.raises(ValueError, match=named):
        drive_scheduler(scheduler, steps=50)
    with pytest.raises(ValueError):
        answer.result()  # and whoever asked for the trial hears it, rather than reading a clean episode


@pytest.mark.timeout(3.0)
def test_trial_seed_reaches_task_reset_and_meta(world):
    """A trial's params draw its scene and identify it: the scene prepare is asked with them, and they land
    in episode meta beside the instruction. A real rig records that it charged inference time, whatever the
    task asked."""
    policy = StubPolicy()
    seeds = []

    def reset(params):
        seeds.append(params.get(keys.EVAL_SEED))
        p['meta_em'].emit({})  # the producer publishes fresh scene meta, recorded into the episode at finalize

    scene = _Scene(reset)
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment(prepare_handlers={keys.SCENE: scene.env_reset}))
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.SCENE], scene.env_reset)

    scheduler = world.start([harness, scene])
    for i in range(2):
        seed = {keys.EVAL_SEED: 7 + i}
        p['perform_task'](
            Task(instruction_source='stack', timeout_sec=0.05, prepare_args={keys.SCENE: seed}, meta=seed)
        )
        drive_scheduler(scheduler, steps=200)

    assert seeds == [7, 8]
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 2
    assert [s.static_data[keys.EVAL_SEED] for s in stops] == [7, 8]
    assert all(s.static_data[keys.TASK] == 'stack' for s in stops)
    assert all(s.static_data[keys.EVAL_UNIVERSE] == 'real' for s in stops)
    assert all(s.static_data[keys.EVAL_EMBODIMENT] == '' for s in stops)
    assert all(s.static_data[keys.EVAL_TIMEOUT] == 0.05 for s in stops)
    assert all(s.static_data[keys.EVAL_CHARGE_INFERENCE_TIME] is True for s in stops)
    assert policy.reset_calls == 2


@pytest.mark.timeout(3.0)
def test_timeout_during_inference_drops_the_chunk(world):
    """A trial whose deadline lapses while the model is still inside its function ends with that function in
    flight: the trajectory it eventually returns is discarded, never emitted past the advertised termination
    point."""
    harness = Harness(
        # the function runs well past the deadline
        ChunkPlayer().wrap(RemoteStubPolicy(wall_sec=0.3, chunk=slow_chunk())),
        make_embodiment(simulated=True),
    )
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    ds_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(ds_recorder)

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    driver = ManualDriver([(partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01), (None, 0.3)])

    scheduler = world.start([harness, driver, _Pacer()])
    perform_task(Task(instruction_source='test', timeout_sec=0.05, charge_inference_time=True))
    drive_scheduler(scheduler, steps=2000)

    stops = [data for _, data in ds_recorder.emitted if data.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0].static_data[keys.EVAL_TERMINATED] is False
    # A trial that times out mid-call plays nothing: the chunk it was waiting on is dropped.
    assert not _emitted_commands(cmd_recorder)
    assert not _emitted_grips(grip_recorder)


@pytest.mark.timeout(3.0)
def test_a_terminal_landing_while_idle_does_not_end_the_next_episode(world):
    """A finish pressed with nothing running belongs to no episode, so the one asked for next runs on."""
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(p['done_em'].emit, OPERATOR_DONE), 0.02),  # nothing is running to finish
        (partial(p['perform_task'], Task(instruction_source='t', timeout_sec=None)), 0.02),
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.05),
        (None, 0.05),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    assert _ds_types(p).count(DsWriterCommandType.START_EPISODE) == 1
    stops = [c for c in _ds_commands(p) if c.type == DsWriterCommandType.STOP_EPISODE]
    assert keys.EVAL_ENDED_BY not in stops[0].static_data, 'the idle terminal ended the episode that followed it'


@pytest.mark.timeout(3.0)
def test_a_call_arriving_mid_episode_is_refused(world):
    """The live episode runs on and the second caller is told why, rather than its ask being dropped."""
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    scheduler = world.start([harness])
    live = p['perform_task'](Task(instruction_source='ep1', timeout_sec=None))
    drive_scheduler(scheduler, steps=5)
    refused = p['perform_task'](Task(instruction_source='ep2', timeout_sec=None))
    drive_scheduler(scheduler, steps=5)

    with pytest.raises(RuntimeError):
        refused.result()
    assert not live.done()
    assert _ds_types(p).count(DsWriterCommandType.START_EPISODE) == 1
    assert policy.reset_calls == 1


class _FrameWatchingSession(_FakeInferenceSession):
    """Reads its camera frame at both ends of a slow function, so a rewrite underneath it shows up as a
    difference."""

    def __init__(self, wall_sec: float):
        super().__init__([], wall_sec)
        self.seen: list[tuple[np.ndarray, np.ndarray]] = []

    def infer(self, obs):
        entry = np.array(obs[CAM])
        super().infer(obs)
        self.seen.append((entry, np.array(obs[CAM])))
        return []


@pytest.mark.timeout(10.0)
def test_a_producer_reusing_its_buffer_cannot_rewrite_a_pending_observation(world):
    """A camera renders into the array behind the adapter it re-emits, and a wall-charged trial keeps the loop
    stepping while the function runs — so the observation handed to that function has to be its own copy."""
    watcher = _FrameWatchingSession(wall_sec=0.3)
    harness = Harness(ChunkPlayer().wrap(ServedPolicy(watcher)), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    frame = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.dtype(np.uint8))

    def emit_frame(fill: int):
        frame.array[:] = np.full((2, 2, 3), fill, dtype=np.uint8)
        p['frame_em'].emit(frame)  # the same adapter every time, as a camera does
        p['robot_em'].emit(robot_state)
        p['grip_em'].emit(0.25)

    driver = ManualDriver([
        (partial(p['perform_task'], Task(instruction_source='t', timeout_sec=None, charge_inference_time=True)), 0.0),
        (partial(emit_frame, 1), 0.01),
        (partial(emit_frame, 9), 0.05),  # rewrites the buffer while the first call is still running
        (None, 0.4),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=60)

    assert watcher.seen, 'the policy was never called'
    entry, exit_ = watcher.seen[0]
    np.testing.assert_array_equal(entry, exit_, 'the observation was rewritten while the function was in flight')


class _AbandonedCallPolicy(ServedPolicy):
    """Records the order of session openings and function completions, with a function that outlives its
    episode."""

    class _Infer(_FakeInferenceSession):
        def __init__(self, events: list[str], wall_sec: float):
            super().__init__([], wall_sec)
            self._events = events

        def infer(self, obs):
            answer = super().infer(obs)
            self._events.append('answered')
            return answer

    def __init__(self, wall_sec: float):
        self.events: list[str] = []
        super().__init__(_AbandonedCallPolicy._Infer(self.events, wall_sec))

    @contextmanager
    def episode(self, context=None):
        self.events.append('open')
        with super().episode(context) as fns:
            yield fns


@pytest.mark.timeout(10.0)
def test_a_new_episode_waits_out_the_call_the_last_one_abandoned(world):
    """An in-process policy is one model across episodes, so opening a session must not overtake a function
    still inside the previous one — ``new_session`` resets the object that function is using."""
    policy = _AbandonedCallPolicy(wall_sec=0.4)
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    emit_obs = partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state)

    driver = ManualDriver([
        (partial(p['perform_task'], Task(instruction_source='ep1', timeout_sec=None)), 0.0),
        (emit_obs, 0.01),  # the observation that starts the function
        (partial(p['done_em'].emit, OPERATOR_DONE), 0.02),  # while that function is still running
        (partial(p['perform_task'], Task(instruction_source='ep2', timeout_sec=None)), 0.02),
        (None, 0.02),
    ])

    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=40)

    assert policy.events[:3] == ['open', 'answered', 'open'], (
        f'the second session opened before the abandoned function answered: {policy.events}'
    )


@pytest.mark.timeout(3.0)
def test_run_calls_policy_reset_with_context(world):
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['perform_task'], Task(instruction_source='test-task', timeout_sec=None)), 0.0),
        (None, 0.01),
    ])

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

    def reset(_params):
        scene['task'] = 'resolved-on-reset'  # the env reports its task only here

    drawing = _Scene(reset)
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment(prepare_handlers={keys.SCENE: drawing.env_reset}))
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.SCENE], drawing.env_reset)

    scheduler = world.start([harness, drawing])
    p['perform_task'](Task(instruction_source=lambda: scene['task'], timeout_sec=0.05, prepare_args={keys.SCENE: {}}))
    drive_scheduler(scheduler, steps=200)

    assert policy.last_reset_context is not None
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
    """Finishing drops the schedule the harness is playing: the chunk's remaining waypoints never reach the
    devices, so nothing is emitted past the recorder's STOP."""
    policy = ChunkPolicy()
    wrapped = ActionTimestamp(fps=5.0).wrap(policy)  # 1.8 s chunk — won't drain before the episode ends
    harness = Harness(ChunkPlayer().wrap(wrapped), make_embodiment())
    events: list[tuple[str, object]] = []
    harness.commands[keys.ROBOT_COMMAND]._bind(_LabeledRecorder(keys.ROBOT_COMMAND, events))
    harness.commands['target_grip']._bind(_LabeledRecorder('target_grip', events))
    harness.ds_command._bind(_LabeledRecorder('ds_command', events))

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)
    done_em = world.pair(harness.done)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    script = [
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
        (partial(done_em.emit, OPERATOR_DONE), 0.0),
        (None, 0.5),
    ]
    scheduler = world.start([harness, ManualDriver(script)])
    drive_scheduler(scheduler, steps=400)

    stops = [i for i, (_, data) in enumerate(events) if getattr(data, 'type', None) is DsWriterCommandType.STOP_EPISODE]
    assert stops, 'finishing did not emit STOP_EPISODE'
    grips_after = [data for lbl, data in events[stops[0] :] if lbl == keys.TARGET_GRIP]
    assert not grips_after, f'the cancelled chunk kept playing past the finish: {grips_after}'


@pytest.mark.timeout(3.0)
def test_empty_trajectory_leaves_every_channel_holding(world):
    """A trajectory with no waypoints schedules nothing on any channel, so every device holds where it
    already is rather than one channel draining on while another stops."""

    class EmptyChunkPolicy(Policy):
        @contextmanager
        def episode(self, context=None):
            yield {INFER: lambda obs: []}

    harness = Harness(ChunkPlayer().wrap(EmptyChunkPolicy()), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands['target_grip']._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    script = [
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
    ]
    scheduler = world.start([harness, ManualDriver(script)])
    drive_scheduler(scheduler, steps=200)

    assert not _emitted_commands(cmd_recorder)  # an empty trajectory schedules nothing
    assert not _emitted_grips(grip_recorder)


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_finish(world):
    policy = ChunkPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['perform_task'](Task(instruction_source='test', timeout_sec=None))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=5)

    assert _last_grip(p) >= 100.0, 'Expected chunk 1'

    p['done_em'].emit(OPERATOR_DONE)
    drive_scheduler(scheduler, steps=3)

    p['perform_task'](Task(instruction_source='test', timeout_sec=None))
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=4)

    assert _last_grip(p) >= 200.0, 'Expected chunk 2; trajectory clearing failed'


@pytest.mark.timeout(3.0)
def test_harness_clears_trajectory_on_run(world):
    policy = ChunkPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    scheduler = world.start([harness])

    p['perform_task'](Task(instruction_source='test', timeout_sec=None))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=5)

    assert _last_grip(p) >= 100.0

    p['perform_task'](Task(instruction_source='test-restart', timeout_sec=None))
    drive_scheduler(scheduler, steps=1)

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
    drive_scheduler(scheduler, steps=20)

    assert _last_grip(p) >= 200.0, 'Expected chunk 2; trajectory clearing on a new episode failed'


@pytest.mark.timeout(3.0)
@pytest.mark.parametrize('unavailable', [RobotStatus.BUSY, RobotStatus.ERROR])
def test_the_stack_keeps_the_model_away_from_an_unavailable_arm(world, unavailable):
    """An arm that will not take a command is not tracking the plan it was given, so ``StopOnFault`` answers
    its observation itself rather than let the model plan against it. Once it is available the model is
    asked again, on a fresh
    chunk."""
    policy = ChunkPolicy()
    harness = Harness((StopOnFault() | ChunkPlayer()).wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    state_ok = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.AVAILABLE)
    state_unavailable = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=unavailable)

    scheduler = world.start([harness])

    p['perform_task'](Task(instruction_source='test', timeout_sec=None))
    drive_scheduler(scheduler, steps=1)
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, steps=3)
    assert _last_grip(p) >= 100.0

    obs_before = len(policy.observations)
    p['robot_em'].emit(state_unavailable)
    drive_scheduler(scheduler, steps=2)
    assert len(policy.observations) == obs_before, 'the model was asked about an arm that is not tracking it'

    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], state_ok)
    drive_scheduler(scheduler, steps=20)  # long enough for the first chunk to play out and the next to land
    assert _last_grip(p) >= 200.0


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


@pytest.mark.parametrize('status', list(RobotStatus))
def test_robot_state_serializer_emits_the_status_beside_the_pose(status):
    state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=status)
    serialized = Serializers.robot_state(state)
    assert set(serialized) == {'.status', '.q', '.dq', '.ee_pose'}
    assert serialized['.status'] is status


@pytest.mark.timeout(3.0)
def test_shutdown_stops_playing_the_live_chunk(world):
    """Shutdown while recording drops the schedule too: the unplayed tail of the live chunk never reaches
    the devices after the recorder's STOP."""
    events: list[tuple[str, object]] = []
    wrapped = ActionTimestamp(fps=5.0).wrap(ChunkPolicy())  # 1.8 s chunk — won't drain before shutdown
    harness = Harness(ChunkPlayer().wrap(wrapped), make_embodiment())
    harness.commands[keys.ROBOT_COMMAND]._bind(_LabeledRecorder(keys.ROBOT_COMMAND, events))
    harness.commands[keys.TARGET_GRIP]._bind(_LabeledRecorder(keys.TARGET_GRIP, events))
    harness.ds_command._bind(_LabeledRecorder('ds_command', events))

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # A call + a complete obs schedules a chunk; the driver then ends, which makes the
    # world signal shutdown while still recording — exercising the run() finalizer.
    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
        (None, 0.1),
    ])
    scheduler = world.start([harness, driver])
    drive_scheduler(scheduler, steps=200)

    stops = [i for i, (_, data) in enumerate(events) if getattr(data, 'type', None) is DsWriterCommandType.STOP_EPISODE]
    assert stops, 'shutdown did not emit STOP_EPISODE'
    assert not [lbl for lbl, _ in events[stops[0] :] if lbl == keys.TARGET_GRIP]


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


def _ask(world: pimm.World, harness: Harness, task: Task) -> None:
    """Deliver one ``perform_task``, for a test that drives the harness's generator rather than a World."""
    caller = pimm.calls.ControlSystemCaller[Task, dict[str, Any]](harness)
    wire_call(world, caller, harness.perform_task)
    caller(task)


@pytest.mark.timeout(3.0)
def test_stop_mid_episode_keeps_episode_open_for_recorder_flush(world, tmp_path):
    """A stop arriving mid-episode winds down through the same close order as ``_end_episode``: the harness
    yields a turn between queueing the recorder's STOP and closing the episode span, so the recorder's
    shutdown-flush ``record.io`` span parents to the episode, not the pass. Driven straight through the
    generator protocol: the yield after the queued STOP is the recorder's flush slot."""
    policy = StubPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    ds_recorder = RecordingEmitter()
    harness.ds_command._bind(ds_recorder)
    # Never ends within the drive: the stop, not the deadline, is what winds this episode down.
    _ask(world, harness, Task(instruction_source='stack', timeout_sec=10.0, meta={keys.EVAL_TRIAL_INDEX: 0}))
    stop = SimpleNamespace(value=False)
    clock = _ManualClock()

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-stop'), _eval_pass('run-stop'):
        gen = harness.run(cast(pimm.SignalReceiver, stop), cast(pimm.Clock, clock))
        for _ in range(20):
            next(gen)
            if any(d.type == DsWriterCommandType.START_EPISODE for _, d in ds_recorder.emitted):
                break
        else:
            pytest.fail('the trial never started')

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
    assert telemetry_keys.ATTR_EPISODE_STEPS in episode.attrs  # sealed via end_episode, not left open
    flushes = [s for s in spans if s.name == telemetry_keys.SPAN_RECORD_IO]
    assert flushes and all(s.parent_id == episode.span_id for s in flushes)


def test_timing_spans_recorded_with_taxonomy(world, tmp_path):
    """Under ``telemetry.bind`` an episode writes the span taxonomy to the harness file: the
    episode parents to the pass, and reset + policy.infer parent to the episode, with the episode carrying its
    index, step count, and virtual duration. Read back from the file so the OTLP encoding is exercised. The
    ``policy.infer`` span is recorded at the remote inference boundary, so the terminal is a ``RemoteStubPolicy``
    (a real wire round trip over a fake inference session)."""
    policy = ChunkPlayer().wrap(RemoteStubPolicy())
    harness = Harness(policy, make_embodiment())
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    # A latched observation set makes the inference of every step run, because the harness reads the latest
    # value. The world ends when the script runs out, so the script covers the rounds one inference takes:
    # one round starts the round trip, and a later round reads its answer.
    producer = ManualDriver(
        [(partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.0)] * 4
    )

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-taxonomy'), _eval_pass('run-taxonomy'):
        scheduler = world.start([harness, producer])
        p['perform_task'](Task(instruction_source='stack', timeout_sec=0.05, meta={keys.EVAL_TRIAL_INDEX: 0}))
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
    # Every round the harness calls the session is one step, so the count covers the round trips it made.
    infers = len(by_name[telemetry_keys.SPAN_POLICY_INFER])
    assert episode.attrs[telemetry_keys.ATTR_EPISODE_STEPS] >= infers > 0
    assert episode.attrs[telemetry_keys.ATTR_EPISODE_VIRTUAL_S] >= 0.0


@pytest.mark.timeout(10.0)
def test_an_inference_outliving_its_episode_parents_to_it(world, tmp_path):
    """A trial whose deadline lapses mid-call ends the episode while the model is still inside it, so the
    ``policy.infer`` span is recorded off the loop thread after the episode span closed. It still parents to
    the episode that asked for it rather than to the pass: charging wall time is the mode that measures real
    inference cost, so that span is the one a reader most wants attributed."""
    harness = Harness(
        ChunkPlayer().wrap(RemoteStubPolicy(wall_sec=0.3)),  # the call runs well past the deadline
        make_embodiment(simulated=True),
    )
    p = _pair_all(world, harness)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    producer = ManualDriver([
        (partial(emit_ready_payload, p['frame_em'], p['robot_em'], p['grip_em'], robot_state), 0.01),
        (None, 0.3),
    ])

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-outlive'), _eval_pass('run-outlive'):
        scheduler = world.start([harness, producer, _Pacer()])
        p['perform_task'](Task(instruction_source='stack', timeout_sec=0.05, charge_inference_time=True))
        drive_scheduler(scheduler, steps=2000)

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episodes = [s for s in spans if s.name == telemetry_keys.SPAN_EPISODE]
    infers = [s for s in spans if s.name == telemetry_keys.SPAN_POLICY_INFER]
    assert len(episodes) == 1 and len(infers) == 1
    assert infers[0].end_ns > episodes[0].end_ns  # the call did outlive the episode
    assert infers[0].parent_id == episodes[0].span_id


@pytest.mark.timeout(3.0)
def test_failed_pass_seals_open_episode_span(world, tmp_path):
    """A ``reset`` raising after the episode span was opened must seal that span before the
    provider flushes on exit. Ending it is what exports it at all: an unended span never leaves the batch
    processor, so its finished ``reset`` child orphans (unknown parent) and the report loses that phase and
    charges the episode's whole wall to ``between_episodes``. Sealed and marked ``episode.partial`` — with its
    step count and virtual duration stamped, like a clean end — the span exports parented to the (failed) pass,
    so the reduce keeps it and its phases attribute."""

    policy = StubPolicy()
    scene = pimm.calls.ControlSystemHandler[Any, None](Passive())
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment(prepare_handlers={keys.SCENE: scene}))
    wire_call(world, harness.prepare[keys.SCENE], scene)
    harness.ds_command._bind(RecordingEmitter())
    task = Task(
        instruction_source='stack', timeout_sec=10.0, prepare_args={keys.SCENE: {}}, meta={keys.EVAL_TRIAL_INDEX: 0}
    )
    _ask(world, harness, task)
    stop = SimpleNamespace(value=False)
    clock = _ManualClock()

    with pytest.raises(RuntimeError, match='reset boom'):
        with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-fail'), _eval_pass('run-fail'):
            for _ in harness.run(cast(pimm.SignalReceiver, stop), cast(pimm.Clock, clock)):
                for call in scene.incoming():
                    call.set_exception(RuntimeError('reset boom'))

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
def test_episode_virtual_duration_starts_when_the_rig_is_ready(world, tmp_path):
    """The rounds a rig spends readying itself advance the virtual clock without stepping the environment.
    The rollout's virtual duration measures from the end of the prepare, so that stretch stays reset work
    instead of inflating the real-time factor the report derives from it."""
    scene = _Scene(lambda _: None, draw_s=0.2)
    harness = Harness(
        ChunkPlayer().wrap(ChunkPolicy()), make_embodiment(prepare_handlers={keys.SCENE: scene.env_reset})
    )
    p = _pair_all(world, harness)
    wire_call(world, harness.prepare[keys.SCENE], scene.env_reset)
    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'run-anchor'), _eval_pass('run-anchor'):
        scheduler = world.start([harness, scene])
        p['perform_task'](Task(instruction_source='test', timeout_sec=None, prepare_args={keys.SCENE: {}}))
        draw_start = world.clock.now()
        for _ in range(1000):
            drive_scheduler(scheduler, steps=1)
            if DsWriterCommandType.START_EPISODE in _ds_types(p):
                break
        draw_s = world.clock.now() - draw_start
        emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], robot_state)
        drive_scheduler(scheduler, steps=10)  # two control systems per round, so a round is two steps
        p['done_em'].emit(OPERATOR_DONE)
        drive_scheduler(scheduler, steps=20)

    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episodes = [s for s in spans if s.name == telemetry_keys.SPAN_EPISODE]
    assert len(episodes) == 1
    virtual_s = episodes[0].attrs[telemetry_keys.ATTR_EPISODE_VIRTUAL_S]
    assert virtual_s > 0.0  # the rounds the policy ran for are measured
    # A handful of rollout rounds against the 0.2 the draw took: anchoring at the ask would swallow the draw
    # into the rollout's virtual duration.
    assert virtual_s < draw_s


@pytest.mark.parametrize(
    ('resume_in_sec', 'sleep_sec'),
    [(0.002, 0.002), (5.0, MAX_ROUND_SEC), (0.0002, MIN_ROUND_SEC), (-0.001, MIN_ROUND_SEC), (None, WAIT_PERIOD_SEC)],
    ids=['asked_for', 'over_the_ceiling', 'under_the_floor', 'already_passed', 'no_answer_yet'],
)
def test_a_real_rig_wakes_at_the_moment_the_session_asked_for(world, resume_in_sec, sleep_sec):
    """The harness sleeps to the instant the session named, inside its own floor and ceiling. A round the
    live session has not answered waits on the call the harness made."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    now_ns = world.clock.now_ns()
    harness._resume_at_ns = None if resume_in_sec is None else now_ns + int(resume_in_sec * 1e9)

    command = harness._pace(world.clock)

    assert isinstance(command, pimm.Sleep)
    assert command.seconds == pytest.approx(sleep_sec, abs=1e-4)


def test_a_real_rig_never_sleeps_past_the_deadline(world):
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    now_ns = world.clock.now_ns()
    harness._resume_at_ns = now_ns + 500_000_000
    harness._deadline_ns = now_ns + 20_000_000

    command = harness._pace(world.clock)

    assert isinstance(command, pimm.Sleep)
    assert command.seconds == pytest.approx(0.02, abs=1e-4)


@pytest.mark.parametrize(('expired', 'emitted'), [(True, False), (False, True)])
def test_a_command_is_emitted_only_while_the_trial_still_has_budget(world, expired, emitted):
    """A trial advertises the instant it stops at. A session call the world passes that instant during
    commands nothing, and ``_run`` finishes the trial on the next round."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    p = _pair_all(world, harness)
    scheduler = world.start([harness])
    p['perform_task'](Task(instruction_source='t', timeout_sec=None))
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], make_robot_state([0.1] * 3, [0.2] * 7))
    drive_scheduler(scheduler, steps=6)
    assert harness._inference is not None, 'the episode never opened'

    # Armed after the round's terminal check, which is the window a slow session call opens.
    harness._deadline_ns = world.clock.now_ns() + (-1_000_000_000 if expired else 1_000_000_000)
    p['command_rx'].read()  # drop what the opening round played
    harness._infer(harness._inference, world.clock, _NEVER_STOPS)

    message = p['command_rx'].read()
    assert (message is not None and message.updated) is emitted


def test_a_policy_that_answers_chunks_refuses_to_open(world):
    """Nothing turns a chunk into commands without a player, and an episode must not start on one."""
    harness = Harness(StubPolicy(), make_embodiment())
    p = _pair_all(world, harness)
    scheduler = world.start([harness])
    call = p['perform_task'](Task(instruction_source='t', timeout_sec=None))
    emit_ready_payload(p['frame_em'], p['robot_em'], p['grip_em'], make_robot_state([0.1] * 3, [0.2] * 7))

    with pytest.raises(NotImplementedError, match='put a ChunkPlayer above it'):
        drive_scheduler(scheduler, steps=20)
    assert call.done()


class _ReplanEarly(Layer):
    """Plays a chunk and asks the model again halfway through it.

    The re-query-before-exhaustion shape (RTC, temporal ensembling) that the substrate exists for: unlike
    ``ChunkPlayer`` it starts the next call while waypoints of the chunk it holds are still due.
    """

    class _Session(Session):
        POLL_SEC = 0.001

        def __init__(self, infer: Fn):
            self._infer = infer
            self._waypoints: deque[tuple[int, dict[str, Any]]] = deque()
            self._replan_at_ns: int | None = None
            self._answer: Answer | None = None

        def __call__(self, obs, time_ns):
            commands: dict[str, Any] = {}
            while self._waypoints and self._waypoints[0][0] <= time_ns:
                commands.update(self._waypoints.popleft()[1])
            if self._replan_at_ns is None or time_ns >= self._replan_at_ns:
                self._replan(obs, time_ns)
            next_waypoint_ns = self._waypoints[0][0] if self._waypoints else None
            due = [at_ns for at_ns in (self._replan_at_ns, next_waypoint_ns) if at_ns is not None and at_ns > time_ns]
            return commands, min(due, default=time_ns + int(self.POLL_SEC * 1e9))

        def _replan(self, obs, time_ns: int) -> None:
            if self._answer is None:
                self._answer = self._infer(obs)
            if not self._answer.done():  # the call it made has still to answer
                return
            chunk, self._answer = self._answer.result(), None
            anchor = time_ns / 1e9
            self._waypoints = deque(
                (int((anchor + action[keys.ACTION_TIMESTAMP]) * 1e9), {keys.TARGET_GRIP: action[keys.TARGET_GRIP]})
                for action in chunk
                if keys.TARGET_GRIP in action
            )
            end_ns = self._waypoints[-1][0] if self._waypoints else time_ns
            self._replan_at_ns = time_ns + (end_ns - time_ns) // 2

    class _Policy(DelegatingPolicy):
        def new_session(self, rt):
            return _ReplanEarly._Session(rt.fns[INFER])

    PLAYS_CHUNKS = True

    def wrap(self, policy: Policy) -> Policy:
        return _ReplanEarly._Policy(policy)


class _TimedRecorder(pimm.SignalEmitter):
    """Records each emission against the world clock, so a test can read when a command went out."""

    def __init__(self, clock: pimm.Clock):
        self._clock = clock
        self.emitted: list[tuple[float, Any]] = []

    def emit(self, data, ts: int = -1):
        self.emitted.append((self._clock.now(), data))


def _run_episode(
    world, policy, layer, *, charge_inference_time, simulated=True, steps=4000, run_sec=1.5
) -> list[tuple[float, Any]]:
    """One trial run with ``charge_inference_time``; returns the grip commands with the world time each went
    out at. A sim trial runs against a pacer, the sole time-master a real rig doesn't need."""
    harness = Harness(layer.wrap(policy), make_embodiment(simulated=simulated))
    grip_recorder = _TimedRecorder(world.clock)
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands[keys.TARGET_GRIP]._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (
            partial(
                perform_task,
                Task(instruction_source='t', timeout_sec=None, charge_inference_time=charge_inference_time),
            ),
            0.0,
        ),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.001),
        (None, run_sec),
    ])
    systems = [harness, driver, _Pacer()] if simulated else [harness, driver]
    drive_scheduler(world.start(systems), steps=steps)
    return grip_recorder.emitted


@pytest.mark.timeout(20.0)
def test_an_uncharged_call_pauses_the_world(world):
    """Sim's default charges nothing: the world does not advance while the model runs, so the chunk is
    anchored at the observation's own instant however long the function really took."""
    policy = RemoteStubPolicy(wall_sec=0.05, chunk=slow_chunk())
    played = _run_episode(world, policy, ChunkPlayer(), charge_inference_time=False)

    assert played, 'no command was played'
    assert played[0][0] < 0.05, f'the world paid for the function: first command at {played[0][0]}s'


@pytest.mark.timeout(20.0)
def test_a_charged_call_costs_its_own_wall_duration(world):
    """``charge_inference_time=True`` charges the world what the model really took, so a slow server is scored
    as slow — at the cost of a trace that inherits the machine's noise."""
    policy = RemoteStubPolicy(wall_sec=0.2, chunk=slow_chunk())
    played = _run_episode(world, policy, ChunkPlayer(), charge_inference_time=True)

    assert played, 'no command was played'
    assert played[0][0] >= 0.2, f'first command at {played[0][0]}s, under the 0.2s the function took'


@pytest.mark.timeout(20.0)
def test_a_real_rig_pays_wall_time_whatever_the_trial_asks_for(world):
    """The knob is sim-only: a real rig pays what its functions take, so a task leaving
    ``charge_inference_time`` unset does not hold the world for them."""
    policy = RemoteStubPolicy(wall_sec=0.2, chunk=slow_chunk())
    played = _run_episode(world, policy, ChunkPlayer(), charge_inference_time=False, simulated=False)

    assert played, 'no command was played'
    assert played[0][0] >= 0.2, f'first command at {played[0][0]}s, under the 0.2s the function took'


class _ObservedTicks(Layer):
    """Records the observation instant of every call that reaches it."""

    def __init__(self):
        self.seen: list[float] = []

    class _Session(DelegatingSession):
        def __init__(self, inner: Session, seen: list[float]):
            super().__init__(inner)
            self._seen = seen

        def __call__(self, obs, time_ns):
            self._seen.append(obs[keys.OBS_TIME_NS] / 1e9)
            return self._inner(obs, time_ns)

    def make_session(self, inner: Session):
        return _ObservedTicks._Session(inner, self.seen)


@pytest.mark.timeout(30.0)
def test_the_layers_see_every_tick(world):
    """What a temporal stack records: nothing keeps an observation from the layers above the scheduler —
    not the machine time inside the function, and not the rounds in between.

    The trial charges its inference, because an uncharged one holds the world and leaves no tick to miss.
    """
    ticks = _ObservedTicks()
    _run_episode(
        world,
        RemoteStubPolicy(wall_sec=0.01, chunk=slow_chunk(0.3, 15)),
        ticks | ChunkPlayer(),
        charge_inference_time=True,
        run_sec=1.0,
    )

    period = 0.005  # the pacer's control period
    gaps = [round(b - a, 4) for a, b in zip(ticks.seen, ticks.seen[1:], strict=False)]
    assert gaps and all(gap <= period for gap in gaps), gaps


@pytest.mark.timeout(20.0)
def test_harness_keeps_playing_while_a_call_is_in_flight(world):
    """A layer that replans before its chunk is exhausted leaves waypoints due during inference, and the
    harness emits them on time instead of standing still until the model answers."""
    played = _run_episode(
        world, RemoteStubPolicy(wall_sec=0.15, chunk=slow_chunk(0.4, 20)), _ReplanEarly(), charge_inference_time=True
    )

    # The first chunk lands at ~0.15s and spans 0.4s; the second call starts halfway through it (~0.28s) and
    # runs for 0.15s the world runs through; the waypoints due in that window have to keep going out.
    during = [t for t, _ in played if 0.29 <= t < 0.4]
    assert len(during) >= 3, f'the harness stopped playing during inference: {[t for t, _ in played]}'


@pytest.mark.timeout(3.0)
@pytest.mark.parametrize('unavailable', [RobotStatus.BUSY, RobotStatus.ERROR])
def test_an_unavailable_arm_reaches_the_policy_with_its_pose(world, unavailable):
    """An unavailable arm is not swallowed as "no observation": its measurements reach the policy beside the status.
    The harness here runs no ``StopOnFault``, so nothing filters it on the way."""
    policy = SpyPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment())
    p = _pair_all(world, harness)

    driver = ManualDriver([
        (partial(p['perform_task'], Task(instruction_source='test', timeout_sec=None)), 0.0),
        (
            partial(
                emit_ready_payload,
                p['frame_em'],
                p['robot_em'],
                p['grip_em'],
                make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=unavailable),
            ),
            0.01,
        ),
        (None, 0.02),
    ])

    drive_scheduler(world.start([harness, driver]), steps=40)

    assert policy.last_obs is not None, 'the status never reached the policy'
    assert policy.last_obs[keys.ROBOT_STATUS] is unavailable
    np.testing.assert_allclose(policy.last_obs[keys.JOINTS], [0.4, 0.5, 0.6])
    np.testing.assert_allclose(policy.last_obs[keys.EE_POSE][:3], [0.1, 0.2, 0.3])


@pytest.mark.timeout(3.0)
def test_every_arm_of_a_bimanual_rig_reports_its_own_status(world):
    """One arm busy and the other faulted reach the stack as themselves: neither channel stands in for
    the other, whichever the embodiment happens to list first."""
    left, right = f'{keys.ROBOT_STATE}.left', f'{keys.ROBOT_STATE}.right'
    embodiment = Embodiment(
        '',
        {
            left: Observation(pimm.ControlSystemEmitter(Passive()), Serializers.robot_state),
            right: Observation(pimm.ControlSystemEmitter(Passive()), Serializers.robot_state),
            keys.GRIP: Observation(pimm.ControlSystemEmitter(Passive()), None),
        },
        {keys.ROBOT_COMMAND: Command(pimm.ControlSystemReceiver(Passive()), Serializers.robot_command)},
        {},
        {},
        pimm.ControlSystemEmitter(Passive()),
    )
    policy = SpyPolicy()
    harness = Harness(ChunkPlayer().wrap(policy), embodiment)
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())
    left_em = world.pair(harness.observations[left])
    right_em = world.pair(harness.observations[right])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    def emit_states():
        left_em.emit(make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.BUSY))
        right_em.emit(make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], status=RobotStatus.ERROR))
        grip_em.emit(0.0)

    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='test', timeout_sec=None)), 0.0),
        (emit_states, 0.01),
        (None, 0.02),
    ])

    drive_scheduler(world.start([harness, driver]), steps=40)

    assert policy.last_obs is not None, 'neither arm reached the policy'
    assert policy.last_obs[f'{right}.status'] is RobotStatus.ERROR
    assert policy.last_obs[f'{left}.status'] is RobotStatus.BUSY


@pytest.mark.timeout(20.0)
def test_a_stop_clears_the_chunk_in_the_round_the_fault_is_seen(world):
    """A stop has no waypoints to place: an arm that faults mid-chunk stops in the round its fault is seen."""
    fault_at, period = 0.5, 0.005
    stack = StopOnFault() | ChunkPlayer()
    harness = Harness(stack.wrap(RemoteStubPolicy(chunk=slow_chunk(1.0, 50))), make_embodiment(simulated=True))
    grip_recorder = _TimedRecorder(world.clock)
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands[keys.TARGET_GRIP]._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    pose, joints = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]
    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, make_robot_state(pose, joints)), fault_at),
        (partial(robot_em.emit, make_robot_state(pose, joints, status=RobotStatus.ERROR)), 0.5),
    ])
    drive_scheduler(world.start([harness, driver, _Pacer(period)]), steps=4000)

    played = [t for t, _ in grip_recorder.emitted]
    assert [t for t in played if t < fault_at], 'the chunk was not playing when the arm faulted'
    late = [t for t in played if t > fault_at + 3 * period]
    assert not late, f'the faulted arm was still being driven at {late}'


@pytest.mark.timeout(20.0)
def test_finish_does_not_wait_for_the_call_in_flight():
    """Finishing ends the episode where it lands: the recording stops while the model is still inside its
    function, and the failure that function ends in reaches the log alone.

    Real time, real rig: a wall-charged trial is the one that leaves the loop running while a function is in
    flight.
    """
    hang_sec = 1.0

    class _HangingInfer(_FakeInferenceSession):
        def infer(self, obs):
            super().infer(obs)
            raise RuntimeError('inference boom')

    with pimm.World() as world:
        harness = Harness(ChunkPlayer().wrap(ServedPolicy(_HangingInfer([], hang_sec))), make_embodiment())
        cmd_recorder = RecordingEmitter()
        ds_recorder = _TimedRecorder(world.clock)
        harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
        harness.commands[keys.TARGET_GRIP]._bind(RecordingEmitter())
        harness.ds_command._bind(ds_recorder)

        frame_em = world.pair(harness.observations[CAM])
        robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
        grip_em = world.pair(harness.observations[keys.GRIP])
        perform_task = world.pair(harness.perform_task)
        done_em = world.pair(harness.done)

        robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        driver = ManualDriver([
            (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
            (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
            (partial(done_em.emit, OPERATOR_DONE), 0.05),
            (None, 0.05),
        ])
        started = world.clock.now()
        drive_scheduler(world.start([harness, driver]), steps=40)

    stops = [(t, data) for t, data in ds_recorder.emitted if data.type == DsWriterCommandType.STOP_EPISODE]
    assert len(stops) == 1
    assert stops[0][0] - started < hang_sec, 'the stop waited for the function to answer'


@pytest.mark.timeout(20.0)
def test_the_run_ends_only_once_the_call_it_abandoned_is_out_of_the_policy():
    """A sweep runs a harness per eval over one shared policy, so a function still inside the model when a run
    ends would meet the next run's ``new_session`` — or ``policy.close()``. The run outlives its own work."""
    hang_sec = 1.0
    left_the_model = threading.Event()

    class _HangingInfer(_FakeInferenceSession):
        def infer(self, obs):
            answer = super().infer(obs)
            left_the_model.set()
            return answer

    with pimm.World() as world:
        harness = Harness(ChunkPlayer().wrap(ServedPolicy(_HangingInfer([], hang_sec))), make_embodiment())
        harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
        harness.commands[keys.TARGET_GRIP]._bind(RecordingEmitter())
        harness.ds_command._bind(RecordingEmitter())

        frame_em = world.pair(harness.observations[CAM])
        robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
        grip_em = world.pair(harness.observations[keys.GRIP])
        perform_task = world.pair(harness.perform_task)
        done_em = world.pair(harness.done)

        robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        driver = ManualDriver([
            (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
            (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
            (partial(done_em.emit, OPERATOR_DONE), 0.05),
            (None, 0.05),
        ])
        drive_scheduler(world.start([harness, driver]), steps=40)

    assert left_the_model.is_set(), 'the run returned with a function still inside the shared policy'


@pytest.mark.timeout(20.0)
def test_the_session_is_closed_only_once_its_call_has_left_it():
    """Ending an episode shuts the websocket the function in flight is talking over, so the episode is
    retired with its runtime and released after it."""
    inside_at_close = []

    class _HangingInfer(_FakeInferenceSession):
        def __init__(self):
            super().__init__([], wall_sec=1.0)
            self.inside = False

        def infer(self, obs):
            self.inside = True
            try:
                return super().infer(obs)
            finally:
                self.inside = False

        def close(self):
            inside_at_close.append(self.inside)

    with pimm.World() as world:
        harness = Harness(ChunkPlayer().wrap(ServedPolicy(_HangingInfer())), make_embodiment())
        harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
        harness.commands[keys.TARGET_GRIP]._bind(RecordingEmitter())
        harness.ds_command._bind(RecordingEmitter())

        frame_em = world.pair(harness.observations[CAM])
        robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
        grip_em = world.pair(harness.observations[keys.GRIP])
        perform_task = world.pair(harness.perform_task)
        done_em = world.pair(harness.done)

        robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        driver = ManualDriver([
            (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
            (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.01),
            (partial(done_em.emit, OPERATOR_DONE), 0.05),
            (None, 0.05),
        ])
        drive_scheduler(world.start([harness, driver]), steps=40)

    assert inside_at_close == [False], 'the session was closed while its own function was still inside it'


@pytest.mark.timeout(3.0)
def test_a_rescheduled_trajectory_clears_the_channels_it_omits(world):
    """A trajectory naming only one channel replaces the whole schedule: the omitted channel stops being
    played rather than draining the previous trajectory's tail."""

    class _GripThenArmPolicy(Policy):
        """First a two-channel chunk, then an arm-only one that must silence the gripper."""

        def __init__(self):
            self._calls = 0

        @contextmanager
        def episode(self, context=None):
            yield {INFER: self._infer}

        def _infer(self, obs):
            self._calls += 1
            pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
            command = CartesianPosition(pose=pose)
            if self._calls == 1:
                return [
                    {keys.ROBOT_COMMAND: command, keys.TARGET_GRIP: 0.5, keys.ACTION_TIMESTAMP: i * 0.01}
                    for i in range(10)
                ]
            return [{keys.ROBOT_COMMAND: command, keys.ACTION_TIMESTAMP: i * 0.01} for i in range(10)]

    harness = Harness(ChunkPlayer().wrap(_GripThenArmPolicy()), make_embodiment())
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(RecordingEmitter())
    harness.commands[keys.TARGET_GRIP]._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.001),
        (None, 0.5),
    ])
    drive_scheduler(world.start([harness, driver]), steps=1000)

    grips = _emitted_grips(grip_recorder)
    assert set(grips) == {0.5}, f'the second chunk kept the gripper playing: {grips}'


@pytest.mark.timeout(3.0)
def test_manual_commands_are_emitted_as_plain_values(world):
    """An operator's command bypasses the schedule: it is the command, not a plan to play."""
    harness = Harness(ChunkPlayer().wrap(StubPolicy()), make_embodiment())
    cmd_recorder = RecordingEmitter()
    grip_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands[keys.TARGET_GRIP]._bind(grip_recorder)
    harness.ds_command._bind(RecordingEmitter())
    manual_em = world.pair(harness.manual_command)

    pose = Transform3D(translation=np.array([0.1, 0.1, 0.1], dtype=np.float32), rotation=Rotation.identity)
    manual = CartesianPosition(pose=pose)
    driver = ManualDriver([(partial(manual_em.emit, {keys.ROBOT_COMMAND: manual}), 0.01), (None, 0.02)])
    drive_scheduler(world.start([harness, driver]), steps=50)

    assert _emitted_commands(cmd_recorder) == [manual]
    assert not _emitted_grips(grip_recorder)


@pytest.mark.timeout(20.0)
def test_finishing_discards_a_call_that_is_still_in_flight(world):
    """Finishing while the model is still inside its call throws that answer away: the trajectory it carries
    never reaches the devices."""
    policy = RemoteStubPolicy(wall_sec=1.0, chunk=slow_chunk())
    harness = Harness(ChunkPlayer().wrap(policy), make_embodiment(simulated=True))
    cmd_recorder = RecordingEmitter()
    harness.commands[keys.ROBOT_COMMAND]._bind(cmd_recorder)
    harness.commands[keys.TARGET_GRIP]._bind(RecordingEmitter())
    harness.ds_command._bind(RecordingEmitter())

    frame_em = world.pair(harness.observations[CAM])
    robot_em = world.pair(harness.observations[keys.ROBOT_STATE])
    grip_em = world.pair(harness.observations[keys.GRIP])
    perform_task = world.pair(harness.perform_task)
    done_em = world.pair(harness.done)

    robot_state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    driver = ManualDriver([
        (partial(perform_task, Task(instruction_source='t', timeout_sec=None, charge_inference_time=True)), 0.0),
        (partial(emit_ready_payload, frame_em, robot_em, grip_em, robot_state), 0.001),
        (None, 0.05),  # well inside the 1.0s the function takes
        (partial(done_em.emit, OPERATOR_DONE), 0.0),
        (None, 0.05),
    ])
    drive_scheduler(world.start([harness, driver, _Pacer()]), steps=2000)

    assert not _emitted_commands(cmd_recorder)
