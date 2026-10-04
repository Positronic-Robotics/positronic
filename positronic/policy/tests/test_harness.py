"""Timing contracts for processor execution and simulated chunk playback."""

import operator
import threading
import time
from collections.abc import Mapping
from contextlib import closing, contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from positronic_wire import wire as session_wire

import pimm
from pimm.tests.testing import Passive, wire_call
from pimm.world import VirtualClock
from positronic import keys, telemetry, telemetry_keys, wire
from positronic.dataset.ds_writer_agent import DsWriterCommandType, TimeMode
from positronic.dataset.episode import Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.dataset.serializers import Serializers
from positronic.dataset.video import LibavEncoder
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.command import CartesianDelta, CartesianPosition, from_wire, to_wire
from positronic.drivers.roboarm.models import DEFAULT_FRAME, EE_LINK, bundled_franka_model
from positronic.drivers.roboarm.tests.fakes import make_robot_state
from positronic.eval import Command, Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.geom import Rotation, Transform3D
from positronic.policy import executor as executor_module
from positronic.policy import keys as policy_keys
from positronic.policy import remote as remote_module
from positronic.policy.base import ARGS, NAME, VERSION, NotAnswered, Policy, PolicyRun, Step
from positronic.policy.codec import Codec
from positronic.policy.executor import Executor, _UnchargedAnswer
from positronic.policy.harness import Harness, Rollout
from positronic.policy.journal import (
    PLAIN_DATA,
    Activity,
    ActivityFailed,
    Capture,
    Closing,
    CommandEmitted,
    Ended,
    Finished,
    Journal,
    Parent,
    PayloadCodec,
    PlainData,
    Primed,
    Published,
    Raised,
    ReplacedResult,
    Returned,
    StartFailed,
    Startup,
    StepReturned,
    Stopped,
    Submitted,
    TurnFailed,
    TurnStarted,
    Wake,
)
from positronic.policy.processors import ChunkedSchedule, PauseOnUnavailable
from positronic.policy.remote import RemotePolicy, round_trip
from positronic.policy.replay import (
    ExecutionRefused,
    MissingInput,
    MissingResult,
    ReplaceResult,
    ReplayDivergence,
    ReplayError,
    RerunActivity,
    Verified,
    branch,
    verify,
)
from positronic.policy.sequential import Sequential

MOTOR = 'motor'
POSITION = 'position'
RESET = 'reset'


class StubPolicy(Policy):
    """Emit a fixed command and retain observations for harness and environment integration tests."""

    def __init__(self, command=None, target_grip=0.33):
        self.command = (
            command
            if command is not None
            else CartesianPosition(pose=Transform3D(translation=np.array([0.4, 0.5, 0.6]), rotation=Rotation.identity))
        )
        self.target_grip = target_grip
        self.observations = []

    def run(self, runtime) -> PolicyRun:
        obs = yield
        while True:
            self.observations.append(obs)
            obs = yield Step(
                {keys.ROBOT_COMMAND: self.command, keys.TARGET_GRIP: self.target_grip}, runtime.time_ns + 100_000_000
            )


class RemoteStubPolicy(StubPolicy):
    """Exercise round-trip telemetry without a network or model dependency."""

    def run(self, runtime):
        session = Mock(
            infer=Mock(return_value=[{keys.ROBOT_COMMAND: self.command, keys.TARGET_GRIP: self.target_grip}]),
            served_timing={},
        )
        return ChunkedSchedule(fps=10).run(
            runtime, lambda obs: cast(list[dict[str, Any]], round_trip(session, obs, compress_images=False))
        )


class Motion(pimm.ControlSystem):
    """Integrate the commanded velocity once per two-millisecond physics step."""

    def __init__(self):
        self.command = pimm.ControlSystemReceiver[int](self)
        self.position = pimm.ControlSystemEmitter[int](self)
        self.reset = pimm.calls.ControlSystemHandler[Any, None](self)
        self.resets = 0
        self.positions = []

    def run(self, should_stop, clock):
        position, velocity = 0, 0
        while not should_stop.value:
            yield pimm.Sleep(0.002)
            for call in self.reset.incoming():
                self.resets += 1
                yield pimm.Sleep(0.004)
                position = 0
                call.set_result(None)
            if (command := pimm.read_updated(self.command)) is not None:
                velocity = command.data
            position += velocity
            self.positions.append((clock.now_ns(), position))
            self.position.emit(position)


class Trace(pimm.SignalEmitter):
    def __init__(self, clock, forward=None):
        self.clock = clock
        self.forward = forward
        self.values = []

    def emit(self, data, ts=-1):
        self.values.append((self.clock.now_ns(), data))
        if self.forward is not None:
            self.forward.emit(data, ts)


@pytest.fixture
def observed_harness():
    calls = []

    class Observe(Policy):
        def run(self, runtime):
            obs = yield
            while True:
                calls.append((obs, runtime.time_ns, runtime.tick))
                obs = yield Step({}, runtime.time_ns + 100_000_000)

    serializers = {name: Mock(side_effect=lambda value: value) for name in (keys.WRIST_IMAGE, POSITION)}
    with pimm.World(virtual_time=True) as world:
        source = Passive()
        embodiment = Embodiment(
            descriptor='test',
            observations={
                name: Observation(pimm.ControlSystemEmitter(source), serializer)
                for name, serializer in serializers.items()
            },
            commands={},
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
        emitters = {}
        for name, receiver in harness.observations.items():
            emitters[name], physical_receiver = world.local_pipe()
            receiver._bind(physical_receiver)
        harness.ds_command._bind(Trace(world.clock))
        harness.deadline_ns._bind(Trace(world.clock))
        runtime = Executor(world.clock.now_ns, simulated=True, charge_inference_time=False)
        policy_run = runtime.start(Observe())
        step = partial(harness._step, Task('test', None), runtime, policy_run, None)
        try:
            yield world, harness, emitters, serializers, calls, step
        finally:
            runtime.close(policy_run)


def test_observations_refresh_on_signal_updates_independently_of_time(observed_harness):
    world, _, emitters, serializers, calls, step = observed_harness
    frame = np.array([1, 2])
    emitters[keys.WRIST_IMAGE].emit(frame)
    assert step() is None
    assert calls == []

    emitters[POSITION].emit(3)
    step()
    first, now, tick = calls[-1]
    assert now == tick == 0
    assert set(first) == {keys.WRIST_IMAGE, POSITION, keys.TASK, keys.DESCRIPTOR}
    assert first[keys.TASK] == 'test'
    assert first[keys.DESCRIPTOR] == 'test'
    np.testing.assert_array_equal(first[keys.WRIST_IMAGE], [1, 2])
    assert all(serializer.call_count == 1 for serializer in serializers.values())

    step()
    assert calls[-1][0][keys.WRIST_IMAGE] is first[keys.WRIST_IMAGE]
    assert all(serializer.call_count == 1 for serializer in serializers.values())

    frame[:] = [4, 5]
    emitters[keys.WRIST_IMAGE].emit(frame)
    step()
    current, now, tick = calls[-1]
    assert now == tick == 0
    assert current[POSITION] == 3
    np.testing.assert_array_equal(current[keys.WRIST_IMAGE], [4, 5])
    np.testing.assert_array_equal(first[keys.WRIST_IMAGE], [1, 2])
    assert serializers[keys.WRIST_IMAGE].call_count == 2
    assert serializers[POSITION].call_count == 1

    cast(VirtualClock, world.clock).advance_to_ns(1_000_000)
    step()
    assert calls[-1][1:] == (1_000_000, 1)
    assert calls[-1][0][keys.WRIST_IMAGE] is current[keys.WRIST_IMAGE]
    assert serializers[keys.WRIST_IMAGE].call_count == 2
    assert serializers[POSITION].call_count == 1


def test_observation_cache_initializes_from_already_read_signals(observed_harness):
    _, harness, emitters, serializers, calls, step = observed_harness
    for name, emitter in emitters.items():
        emitter.emit(1)
        harness.observations[name].read()
    step()
    assert len(calls) == 1
    assert all(serializer.call_count == 1 for serializer in serializers.values())


def test_updated_observations_remove_fields_the_serializer_no_longer_returns(observed_harness):
    _, _, emitters, _, calls, step = observed_harness
    emitters[keys.WRIST_IMAGE].emit(np.array([1]))
    emitters[POSITION].emit({'': 2, '.extra': 3})
    step()
    first = calls[-1][0]
    assert first[POSITION + '.extra'] == 3

    emitters[POSITION].emit({'': 4, '.extra': None})
    step()
    assert calls[-1][0][POSITION] == 4
    assert POSITION + '.extra' not in calls[-1][0]
    assert first[POSITION] == 2
    assert first[POSITION + '.extra'] == 3

    emitters[POSITION].emit(None)
    step()
    assert POSITION not in calls[-1][0]
    assert calls[-1][0][keys.WRIST_IMAGE] is first[keys.WRIST_IMAGE]


def test_observation_conversion_errors_propagate(observed_harness):
    _, _, emitters, serializers, calls, step = observed_harness
    emitters[keys.WRIST_IMAGE].emit(np.array([1]))
    emitters[POSITION].emit(1)
    step()
    serializers[POSITION].side_effect = pimm.NoValueException('conversion failed')
    emitters[POSITION].emit(2)
    with pytest.raises(pimm.NoValueException, match='conversion failed'):
        step()
    assert len(calls) == 1


@pytest.fixture
def episode_harness():
    with pimm.World(virtual_time=True) as world:
        source = Passive()
        serializer = Mock(side_effect=lambda value: value)
        preparation = {name: pimm.calls.ControlSystemHandler(source) for name in (RESET, eval_keys.SCENE)}
        embodiment = Embodiment(
            descriptor='test',
            observations={POSITION: Observation(pimm.ControlSystemEmitter(source), serializer)},
            commands={MOTOR: Command(pimm.ControlSystemReceiver(source), None)},
            prepare_handlers=preparation,
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
        for name, handler in preparation.items():
            wire_call(world, harness.prepare[name], handler)
        caller = pimm.calls.ControlSystemCaller[Rollout, dict[str, Any]](source)
        wire_call(world, caller, harness.perform_task)
        emitters = []
        for receiver in (harness.manual_command, harness.done, harness.observations[POSITION]):
            emitter, physical_receiver = world.local_pipe()
            receiver._bind(physical_receiver)
            emitters.append(emitter)
        manual, done, observation = emitters
        ports = SimpleNamespace(
            world=world,
            harness=harness,
            embodiment=embodiment,
            prepare=preparation,
            loop=harness.run(world.should_stop_reader(), world.clock),
            caller=caller,
            manual=manual,
            done=done,
            observation=observation,
            serializer=serializer,
            records=Trace(world.clock),
            deadlines=Trace(world.clock),
            commands=Trace(world.clock),
        )
        harness.ds_command._bind(ports.records)
        harness.deadline_ns._bind(ports.deadlines)
        harness.commands[MOTOR]._bind(ports.commands)
        try:
            yield ports
        finally:
            world.request_stop()
            list(ports.loop)


def test_episode_completion_then_shutdown_with_fresh_observations(episode_harness):
    h = episode_harness
    observations, closed = [], []

    class Observe(Policy):
        def run(self, runtime):
            try:
                obs = yield
                while True:
                    observations.append(obs)
                    obs = yield Step({MOTOR: int(obs[POSITION][0])}, runtime.time_ns + 100_000_000)
            finally:
                closed.append(True)

    frame = np.array([1])
    h.observation.emit(frame)
    h.done.emit({'stale': True})
    first = h.caller(Rollout(Task('first', 0.01), Observe(), None))
    next(h.loop)
    assert h.deadlines.values == [(0, 10_000_000)]

    h.manual.emit({MOTOR: 99})
    overlapping = h.caller(Rollout(Task('overlapping', None), Observe(), None))
    h.done.emit({'success': True}, ts=5_000_000)
    h.world.clock.advance_to_ns(12_000_000)
    next(h.loop)
    assert not first.done()  # The recorder gets a turn before the caller is answered.
    assert closed == [True]
    assert h.deadlines.values[-1] == (12_000_000, None)
    with pytest.raises(RuntimeError, match='already running'):
        overlapping.result()
    next(h.loop)
    assert first.result() == {'success': True, eval_keys.TERMINATED: True}
    assert h.records.values[-1][1].static_data[keys.TASK] == 'first'

    frame[0] = 2  # No new signal: the next episode must rebuild its observation cache.
    second = h.caller(Rollout(Task('second', None), Observe(), None))
    next(h.loop)
    assert h.serializer.call_count == 2
    assert [obs[POSITION][0] for obs in observations] == [1, 2]
    assert h.commands.values == [(0, 1), (12_000_000, 2)]
    assert not second.done()

    h.world.request_stop()
    list(h.loop)
    with pytest.raises(pimm.calls.HandlerStopped):
        second.result()
    assert closed == [True, True]
    assert h.deadlines.values[-1][1] is None
    assert [command.type for _, command in h.records.values] == [
        DsWriterCommandType.START_EPISODE,
        DsWriterCommandType.STOP_EPISODE,
        DsWriterCommandType.START_EPISODE,
        DsWriterCommandType.STOP_EPISODE,
    ]


@pytest.mark.parametrize('done_at_ns, terminated', [(10_000_000, True), (11_000_000, False)])
def test_episode_deadline_uses_the_done_signal_timestamp(episode_harness, done_at_ns, terminated):
    h = episode_harness

    class Wait(Policy):
        def run(self, runtime):
            yield
            while True:
                yield Step({}, runtime.time_ns + 100_000_000)

    h.observation.emit(1)
    answer = h.caller(Rollout(Task('test', 0.01), Wait(), None))
    next(h.loop)
    h.done.emit({'success': True}, ts=done_at_ns)
    h.world.clock.advance_to_ns(12_000_000)
    next(h.loop)
    next(h.loop)
    assert answer.result()[eval_keys.TERMINATED] is terminated


@pytest.mark.parametrize('failure', ['startup', 'policy', 'conversion'])
def test_episode_failures_close_the_generator_and_answer_the_caller(episode_harness, failure):
    h = episode_harness
    closed = []

    class Failing(Policy):
        def run(self, runtime):
            try:
                if failure == 'startup':
                    raise ValueError('startup failed')
                yield
                raise ValueError('policy failed')
            finally:
                closed.append(True)

    h.observation.emit(1)
    error = ValueError
    if failure == 'conversion':
        error = pimm.NoValueException
        h.serializer.side_effect = error('conversion failed')
    answer = h.caller(Rollout(Task('test', None), Failing(), None))
    with pytest.raises(error, match=f'{failure} failed'):
        next(h.loop)
    assert closed == [True]
    with pytest.raises(pimm.calls.HandlerStopped):
        answer.result()


@contextmanager
def policy_world(policy, *, simulated=True, charged=False):
    """Drive the harness and its worker threads against a deterministic one-millisecond world clock."""
    with pimm.World(virtual_time=True) as world:
        timer = Passive()
        embodiment = Embodiment(
            descriptor='test',
            observations={},
            commands={MOTOR: Command(pimm.ControlSystemReceiver(timer), None)},
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=simulated,
        )
        harness = Harness(embodiment)
        caller = world.pair(harness.perform_task)
        world.pair(harness.manual_command)
        world.pair(harness.done)
        commands = Trace(world.clock)
        harness.commands[MOTOR]._bind(commands)
        harness.ds_command._bind(Trace(world.clock))
        harness.deadline_ns._bind(Trace(world.clock))
        loop = world.start([harness, timer])
        caller(Rollout(Task('test', None, charge_inference_time=charged), policy, None))
        try:
            yield world, loop, commands
        finally:
            world.request_stop()
            list(loop)


def test_completion_chains_reenter_the_stack_without_advancing_time():
    calls = []
    results = []

    class Outer(Policy):
        def run(self, runtime, inner):
            obs = yield
            while True:
                calls.append((obs, runtime.time_ns, runtime.tick))
                obs = yield inner.send(obs)

    class Chain(Policy):
        def run(self, runtime):
            yield
            for i in range(50):
                answer = runtime.submit(lambda value=i: (time.sleep(0.001), value)[1])
                while not answer.done():
                    yield Step({}, runtime.time_ns + 1_000_000_000)
                results.append(answer.result())
            yield Step({MOTOR: len(results)}, runtime.time_ns + 1_000_000_000)
            while True:
                yield Step({}, runtime.time_ns + 1_000_000_000)

    with policy_world(Sequential(Outer(), Chain())) as (world, loop, commands):
        next(loop)
        assert results == list(range(50))
        assert commands.values == [(0, 50)]
        assert len(calls) >= 51
        assert all(obs == calls[0][0] and now == tick == 0 for obs, now, tick in calls)
        assert world.clock.now_ns() == 1_000_000


def test_an_unread_completion_does_not_wake_again_on_later_ticks():
    calls = []

    class IgnoreResult(Policy):
        def run(self, runtime):
            yield
            runtime.submit(lambda: 42)
            while True:
                calls.append(runtime.time_ns)
                yield Step({}, runtime.time_ns + 10_000_000)

    with policy_world(IgnoreResult()) as (world, loop, _):
        while world.clock.now_ns() <= 11_000_000:
            next(loop)
        assert calls == [0, 0, 10_000_000]


def test_real_completion_wakes_before_the_policy_deadline():
    release = threading.Event()
    answers = []
    calls = []

    class Waiting(Policy):
        def run(self, runtime):
            yield
            answer = runtime.submit(lambda: (release.wait(timeout=2), 42)[1])
            answers.append(answer)
            calls.append(runtime.time_ns)
            yield Step({}, runtime.time_ns + 1_000_000_000)
            calls.append(runtime.time_ns)
            yield Step({MOTOR: answer.result()}, runtime.time_ns + 1_000_000_000)
            while True:
                yield Step({}, runtime.time_ns + 1_000_000_000)

    with policy_world(Waiting(), simulated=False) as (world, loop, commands):
        try:
            next(loop)
            assert calls == [0]
            release.set()
            assert cast(_UnchargedAnswer[int], answers[0]).call.result(timeout=1) == 42
            while world.clock.now_ns() <= 6_000_000:
                next(loop)
            assert calls == [0, 5_000_000]
            assert commands.values == [(5_000_000, 42)]
        finally:
            release.set()


def test_charged_completion_waits_for_its_simulated_time(monkeypatch):
    wall = [0]
    monkeypatch.setattr(executor_module.time, 'monotonic_ns', lambda: wall[0])
    answers = []
    calls = []

    def infer():
        wall[0] = 10_000_000
        return 42

    class Waiting(Policy):
        def run(self, runtime):
            yield
            answer = runtime.submit(infer)
            answers.append(answer)
            calls.append(runtime.time_ns)
            yield Step({}, runtime.time_ns + 1_000_000_000)
            calls.append(runtime.time_ns)
            yield Step({MOTOR: answer.result()}, runtime.time_ns + 1_000_000_000)
            while True:
                yield Step({}, runtime.time_ns + 1_000_000_000)

    with policy_world(Waiting(), charged=True) as (world, loop, commands):
        next(loop)
        assert cast(_UnchargedAnswer[int], answers[0]).call.result(timeout=1) == 42
        while world.clock.now_ns() < 10_000_000:
            next(loop)
        assert calls == [0]
        next(loop)
        assert calls == [0, 10_000_000]
        assert commands.values == [(10_000_000, 42)]


def test_uncharged_failures_reach_the_policy_at_the_same_time():
    calls = []

    def infer():
        raise ValueError('model failed')

    class Waiting(Policy):
        def run(self, runtime):
            yield
            answer = runtime.submit(infer)
            calls.append(runtime.time_ns)
            yield Step({}, runtime.time_ns + 1_000_000_000)
            calls.append(runtime.time_ns)
            answer.result()

    with policy_world(Waiting()) as (_, loop, _):
        with pytest.raises(ValueError, match='model failed'):
            next(loop)
        assert calls == [0, 0]


@pytest.mark.parametrize('delay', [0.0, 0.003, 0.025])
@pytest.mark.parametrize('prepare', [False, True])
def test_simulated_act_cadence_and_uncharged_boundaries(delay, prepare):
    requests = []

    class CompletePolicy(Policy):
        def run(self, runtime):
            def infer(obs):
                requests.append((runtime.time_ns, obs[POSITION]))
                time.sleep(delay)
                return [{MOTOR: i} for i in range(1, 51)]

            return Sequential(PauseOnUnavailable(), ChunkedSchedule(fps=15, horizon_sec=1.0)).run(runtime, infer)

    definition = CompletePolicy()
    with pimm.World(virtual_time=True) as world:
        motion = Motion()
        embodiment = Embodiment(
            descriptor='test',
            observations={POSITION: Observation(motion.position, None)},
            commands={MOTOR: Command(motion.command, None)},
            prepare_handlers={RESET: motion.reset},
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
        caller = world.pair(harness.perform_task)
        world.pair(harness.manual_command)
        world.pair(harness.done)
        observations = world.pair(harness.observations[POSITION])
        motion.position._bind(observations)
        commands = Trace(world.clock, world.pair(motion.command))
        harness.commands[MOTOR]._bind(commands)
        records = Trace(world.clock)
        harness.ds_command._bind(records)
        harness.deadline_ns._bind(Trace(world.clock))
        world.connect(harness.prepare[RESET], motion.reset)
        loop = world.start([harness, motion])
        observations.emit(0)
        answer = caller(
            Rollout(
                Task('move', 2.0, prepare_args={RESET: None} if prepare else {}, charge_inference_time=False),
                definition,
                None,
            )
        )
        try:
            for _ in range(1100):
                next(loop)
                if answer.done():
                    break
            assert answer.done()
            assert motion.resets == (2 if prepare else 0)
            start_ns = records.values[0][0]
            assert start_ns == (8_000_000 if prepare else 0)
            fixture = Path(__file__).resolve().parents[3] / 'integration_tests/fixtures/act_stack/seed_4.npz'
            with np.load(fixture, allow_pickle=False) as reference:
                chunk_ns = reference[keys.TARGET_GRIP + '.time_ns'][:15]
            expected_ns = np.concatenate((chunk_ns, chunk_ns + 1_000_000_000))
            assert [timestamp - start_ns for timestamp, _ in commands.values] == expected_ns.tolist()
            assert [command for _, command in commands.values] == list(range(1, 16)) * 2
            assert [timestamp - start_ns for timestamp, _ in requests] == [0, 1_000_000_000]
            boundary_ns = start_ns + 1_000_000_000
            before_boundary = [position for ns, position in motion.positions if ns < boundary_ns]
            # Inference reads the last completed physics step, then its first command applies immediately.
            assert requests[1][1] == before_boundary[-1]
            assert dict(motion.positions)[boundary_ns] == requests[1][1] + commands.values[15][1]
        finally:
            world.request_stop()
            list(loop)


class Hold(Policy):
    def run(self, runtime):
        yield
        while True:
            yield Step({}, runtime.time_ns + 100_000_000)

    def meta(self):
        return {'config': {'name': 'hold'}}


@pytest.mark.parametrize('record', [False, True])
def test_recording_path_and_final_metadata(episode_harness, tmp_path, record):
    h = episode_harness
    h.embodiment.static_meta['rig'] = 'test-rig'
    output = tmp_path if record else None
    task = Task('move', None, meta={'seed': 42})
    answer = h.caller(Rollout(task, Hold(), output))
    next(h.loop)
    assert h.records.values[0][1].output_path == output
    h.done.emit({eval_keys.SUCCESS: True})
    next(h.loop)
    next(h.loop)
    meta = h.records.values[-1][1].static_data
    assert meta['rig'] == 'test-rig'
    assert meta['seed'] == 42
    assert meta['inference.policy.config.name'] == 'hold'
    assert meta[keys.TASK] == 'move'
    assert meta[eval_keys.UNIVERSE] == 'sim'
    assert eval_keys.TIMEOUT not in meta
    assert meta[eval_keys.SUCCESS] is True
    assert answer.result()[eval_keys.TERMINATED] is True


def test_run_metadata_overrides_definition_and_is_snapshotted_before_cleanup(episode_harness, tmp_path):
    class Record(Policy):
        def meta(self):
            return {'config.name': 'record', 'config.status': 'initial'}

        def run(self, runtime):
            events = runtime.metadata.setdefault('events', [])
            runtime.metadata['config'] = {'status': 'active'}
            try:
                yield
                events.append('started')
                while True:
                    yield Step({}, runtime.time_ns + 100_000_000)
            finally:
                events.append('closed')

    h = episode_harness
    policy = Record()
    h.observation.emit(0)
    for _ in range(2):
        answer = h.caller(Rollout(Task('move', None), policy, tmp_path))
        next(h.loop)
        h.done.emit({eval_keys.SUCCESS: True})
        next(h.loop)
        next(h.loop)
        assert answer.done()
        meta = h.records.values[-1][1].static_data
        assert meta['inference.policy.config.name'] == 'record'
        assert meta['inference.policy.config.status'] == 'active'
        assert meta['inference.policy.events'] == ['started']


def test_preparation_precedes_budget_and_return_skips_scene(episode_harness):
    h = episode_harness
    task = Task('move', 0.01, prepare_args={RESET: 'home', eval_keys.SCENE: 42})
    answer = h.caller(Rollout(task, Hold(), None))
    next(h.loop)
    assert h.deadlines.values == h.records.values == []
    for name, request in task.prepare_args.items():
        call = next(h.prepare[name].incoming())
        assert call.request == request
        call.set_result(None)
    h.world.clock.advance_to_ns(1_000_000_000)
    next(h.loop)
    assert h.deadlines.values == [(1_000_000_000, 1_010_000_000)]
    h.world.clock.advance_to_ns(1_010_000_000)
    next(h.loop)
    next(h.loop)
    assert not answer.done()
    back = next(h.prepare[RESET].incoming())
    assert back.request == 'home'
    assert list(h.prepare[eval_keys.SCENE].incoming()) == []
    back.set_result(None)
    next(h.loop)
    assert answer.result() == {eval_keys.TERMINATED: False}


@pytest.mark.parametrize('failure', ['unknown', 'prepare', 'return'])
def test_preparation_errors_and_failed_return(episode_harness, failure, caplog):
    h = episode_harness
    name = 'unknown' if failure == 'unknown' else RESET
    answer = h.caller(Rollout(Task('move', 0.01, prepare_args={name: None}), Hold(), None))
    if failure == 'unknown':
        with pytest.raises(ValueError, match='unknown'):
            next(h.loop)
    else:
        next(h.loop)
        call = next(h.prepare[RESET].incoming())
        if failure == 'prepare':
            call.set_exception(ValueError('prepare failed'))
            with pytest.raises(ValueError, match='prepare failed'):
                next(h.loop)
        else:
            call.set_result(None)
            next(h.loop)
            h.done.emit({eval_keys.SUCCESS: True})
            next(h.loop)
            next(h.loop)
            back = next(h.prepare[RESET].incoming())
            back.set_exception(ValueError('return failed'))
            next(h.loop)
            assert answer.result()[eval_keys.SUCCESS] is True
            assert 'return failed' in caplog.text
            return
    with pytest.raises(pimm.calls.HandlerStopped):
        answer.result()
    assert h.records.values == []


def test_idle_manual_commands_pass_through(episode_harness):
    h = episode_harness
    h.manual.emit({MOTOR: 3})
    next(h.loop)
    assert h.commands.values == [(0, 3)]


@pytest.mark.parametrize('ending', ['done', 'shutdown', 'failure'])
def test_episode_spans_include_reset_and_recorder_flush(episode_harness, tmp_path, ending):
    h = episode_harness
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'test-episode'):
        h.caller(Rollout(Task('move', None), Hold(), None))
        next(h.loop)
        h.world.clock.advance_to_ns(100_000_000)
        if ending == 'failure':
            h.observation.emit(1)
            h.serializer.side_effect = ValueError('failed')
            with pytest.raises(ValueError, match='failed'):
                next(h.loop)
        else:
            if ending == 'shutdown':
                h.world.request_stop()
            else:
                h.done.emit({eval_keys.SUCCESS: True})
            next(h.loop)
            with telemetry.span(telemetry_keys.SPAN_RECORD_IO):
                pass
            h.world.request_stop()
            list(h.loop)
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episode = next(s for s in spans if s.name == telemetry_keys.SPAN_EPISODE)
    reset = next(s for s in spans if s.name == telemetry_keys.SPAN_RESET)
    assert reset.parent_id == episode.span_id
    assert episode.attrs[telemetry_keys.ATTR_EPISODE_VIRTUAL_S] == pytest.approx(0.1)
    if ending == 'failure':
        assert episode.attrs[telemetry_keys.ATTR_EPISODE_PARTIAL] is True
    else:
        flush = next(s for s in spans if s.name == telemetry_keys.SPAN_RECORD_IO)
        assert flush.parent_id == episode.span_id


def test_step_spans_carry_the_step_durations_and_parent_the_policy(episode_harness, tmp_path):
    h = episode_harness

    class EveryTenMs(Policy):
        def run(self, runtime):
            yield
            while True:
                yield Step({MOTOR: 1}, runtime.time_ns + 10_000_000)

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'test-steps'):
        h.observation.emit(1)
        h.caller(Rollout(Task('move', None), EveryTenMs(), None))
        next(h.loop)
        h.world.clock.advance_to_ns(13_000_000)
        next(h.loop)
        h.world.request_stop()
        list(h.loop)
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    episode = next(s for s in spans if s.name == telemetry_keys.SPAN_EPISODE)
    first, second = sorted((s for s in spans if s.name == telemetry_keys.SPAN_HARNESS_STEP), key=lambda s: s.start_ns)
    read_key = telemetry_keys.ATTR_STEP_READ_MS_PREFIX + POSITION
    convert_key = telemetry_keys.ATTR_STEP_CONVERT_MS_PREFIX + POSITION
    durations = (
        telemetry_keys.ATTR_STEP_OBSERVE_MS,
        telemetry_keys.ATTR_STEP_POLICY_MS,
        telemetry_keys.ATTR_STEP_EMIT_MS,
    )
    for step in (first, second):
        assert step.parent_id == episode.span_id
        assert all(step.attrs[key] >= 0 for key in (*durations, read_key))
    assert telemetry_keys.ATTR_STEP_LATE_MS not in first.attrs
    assert second.attrs[telemetry_keys.ATTR_STEP_LATE_MS] == pytest.approx(3.0)
    assert convert_key in first.attrs and convert_key not in second.attrs
    policy_spans = [s for s in spans if s.name == telemetry.component_name(EveryTenMs())]
    assert {s.parent_id for s in policy_spans} == {first.span_id, second.span_id}


def test_a_step_without_an_observation_records_only_the_observe_values(episode_harness, tmp_path):
    h = episode_harness
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'test-missing-observation'):
        h.caller(Rollout(Task('move', None), Hold(), None))
        next(h.loop)
        h.world.request_stop()
        list(h.loop)
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    steps = [s for s in spans if s.name == telemetry_keys.SPAN_HARNESS_STEP]
    assert steps
    for step in steps:
        assert telemetry_keys.ATTR_STEP_OBSERVE_MS in step.attrs
        assert telemetry_keys.ATTR_STEP_POLICY_MS not in step.attrs
        assert telemetry_keys.ATTR_STEP_EMIT_MS not in step.attrs


def test_shutdown_drains_work_before_closing_policy_resources(episode_harness):
    h = episode_harness
    started, release = threading.Event(), threading.Event()
    order = []

    class Pending(Policy):
        def run(self, runtime):
            def infer():
                started.set()
                assert release.wait(timeout=2)
                order.append('inference finished')

            try:
                yield
                runtime.submit(infer)
                while True:
                    yield Step({}, runtime.time_ns + 1_000_000_000)
            finally:
                order.append('policy closed')

    # A real rig can end its episode while a worker is still answering.
    h.embodiment.simulated = False
    h.observation.emit(1)
    answer = h.caller(Rollout(Task('move', None), Pending(), None))
    next(h.loop)
    assert started.wait(timeout=1)
    h.world.request_stop()
    releaser = threading.Timer(0.01, release.set)
    releaser.start()
    try:
        list(h.loop)
    finally:
        release.set()
        releaser.join()
    assert order == ['inference finished', 'policy closed']
    with pytest.raises(pimm.calls.HandlerStopped):
        answer.result()


def test_rollout_records_commands_and_the_state_they_produce(tmp_path):
    class Move(Policy):
        def run(self, runtime):
            return ChunkedSchedule(fps=10).run(runtime, lambda obs: [{MOTOR: 1}, {MOTOR: 2}])

    with pimm.World(virtual_time=True) as world:
        motion = Motion()
        embodiment = Embodiment(
            descriptor='recording-test',
            observations={POSITION: Observation(motion.position, None)},
            commands={MOTOR: Command(motion.command, None)},
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
        recorder = wire.wire_embodiment(world, harness, embodiment, TimeMode.MESSAGE)
        assert recorder is not None
        world.connect(harness.ds_command, recorder.command)
        caller = world.pair(harness.perform_task)
        loop = world.start([harness, motion, recorder])
        answer = caller(Rollout(Task('move', 0.21, charge_inference_time=False), Move(), tmp_path))
        try:
            for _ in range(1000):
                next(loop)
                if answer.done():
                    break
            assert answer.result() == {eval_keys.TERMINATED: False}
        finally:
            world.request_stop()
            list(loop)
    episode = LocalDataset(tmp_path)[0]
    assert isinstance(episode, Episode)
    commands = episode[MOTOR]
    assert list(commands.values()) == [1, 2, 1]
    np.testing.assert_array_equal(np.diff(list(commands.keys())), [100_000_000, 100_000_000])
    positions = episode[POSITION]
    recorded = dict(zip(positions.keys(), positions.values(), strict=True))
    assert recorded
    assert all(recorded[ns] == value for ns, value in motion.positions if ns in recorded)
    assert 1 in np.diff(list(positions.values()))
    assert 2 in np.diff(list(positions.values()))
    schedule = f'{policy_keys.POLICY_META}.{eval_keys.SCHEDULE}'
    assert episode.static[f'{schedule}.{eval_keys.SCHEDULED}'] == 4
    assert episode.static[f'{schedule}.{eval_keys.EMITTED}'] == 3
    assert episode.static[f'{schedule}.{eval_keys.DROPPED}'] == 0


class JournaledMove(Policy):
    """Play the chunks that ``infer`` returns at 10 Hz."""

    def __init__(self, infer):
        self.infer = Activity('step_plan', 1, infer)

    def run(self, runtime):
        return ChunkedSchedule(fps=10).run(runtime, self.infer)


class Feedback(Policy):
    """Ask ``infer`` for a velocity from the position and the previous velocity, every ``period_ns``."""

    def __init__(
        self, infer, *, operation='velocity', version=1, offset=0, period_ns=50_000_000, capture=None, codec=PLAIN_DATA
    ):
        self.infer = Activity(operation, version, infer, capture or Capture.INPUT_AND_RESULT, codec)
        self.offset = offset
        self.period_ns = period_ns

    def run(self, runtime):
        answer, velocity = None, 0
        obs = yield
        while True:
            if answer is None:
                answer = runtime.submit(self.infer, obs[POSITION] + self.offset, velocity)
            commands = {}
            if answer.done():
                velocity, answer = answer.result(), None
                commands = {MOTOR: velocity}
            obs = yield Step(commands, runtime.time_ns + self.period_ns)


def accelerate(position, velocity):
    return velocity + 1


def unreachable(*args):
    raise AssertionError('Replay ran a recorded activity')


def journaled_rollout(journal: Journal, policy: Policy) -> dict[str, Any]:
    """Run ``policy`` on the motor for 0.21 s with a journal and no dataset; return the terminal payload."""
    with pimm.World(virtual_time=True) as world:
        motion = Motion()
        embodiment = Embodiment(
            descriptor='journal-test',
            observations={POSITION: Observation(motion.position, None)},
            commands={MOTOR: Command(motion.command, None)},
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
        wire.wire_embodiment(world, harness, embodiment, record=False)
        caller = world.pair(harness.perform_task)
        loop = world.start([harness, motion])
        answer = caller(Rollout(Task('move', 0.21, charge_inference_time=False), policy, None, journal))
        try:
            for _ in range(1000):
                next(loop)
                if answer.done():
                    break
            return answer.result()
        finally:
            world.request_stop()
            list(loop)


def returned_commands(journal: Journal, events) -> list:
    recording = journal.read()
    return [journal.commands.decode(recording.payload(e.commands)) for e in events if isinstance(e, StepReturned)]


def test_journaled_rollout_replays_offline_without_running_inference(tmp_path):
    journal = Journal(tmp_path / 'journal')
    assert journaled_rollout(journal, JournaledMove(lambda obs: [{MOTOR: 1}, {MOTOR: 2}])) == {
        eval_keys.TERMINATED: False
    }
    recording = journal.read()
    turns = [e for e in recording.events if isinstance(e, TurnStarted)]
    assert [turn.wake for turn in turns] == [Wake.FIRST, Wake.COMPLETION, Wake.DUE, Wake.DUE, Wake.COMPLETION]
    assert [turn.invocation for turn in turns] == list(range(5))
    assert (turns[1].time_ns, turns[1].tick) == (turns[0].time_ns, turns[0].tick)
    assert returned_commands(journal, recording.events) == [{}, {MOTOR: 1}, {MOTOR: 2}, {}, {MOTOR: 1}]
    emitted = [(e.invocation, e.command) for e in recording.events if isinstance(e, CommandEmitted)]
    assert emitted == [(1, MOTOR), (2, MOTOR), (4, MOTOR)]
    assert isinstance(ended := recording.events[-1], Ended)
    assert verify(JournaledMove(unreachable), journal) == Verified(5, ended.termination)


class Gain(Codec):
    """Scale the position that the model receives and the velocities that it returns by ``gain``."""

    WIRE_NAME = 'gain'

    def __init__(self, gain):
        self.gain = gain

    def encode(self, data):
        return {POSITION: data[POSITION] * self.gain}

    def _decode_single(self, data):
        return {MOTOR: data[MOTOR] * self.gain}

    def to_spec(self):
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: {'gain': self.gain}}


class GainMove(Policy):
    """Play at 10 Hz the chunks of ``infer``, with ``Gain(gain)`` between the schedule and the activity."""

    def __init__(self, infer, gain, codec=PLAIN_DATA):
        self.infer = Activity('step_plan', 1, infer, codec=codec)
        self.stack = Sequential(ChunkedSchedule(fps=10), Gain(gain))

    def run(self, runtime):
        return self.stack.run(runtime, self.infer)


class OuterGainMove(Policy):
    """Play at 10 Hz the chunks of an ``Activity`` around ``Gain(gain).wrap(infer)``, recorded with ``codec``."""

    def __init__(self, infer, gain, codec):
        self.infer = Activity('gained_step_plan', 1, Gain(gain).wrap(infer), codec=codec)

    def run(self, runtime):
        return ChunkedSchedule(fps=10).run(runtime, self.infer)


class Keyed(PlainData):
    """Plain data that refuses a mapping with a key outside ``allowed``."""

    NAME = 'keyed'

    def __init__(self, allowed):
        self.allowed = allowed

    def encode(self, value):
        self._check(value)
        return super().encode(value)

    def _check(self, value):
        if isinstance(value, Mapping):
            if unknown := set(value) - self.allowed:
                raise ValueError(f'{sorted(unknown)} are not allowed keys')
            items = value.values()
        elif isinstance(value, list | tuple):
            items = value
        else:
            return
        for item in items:
            self._check(item)


MODEL_KEYS = {POSITION, MOTOR}
ROBOT_KEYS = {POSITION, MOTOR, keys.TASK, keys.DESCRIPTOR}


def test_a_codec_refuses_an_activity_whose_payload_codec_serves_the_model_side(tmp_path):
    requests = []

    def infer(request):
        requests.append(request)
        return [{MOTOR: 1}, {MOTOR: 2}]

    refused = Journal(tmp_path / 'refused')
    with pytest.raises(TypeError, match='keyed is declared for the model input and output'):
        journaled_rollout(refused, GainMove(infer, gain=3, codec=Keyed(MODEL_KEYS)))
    assert requests == []
    assert [e.kind for e in refused.read().events] == ['started', 'startup', 'start_failed', 'closing', 'ended']

    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, OuterGainMove(infer, 3, Keyed(ROBOT_KEYS)))
    recording = journal.read()
    first = recording.submissions()[0]
    assert (first.operation, first.codec.name) == ('gained_step_plan', 'keyed')
    [obs], _ = PLAIN_DATA.decode(recording.payload(first.input))
    assert set(obs) == {POSITION, keys.TASK, keys.DESCRIPTOR} and requests[0] == {POSITION: 3 * obs[POSITION]}
    assert returned_commands(journal, recording.events)[1:3] == [{MOTOR: 3}, {MOTOR: 6}]
    assert verify(OuterGainMove(unreachable, 3, Keyed(ROBOT_KEYS)), journal).complete


def test_a_rerun_of_a_wrapped_activity_takes_the_observation_and_returns_the_decoded_result(tmp_path):
    source = Journal(tmp_path / 'source')
    journaled_rollout(source, GainMove(lambda request: [{MOTOR: 1}, {MOTOR: 2}], gain=3))
    recording = source.read()
    [recorded], _ = PLAIN_DATA.decode(recording.payload(recording.submissions()[0].input))
    received = []

    def rerun(obs):
        received.append(obs)
        return [{MOTOR: -1}, {MOTOR: -2}]

    changes = [RerunActivity(0, rerun, version=2, allow_execution=True)]
    reran = branch(GainMove(unreachable, gain=3), source, tmp_path / 'rerun', changes)
    assert received == [recorded] and set(recorded) == {POSITION, keys.TASK, keys.DESCRIPTOR}
    assert [returned_commands(reran.journal, d.branch) for d in reran.differences] == [[{MOTOR: -1}], [{MOTOR: -2}]]

    with pytest.raises(MissingResult):
        branch(GainMove(unreachable, gain=2), source, tmp_path / 'changed', changes)
    assert received == [recorded]


def test_a_codec_between_a_schedule_and_its_activity_is_journaled_under_the_codec_spec(tmp_path):
    requests = []

    def infer(request):
        requests.append(request)
        return [{MOTOR: 1}, {MOTOR: 2}]

    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, GainMove(infer, gain=3))
    recording = journal.read()
    submissions = recording.submissions()
    assert {(s.operation, s.version) for s in submissions} == {
        ('{"args":{"gain":3},"name":"gain","version":1}(step_plan)', 1)
    }
    positions = [PLAIN_DATA.decode(recording.payload(s.input))[0][0][POSITION] for s in submissions]
    assert positions[1] > 0 and requests == [{POSITION: 3 * position} for position in positions]
    results = [e.outcome for e in recording.events if isinstance(e, Published)]
    assert all(isinstance(r, Returned) for r in results)
    assert [PLAIN_DATA.decode(recording.payload(cast(Returned, r).result)) for r in results] == [
        [{MOTOR: 3}, {MOTOR: 6}]
    ] * len(submissions)
    assert verify(GainMove(unreachable, gain=3), journal).complete

    with pytest.raises(ReplayDivergence) as raised:
        verify(GainMove(unreachable, gain=2), journal)
    assert raised.value.expected == submissions[0]
    assert isinstance(actual := raised.value.actual, Submitted) and actual.operation.startswith('{"args":{"gain":2}')


@pytest.mark.parametrize(
    'change, differs',
    [
        ({'operation': 'other'}, 'submitted'),
        ({'version': 2}, 'submitted'),
        ({'offset': 1}, 'submitted'),
        ({'capture': Capture.RESULT}, 'submitted'),
        ({'period_ns': 60_000_000}, 'step'),
    ],
)
def test_replay_stops_at_the_first_divergence(tmp_path, change, differs):
    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, Feedback(accelerate))
    assert verify(Feedback(unreachable), journal).complete
    with pytest.raises(ReplayDivergence) as raised:
        verify(Feedback(unreachable, **change), journal)
    assert raised.value.expected is not None and raised.value.actual is not None
    assert (raised.value.expected.kind, raised.value.actual.kind) == (differs, differs)
    assert raised.value.index == next(i for i, e in enumerate(journal.read().events) if e.kind == differs)


def test_branch_replaces_or_reruns_one_result_and_keeps_the_source(tmp_path):
    source = Journal(tmp_path / 'source')
    journaled_rollout(source, JournaledMove(lambda obs: [{MOTOR: 1}, {MOTOR: 2}]))
    recorded = {path: path.read_bytes() for path in source.path.rglob('*') if path.is_file()}
    first = source.read().submissions()[0]

    replaced = branch(JournaledMove(unreachable), source, tmp_path / 'replaced', [ReplaceResult(0, [{MOTOR: 0}] * 2)])
    assert [d.invocation for d in replaced.differences] == [1, 2]
    assert [returned_commands(replaced.journal, d.branch) for d in replaced.differences] == [[{MOTOR: 0}]] * 2
    assert replaced.journal.read().started.parent == Parent(
        journal=source.read().started.journal, path=source.path, changes=(ReplacedResult(submission=0),)
    )
    assert verify(JournaledMove(unreachable), replaced.journal).complete

    inputs = []

    def infer_v2(obs):
        inputs.append(obs)
        return [{MOTOR: -1}, {MOTOR: -2}]

    with pytest.raises(ExecutionRefused):
        branch(JournaledMove(unreachable), source, tmp_path / 'refused', [RerunActivity(0, infer_v2, version=2)])
    assert inputs == [] and not (tmp_path / 'refused').exists()
    rerun = RerunActivity(0, infer_v2, version=2, allow_execution=True)
    reran = branch(JournaledMove(unreachable), source, tmp_path / 'rerun', [rerun])
    retained_args, _ = PLAIN_DATA.decode(source.read().payload(first.input))
    assert inputs == [retained_args[0]]
    assert [returned_commands(reran.journal, d.branch) for d in reran.differences] == [[{MOTOR: -1}], [{MOTOR: -2}]]
    assert {path: path.read_bytes() for path in source.path.rglob('*') if path.is_file()} == recorded


def test_branch_refuses_unknown_unretained_or_unmatched_submissions(tmp_path):
    source = Journal(tmp_path / 'source')
    journaled_rollout(source, Feedback(accelerate, capture=Capture.RESULT))
    policy = Feedback(unreachable, capture=Capture.RESULT)
    with pytest.raises(ReplayError, match='no submission 99'):
        branch(policy, source, tmp_path / 'unknown', [ReplaceResult(99, 0)])
    with pytest.raises(MissingInput):
        branch(policy, source, tmp_path / 'unretained', [RerunActivity(0, accelerate, version=2, allow_execution=True)])
    with pytest.raises(ReplayError, match='does not encode'):
        branch(policy, source, tmp_path / 'unencodable', [ReplaceResult(0, object())])
    # The next request carries the changed velocity, which no recorded submission saw.
    with pytest.raises(MissingResult, match='submission=1'):
        branch(policy, source, tmp_path / 'unmatched', [ReplaceResult(0, 10)])


def test_replay_refuses_a_journal_it_cannot_read(tmp_path):
    class Other(PayloadCodec):
        NAME = 'other'

        def encode(self, value):
            return b''

        def decode(self, payload):
            return None

        def decode_frozen(self, payload):
            return None

    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, Feedback(accelerate))
    with pytest.raises(ValueError, match='other'):
        verify(Feedback(unreachable), Journal(journal.path, commands=Other()))
    with pytest.raises(ReplayError, match='records policy'):
        verify(JournaledMove(unreachable), journal)
    events = journal.path / 'events.jsonl'
    lines = events.read_bytes().splitlines(keepends=True)
    events.write_bytes(b''.join(lines[:3]) + b'{"kind":"turn"}\n' + b''.join(lines[3:]))
    with pytest.raises(ValueError, match='validation error'):
        verify(Feedback(unreachable), journal)
    events.write_bytes(b''.join(line for line in lines if b'"kind":"closing"' not in line))
    with pytest.raises(ValueError, match='cannot follow'):
        verify(Feedback(unreachable), journal)
    events.write_bytes(b''.join(lines).replace(b'"format":1', b'"format":2', 1))
    with pytest.raises(ValueError, match='format 2'):
        verify(Feedback(unreachable), journal)


def test_replay_checks_a_torn_journal_through_its_last_closed_scope(tmp_path):
    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, Feedback(accelerate))
    events = journal.path / 'events.jsonl'
    lines = events.read_bytes().splitlines(keepends=True)
    turns = [i for i, line in enumerate(lines) if b'"kind":"turn"' in line]
    for kept, played in ((len(lines) - 1, len(turns)), (turns[2] + 1, 2), (2, 0)):
        events.write_bytes(b''.join(lines[:kept]) + lines[kept][:-5])
        assert verify(Feedback(unreachable), journal) == Verified(played, None)
    events.write_bytes(b''.join(lines[: turns[2] + 1]))
    with pytest.raises(ReplayDivergence):
        verify(Feedback(unreachable, offset=1), journal)


class ReadsAtClose(Policy):
    """Submit ``infer`` at the first turn, and put its result into the metadata when the policy closes."""

    def __init__(self, infer, period_ns=50_000_000):
        self.infer = Activity('velocity', 1, infer)
        self.period_ns = period_ns

    def run(self, runtime):
        obs = yield
        answer = runtime.submit(self.infer, obs[POSITION], 0)
        try:
            while True:
                obs = yield Step({}, runtime.time_ns + self.period_ns)
        finally:
            runtime.metadata['velocity'] = answer.result()


def test_a_finalizer_outside_the_recorded_close_does_not_change_what_a_replay_reports(tmp_path, caplog):
    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, ReadsAtClose(accelerate))
    assert verify(ReadsAtClose(unreachable), journal).complete
    events = journal.path / 'events.jsonl'
    lines = events.read_bytes().splitlines(keepends=True)
    published = next(i for i, line in enumerate(lines) if b'"kind":"published"' in line)
    events.write_bytes(b''.join(lines[:published]) + lines[published][:-5])

    def close_errors():
        records = [r for r in caplog.records if 'outside the recorded part of the journal' in r.getMessage()]
        caplog.clear()
        return [type(r.exc_info[1]) for r in records if r.exc_info is not None]

    assert verify(ReadsAtClose(unreachable), journal) == Verified(1, None)
    assert close_errors() == [NotAnswered]
    with pytest.raises(ReplayDivergence) as raised:
        verify(ReadsAtClose(unreachable, period_ns=60_000_000), journal)
    assert isinstance(raised.value.expected, StepReturned)
    assert close_errors() == [NotAnswered]


class Composite(Policy):
    """Start ``child`` in the policy's own run, and pass each observation to it."""

    def __init__(self, child: Policy):
        self.child = child

    def run(self, runtime):
        with closing(runtime.start(self.child)) as child:
            obs = yield
            while True:
                obs = yield child.send(obs)


class Idle(Policy):
    def run(self, runtime, *dependencies):
        yield
        while True:
            yield Step({}, runtime.time_ns + 50_000_000)


@pytest.fixture
def remote(monkeypatch):
    """A ``RemotePolicy`` whose mocked server declares an ``Idle`` stack."""
    monkeypatch.setattr(remote_module, 'declared_stack', lambda meta, protocol_version: Idle())
    policy = RemotePolicy('websocket', session_wire.HostPortAddress('localhost', 0, session_wire.SESSION_PATH, ''))
    policy._client = Mock()
    policy._client.new_session.return_value = Mock(metadata={})
    return policy


def test_a_remote_policy_opens_its_session_inside_a_custom_processor_without_a_journal(remote):
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    run = runtime.start(Composite(remote))
    try:
        assert run.send({}) == Step({}, 50_000_000)
    finally:
        runtime.close(run)
    assert remote._client.new_session.call_count == 1


def test_a_journal_refuses_a_remote_policy_inside_a_custom_processor_before_its_session_opens(remote, tmp_path):
    journal = Journal(tmp_path / 'journal')
    with pytest.raises(TypeError, match='cannot record a RemotePolicy'):
        journaled_rollout(journal, Composite(remote))
    remote._client.new_session.assert_not_called()
    assert [e.kind for e in journal.read().events] == ['started', 'startup', 'start_failed', 'closing', 'ended']


def test_replay_refuses_a_remote_policy_inside_a_custom_processor_before_its_session_opens(remote, tmp_path):
    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, Composite(Idle()))
    assert verify(Composite(Idle()), journal).complete
    with pytest.raises(ReplayDivergence, match='cannot record a RemotePolicy'):
        verify(Composite(remote), journal)
    remote._client.new_session.assert_not_called()


class WarmUp(Policy):
    """Warm the model up at startup with the startup time, then command what it returned."""

    def __init__(self, warm_up):
        self.warm_up = Activity('warm_up', 1, warm_up)

    def run(self, runtime):
        started_ns = runtime.time_ns
        answer = runtime.submit(self.warm_up, started_ns)
        yield
        while True:
            commands = {MOTOR: answer.result()} if answer.done() else {}
            yield Step(commands, runtime.time_ns + 50_000_000)


def test_journal_replays_the_startup_time_and_work(tmp_path):
    journal = Journal(tmp_path / 'journal')
    journaled_rollout(journal, WarmUp(lambda started_ns: started_ns + 1))
    recording = journal.read()
    startup, submitted = recording.events[1:3]
    assert isinstance(startup, Startup) and isinstance(submitted, Submitted) and submitted.invocation == -1
    assert PLAIN_DATA.decode(recording.payload(submitted.input)) == [[startup.time_ns], {}]
    assert returned_commands(journal, recording.events)[1] == {MOTOR: startup.time_ns + 1}
    assert verify(WarmUp(unreachable), journal).complete


class Recover(Policy):
    """Note what a failed ``infer`` shows the policy, and stop the motor."""

    def __init__(self, infer, seen):
        self.infer = Activity('velocity', 1, infer)
        self.seen = seen

    def run(self, runtime):
        obs = yield
        answer = runtime.submit(self.infer, obs[POSITION])
        while True:
            commands = {}
            if answer.done():
                try:
                    answer.result()
                except ActivityFailed as exc:
                    self.seen.append((exc.args, exc.__cause__, exc.__context__))
                    commands = {MOTOR: 0}
            obs = yield Step(commands, runtime.time_ns + 50_000_000)


def test_a_failed_activity_shows_the_same_error_live_and_in_replay(tmp_path):
    def fail(position):
        raise ValueError(f'no velocity at {position}')

    journal = Journal(tmp_path / 'journal')
    live, replayed = [], []
    journaled_rollout(journal, Recover(fail, live))
    assert verify(Recover(unreachable, replayed), journal).complete
    assert live[0] == (('ValueError: no velocity at 0',), None, None)
    assert replayed == live


def test_a_failed_rerun_shows_the_policy_the_failure_at_the_recorded_turn(tmp_path):
    def fail(position):
        raise ValueError(f'no velocity at {position}')

    source = Journal(tmp_path / 'source')
    live, branched, replayed = [], [], []
    journaled_rollout(source, Recover(lambda position: position + 1, live))
    recorded = {path: path.read_bytes() for path in source.path.rglob('*') if path.is_file()}
    rerun = RerunActivity(0, fail, version=2, allow_execution=True)
    result = branch(Recover(unreachable, branched), source, tmp_path / 'branch', [rerun])
    assert live == [] and branched and set(branched) == {(('ValueError: no velocity at 0',), None, None)}
    [published] = [e for e in result.differences[0].branch if isinstance(e, Published)]
    assert isinstance(published.outcome, Raised) and published.outcome.error == 'ValueError: no velocity at 0'
    assert [returned_commands(result.journal, d.branch) for d in result.differences] == [[{MOTOR: 0}]] * len(
        result.differences
    )
    assert verify(Recover(unreachable, replayed), result.journal).complete
    assert replayed == branched
    assert {path: path.read_bytes() for path in source.path.rglob('*') if path.is_file()} == recorded


class Strict(Policy):
    """Command the result of ``infer``, which must be an integer, and note in ``closed`` the time it closes at."""

    def __init__(self, infer, closed):
        self.infer = Activity('value', 1, infer)
        self.closed = closed

    def run(self, runtime):
        yield
        answer = runtime.submit(self.infer)
        try:
            while True:
                commands = {MOTOR: operator.index(answer.result())} if answer.done() else {}
                yield Step(commands, runtime.time_ns + 50_000_000)
        finally:
            self.closed.append(runtime.time_ns)


def ending(journal: Journal) -> tuple[Closing, Ended]:
    *_, closing, ended = (e for e in journal.read().events if isinstance(e, Closing | Ended))
    assert isinstance(closing, Closing) and isinstance(ended, Ended)
    return closing, ended


def fail():
    raise ValueError('rerun failed')


@pytest.mark.parametrize(
    'change, error',
    [
        (RerunActivity(0, fail, version=2, allow_execution=True), 'ActivityFailed: ValueError: rerun failed'),
        (ReplaceResult(0, 'one'), 'TypeError: '),
    ],
    ids=['failed-rerun', 'replaced-result'],
)
def test_a_turn_failure_that_the_source_does_not_record_ends_the_branch_at_that_turn(tmp_path, change, error):
    source = Journal(tmp_path / 'source')
    assert journaled_rollout(source, Strict(lambda: 1, [])) == {eval_keys.TERMINATED: False}
    source_closing, _ = ending(source)

    closed = []
    result = branch(Strict(unreachable, closed), source, tmp_path / 'branch', [change])
    events = result.journal.read().events
    [failed] = [e for e in events if isinstance(e, TurnFailed)]
    [turn] = [e for e in events if isinstance(e, TurnStarted) and e.invocation == failed.invocation]
    closing, ended = ending(result.journal)
    assert failed.error.startswith(error) and turn.time_ns < source_closing.time_ns
    assert closing.time_ns == turn.time_ns and closed == [turn.time_ns]
    assert isinstance(ended.termination, Raised) and ended.termination.error == failed.error
    assert verify(Strict(unreachable, []), result.journal) == Verified(failed.invocation + 1, ended.termination)


def test_a_branch_keeps_the_recorded_ending_when_its_turns_end_as_the_source_turns_did(tmp_path):
    succeeded = Journal(tmp_path / 'succeeded')
    journaled_rollout(succeeded, Strict(lambda: 1, []))
    rerun = RerunActivity(0, lambda: 2, version=2, allow_execution=True)
    reran = branch(Strict(unreachable, []), succeeded, tmp_path / 'reran', [rerun])
    assert [returned_commands(reran.journal, d.branch) for d in reran.differences][0] == [{MOTOR: 2}]
    assert ending(reran.journal) == ending(succeeded)

    failed = Journal(tmp_path / 'failed')
    with pytest.raises(TypeError):
        journaled_rollout(failed, Strict(lambda: 'one', []))
    _, recorded = ending(failed)
    assert isinstance(recorded.termination, Raised)
    assert verify(Strict(unreachable, []), failed).termination == recorded.termination
    unchanged = branch(Strict(unreachable, []), failed, tmp_path / 'unchanged', [])
    assert unchanged.differences == () and ending(unchanged.journal) == ending(failed)


class Follow(Policy):
    """Command the observed position, also on a channel the rig lacks at position 2. Fail when it is negative."""

    def run(self, runtime):
        obs = yield
        while True:
            if (position := obs[POSITION]) < 0:
                raise ValueError('negative position')
            commands = {MOTOR: position} | ({'unknown': position} if position == 2 else {})
            obs = yield Step(commands, runtime.time_ns + 1_000_000)


class FailingStart(Policy):
    """Fail at startup when ``fail`` is set; otherwise wait."""

    def __init__(self, fail):
        self.fail = fail

    def run(self, runtime):
        if self.fail:
            raise ValueError('startup failed')
        yield
        while True:
            yield Step({}, runtime.time_ns + 1_000_000)


def test_journal_records_a_failed_startup_and_its_replay_must_fail_too(episode_harness, tmp_path):
    h = episode_harness
    journal = Journal(tmp_path / 'journal')
    h.observation.emit(1)
    h.caller(Rollout(Task('test', None), FailingStart(True), None, journal))
    with pytest.raises(ValueError, match='startup failed'):
        next(h.loop)
    assert [e.kind for e in journal.read().events] == ['started', 'startup', 'start_failed', 'closing', 'ended']
    assert isinstance(ended := journal.read().events[-1], Ended) and isinstance(ended.termination, Raised)
    assert verify(FailingStart(True), journal) == Verified(0, ended.termination)
    with pytest.raises(ReplayDivergence, match='StartFailed'):
        verify(FailingStart(False), journal)


class Opens(Policy):
    """Fail at startup when ``fail_start`` is set; otherwise note in ``events`` that it opens and closes.

    Its finalizer raises when ``fail_close`` is set.
    """

    def __init__(self, events, *, fail_start=False, fail_close=False):
        self.events = events
        self.fail_start = fail_start
        self.fail_close = fail_close

    def run(self, runtime):
        if self.fail_start:
            raise ValueError('startup failed')
        self.events.append('opened')
        try:
            yield
            while True:
                yield Step({}, runtime.time_ns + 50_000_000)
        finally:
            self.events.append('closed')
            if self.fail_close:
                raise RuntimeError('the policy failed to release its resource')


@pytest.mark.parametrize('fail_close', [False, True])
def test_a_replay_that_primes_where_the_journal_records_a_failed_start_closes_the_policy_at_once(
    tmp_path, caplog, fail_close
):
    journal = Journal(tmp_path / 'journal')
    with pytest.raises(ValueError, match='startup failed'):
        journaled_rollout(journal, Opens([], fail_start=True))
    assert verify(Opens([], fail_start=True), journal).complete

    events = []
    with pytest.raises(ReplayDivergence) as raised:
        verify(Opens(events, fail_close=fail_close), journal)
    assert events == ['opened', 'closed']
    assert isinstance(raised.value.expected, StartFailed) and isinstance(raised.value.actual, Primed)
    assert ('the policy failed to release its resource' in caplog.text) is fail_close


def test_a_startup_failure_that_the_source_does_not_record_ends_the_branch_at_the_startup(tmp_path):
    source = Journal(tmp_path / 'source')
    journaled_rollout(source, Opens([]))
    [startup] = [e for e in source.read().events if isinstance(e, Startup)]
    source_closing, source_ended = ending(source)
    assert isinstance(source_ended.termination, Finished) and startup.time_ns < source_closing.time_ns
    unchanged = branch(Opens([]), source, tmp_path / 'unchanged', [])
    assert unchanged.differences == () and ending(unchanged.journal) == ending(source)

    result = branch(Opens([], fail_start=True), source, tmp_path / 'branch', [])
    kinds = [e.kind for e in result.journal.read().events]
    closing, ended = ending(result.journal)
    assert kinds == ['started', 'startup', 'start_failed', 'closing', 'ended']
    assert closing.time_ns == startup.time_ns
    assert isinstance(ended.termination, Raised) and ended.termination.error == 'ValueError: startup failed'
    assert verify(Opens([], fail_start=True), result.journal) == Verified(0, ended.termination)

    failed = Journal(tmp_path / 'failed')
    with pytest.raises(ValueError, match='startup failed'):
        journaled_rollout(failed, Opens([], fail_start=True))
    kept = branch(Opens([], fail_start=True), failed, tmp_path / 'kept', [])
    assert kept.differences == () and ending(kept.journal) == ending(failed)


def failed_turn_source(path: Path) -> Journal:
    source = Journal(path)
    with pytest.raises(TypeError):
        journaled_rollout(source, Strict(lambda: 'one', []))
    return source


def failed_startup_source(path: Path) -> Journal:
    source = Journal(path)
    with pytest.raises(ValueError, match='startup failed'):
        journaled_rollout(source, Opens([], fail_start=True))
    return source


@pytest.mark.parametrize(
    'record, policy, changes, last_scope, closes',
    [
        (
            failed_turn_source,
            lambda closed: Strict(unreachable, closed),
            [ReplaceResult(0, 1)],
            StepReturned,
            lambda closing_ns: [closing_ns],
        ),
        (failed_startup_source, Opens, [], Primed, lambda closing_ns: ['opened', 'closed']),
    ],
    ids=['turn', 'startup'],
)
def test_a_branch_that_goes_on_where_the_source_failed_ends_without_an_ending(
    tmp_path, record, policy, changes, last_scope, closes
):
    source = record(tmp_path / 'source')
    _, recorded = ending(source)
    assert isinstance(recorded.termination, Raised)

    closed = []
    result = branch(policy(closed), source, tmp_path / 'branch', changes)
    events = result.journal.read().events
    assert not any(isinstance(e, TurnFailed | StartFailed | Ended) for e in events)
    *_, last, closing = (e for e in events if isinstance(e, Primed | StepReturned | Closing))
    assert isinstance(last, last_scope) and isinstance(closing, Closing)
    started = [e.time_ns for e in events if isinstance(e, Startup | TurnStarted)]
    assert closing.time_ns == started[-1] and closed == closes(closing.time_ns)
    verified = verify(policy([]), result.journal)
    assert verified == Verified(len(started) - 1, None) and not verified.complete


@pytest.mark.parametrize('done_at_ns, terminated', [(5_000_000, True), (None, False)])
def test_journal_ends_with_the_terminal_payload(episode_harness, tmp_path, done_at_ns, terminated):
    h = episode_harness
    journal = Journal(tmp_path / 'journal')
    h.observation.emit(1)
    answer = h.caller(Rollout(Task('test', 0.01), Follow(), None, journal))
    next(h.loop)
    if done_at_ns is not None:
        h.done.emit({'success': True}, ts=done_at_ns)
    h.world.clock.advance_to_ns(12_000_000)
    next(h.loop)
    next(h.loop)
    assert answer.result()[eval_keys.TERMINATED] is terminated
    result = verify(Follow(), journal)
    assert isinstance(finished := result.termination, Finished)
    assert PLAIN_DATA.decode(journal.read().payload(finished.payload)) == answer.result()


def test_journal_ends_stopped_when_the_world_stops(episode_harness, tmp_path):
    h = episode_harness
    journal = Journal(tmp_path / 'journal')
    h.observation.emit(1)
    answer = h.caller(Rollout(Task('test', None), Follow(), None, journal))
    next(h.loop)
    h.world.request_stop()
    list(h.loop)
    with pytest.raises(pimm.calls.HandlerStopped):
        answer.result()
    assert verify(Follow(), journal).termination == Stopped()


@pytest.mark.parametrize(
    'observation, error, emitted', [(-1, 'ValueError: negative position', []), (2, "KeyError: 'unknown", [MOTOR])]
)
def test_journal_ends_with_the_error_of_a_failed_turn_or_emission(
    episode_harness, tmp_path, observation, error, emitted
):
    h = episode_harness
    journal = Journal(tmp_path / 'journal')
    h.observation.emit(observation)
    h.caller(Rollout(Task('test', None), Follow(), None, journal))
    with pytest.raises((ValueError, KeyError)):
        next(h.loop)
    events = journal.read().events
    assert [e.command for e in events if isinstance(e, CommandEmitted)] == emitted
    result = verify(Follow(), journal)
    assert isinstance(raised := result.termination, Raised) and raised.error.startswith(error)


class Undecodable(PlainData):
    """Plain data whose frozen decode refuses the values that ``refuses`` selects."""

    NAME = 'undecodable'

    def __init__(self, refuses):
        self.refuses = refuses

    def decode_frozen(self, payload):
        value = super().decode_frozen(payload)
        if self.refuses(value):
            raise ValueError('the codec refused the value')
        return value


class NotesClose(Policy):
    """Run ``Feedback`` on ``infer``, and note in ``events`` each return of the activity and the close of the policy."""

    def __init__(self, infer, events, **options):
        def noted(*args):
            events.append('returned')
            return infer(*args)

        self.feedback = Feedback(noted, **options)
        self.events = events

    def run(self, runtime):
        try:
            yield from self.feedback.run(runtime)
        finally:
            self.events.append('closed')


@pytest.mark.parametrize(
    'journal_codec, activity_codec, last_scope',
    [
        (Undecodable(lambda obs: obs[POSITION] > 0), PLAIN_DATA, ('step', 'emitted')),
        (PLAIN_DATA, Undecodable(lambda result: True), ('failed',)),
    ],
    ids=['observation', 'publication'],
)
def test_a_codec_failure_at_turn_entry_closes_the_policy_and_ends_the_journal_with_it(
    tmp_path, journal_codec, activity_codec, last_scope
):
    journal = Journal(tmp_path / 'journal', journal_codec)
    events = []
    with pytest.raises(ValueError, match='the codec refused the value'):
        journaled_rollout(journal, NotesClose(accelerate, events, codec=activity_codec))
    recording = journal.read()
    assert events.count('closed') == 1 and events[-1] == 'closed'
    assert events.count('returned') == len(recording.submissions())
    kinds = [e.kind for e in recording.events]
    assert kinds[-3] in last_scope and kinds[-2:] == ['closing', 'ended']
    assert isinstance(ended := recording.events[-1], Ended) and isinstance(ended.termination, Raised)
    assert ended.termination.error == 'ValueError: the codec refused the value'
    assert verify(NotesClose(unreachable, [], codec=activity_codec), journal).termination == ended.termination


def test_recorder_refuses_an_encoder_this_host_cannot_run():
    class AbsentEncoder(LibavEncoder):
        def ensure_available(self) -> None:
            raise RuntimeError('no such encoder here')

    with pimm.World(virtual_time=True) as world:
        motion = Motion()
        embodiment = Embodiment(
            descriptor='recording-test',
            observations={POSITION: Observation(motion.position, None)},
            commands={MOTOR: Command(motion.command, None)},
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=True,
            video_encoder=AbsentEncoder(),
        )
        with pytest.raises(RuntimeError, match='no such encoder here'):
            wire.wire_embodiment(world, Harness(embodiment), embodiment, TimeMode.MESSAGE)


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


@pytest.mark.parametrize('late', [False, True])
@pytest.mark.parametrize(
    'model',
    [
        {roboarm_keys.URDF: bundled_franka_model()[roboarm_keys.URDF], roboarm_keys.CONTROL_FRAME: EE_LINK},
        {roboarm_keys.URDF: '<robot name="r"><link name="base"/></robot>', roboarm_keys.CONTROL_FRAME: DEFAULT_FRAME},
    ],
)
def test_observations_reject_an_invalid_control_frame(observed_harness, model, late):
    world, harness, _, _, _, step = observed_harness
    if late:
        step()
        emitter, receiver = world.local_pipe()
        harness.robot_meta_in._bind(receiver)
        emitter.emit(model)
    else:
        harness._embodiment.static_meta.update(model)
    with pytest.raises(ValueError, match=model[roboarm_keys.CONTROL_FRAME]):
        step()


@pytest.mark.parametrize('requested_ns, expected_ns', [(-1, 5_000_000), (100_000_000, 100_000_000), (10**12, 10**9)])
def test_wake_interval_is_clamped_from_call_start(episode_harness, requested_ns, expected_ns):
    h = episode_harness
    h.observation.emit(1)
    now = [0]

    class Slow(Policy):
        def run(self, runtime):
            yield
            now[0] = 500_000_000
            yield Step({}, requested_ns)

    runtime = Executor(lambda: now[0], simulated=False, charge_inference_time=False)
    run = runtime.start(Slow())
    try:
        assert h.harness._step(Task('test', None), runtime, run, None) == expected_ns
    finally:
        runtime.close()
        run.close()


def test_robot_observation_serialization_and_typed_command_emission():
    with pimm.World(virtual_time=True) as world:
        device = Passive()
        embodiment = Embodiment(
            descriptor='robot',
            observations={keys.ROBOT_STATE: Observation(pimm.ControlSystemEmitter(device), Serializers.robot_state)},
            commands={
                keys.ROBOT_COMMAND: Command(pimm.ControlSystemReceiver(device), Serializers.robot_command),
                keys.TARGET_GRIP: Command(pimm.ControlSystemReceiver(device), None),
            },
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
        emitter, receiver = world.local_pipe()
        harness.observations[keys.ROBOT_STATE]._bind(receiver)
        emitted = Trace(world.clock)
        harness.commands[keys.ROBOT_COMMAND]._bind(emitted)
        harness.commands[keys.TARGET_GRIP]._bind(Trace(world.clock))
        state = make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        emitter.emit(state)
        policy = StubPolicy()
        runtime = Executor(world.clock.now_ns, simulated=True, charge_inference_time=False)
        run = runtime.start(policy)
        try:
            harness._step(Task('move', None), runtime, run, None)
        finally:
            runtime.close()
            run.close()
        obs = policy.observations[0]
        np.testing.assert_allclose(obs[keys.EE_POSE][:3], state.ee_pose.translation)
        np.testing.assert_allclose(obs[keys.JOINTS], state.q)
        np.testing.assert_allclose(obs[keys.JOINT_VEL], state.dq)
        assert obs[keys.ROBOT_STATUS] == state.status
        assert obs[keys.TASK] == 'move'
        assert obs[keys.DESCRIPTOR] == 'robot'
        assert 'obs_time_ns' not in obs and 'wall_time_ns' not in obs
        assert emitted.values == [(0, policy.command)]


def test_real_sleep_stops_at_episode_deadline(episode_harness):
    h = episode_harness
    h.harness._embodiment = replace(h.embodiment, simulated=False)
    h.observation.emit(1)

    class SlowPoll(Policy):
        def run(self, runtime):
            yield
            while True:
                yield Step({}, runtime.time_ns + 10**9)

    answer = h.caller(Rollout(Task('test', 0.02), SlowPoll(), None))
    wake = next(h.loop)
    assert isinstance(wake, pimm.Sleep)
    assert wake.seconds == pytest.approx(0.02)
    h.world.clock.advance_to_ns(20_000_000)
    next(h.loop)
    assert h.deadlines.values[-1] == (20_000_000, None)
    next(h.loop)
    assert answer.result() == {eval_keys.TERMINATED: False}
