"""Timing contracts for processor execution and simulated chunk playback."""

import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

import pimm
from pimm.tests.testing import Passive, wire_call
from pimm.time import EMITTED_WALL, EMITTED_WORLD, RECEIVED_WALL, RECEIVED_WORLD
from pimm.world import VirtualClock
from positronic import keys, telemetry, telemetry_keys, wire
from positronic.cli.eval.run import run_world
from positronic.dataset.dataset import DatasetWriter
from positronic.dataset.episode import Episode, EpisodeWriter
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.serializers import Serializers
from positronic.dataset.time import HARNESS_WALL, HARNESS_WORLD
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
from positronic.policy.base import Policy, PolicyRun, Step
from positronic.policy.executor import Executor, _UnchargedAnswer
from positronic.policy.harness import POLL_PERIOD_SEC, Harness, Rollout
from positronic.policy.processors import ChunkedSchedule, PauseOnUnavailable
from positronic.policy.remote import round_trip
from positronic.policy.sequential import Sequential

MOTOR = 'motor'
POSITION = 'position'
RESET = 'reset'
CAMERA = 'camera'
ARM = 'arm'
GROUND_TRUTH = 'ground_truth'


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

    def _emit(self, data, time: pimm.Time):
        self.values.append((self.clock.now_ns(), data))
        if self.forward is not None:
            self.forward._emit(data, time)


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
        harness.deadline_ns._bind(Trace(world.clock), clock=world.clock)
        runtime = Executor(world.clock.now_ns, simulated=True, charge_inference_time=False)
        policy_run = runtime.start(Observe())
        step = partial(harness._step, Task('test', None), runtime, policy_run, None)
        try:
            yield world, harness, emitters, serializers, calls, step
        finally:
            runtime.close()
            policy_run.close()


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


class MemoryEpisode(EpisodeWriter):
    def __init__(self):
        self.records = []
        self.static = {}
        self.closed = False
        self.aborted = False

    def append(self, signal_name, data, timestamps):
        self.records.append((signal_name, data, timestamps))

    def set_static(self, name, data):
        self.static[name] = data

    def abort(self):
        self.aborted = True

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        self.closed = True


class MemoryDataset(DatasetWriter):
    def __init__(self):
        self.episodes = []

    def new_episode(self):
        episode = MemoryEpisode()
        self.episodes.append(episode)
        return episode

    def __exit__(self, exc_type, exc, tb):
        pass


@contextmanager
def _episode_ports(devices: tuple[str, ...] = (), observed: tuple[str, ...] = (), privileged: tuple[str, ...] = ()):
    """A harness whose loop the test drives, with a ready handler for each of ``devices``, and an observation of the
    same name for each of ``observed``."""
    with pimm.World(virtual_time=True) as world:
        source = Passive()
        serializer = Mock(side_effect=lambda value: value)
        preparation = {name: pimm.calls.ControlSystemHandler(source) for name in (RESET, eval_keys.SCENE)}
        readiness = {name: pimm.calls.ControlSystemHandler[None, None](source) for name in devices}
        embodiment = Embodiment(
            descriptor='test',
            observations={
                POSITION: Observation(pimm.ControlSystemEmitter(source), serializer),
                **{name: Observation(pimm.ControlSystemEmitter(source), None) for name in observed},
            },
            commands={MOTOR: Command(pimm.ControlSystemReceiver(source), None)},
            prepare_handlers=preparation,
            static_meta={},
            meta_source=None,
            ready_handlers=readiness,
            simulated=True,
        )
        dataset = MemoryDataset()
        dataset_factory = Mock(return_value=dataset)
        privileged_specs = {name: Observation(pimm.ControlSystemEmitter(source), None) for name in privileged}
        harness = Harness(embodiment, dataset_factory=dataset_factory, privileged=privileged_specs)
        for name, handler in readiness.items():
            wire_call(world, harness.ready[name], handler)
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
        device_observations = {}
        for name in observed:
            emitter, physical_receiver = world.local_pipe()
            harness.observations[name]._bind(physical_receiver)
            device_observations[name] = emitter
        privileged_emitters = {}
        for name, receiver in harness.privileged.items():
            emitter, physical_receiver = world.local_pipe()
            receiver._bind(physical_receiver)
            privileged_emitters[name] = emitter
        ports = SimpleNamespace(
            world=world,
            harness=harness,
            embodiment=embodiment,
            ready=readiness,
            prepare=preparation,
            loop=harness.run(world.should_stop_reader(), world.clock),
            caller=caller,
            manual=manual,
            done=done,
            observation=observation,
            device_observations=device_observations,
            privileged=privileged_emitters,
            serializer=serializer,
            dataset=dataset,
            dataset_factory=dataset_factory,
            output_path=Path('recording'),
            deadlines=Trace(world.clock),
            commands=Trace(world.clock),
        )
        harness.deadline_ns._bind(ports.deadlines, clock=world.clock)
        harness.commands[MOTOR]._bind(ports.commands, clock=world.clock)
        try:
            yield ports
        finally:
            world.request_stop()
            list(ports.loop)


@pytest.fixture
def episode_harness():
    with _episode_ports() as ports:
        yield ports


@pytest.fixture
def ready_harness():
    with _episode_ports(devices=(CAMERA, ARM)) as ports:
        yield ports


@pytest.fixture
def repair_harness():
    with _episode_ports(devices=(CAMERA, ARM), observed=(CAMERA,)) as ports:
        yield ports


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
    first = h.caller(Rollout(Task('first', 0.01), Observe(), h.output_path))
    next(h.loop)
    assert h.deadlines.values == [(0, 10_000_000)]

    h.manual.emit({MOTOR: 99})
    overlapping = h.caller(Rollout(Task('overlapping', None), Observe(), h.output_path))
    h.world.clock.advance_to_ns(5_000_000)
    h.done.emit({'success': True})
    h.world.clock.advance_to_ns(12_000_000)
    next(h.loop)
    assert first.done()
    assert h.dataset.episodes[0].closed
    assert closed == [True]
    assert h.deadlines.values[-1] == (12_000_000, None)
    with pytest.raises(RuntimeError, match='already running'):
        overlapping.result()
    next(h.loop)
    assert first.result() == {'success': True, eval_keys.TERMINATED: True}
    assert h.dataset.episodes[-1].static[keys.TASK] == 'first'

    frame[0] = 2  # No new signal: the next episode must rebuild its observation cache.
    second = h.caller(Rollout(Task('second', None), Observe(), h.output_path))
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
    assert len(h.dataset.episodes) == 2
    assert all(episode.closed and not episode.aborted for episode in h.dataset.episodes)


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
    h.world.clock.advance_to_ns(done_at_ns)
    h.done.emit({'success': True})
    h.world.clock.advance_to_ns(12_000_000)
    next(h.loop)
    next(h.loop)
    assert answer.result()[eval_keys.TERMINATED] is terminated


@pytest.mark.parametrize('failure', ['startup', 'policy', 'conversion'])
def test_a_failed_episode_closes_the_generator_answers_its_call_and_the_next_call_runs(episode_harness, failure):
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
    answer = h.caller(Rollout(Task('test', 0.01), Failing(), h.output_path))
    next(h.loop)
    assert closed == [True]
    with pytest.raises(error, match=f'{failure} failed'):
        answer.result()
    opened = failure != 'startup'
    assert len(h.dataset.episodes) == int(opened)
    assert all(episode.closed and episode.aborted for episode in h.dataset.episodes)
    assert [deadline for _, deadline in h.deadlines.values] == ([10_000_000, None] if opened else [])

    h.serializer.side_effect = lambda value: value
    h.caller(Rollout(Task('next', None), Hold(), h.output_path))
    next(h.loop)
    assert not h.dataset.episodes[-1].closed


@pytest.mark.parametrize('ending', ['stop', 'interrupt'])
def test_an_episode_that_does_not_fail_on_its_own_answers_handler_stopped(episode_harness, ending):
    h = episode_harness

    class Interrupted(Policy):
        def run(self, runtime):
            yield
            raise KeyboardInterrupt

    h.observation.emit(1)
    if ending == 'stop':
        answer = h.caller(Rollout(Task('move', None, prepare_args={RESET: 'home'}), Hold(), None))
        next(h.loop)
        assert next(h.prepare[RESET].incoming()).request == 'home'
        h.world.request_stop()
        list(h.loop)
    else:
        answer = h.caller(Rollout(Task('move', None), Interrupted(), None))
        with pytest.raises(KeyboardInterrupt):
            next(h.loop)
    with pytest.raises(pimm.calls.HandlerStopped):
        answer.result()


class Positions(Policy):
    """Record the position each step observes."""

    def __init__(self):
        self.seen = []
        self.observations = []
        self.closed = False

    def run(self, runtime):
        try:
            obs = yield
            while True:
                self.seen.append(obs[POSITION])
                self.observations.append(obs)
                obs = yield Step({}, runtime.time_ns + 100_000_000)
        finally:
            self.closed = True


@pytest.mark.parametrize('lost', [True, False])
def test_an_observation_error_discards_the_episode_and_an_old_value_does_not(episode_harness, lost):
    h = episode_harness
    h.harness._embodiment = replace(h.embodiment, simulated=False)
    policy = Positions()
    h.observation.emit(1)
    answer = h.caller(Rollout(Task('move', None), policy, h.output_path))
    next(h.loop)
    error = pimm.SignalError('camera lost')
    if lost:
        h.observation.emit(error)
    h.world.clock.advance_to_ns(5_000_000_000)  # no new data for 5 s either way
    next(h.loop)
    episode = h.dataset.episodes[-1]
    if not lost:
        assert not answer.done()
        assert policy.seen == [1, 1]
        assert not episode.closed
        return
    with pytest.raises(pimm.SignalError) as raised:
        answer.result()
    assert raised.value is error
    assert policy.seen == [1]
    assert policy.closed
    assert episode.closed and episode.aborted
    assert h.deadlines.values[-1][1] is None
    assert isinstance(next(h.loop), pimm.Sleep)


def test_an_ask_while_an_observation_carries_an_error_is_refused_and_the_next_one_runs(episode_harness):
    h = episode_harness
    policy = Positions()
    h.observation.emit(pimm.SignalError('camera absent'))
    refused = h.caller(Rollout(Task('move', None), policy, None))
    next(h.loop)
    with pytest.raises(pimm.SignalError, match='camera absent'):
        refused.result()
    assert policy.seen == []

    h.observation.emit(1)
    accepted = h.caller(Rollout(Task('move', None), policy, None))
    next(h.loop)
    assert policy.seen == [1]
    assert not accepted.done()


def test_each_ask_refused_on_one_error_starts_a_new_traceback(episode_harness):
    h = episode_harness
    error = pimm.SignalError('camera absent')
    h.observation.emit(error)
    depths = []
    for _ in range(3):
        refused = h.caller(Rollout(Task('move', None), Positions(), None))
        next(h.loop)
        with pytest.raises(pimm.SignalError):
            refused.result()
        depths.append(len(traceback.extract_tb(error.__traceback__)))
    assert depths[0] == depths[-1]


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
        harness.commands[MOTOR]._bind(commands, clock=world.clock)
        harness.deadline_ns._bind(Trace(world.clock), clock=world.clock)
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
            with pytest.raises(ValueError, match='model failed'):
                answer.result()
            calls.append(runtime.time_ns)
            while True:
                yield Step({}, runtime.time_ns + 1_000_000_000)

    with policy_world(Waiting()) as (_, loop, _):
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
        motion.position._bind(observations, clock=world.clock)
        commands = Trace(world.clock, world.pair(motion.command))
        harness.commands[MOTOR]._bind(commands, clock=world.clock)
        deadlines = Trace(world.clock)
        harness.deadline_ns._bind(deadlines, clock=world.clock)
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
            start_ns = deadlines.values[0][0]
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
    if record:
        h.dataset_factory.assert_called_once_with(output)
    else:
        h.dataset_factory.assert_not_called()
    h.done.emit({eval_keys.SUCCESS: True})
    next(h.loop)
    next(h.loop)
    assert answer.result()[eval_keys.TERMINATED] is True
    if not record:
        assert h.dataset.episodes == []
        return
    meta = h.dataset.episodes[-1].static
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
        meta = h.dataset.episodes[-1].static
        assert meta['inference.policy.config.name'] == 'record'
        assert meta['inference.policy.config.status'] == 'active'
        assert meta['inference.policy.events'] == ['started']


def test_preparation_precedes_budget_and_return_skips_scene(episode_harness):
    h = episode_harness
    task = Task('move', 0.01, prepare_args={RESET: 'home', eval_keys.SCENE: 42})
    answer = h.caller(Rollout(task, Hold(), None))
    next(h.loop)
    assert h.deadlines.values == h.dataset.episodes == []
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
        next(h.loop)
        with pytest.raises(ValueError, match='unknown'):
            answer.result()
    else:
        next(h.loop)
        call = next(h.prepare[RESET].incoming())
        if failure == 'prepare':
            call.set_exception(ValueError('prepare failed'))
            next(h.loop)
            with pytest.raises(ValueError, match='prepare failed'):
                answer.result()
            h.caller(Rollout(Task('move', 0.01, prepare_args={RESET: 'again'}), Hold(), None))
            next(h.loop)
            assert next(h.prepare[RESET].incoming()).request == 'again'
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
    assert h.dataset.episodes == []


def test_every_device_is_asked_to_be_ready_before_each_episode_and_before_its_preparation(ready_harness):
    h = ready_harness
    task = Task('move', 0.01, prepare_args={RESET: 'home'})
    for episode in range(2):
        answer = h.caller(Rollout(task, Hold(), None))
        next(h.loop)
        asked = [next(handler.incoming()) for handler in h.ready.values()]
        assert [call.request for call in asked] == [None, None]
        assert list(h.prepare[RESET].incoming()) == []
        for call in asked:
            call.set_result(None)
        next(h.loop)
        next(h.prepare[RESET].incoming()).set_result(None)
        start_ns = (episode + 1) * 1_000_000_000
        h.world.clock.advance_to_ns(start_ns)
        next(h.loop)
        h.world.clock.advance_to_ns(start_ns + 10_000_000)
        next(h.loop)
        next(h.loop)
        next(h.prepare[RESET].incoming()).set_result(None)
        next(h.loop)
        assert answer.result() == {eval_keys.TERMINATED: False}


@pytest.mark.parametrize('error', [pimm.SignalError('camera lost'), RuntimeError('the arm holds a fault')])
def test_a_device_that_answers_ready_with_an_error_fails_the_ask_and_is_asked_again(ready_harness, error):
    h = ready_harness
    task = Task('move', None, prepare_args={RESET: 'home'})
    refused = h.caller(Rollout(task, Hold(), None))
    next(h.loop)
    next(h.ready[CAMERA].incoming()).set_exception(error)
    next(h.ready[ARM].incoming()).set_result(None)
    next(h.loop)
    with pytest.raises(type(error)) as raised:
        refused.result()
    assert raised.value is error
    assert list(h.prepare[RESET].incoming()) == []
    assert h.dataset.episodes == []

    h.caller(Rollout(task, Hold(), None))
    next(h.loop)
    assert [call.request for handler in h.ready.values() for call in handler.incoming()] == [None, None]


def _ready_calls(h) -> dict[str, int]:
    return {name: len(list(handler.incoming())) for name, handler in h.ready.items()}


def test_an_idle_device_in_error_is_asked_to_be_ready_with_no_ask_until_it_repairs(repair_harness):
    h = repair_harness
    h.device_observations[CAMERA].emit(pimm.SignalError('camera lost'))
    next(h.loop)
    first = next(h.ready[CAMERA].incoming())
    next(h.loop)
    assert _ready_calls(h) == {CAMERA: 0, ARM: 0}  # one call at a time

    first.set_exception(pimm.SignalError('camera did not open'))
    next(h.loop)
    second = next(h.ready[CAMERA].incoming())  # the error stays, so the harness asks again

    h.device_observations[CAMERA].emit(np.array([1]))
    second.set_result(None)
    next(h.loop)
    next(h.loop)
    assert _ready_calls(h) == {CAMERA: 0, ARM: 0}


def test_a_healthy_idle_device_is_not_asked_to_be_ready(repair_harness):
    h = repair_harness
    h.device_observations[CAMERA].emit(np.array([1]))
    for _ in range(3):
        next(h.loop)
    assert _ready_calls(h) == {CAMERA: 0, ARM: 0}


def test_an_idle_error_on_an_observation_with_no_ready_handler_of_its_name_asks_no_device(repair_harness):
    h = repair_harness
    h.device_observations[CAMERA].emit(np.array([1]))
    h.observation.emit(pimm.SignalError('no position'))
    for _ in range(3):
        next(h.loop)
    assert _ready_calls(h) == {CAMERA: 0, ARM: 0}


def test_idle_manual_commands_pass_through(episode_harness):
    h = episode_harness
    h.manual.emit({MOTOR: 3})
    next(h.loop)
    assert h.commands.values == [(0, 3)]


@pytest.mark.parametrize('ending', ['done', 'shutdown', 'failure'])
def test_episode_spans_include_reset_and_recorder_flush(episode_harness, tmp_path, ending):
    h = episode_harness
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'test-episode'):
        h.caller(Rollout(Task('move', None), Hold(), h.output_path))
        next(h.loop)
        h.world.clock.advance_to_ns(100_000_000)
        if ending == 'failure':
            h.observation.emit(1)
            h.serializer.side_effect = ValueError('failed')
            next(h.loop)
        else:
            if ending == 'shutdown':
                h.world.request_stop()
            else:
                h.done.emit({eval_keys.SUCCESS: True})
            if ending == 'shutdown':
                list(h.loop)
            else:
                next(h.loop)
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
        wire.wire_embodiment(world, harness, embodiment)
        caller = world.pair(harness.perform_task)
        loop = world.start([harness, motion])
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
    np.testing.assert_array_equal(np.diff(commands.timestamps(EMITTED_WORLD)), [100_000_000, 100_000_000])
    positions = episode[POSITION]
    np.testing.assert_array_equal(np.diff(positions.timestamps(EMITTED_WORLD)), 2_000_000)
    recorded = dict(zip(positions.timestamps(EMITTED_WORLD), positions.values(), strict=True))
    assert recorded
    assert all(recorded[ns] == value for ns, value in motion.positions if ns in recorded)
    assert 1 in np.diff(list(positions.values()))
    assert 2 in np.diff(list(positions.values()))
    schedule = f'{policy_keys.POLICY_META}.{eval_keys.SCHEDULE}'
    assert episode.static[f'{schedule}.{eval_keys.SCHEDULED}'] == 4
    assert episode.static[f'{schedule}.{eval_keys.EMITTED}'] == 3
    assert episode.static[f'{schedule}.{eval_keys.DROPPED}'] == 0


def test_recorder_refuses_an_encoder_this_host_cannot_run():
    class AbsentEncoder(LibavEncoder):
        def ensure_available(self) -> None:
            raise RuntimeError('no such encoder here')

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
        run_world(embodiment, Passive())


def test_a_command_without_a_device_records_simulation_timestamps(tmp_path):
    class Move(Policy):
        def run(self, runtime):
            yield
            while True:
                yield Step({MOTOR: 1}, runtime.time_ns + 100_000_000)

    timer = Passive()
    embodiment = Embodiment(
        descriptor='no-device',
        observations={},
        commands={MOTOR: Command(pimm.FakeReceiver(timer), None)},
        prepare_handlers={},
        static_meta={},
        meta_source=None,
        simulated=True,
    )
    harness = Harness(embodiment)
    with pimm.World(virtual_time=True) as world:
        wire.wire_embodiment(world, harness, embodiment)
        caller = world.pair(harness.perform_task)
        loop = world.start([harness, timer])
        answer = caller(Rollout(Task('move', 0.005), Move(), tmp_path))
        try:
            while not answer.done():
                next(loop)
            answer.result()
        finally:
            world.request_stop()
            list(loop)
    command = LocalDataset(tmp_path)[0][MOTOR]
    assert list(command.timestamps(EMITTED_WORLD)) == [0]
    assert set(command.timelines) == {EMITTED_WALL, EMITTED_WORLD, HARNESS_WALL, HARNESS_WORLD}


def test_policy_completion_resets_the_sampling_deadline(episode_harness):
    h = episode_harness
    h.harness._embodiment = replace(h.embodiment, simulated=False)
    release = threading.Event()
    answers = []
    observations = []

    class Await(Policy):
        def run(self, runtime):
            obs = yield
            answers.append(runtime.submit(release.wait))
            while True:
                observations.append(obs[POSITION])
                if answers[0].done():
                    answers[0].result()
                obs = yield Step({}, runtime.time_ns + 100_000_000)

    h.observation.emit(0)
    h.caller(Rollout(Task('sample', None), Await(), h.output_path))
    try:
        assert next(h.loop) == pimm.Sleep(0.001)
        h.world.clock.advance_to_ns(800_000)
        h.observation.emit(1)
        release.set()
        cast(_UnchargedAnswer, answers[0]).call.result(timeout=5)
        assert next(h.loop) == pimm.Sleep(0.001)
        assert observations == [0, 1]

        h.world.clock.advance_to_ns(1_000_000)
        h.observation.emit(2)
        assert next(h.loop) == pimm.Sleep(0.0008)
        assert h.serializer.call_count == 2

        h.world.clock.advance_to_ns(1_800_000)
        assert next(h.loop) == pimm.Sleep(0.001)
        assert h.serializer.call_count == 3
        assert observations == [0, 1]
        assert [data for name, data, _ in h.dataset.episodes[0].records if name == POSITION] == [0, 1, 2]
    finally:
        release.set()


def test_sampling_records_privileged_data_without_exposing_it_to_the_policy():
    with _episode_ports(privileged=(GROUND_TRUTH,)) as h:
        policy = Positions()
        h.observation.emit(0)
        h.privileged[GROUND_TRUTH].emit(np.array([10]))
        answer = h.caller(Rollout(Task('sample', None), policy, h.output_path))
        next(h.loop)
        episode = h.dataset.episodes[0]
        assert [name for name, _, _ in episode.records] == [POSITION, GROUND_TRUTH]

        h.world.clock.advance_to_ns(1_000_000)
        frame = np.array([20])
        h.observation.emit(1)
        h.privileged[GROUND_TRUTH].emit(frame)
        next(h.loop)
        assert policy.seen == [0]
        frame[0] = 99
        h.world.clock.advance_to_ns(100_000_000)
        next(h.loop)
        assert policy.seen == [0, 1]
        assert h.serializer.call_count == 2

        h.world.clock.advance_to_ns(101_000_000)
        h.privileged[GROUND_TRUTH].emit(np.array([30]))
        h.done.emit({eval_keys.SUCCESS: True})
        next(h.loop)
        assert answer.done() and episode.closed
        assert all(GROUND_TRUTH not in obs for obs in policy.observations)
        assert [data for name, data, _ in episode.records if name == POSITION] == [0, 1]
        assert [data.tolist() for name, data, _ in episode.records if name == GROUND_TRUTH] == [[10], [20], [30]]
        for _, _, timestamps in episode.records:
            assert timestamps[HARNESS_WORLD] == timestamps[RECEIVED_WORLD]
            assert timestamps[HARNESS_WALL] == timestamps[RECEIVED_WALL]


def test_disabled_recording_does_not_read_privileged_data_or_add_sampling_wakeups():
    with _episode_ports(privileged=(GROUND_TRUTH,)) as h:
        h.harness._embodiment = replace(h.embodiment, simulated=False)
        h.harness.privileged[GROUND_TRUTH].read = Mock(side_effect=AssertionError('unexpected read'))
        h.observation.emit(0)
        h.caller(Rollout(Task('move', None), Hold(), None))
        assert next(h.loop) == pimm.Sleep(0.1)
        h.dataset_factory.assert_not_called()


def test_every_command_is_recorded_when_completions_reenter_at_one_simulated_instant(episode_harness, tmp_path):
    h = episode_harness
    h.harness._dataset_factory = LocalDatasetWriter

    class Chain(Policy):
        def run(self, runtime):
            yield
            for value in range(5):
                answer = runtime.submit(lambda: None)
                yield Step({MOTOR: value}, runtime.time_ns + 100_000_000)
                answer.result()
            while True:
                yield Step({}, runtime.time_ns + 100_000_000)

    h.observation.emit(0)
    answer = h.caller(Rollout(Task('move', None, charge_inference_time=False), Chain(), tmp_path))
    next(h.loop)
    assert h.world.clock.now_ns() == 0
    h.done.emit({eval_keys.SUCCESS: True})
    next(h.loop)
    assert answer.result() == {eval_keys.SUCCESS: True, eval_keys.TERMINATED: True}
    episode = LocalDataset(tmp_path)[0]
    commands = episode[MOTOR]
    assert list(commands.values()) == list(range(5))
    assert set(commands.timelines) == {EMITTED_WALL, EMITTED_WORLD, HARNESS_WALL, HARNESS_WORLD}
    assert list(commands.timestamps(EMITTED_WORLD)) == [0] * 5
    assert np.all(np.diff(commands.timestamps(EMITTED_WALL)) > 0)
    assert len(episode[POSITION]) == 1


def test_inputs_are_written_before_the_policy_and_commands_after_emission(episode_harness):
    h = episode_harness
    commands = np.array([0])
    calls = []

    class Move(Policy):
        def run(self, runtime):
            obs = yield
            while True:
                episode = h.dataset.episodes[0]
                assert episode.records[-1][0] == POSITION
                assert episode.records[-1][1][0] == obs[POSITION][0]
                commands[0] += 1
                calls.append(obs)
                obs = yield Step({MOTOR: commands}, runtime.time_ns + 5_000_000)

    frame = np.array([10])
    h.observation.emit(frame)
    h.caller(Rollout(Task('move', None), Move(), h.output_path))
    next(h.loop)
    episode = h.dataset.episodes[0]
    append = episode.append

    def check_delivery(name, value, timestamps):
        if name == MOTOR:
            np.testing.assert_array_equal(h.commands.values[-1][1], value)
        append(name, value, timestamps)

    episode.append = check_delivery
    frame[0] = 20
    h.world.clock.advance_to_ns(5_000_000)
    h.observation.emit(frame)
    next(h.loop)
    assert [data.tolist() for _, data, _ in episode.records] == [[10], [1], [20], [2]]
    for name, _, ts in episode.records:
        expected = {EMITTED_WALL, EMITTED_WORLD, HARNESS_WALL, HARNESS_WORLD}
        if name == POSITION:
            expected |= {RECEIVED_WALL, RECEIVED_WORLD}
            assert ts[HARNESS_WORLD] == ts[RECEIVED_WORLD]
            assert ts[HARNESS_WALL] == ts[RECEIVED_WALL]
        else:
            assert ts[HARNESS_WORLD] == ts[EMITTED_WORLD]
            assert ts[HARNESS_WALL] == ts[EMITTED_WALL]
        assert set(ts) == expected


def test_recorded_arrays_do_not_change_when_the_policy_mutates_inputs(episode_harness, tmp_path):
    h = episode_harness
    h.harness._dataset_factory = LocalDatasetWriter
    observations = []

    class Normalize(Policy):
        def run(self, runtime):
            obs = yield
            while True:
                observations.append(obs[POSITION])
                obs[POSITION][:] = 99
                obs = yield Step({}, runtime.time_ns + 5_000_000)

    frame = np.array([10])
    h.observation.emit(frame)
    answer = h.caller(Rollout(Task('move', None), Normalize(), tmp_path))
    next(h.loop)
    frame[:] = 20
    h.world.clock.advance_to_ns(5_000_000)
    h.observation.emit(frame)
    next(h.loop)
    h.done.emit({eval_keys.SUCCESS: True})
    next(h.loop)
    assert answer.result()[eval_keys.TERMINATED]
    np.testing.assert_array_equal(observations, [[99], [99]])
    np.testing.assert_array_equal(LocalDataset(tmp_path)[0][POSITION].values(), [[10], [20]])


def test_recording_keeps_first_receipt_times_and_does_not_change_messages(episode_harness):
    h = episode_harness
    h.observation.emit(1)
    h.world.clock.advance_to_ns(1_000_000)
    message = h.harness.observations[POSITION].read()
    original_time = message.time
    h.world.clock.advance_to_ns(2_000_000)
    h.caller(Rollout(Task('move', None), Hold(), h.output_path))
    next(h.loop)
    h.world.clock.advance_to_ns(3_000_000)
    next(h.loop)
    records = h.dataset.episodes[0].records
    assert len(records) == 1
    _, _, timestamps = records[0]
    assert timestamps[HARNESS_WORLD] == original_time[RECEIVED_WORLD] == 1_000_000
    assert timestamps[HARNESS_WALL] == original_time[RECEIVED_WALL]
    assert message.time is original_time
    assert HARNESS_WORLD not in message.time and HARNESS_WALL not in message.time


@pytest.mark.parametrize('timeline', [HARNESS_WORLD, HARNESS_WALL])
def test_producer_cannot_overwrite_harness_recording_timestamps(episode_harness, timeline):
    h = episode_harness
    h.observation.emit(1, time=pimm.Time(**{timeline: 123}))
    answer = h.caller(Rollout(Task('move', None), Hold(), h.output_path))
    next(h.loop)
    with pytest.raises(TypeError, match=timeline):
        answer.result()
    assert h.dataset.episodes[0].aborted


@pytest.mark.parametrize('failure', ['append', 'metadata', 'close'])
def test_writer_failure_reaches_the_episode_caller(episode_harness, failure, monkeypatch):
    h = episode_harness
    if failure == 'close':
        monkeypatch.setattr(MemoryEpisode, '__exit__', Mock(side_effect=OSError('disk failed')))
    h.observation.emit(0)
    answer = h.caller(Rollout(Task('fail', None), Hold(), h.output_path))
    next(h.loop)
    episode = h.dataset.episodes[0]
    if failure == 'append':
        episode.append = Mock(side_effect=OSError('disk failed'))
        h.world.clock.advance_to_ns(1_000_000)
        h.observation.emit(1)
    else:
        if failure == 'metadata':
            episode.set_static = Mock(side_effect=OSError('disk failed'))
        h.done.emit({eval_keys.SUCCESS: True})
    next(h.loop)
    with pytest.raises(OSError, match='disk failed'):
        answer.result()
    if failure != 'close':
        assert episode.closed and episode.aborted
    assert h.deadlines.values[-1][1] is None


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
        harness.commands[keys.ROBOT_COMMAND]._bind(emitted, clock=world.clock)
        harness.commands[keys.TARGET_GRIP]._bind(Trace(world.clock), clock=world.clock)
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


def test_a_missing_observation_is_read_again_after_one_poll_period(episode_harness):
    h = episode_harness
    h.harness._embodiment = replace(h.embodiment, simulated=False)
    observed_at = []

    class Record(Policy):
        def run(self, runtime):
            yield
            while True:
                observed_at.append(runtime.time_ns)
                yield Step({}, runtime.time_ns + 10**9)

    h.caller(Rollout(Task('test', 1.0), Record(), None))
    wake = next(h.loop)
    assert wake.seconds == pytest.approx(POLL_PERIOD_SEC)
    h.observation.emit(1)
    h.world.clock.advance_to_ns(100_000_000)
    next(h.loop)
    assert observed_at == [100_000_000]


def test_a_missing_observation_still_ends_the_episode_at_its_deadline(episode_harness):
    h = episode_harness
    h.harness._embodiment = replace(h.embodiment, simulated=False)

    answer = h.caller(Rollout(Task('test', 0.05), Hold(), None))
    wake = next(h.loop)
    assert wake.seconds == pytest.approx(0.05)
    h.world.clock.advance_to_ns(50_000_000)
    next(h.loop)
    assert h.deadlines.values[-1] == (50_000_000, None)
    next(h.loop)
    assert answer.result() == {eval_keys.TERMINATED: False}
