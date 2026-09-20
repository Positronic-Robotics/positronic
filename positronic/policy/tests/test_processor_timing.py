"""Timing contracts for processor execution and simulated chunk playback."""

import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

import pimm
from pimm.tests.testing import Passive, wire_call
from pimm.world import VirtualClock
from positronic import keys
from positronic.dataset.ds_writer_agent import DsWriterCommandType
from positronic.eval import Command, Embodiment, Observation, Task
from positronic.eval import keys as eval_keys
from positronic.policy import executor as executor_module
from positronic.policy.base import Policy, Sequential, Step
from positronic.policy.executor import Executor, _UnchargedAnswer
from positronic.policy.harness import Harness, Rollout
from positronic.policy.layers import ChunkedSchedule, StopOnFault

MOTOR = 'motor'
POSITION = 'position'
RESET = 'reset'


@pytest.fixture
def execution(monkeypatch):
    with pimm.World(virtual_time=True) as world:
        runtime = Executor(world.clock.now_ns, simulated=True, charge_inference_time=False)

        def submit(function, *args, **kwargs):
            future = Future()
            future.set_result(function(*args, **kwargs))
            return _UnchargedAnswer(future)

        monkeypatch.setattr(runtime, 'submit', submit)
        try:
            yield runtime, cast(VirtualClock, world.clock), world.should_stop_reader()
        finally:
            runtime.close()


def test_ready_answer_emits_without_an_initial_yield(execution, monkeypatch):
    runtime, _, _ = execution
    future = Future()
    future.set_result([{MOTOR: 1}])
    monkeypatch.setattr(runtime, 'submit', lambda *args: _UnchargedAnswer(future))
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [])
    assert run.send({}) == Step({MOTOR: 1}, 100_000_000)
    run.close()


def test_last_action_keeps_its_full_period(execution):
    runtime, clock, stop = execution
    requests = []

    def infer(obs):
        requests.append(runtime.time_ns)
        return [{MOTOR: i} for i in (1, 2, 3)]

    run = runtime.start(ChunkedSchedule(fps=10), infer)
    for index in range(4):
        clock.advance_to_ns(index * 100_000_000)
        step = run.send({})
        assert step == Step({MOTOR: index % 3 + 1}, (index + 1) * 100_000_000)
    assert requests == [0, 300_000_000]
    run.close()


def test_late_wake_merges_due_channels_and_keeps_absolute_deadlines(execution):
    runtime, clock, stop = execution
    chunk = [{MOTOR: 1}, {POSITION: 2}, {MOTOR: 3}, {MOTOR: 4}]
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: chunk)
    run.send({})
    clock.advance_to_ns(225_000_000)
    assert run.send({}) == Step({POSITION: 2, MOTOR: 3}, 300_000_000)
    run.close()


def test_horizon_cuts_commands_but_preserves_boundary(execution):
    runtime, clock, stop = execution
    run = runtime.start(ChunkedSchedule(fps=10, horizon_sec=0.15), lambda obs: [{MOTOR: i} for i in range(5)])
    assert run.send({}) == Step({MOTOR: 0}, 100_000_000)
    clock.advance_to_ns(100_000_000)
    assert run.send({}) == Step({MOTOR: 1}, 150_000_000)
    clock.advance_to_ns(150_000_000)
    assert run.send({}) == Step({MOTOR: 0}, 250_000_000)
    run.close()


def test_empty_chunks_request_an_immediate_retry(execution):
    runtime, clock, stop = execution
    requests = []

    def infer(obs):
        requests.append(runtime.time_ns)
        return []

    run = runtime.start(ChunkedSchedule(fps=10), infer)
    for now in (0, 0):
        clock.advance_to_ns(now)
        assert run.send({}) == Step({}, now)
    assert requests == [0, 0]
    run.close()


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
        step = partial(harness._step, Task('test', None), runtime, policy_run)
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


@pytest.fixture
def episode_harness():
    with pimm.World(virtual_time=True) as world:
        source = Passive()
        serializer = Mock(side_effect=lambda value: value)
        embodiment = Embodiment(
            descriptor='test',
            observations={POSITION: Observation(pimm.ControlSystemEmitter(source), serializer)},
            commands={MOTOR: Command(pimm.ControlSystemReceiver(source), None)},
            prepare_handlers={},
            static_meta={},
            meta_source=None,
            simulated=True,
        )
        harness = Harness(embodiment)
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

            return Sequential(StopOnFault(), ChunkedSchedule(fps=15, horizon_sec=1.0)).run(runtime, infer)

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
        answer = caller(Rollout(Task('move', 2.0, prepare_args={RESET: None} if prepare else {}), definition, None))
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
            assert requests[1][1] == (3995 if prepare else 3994)
            assert dict(motion.positions)[start_ns + 1_000_000_000] == (3996 if prepare else 3995)
        finally:
            world.request_stop()
            list(loop)
