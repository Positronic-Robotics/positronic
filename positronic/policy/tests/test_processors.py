"""Scheduling, fault gating, temporal history, and deliverable stack specifications."""

import json
from concurrent.futures import Future
from datetime import date
from sys import maxsize
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from positronic_model_server.spec import ARGS, LATEST, NAME, VERSION, component, parallel, sequence

import pimm
from pimm.world import VirtualClock
from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianPosition, JointPosition, PositionControl
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame
from positronic.eval import keys as eval_keys
from positronic.geom import Transform3D
from positronic.policy import spec
from positronic.policy.base import Answer, Policy, PolicyEnd, Step
from positronic.policy.codecs import (
    ACTION,
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    EncodeImages,
    EndCommand,
    FlipGrip,
    Metadata,
    Resample,
    ResizeMode,
    TransformImages,
)
from positronic.policy.codecs.action import (
    AbsoluteJointsAction,
    AbsolutePositionAction,
    CartesianDeltaAction,
    JointDeltaAction,
    JointPositionAction,
    SetControlMode,
    SplitActionChunk,
)
from positronic.policy.codecs.geometry import ConvertPose
from positronic.policy.codecs.observation import (
    PackFields,
    RenameObservationFields,
    SelectObservationFields,
    StateObservationCodec,
)
from positronic.policy.compatibility import ObservationCodecV1
from positronic.policy.executor import Executor, _UnchargedAnswer
from positronic.policy.keys import END, RTC
from positronic.policy.processors import (
    ChunkedSchedule,
    ConstantDuration,
    DelayEstimate,
    DurationEstimate,
    DurationEstimateRun,
    EndOnField,
    PauseOnUnavailable,
    RTCSchedule,
    TemporalStack,
)
from positronic.policy.sequential import Sequential
from positronic.utils.versions import Deprecation, Version

MOTOR = 'motor'
POSITION = 'position'


class Echo(Policy):
    def run(self, runtime):
        obs = yield
        while True:
            obs = yield Step(obs, runtime.time_ns + 100_000_000)


@pytest.mark.parametrize('status', [RobotStatus.ERROR, RobotStatus.BUSY])
@pytest.mark.parametrize('channel', [keys.ROBOT_STATUS, 'robot_state.left.status', 'robot_state.right.status'])
def test_unavailability_withholds_commands_and_resumes_on_recovery(execution, status, channel):
    runtime, _ = execution
    inner = Mock(send=Mock(side_effect=lambda obs: Step(obs, 100_000_000)))
    run = runtime.start(PauseOnUnavailable(), inner)
    try:
        obs = {keys.ROBOT_STATUS: RobotStatus.AVAILABLE, channel: status}
        assert run.send(obs) == Step({}, 1_000_000)
        inner.send.assert_not_called()
        obs[channel] = RobotStatus.AVAILABLE
        assert run.send(obs) == Step(obs, 100_000_000)
        assert run.send({MOTOR: 1}) == Step({MOTOR: 1}, 100_000_000)
    finally:
        run.close()


@pytest.mark.parametrize('wrapper', [PauseOnUnavailable(), TemporalStack((MOTOR,), (0.0,)), EndOnField()])
def test_a_wrapper_returns_what_its_inner_run_returned(execution, wrapper):
    runtime, _ = execution

    def inner():
        yield
        return PolicyEnd.COMPLETE

    child = inner()
    next(child)
    run = runtime.start(wrapper, child)
    with pytest.raises(StopIteration) as end:
        run.send({MOTOR: 1})
    assert end.value.value is PolicyEnd.COMPLETE


@pytest.mark.parametrize('pad_start', [True, False])
def test_temporal_history_padding_sampling_and_fresh_runs(execution, pad_start):
    runtime, clock = execution
    definition = Sequential(TemporalStack((POSITION,), (-0.2, -0.1, 0.0), pad_start), Echo())
    run = runtime.start(definition)
    frame = np.array([1])
    first = run.send({POSITION: frame})
    np.testing.assert_array_equal(first.commands[POSITION][:, 0], [1, 1, 1] if pad_start else [1])
    frame[0] = 2
    clock.advance_to_ns(100_000_000)
    second = run.send({POSITION: frame})
    np.testing.assert_array_equal(second.commands[POSITION][:, 0], [1, 1, 2] if pad_start else [1, 2])
    frame[0] = 3
    clock.advance_to_ns(200_000_000)
    third = run.send({POSITION: frame})
    np.testing.assert_array_equal(third.commands[POSITION][:, 0], [1, 2, 3])
    run.close()
    fresh = runtime.start(definition)
    try:
        reset = fresh.send({POSITION: frame})
        np.testing.assert_array_equal(reset.commands[POSITION][:, 0], [3, 3, 3] if pad_start else [3])
    finally:
        fresh.close()


def test_temporal_history_carries_the_last_sample_before_each_offset(execution):
    runtime, clock = execution
    run = runtime.start(Sequential(TemporalStack((POSITION,), (-0.2, -0.1, 0.0)), Echo()))
    try:
        for ns, value in [(0, 1), (150_000_000, 2), (300_000_000, 3)]:
            clock.advance_to_ns(ns)
            result = run.send({POSITION: np.array([value])})
        np.testing.assert_array_equal(result.commands[POSITION][:, 0], [1, 2, 3])
    finally:
        run.close()


def test_temporal_stack_builds_a_stack_on_the_first_read_of_its_key(execution, monkeypatch):
    runtime, clock = execution
    requests = []
    monkeypatch.setattr(runtime, 'submit', lambda function, obs: requests.append(obs) or _UnchargedAnswer(Future()))
    stacks = []
    stack = np.stack
    monkeypatch.setattr(np, 'stack', lambda arrays: stacks.append(1) or stack(arrays))
    run = runtime.start(Sequential(TemporalStack((POSITION,), (-0.1, 0.0)), ChunkedSchedule(fps=10)), Mock())
    try:
        obs = {}  # One dict for every tick, changed in place, so the request must not read it late.
        for tick in range(4):
            clock.advance_to_ns(tick * 5_000_000)
            obs.update({POSITION: np.array([tick]), keys.TASK: f'tick {tick}'})
            run.send(obs)
        assert len(requests) == 1 and stacks == []
        # Tick 0 sampled this window. Later ticks appended to the buffer and did not change it.
        np.testing.assert_array_equal(requests[0][POSITION][:, 0], [0, 0])
        assert len(stacks) == 1
        assert dict(requests[0]).keys() == {POSITION, keys.TASK} and requests[0][keys.TASK] == 'tick 0'
    finally:
        run.close()


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
            yield runtime, cast(VirtualClock, world.clock)
        finally:
            runtime.close()


def test_ready_answer_emits_without_an_initial_yield(execution, monkeypatch):
    runtime, _ = execution
    future = Future()
    future.set_result([{MOTOR: 1}])
    monkeypatch.setattr(runtime, 'submit', lambda *args: _UnchargedAnswer(future))
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [])
    assert run.send({}) == Step({MOTOR: 1}, 100_000_000)
    run.close()


def test_pending_inference_is_not_resubmitted_on_each_tick(execution, monkeypatch):
    runtime, clock = execution
    future = Future()
    submit = Mock(return_value=_UnchargedAnswer(future))
    monkeypatch.setattr(runtime, 'submit', submit)
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [])
    try:
        for now in (0, 5_000_000, 10_000_000):
            clock.advance_to_ns(now)
            assert run.send({}) == Step({}, now)
        submit.assert_called_once()
        future.set_result([{MOTOR: 1}])
        assert run.send({}) == Step({MOTOR: 1}, 110_000_000)
        submit.assert_called_once()
    finally:
        run.close()


def test_last_action_keeps_its_full_period(execution):
    runtime, clock = execution
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
    runtime, clock = execution
    chunk = [{MOTOR: 1}, {POSITION: 2}, {MOTOR: 3}, {MOTOR: 4}]
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: chunk)
    run.send({})
    clock.advance_to_ns(225_000_000)
    assert run.send({}) == Step({POSITION: 2, MOTOR: 3}, 300_000_000)
    run.close()
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 1


def test_an_overrun_skips_all_but_the_last_due_waypoint_and_counts_the_skip(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [{MOTOR: i} for i in range(5)])
    emitted = []
    for now in (0, 100_000_000, 350_000_000, 400_000_000):
        clock.advance_to_ns(now)
        emitted.append(run.send({}).commands[MOTOR])
    assert emitted == [0, 1, 3, 4]
    prefix = eval_keys.SCHEDULE
    assert runtime.metadata_snapshot() == {
        f'{prefix}.{eval_keys.SCHEDULED}': 5,
        f'{prefix}.{eval_keys.EMITTED}': 4,
        f'{prefix}.{eval_keys.DROPPED}': 1,
        f'{prefix}.{eval_keys.LATE_P50_MS}': 0.0,
        f'{prefix}.{eval_keys.LATE_P90_MS}': pytest.approx(35.0),
        f'{prefix}.{eval_keys.LATE_MAX_MS}': 50.0,
        f'{prefix}.{eval_keys.GAP_MAX_MS}': 250.0,
    }
    run.close()


def test_a_schedule_without_stats_writes_no_metadata(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10, record_stats=False), lambda obs: [{MOTOR: i} for i in range(5)])
    for now in (0, 100_000_000, 350_000_000):
        clock.advance_to_ns(now)
        run.send({})
    run.close()
    assert runtime.metadata_snapshot() == {}


def test_a_new_chunk_counts_the_due_waypoints_it_replaces_as_dropped(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [{MOTOR: i} for i in range(5)])
    emitted = []
    for now in (0, 600_000_000):
        clock.advance_to_ns(now)
        emitted.append(run.send({}).commands[MOTOR])
    run.close()
    assert emitted == [0, 0]
    prefix = eval_keys.SCHEDULE
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.SCHEDULED}'] == 10
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.EMITTED}'] == 2
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.DROPPED}'] == 4


def test_a_new_chunk_excludes_the_gap_from_the_previous_chunk(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [{MOTOR: i} for i in range(3)])
    emitted = []
    try:
        for now in (0, 100_000_000, 300_000_000, 305_000_000):
            clock.advance_to_ns(now)
            emitted.append(run.send({}).commands.get(MOTOR))
    finally:
        run.close()
    assert emitted == [0, 1, 0, None]
    prefix = eval_keys.SCHEDULE
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.DROPPED}'] == 1
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.GAP_MAX_MS}'] == 100.0


@pytest.mark.parametrize('late_ms', [[0], [0, 10], [0, 5, 30, 10, 20], list(range(0, 1000, 7))])
def test_late_percentiles_match_numpy(execution, late_ms):
    # The first waypoint is due when the answer is read, so it is never late.
    runtime, clock = execution
    chunk = [{MOTOR: i} for i in range(len(late_ms))]
    run = runtime.start(ChunkedSchedule(fps=1), lambda obs: chunk)
    for index, late in enumerate(late_ms):
        clock.advance_to_ns(index * 1_000_000_000 + late * 1_000_000)
        run.send({})
    run.close()
    prefix = eval_keys.SCHEDULE
    p50, p90 = np.percentile(late_ms, (50, 90))
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.LATE_P50_MS}'] == pytest.approx(p50)
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.LATE_P90_MS}'] == pytest.approx(p90)
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.LATE_MAX_MS}'] == max(late_ms)


def test_horizon_cuts_commands_but_preserves_boundary(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10, horizon_sec=0.15), lambda obs: [{MOTOR: i} for i in range(5)])
    assert run.send({}) == Step({MOTOR: 0}, 100_000_000)
    clock.advance_to_ns(100_000_000)
    assert run.send({}) == Step({MOTOR: 1}, 150_000_000)
    clock.advance_to_ns(150_000_000)
    assert run.send({}) == Step({MOTOR: 0}, 250_000_000)
    run.close()


def test_empty_chunks_request_an_immediate_retry(execution):
    runtime, clock = execution
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


PERIOD_NS = 100_000_000


class SlowModel:
    """Answers each call only when the test gives the answer, and records each call."""

    def __init__(self, runtime):
        self._runtime = runtime
        self.calls: list[tuple[int, list]] = []
        self.prefixes: list[list] = []
        self.requests: list[dict] = []
        self._pending: Future | None = None

    def submit(self, function, obs):
        self.requests.append(dict(obs))
        prefix = obs[RTC.ACTION_PREFIX][RTC.PREFIX_ACTIONS] if RTC.ACTION_PREFIX in obs else []
        self.calls.append((self._runtime.time_ns, [c.get(MOTOR) for c in prefix]))
        self.prefixes.append(list(prefix))
        self._pending = Future()
        return _UnchargedAnswer(self._pending)

    def answer(self, name, length=10):
        self.answer_with([{MOTOR: f'{name}{i}'} for i in range(length)])

    def answer_with(self, actions):
        assert self._pending is not None
        self._pending.set_result(actions)
        self._pending = None


class Rig:
    """Drives a schedule as ``Harness`` does: a round reads the observation, then sends the step's commands.

    Each observation carries the last command the rig received, so a test can see what a call saw.
    """

    def __init__(self, runtime, clock, schedule):
        self._clock = clock
        self.run = runtime.start(schedule, Mock())
        self.position = '--'
        self.executed: list = []

    def tick(self, now_ns: int) -> Step:
        self._clock.advance_to_ns(now_ns)
        step = self.run.send({POSITION: self.position})
        self.position = step.commands.get(MOTOR, self.position)
        self.executed.append(step.commands.get(MOTOR, '--'))
        return step


class ObservingModel(SlowModel):
    """A ``SlowModel`` that records the last command each call saw."""

    def __init__(self, runtime):
        super().__init__(runtime)
        self.seen: list = []

    def submit(self, function, obs):
        self.seen.append(obs[POSITION])
        return super().submit(function, obs)


@pytest.mark.parametrize('status', list(RobotStatus))
@pytest.mark.parametrize('arm', [None, *keys.BIMANUAL_ARMS])
@pytest.mark.parametrize('replan_on_error', [False, True])
def test_errors_bring_inference_and_playback_forward_when_enabled(execution, status, arm, replan_on_error):
    runtime, clock = execution
    requests = []
    channel = f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.STATUS_SUFFIX}'

    def infer(obs):
        requests.append(dict(obs))
        return [{MOTOR: (len(requests), index)} for index in range(5)]

    run = runtime.start(ChunkedSchedule(fps=10, replan_on_error=replan_on_error), infer)
    try:
        assert run.send({}).commands == {MOTOR: (1, 0)}
        clock.advance_to_ns(PERIOD_NS)
        obs = {keys.ROBOT_STATUS: RobotStatus.AVAILABLE.value, channel: status.value, POSITION: 42}
        step = run.send(obs)
        if replan_on_error and status is RobotStatus.ERROR:
            assert step == Step({MOTOR: (2, 0)}, 2 * PERIOD_NS)
            assert requests == [{}, obs]
            assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 4
        else:
            assert step == Step({MOTOR: (1, 1)}, 2 * PERIOD_NS)
            assert requests == [{}]
    finally:
        run.close()


@pytest.mark.parametrize('initial_status', [RobotStatus.AVAILABLE, RobotStatus.ERROR])
def test_repeated_errors_keep_one_recovery_request_and_play_its_answer(execution, monkeypatch, initial_status):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(ChunkedSchedule(fps=10), Mock())
    try:
        run.send({POSITION: 0, keys.ROBOT_STATUS: initial_status})
        model.answer('a', length=5)
        assert run.send({POSITION: 0, keys.ROBOT_STATUS: initial_status}).commands == {MOTOR: 'a0'}
        fault = {POSITION: 1, keys.ROBOT_STATUS: RobotStatus.ERROR}
        for now_ns in (PERIOD_NS, PERIOD_NS + 1, 2 * PERIOD_NS):
            clock.advance_to_ns(now_ns)
            step = run.send(fault)
            assert step.commands == {}
            assert len(model.calls) == 2
        assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 4
        assert model.seen == [0, 1]
        model.answer('b', length=5)
        assert run.send(fault) == Step({MOTOR: 'b0'}, 3 * PERIOD_NS)
        assert len(model.calls) == 2
        clock.advance_to_ns(3 * PERIOD_NS)
        run.send(fault)
        assert len(model.calls) == 3
    finally:
        run.close()


@pytest.mark.parametrize('ready_at_error', [False, True])
def test_errors_keep_an_existing_request_and_execute_its_answer_immediately(execution, monkeypatch, ready_at_error):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.3)), Mock())
    try:
        run.send({POSITION: 0})
        model.answer('a', length=5)
        run.send({POSITION: 0})
        clock.advance_to_ns(2 * PERIOD_NS)
        run.send({POSITION: 2})
        assert len(model.calls) == 2
        if ready_at_error:
            model.answer('b')
        clock.advance_to_ns(250_000_000)
        fault = {POSITION: 25, keys.ROBOT_STATUS: RobotStatus.ERROR}
        step = run.send(fault)
        if not ready_at_error:
            assert step.commands == {}
            assert len(model.calls) == 2
            clock.advance_to_ns(3 * PERIOD_NS)
            assert run.send(fault).commands == {}
            model.answer('b')
            step = run.send(fault)
        assert step.commands == {MOTOR: 'b0'}
        assert model.seen == [0, 2]
        clock.advance_to_ns(4 * PERIOD_NS)
        assert run.send(fault).commands == {}
        assert len(model.calls) == 3
    finally:
        run.close()


def test_errors_replace_a_prefetched_chunk_before_its_playback_starts(execution):
    runtime, clock = execution
    infer = Mock(side_effect=[[{MOTOR: name} for _ in range(5)] for name in ('a', 'prefetched', 'fresh')])
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.3)), infer)
    try:
        run.send({})
        clock.advance_to_ns(2 * PERIOD_NS)
        assert run.send({}).commands == {MOTOR: 'a'}
        assert infer.call_count == 2
        clock.advance_to_ns(250_000_000)
        assert run.send({keys.ROBOT_STATUS: RobotStatus.ERROR}).commands == {MOTOR: 'fresh'}
        assert infer.call_count == 3
    finally:
        run.close()


def test_a_failed_request_before_the_error_still_raises(execution, monkeypatch):
    runtime, _ = execution
    future = Future()
    monkeypatch.setattr(runtime, 'submit', Mock(return_value=_UnchargedAnswer(future)))
    run = runtime.start(ChunkedSchedule(fps=10), Mock())
    run.send({})
    future.set_exception(RuntimeError('inference failed'))
    with pytest.raises(RuntimeError, match='inference failed'):
        run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})


@pytest.mark.parametrize('replan_on_error', [False, True])
def test_serialized_error_replanning_preserves_the_behavior(execution, replan_on_error):
    runtime, clock = execution
    definition = ChunkedSchedule(fps=10, replan_on_error=replan_on_error)
    encoded = json.loads(json.dumps(definition.to_spec()))
    assert encoded[VERSION] == 3
    rebuilt = spec.from_spec(encoded)
    assert rebuilt.to_spec() == encoded
    infer = Mock(return_value=[{MOTOR: 1}] * 5)
    run = runtime.start(rebuilt, infer)
    try:
        run.send({})
        clock.advance_to_ns(PERIOD_NS)
        run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})
        assert infer.call_count == (2 if replan_on_error else 1)
    finally:
        run.close()


def test_v2_keeps_its_chunk_and_inference_deadline_on_error(execution):
    runtime, clock = execution
    declared = {NAME: ChunkedSchedule.WIRE_NAME, VERSION: 2, ARGS: {ChunkedSchedule.FPS_ARG: 10}}
    schedule = spec.from_spec(declared)
    assert schedule.to_spec() == ChunkedSchedule(fps=10, replan_on_error=False).to_spec()
    infer = Mock(return_value=[{MOTOR: index} for index in range(5)])
    run = runtime.start(schedule, infer)
    try:
        for tick in range(6):
            clock.advance_to_ns(tick * PERIOD_NS)
            assert run.send({keys.ROBOT_STATUS: RobotStatus.ERROR}).commands == {MOTOR: tick % 5}
        assert infer.call_count == 2
    finally:
        run.close()


def test_a_slow_answer_is_asked_for_when_the_last_waypoint_is_due(execution, monkeypatch):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    rig = Rig(runtime, clock, ChunkedSchedule(fps=10, inference_lead=DelayEstimate.mean(last=1, max_sec=0.1)))
    answers = {2: 'a', 6: 'b', 10: 'c'}
    try:
        for index in range(12):
            if index in answers:
                model.answer(answers[index], length=3)
            rig.tick(index * PERIOD_NS)
    finally:
        rig.run.close()
    # Inference starts on the final waypoint's tick, using the observation read before that command.
    assert [command for command in rig.executed if command != '--'] == ['a0', 'a1', 'a2', 'b0', 'b1', 'b2', 'c0', 'c1']
    assert model.seen == ['--', 'a1', 'b1']
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 0


def test_a_fast_answer_is_asked_for_its_delay_before_the_chunk_ends(execution, monkeypatch):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    rig = Rig(runtime, clock, ChunkedSchedule(fps=10, inference_lead=DelayEstimate.mean(last=1, max_sec=0.1)))
    try:
        rig.tick(0)
        model.answer('a', length=3)
        assert rig.tick(50_000_000) == Step({MOTOR: 'a0'}, 150_000_000)
        assert rig.tick(250_000_000) == Step({MOTOR: 'a2'}, 300_000_000)
        # The chunk ends at 350 ms and the last call took 50 ms, so the next call starts at 300 ms, after a2 went
        # out.
        rig.tick(300_000_000)
        model.answer('b', length=3)
        # The answer is early: the last waypoint keeps its full period.
        assert rig.tick(320_000_000) == Step({}, 350_000_000)
        assert rig.tick(350_000_000) == Step({MOTOR: 'b0'}, 450_000_000)
    finally:
        rig.run.close()
    assert [time_ns for time_ns, _ in model.calls] == [0, 300_000_000]
    assert model.seen == ['--', 'a2']


def test_a_lead_of_the_whole_chunk_calls_again_in_the_round_that_reads_the_answer(execution, monkeypatch):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    # One command for each call, played at once, as the teleop service declares.
    rig = Rig(runtime, clock, ChunkedSchedule(fps=100, inference_lead=ConstantDuration(0.01)))
    try:
        rig.tick(0)
        model.answer('a', length=1)
        assert rig.tick(66_000_000).commands == {MOTOR: 'a0'}
        model.answer('b', length=1)
        assert rig.tick(132_000_000).commands == {MOTOR: 'b0'}
    finally:
        rig.run.close()
    assert [time_ns for time_ns, _ in model.calls] == [0, 66_000_000, 132_000_000]
    assert model.seen == ['--', '--', 'a0']


def test_an_early_answer_keeps_a_late_final_waypoint_and_observes_the_previous_one(execution, monkeypatch):
    runtime, clock = execution
    first = Future()
    answers = iter([first, None])
    seen = []

    def submit(function, obs):
        seen.append(obs[POSITION])
        pending = next(answers)
        if pending is None:
            pending = Future()
            pending.set_result([{MOTOR: f'b{i}'} for i in range(3)])
        return _UnchargedAnswer(pending)

    monkeypatch.setattr(runtime, 'submit', submit)
    rig = Rig(runtime, clock, ChunkedSchedule(fps=10, inference_lead=DelayEstimate.mean(last=1, max_sec=0.1)))
    try:
        rig.tick(0)
        first.set_result([{MOTOR: f'a{i}'} for i in range(3)])
        for now_ns in (200_000_000, 300_000_000, 450_000_000, 500_000_000, 600_000_000, 700_000_000):
            rig.tick(now_ns)
    finally:
        rig.run.close()
    # The late tick requests inference before emitting a2; the ready answer waits until the chunk ends.
    assert seen == ['--', 'a1']
    assert [command for command in rig.executed if command != '--'] == ['a0', 'a1', 'a2', 'b0', 'b1', 'b2']
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 0
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.GAP_MAX_MS}'] == 150.0


@pytest.mark.parametrize('first_new_ms, new_index', [(500, 0), (650, 1)])
def test_prefetched_chunk_resets_gap_when_execution_crosses_its_boundary(execution, first_new_ms, new_index):
    runtime, clock = execution
    calls = []

    def infer(obs):
        calls.append(runtime.time_ns)
        return [{MOTOR: (len(calls), i)} for i in range(5)]

    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.2)), infer)
    try:
        for index in range(5):
            clock.advance_to_ns(index * PERIOD_NS)
            assert run.send({}).commands == {MOTOR: (1, index)}
        clock.advance_to_ns(first_new_ms * 1_000_000)
        assert run.send({}).commands == {MOTOR: (2, new_index)}
    finally:
        run.close()
    assert calls == [0, 300_000_000]
    metadata = runtime.metadata_snapshot()
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.GAP_MAX_MS}'] == 100.0
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.SCHEDULED}'] == 10
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.EMITTED}'] == 6
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == new_index


def test_answer_between_emits_keeps_the_old_chunk_gap_history(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.15)), Mock())
    try:
        run.send({})
        clock.advance_to_ns(100_000_000)
        model.answer('a', length=3)
        assert run.send({}).commands == {MOTOR: 'a0'}
        clock.advance_to_ns(200_000_000)
        assert run.send({}).commands == {MOTOR: 'a1'}
        clock.advance_to_ns(250_000_000)
        assert run.send({}).commands == {}
        clock.advance_to_ns(260_000_000)
        model.answer('b', length=3)
        assert run.send({}).commands == {}
        clock.advance_to_ns(350_000_000)
        assert run.send({}).commands == {MOTOR: 'a2'}
    finally:
        run.close()
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.GAP_MAX_MS}'] == 150.0


def test_an_early_answer_keeps_every_remaining_action_and_wakes_between_waypoints(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.25)), Mock())
    try:
        run.send({})
        clock.advance_to_ns(100_000_000)
        model.answer('a', length=5)
        assert run.send({}) == Step({MOTOR: 'a0'}, 200_000_000)
        clock.advance_to_ns(200_000_000)
        assert run.send({}) == Step({MOTOR: 'a1'}, 300_000_000)
        clock.advance_to_ns(300_000_000)
        assert run.send({}) == Step({MOTOR: 'a2'}, 350_000_000)
        clock.advance_to_ns(350_000_000)
        assert run.send({}) == Step({}, 350_000_000)
        clock.advance_to_ns(360_000_000)
        model.answer('b', length=3)
        assert run.send({}) == Step({}, 400_000_000)
        for now, action, wake in [(400, 'a3', 500), (500, 'a4', 600), (600, 'b0', 650)]:
            clock.advance_to_ns(now * 1_000_000)
            assert run.send({}) == Step({MOTOR: action}, wake * 1_000_000)
    finally:
        run.close()
    assert [time for time, _ in model.calls] == [0, 350_000_000]
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 0


@pytest.mark.parametrize('lead_sec', [0.3, 10.0, 1e308])
def test_a_lead_cannot_queue_more_than_the_next_chunk(execution, lead_sec):
    runtime, clock = execution
    calls = []

    def infer(obs):
        calls.append(runtime.time_ns)
        return [{MOTOR: (len(calls), i)} for i in range(3)]

    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(lead_sec)), infer)
    try:
        assert run.send({}) == Step({MOTOR: (1, 0)}, 0)
        assert run.send({}) == Step({}, PERIOD_NS)
        assert run.send({}) == Step({}, PERIOD_NS)
        assert calls == [0, 0]
        for index in range(1, 9):
            clock.advance_to_ns(index * PERIOD_NS)
            assert run.send({}).commands == {MOTOR: (index // 3 + 1, index % 3)}
        assert calls == [0, 0, 3 * PERIOD_NS, 6 * PERIOD_NS]
    finally:
        run.close()


@pytest.mark.parametrize(
    'lead, call_ns', [(DelayEstimate.mean(last=2), 1_050_000_000), (DelayEstimate.max(last=2), 1_000_000_000)]
)
def test_delay_based_leads_use_completed_calls(execution, monkeypatch, lead, call_ns):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=lead), Mock())
    try:
        run.send({})
        clock.advance_to_ns(100_000_000)
        model.answer('a', length=5)
        run.send({})
        for tick in range(2, 6):
            clock.advance_to_ns(tick * PERIOD_NS)
            run.send({})
        clock.advance_to_ns(700_000_000)
        model.answer('b', length=5)
        run.send({})
        clock.advance_to_ns(call_ns - 1)
        assert run.send({}).resume_at_ns == call_ns
        assert len(model.calls) == 2
        clock.advance_to_ns(call_ns)
        run.send({})
        assert [time for time, _ in model.calls] == [0, 500_000_000, call_ns]
    finally:
        run.close()


class PeakDuration(DurationEstimate):
    class _Run(DurationEstimateRun):
        def __init__(self):
            self._peak = 0.0

        def update(self, delay_sec):
            self._peak = max(self._peak, delay_sec)
            return self._peak

    def run(self):
        return self._Run()


@pytest.mark.parametrize('estimate_factory', [DelayEstimate.mean, PeakDuration])
@pytest.mark.parametrize('rtc', [False, True])
def test_estimator_histories_are_episode_local(execution, monkeypatch, rtc, estimate_factory):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    estimate = estimate_factory()
    estimate.run().update(0.4)
    schedule = (
        RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=estimate)
        if rtc
        else ChunkedSchedule(fps=10, inference_lead=estimate)
    )
    for start_ns, delay_ns in ((0, 300_000_000), (2_000_000_000, 100_000_000)):
        clock.advance_to_ns(start_ns)
        run = runtime.start(schedule, Mock())
        try:
            run.send({})
            clock.advance_to_ns(start_ns + delay_ns)
            model.answer('a')
            run.send({})
            call_ns = start_ns + delay_ns + (500_000_000 if rtc else 1_000_000_000 - delay_ns)
            for now_ns in range(start_ns + delay_ns + PERIOD_NS, call_ns + 1, PERIOD_NS):
                clock.advance_to_ns(now_ns)
                run.send({})
            assert model.calls[-1][0] == call_ns
            if rtc:
                assert model.requests[-1][RTC.ESTIMATED_LATENCY_SEC] == pytest.approx(delay_ns / 1e9)
                assert len(model.calls[-1][1]) == 5
        finally:
            run.close()


def test_an_empty_chunk_resumes_inference_without_using_the_lead_as_a_delay(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(1)), Mock())
    try:
        run.send({})
        clock.advance_to_ns(PERIOD_NS)
        model.answer('a', length=0)
        assert run.send({}) == Step({}, PERIOD_NS)
        run.send({})
        assert [time for time, _ in model.calls] == [0, PERIOD_NS]
    finally:
        run.close()


def test_an_early_empty_answer_retries_while_the_old_chunk_keeps_running(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.25)), Mock())
    try:
        run.send({})
        clock.advance_to_ns(100_000_000)
        model.answer('a', length=5)
        assert run.send({}).commands == {MOTOR: 'a0'}
        for tick in (2, 3):
            clock.advance_to_ns(tick * PERIOD_NS)
            assert run.send({}).commands == {MOTOR: f'a{tick - 1}'}
        clock.advance_to_ns(350_000_000)
        run.send({})
        clock.advance_to_ns(360_000_000)
        model.answer('empty', length=0)
        assert run.send({}) == Step({}, 360_000_000)
        run.send({})
        clock.advance_to_ns(370_000_000)
        model.answer('b', length=5)
        assert run.send({}) == Step({}, 400_000_000)
        for now, action in ((400, 'a3'), (500, 'a4'), (600, 'b0')):
            clock.advance_to_ns(now * 1_000_000)
            assert run.send({}).commands == {MOTOR: action}
    finally:
        run.close()
    assert [time for time, _ in model.calls] == [0, 350_000_000, 360_000_000]
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 0


def test_lead_and_handoff_use_the_horizon_truncated_end(execution):
    runtime, clock = execution
    calls = []

    def infer(obs):
        calls.append(runtime.time_ns)
        return [{MOTOR: i} for i in range(5)]

    schedule = ChunkedSchedule(fps=10, horizon_sec=0.15, inference_lead=ConstantDuration(0.1))
    run = runtime.start(schedule, infer)
    try:
        assert run.send({}) == Step({MOTOR: 0}, 50_000_000)
        clock.advance_to_ns(50_000_000)
        assert run.send({}) == Step({}, 100_000_000)
        clock.advance_to_ns(100_000_000)
        assert run.send({}) == Step({MOTOR: 1}, 150_000_000)
        clock.advance_to_ns(150_000_000)
        assert run.send({}) == Step({MOTOR: 0}, 200_000_000)
        assert calls == [0, 50_000_000]
    finally:
        run.close()


def _rtc(execution, monkeypatch, call_after_sec, latency_estimate):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = RTCSchedule(fps=10, call_after_sec=call_after_sec, latency_estimate=latency_estimate)
    run = runtime.start(schedule, Mock())

    def step(index):
        clock.advance_to_ns(index * PERIOD_NS)
        return run.send({})

    return model, run, step


def test_rtc_executes_old_plan_until_the_answer_then_skips_elapsed_new_actions(execution, monkeypatch):
    model, run, step = _rtc(execution, monkeypatch, 0.5, DelayEstimate.mean())
    answers = {3: 'o', 11: 'n'}
    executed = []
    try:
        for index in range(18):
            if index in answers:
                model.answer(answers[index])
            result = step(index)
            executed.append(result.commands.get(MOTOR, '--'))
            if index == 3:
                assert result.resume_at_ns == 4 * PERIOD_NS
    finally:
        run.close()
    assert executed == ['--'] * 3 + [f'o{i}' for i in range(8)] + [f'n{i}' for i in range(3, 10)]
    assert model.calls == [
        (0, []),
        (8 * PERIOD_NS, [f'o{i}' for i in range(5, 10)]),
        (13 * PERIOD_NS, [f'n{i}' for i in range(5, 10)]),
    ]


def test_rtc_executes_nothing_between_the_old_chunk_end_and_a_slow_answer(execution, monkeypatch):
    model, run, step = _rtc(execution, monkeypatch, 0.2, DelayEstimate.mean())
    answers = {1: ('o', 4), 8: ('n', 10)}
    executed = []
    try:
        for index in range(10):
            if index in answers:
                model.answer(*answers[index])
            executed.append(step(index).commands.get(MOTOR, '--'))
    finally:
        run.close()
    assert executed == ['--', 'o0', 'o1', 'o2', 'o3', '--', '--', '--', 'n5', 'n6']
    assert model.calls == [(0, []), (3 * PERIOD_NS, ['o2', 'o3']), (8 * PERIOD_NS, [f'n{i}' for i in range(5, 10)])]


@pytest.mark.parametrize('estimate', [DelayEstimate.mean(), DelayEstimate.max(), ConstantDuration(0.1)])
def test_rtc_sends_the_remaining_plan_independently_of_its_latency_estimate(execution, monkeypatch, estimate):
    model, run, step = _rtc(execution, monkeypatch, 0.2, estimate)
    answers = {1: 'o', 6: 'n'}
    executed = []
    try:
        for index in range(7):
            if index in answers:
                model.answer(answers[index])
            executed.append(step(index).commands.get(MOTOR, '--'))
    finally:
        run.close()
    assert executed == ['--', 'o0', 'o1', 'o2', 'o3', 'o4', 'n3']
    assert model.calls == [
        (0, []),
        (3 * PERIOD_NS, [f'o{i}' for i in range(2, 10)]),
        (6 * PERIOD_NS, [f'n{i}' for i in range(3, 10)]),
    ]


@pytest.mark.parametrize('call_after_sec', [0.4, 0.42, 0.47, 0.52, 0.6, 0.7])
def test_rtc_keeps_original_actions_and_reports_their_time_offsets(execution, monkeypatch, call_after_sec):
    model, run, step = _rtc(execution, monkeypatch, call_after_sec, ConstantDuration(0.1))
    _, clock = execution
    actions = [{MOTOR: float(i), 'joints': JointPosition(np.array([i, 2 * i]))} for i in range(6)]
    call_ns = round((0.1 + call_after_sec) * 1e9)
    try:
        step(0)
        model.answer_with(actions)
        step(1)
        clock.advance_to_ns(call_ns)
        run.send({})
        prefix = model.requests[-1][RTC.ACTION_PREFIX]
        first = min(int(round(call_after_sec * 1e9) // PERIOD_NS), len(actions))
        assert [action[MOTOR] for action in prefix[RTC.PREFIX_ACTIONS]] == list(range(first, len(actions)))
        if prefix[RTC.PREFIX_ACTIONS]:
            assert prefix[RTC.CHUNK_OFFSET_SEC] == pytest.approx(first / 10)
            assert prefix[RTC.OBSERVATION_OFFSET_SEC] == pytest.approx(first / 10 - call_after_sec)
            for expected, actual in zip(actions[first:], prefix[RTC.PREFIX_ACTIONS], strict=True):
                assert actual['joints'] is expected['joints']
    finally:
        run.close()


@pytest.mark.parametrize('duration', [DelayEstimate.mean, DelayEstimate.max])
def test_delay_estimates_refuse_a_negative_cap(duration):
    with pytest.raises(ValueError, match='max_sec'):
        duration(max_sec=-0.1)


@pytest.mark.parametrize(
    ('latency_estimate', 'expected'),
    [
        (DelayEstimate.mean(last=2), 0.2),
        (DelayEstimate.max(last=2), 0.3),
        (DelayEstimate.mean(last=2, max_sec=0.15), 0.15),
        (DelayEstimate.max(max_sec=0.4), 0.4),
    ],
)
def test_delay_estimates_read_the_last_delays_and_cap_them(latency_estimate, expected):
    estimate = latency_estimate.run()
    estimate.update(0.9)
    estimate.update(0.1)
    assert estimate.update(0.3) == pytest.approx(expected)


@pytest.mark.parametrize('strategy', [DelayEstimate.mean(last=2), DelayEstimate.max(last=2), PeakDuration()])
def test_duration_estimate_runs_keep_independent_histories(strategy):
    first, second = strategy.run(), strategy.run()
    assert first.update(0.3) == pytest.approx(0.3)
    assert second.update(0.1) == pytest.approx(0.1)
    assert first.update(0.3) == pytest.approx(0.3)


@pytest.mark.parametrize('seconds', [-0.1, float('nan'), float('inf')])
def test_constant_durations_refuse_invalid_seconds(seconds):
    with pytest.raises(ValueError, match='seconds'):
        ConstantDuration(seconds)


def test_constant_durations_ignore_delay_history():
    duration = ConstantDuration(0.7).run()
    for delay in (0.1, 0.2, 0.9):
        assert duration.update(delay) == 0.7


def test_rtc_accepts_a_constant_duration(execution, monkeypatch):
    model, run, step = _rtc(execution, monkeypatch, 0.5, ConstantDuration(0.2))
    try:
        for tick in range(9):
            if tick == 3:
                model.answer('o')
            step(tick)
        assert model.calls == [(0, []), (8 * PERIOD_NS, [f'o{i}' for i in range(5, 10)])]
    finally:
        run.close()


@pytest.mark.parametrize(
    'lead, call_ns',
    [
        (ConstantDuration(), 600_000_000),
        (ConstantDuration(0.25), 350_000_000),
        (DelayEstimate.mean(last=1), 500_000_000),
        (DelayEstimate.max(last=2), 500_000_000),
    ],
)
def test_serialized_inference_leads_preserve_the_request_deadline(execution, monkeypatch, lead, call_ns):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = ChunkedSchedule(fps=10, inference_lead=lead)
    encoded = json.loads(json.dumps(schedule.to_spec(), allow_nan=False))
    rebuilt = spec.from_spec(encoded)
    assert rebuilt.to_spec() == encoded
    run = runtime.start(rebuilt, Mock())
    try:
        run.send({})
        clock.advance_to_ns(100_000_000)
        model.answer('a', length=5)
        run.send({})
        clock.advance_to_ns(call_ns - 1)
        run.send({})
        assert len(model.calls) == 1
        clock.advance_to_ns(call_ns)
        run.send({})
        assert [time for time, _ in model.calls] == [0, call_ns]
    finally:
        run.close()


@pytest.mark.parametrize('estimate', [ConstantDuration(0.25), DelayEstimate.mean(last=2), DelayEstimate.max(last=2)])
def test_estimator_specs_roundtrip_configuration_without_history(estimate):
    declared = json.loads(json.dumps(estimate.to_spec(), allow_nan=False))
    estimate.run().update(0.4)
    assert estimate.to_spec() == declared
    restored = DurationEstimate.from_spec(declared)
    assert type(restored) is type(estimate)
    assert restored.to_spec() == declared
    assert restored.run().update(0.1) == pytest.approx(0.25 if isinstance(estimate, ConstantDuration) else 0.1)


def test_duration_specs_reject_unknown_fields():
    declared = {**DelayEstimate.mean().to_spec(), 'unexpected': 0.2}
    with pytest.raises(TypeError, match='unexpected'):
        DurationEstimate.from_spec(declared)


@pytest.mark.parametrize('last', [2.0, 2.5, True, False, 0, -1, '2', None, maxsize + 1])
def test_duration_specs_reject_invalid_windows_before_starting_a_run(last):
    declared = DelayEstimate.mean().to_spec()
    declared[DelayEstimate.LAST_ARG] = last
    with pytest.raises(ValueError, match='last must be an integer'):
        DurationEstimate.from_spec(declared)


def test_local_duration_strategies_refuse_wire_serialization():
    schedule = ChunkedSchedule(fps=10, inference_lead=PeakDuration())
    with pytest.raises(NotImplementedError, match='PeakDuration has no wire spec'):
        schedule.to_spec()


def test_a_spec_without_a_lead_waits_until_the_chunk_end_even_after_a_slow_answer(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    declared = {NAME: ChunkedSchedule.WIRE_NAME, VERSION: 3, ARGS: {ChunkedSchedule.FPS_ARG: 10}}
    schedule = spec.from_spec(declared)
    assert type(schedule) is ChunkedSchedule
    assert schedule.to_spec() == declared
    run = runtime.start(schedule, Mock())
    try:
        run.send({})
        clock.advance_to_ns(2 * PERIOD_NS)
        model.answer('a', length=3)
        assert run.send({}) == Step({MOTOR: 'a0'}, 3 * PERIOD_NS)
        for tick in (3, 4):
            clock.advance_to_ns(tick * PERIOD_NS)
            assert run.send({}) == Step({MOTOR: f'a{tick - 2}'}, (tick + 1) * PERIOD_NS)
            assert len(model.calls) == 1
        clock.advance_to_ns(5 * PERIOD_NS)
        run.send({})
        assert [time for time, _ in model.calls] == [0, 5 * PERIOD_NS]
    finally:
        run.close()


@pytest.mark.parametrize(
    'definition',
    [
        Sequential(TemporalStack(('a', 'b'), (-0.5, 0.0), pad_start=False), ChunkedSchedule(fps=10, horizon_sec=0.5)),
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps=10), TransformImages(64, 48, mode=ResizeMode.BOUND)),
        Sequential(
            PauseOnUnavailable(),
            ChunkedSchedule(fps=10),
            TransformImages(64, 48, resample=Resample.AREA, mode=ResizeMode.BOUND),
        ),
        ChunkedSchedule(fps=10, record_stats=False),
    ],
)
def test_stack_specs_round_trip(definition):
    assert spec.from_spec(definition.to_spec()).to_spec() == definition.to_spec()


@pytest.mark.parametrize(
    'node, error',
    [
        ({'seq': []}, ValueError),
        ({'par': []}, ValueError),
        ({'par': [{'name': 'stop_on_fault'}]}, ValueError),
        ({'name': 'unknown'}, ValueError),
        (
            {'name': 'temporal_stack', 'version': 2, 'args': {'keys': ['v'], 'offsets_sec': [0.0], 'bogus': 1}},
            TypeError,
        ),
    ],
)
def test_invalid_stack_specs_are_rejected(node, error):
    with pytest.raises(error):
        spec.from_spec(node)


def test_wire_names_match_the_registered_components():
    instances = {
        'chunked_schedule': ChunkedSchedule(fps=10),
        'encode_images': EncodeImages([['camera']]),
        'stop_on_fault': PauseOnUnavailable(),
        'end_on_field': EndOnField(),
        'end_command': EndCommand(),
        'temporal_stack': TemporalStack(('v',), (0.0,)),
        'binarize_grip_training': BinarizeGripTraining(('grip',)),
        'binarize_grip_inference': BinarizeGripInference(),
        'flip_grip': FlipGrip(arms=keys.BIMANUAL_ARMS),
        'metadata': Metadata({'action_fps': 15}),
        'transform_images': TransformImages(640, 640, mode=ResizeMode.BOUND),
        'observation_codec': ObservationCodecV1(state={}, images={}),
        'pack_fields': PackFields(encode_layout={'state': ['input']}),
        'split_action_chunk': SplitActionChunk(),
        'rtc_schedule': RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=DelayEstimate.max()),
        'state_observation': StateObservationCodec(state={}),
        'select_observation_fields': SelectObservationFields(mapping={}),
        'rename_observation_fields': RenameObservationFields({'state': 'input'}),
        'absolute_position_action': AbsolutePositionAction(keys.TARGET_EE_POSE, keys.TARGET_GRIP),
        'cartesian_delta_action': CartesianDeltaAction(),
        'absolute_joints_action': AbsoluteJointsAction(keys.TARGET_JOINTS, keys.TARGET_GRIP),
        'joint_delta_action': JointDeltaAction(),
        'joint_position_action': JointPositionAction('joints', 'grip'),
        'set_control_mode': SetControlMode(PositionControl()),
        'change_ee_frame': ChangeEEFrame(EEFrame(site=GRIPPER_SITE)),
        'convert_pose': ConvertPose('rot6d'),
    }
    registered = spec.COMPONENTS
    assert set(instances) | {'restrict_image_size', 'pack_observation_fields', 'unpack_action_chunk'} == set(registered)
    for name, instance in instances.items():
        assert instance.to_spec()['name'] == name
        factory = registered[name][instance.WIRE_VERSION].implementation
        assert factory is not None
        assert isinstance(factory(**instance.to_spec().get('args', {})), type(instance))
        assert instance.to_spec()['version'] == instance.WIRE_VERSION
        assert spec.from_spec({**instance.to_spec(), VERSION: LATEST}).to_spec() == instance.to_spec()


@pytest.mark.parametrize('version', [None, 1, LATEST])
def test_legacy_configs_translate_inside_nested_compositions(version):
    packing = {NAME: 'pack_observation_fields', ARGS: {'layout': {'q': 'joints'}}}
    resizing = {NAME: 'restrict_image_size', ARGS: {'width': 4, 'height': 4}}
    unpacking = {NAME: 'unpack_action_chunk', ARGS: {'fields': {'q': [0, 'q']}, 'squeeze_dims': 1}}
    if version is not None:
        for node in (packing, resizing, unpacking):
            node[VERSION] = version
    description = sequence(
        parallel(packing, SelectObservationFields({'image': 'camera'}).to_spec()), resizing, unpacking
    )
    original = json.loads(json.dumps(description))
    codec = spec.from_spec(description)
    assert isinstance(codec, Codec)
    assert description == original
    canonical = codec.to_spec()
    assert spec.from_spec(canonical).to_spec() == canonical
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    encoded = codec.encode({'joints': np.arange(7), 'camera': image})
    assert set(encoded) == {'q', 'image'}
    assert encoded['image'].shape == (2, 4, 3)
    np.testing.assert_array_equal(encoded['q'], np.arange(7))
    predicted = np.arange(21).reshape(1, 3, 7)
    rows = codec.decode(({'q': predicted}, {}))
    assert len(rows) == 3
    for index, row in enumerate(rows):
        assert set(row) == {'q'}
        np.testing.assert_array_equal(row['q'], predicted[0, index])


@pytest.mark.parametrize(
    'name', ['pack_observation_fields', 'unpack_action_chunk', 'restrict_image_size', 'chunked_schedule']
)
@pytest.mark.parametrize('version', [0, True, 1.5, '1', 99])
def test_translations_do_not_bypass_version_validation(name, version):
    with pytest.raises(ValueError, match='version'):
        spec.from_spec({NAME: name, VERSION: version})


def test_legacy_translation_pins_its_replacement_version(monkeypatch):
    versions = {**spec.COMPONENTS[PackFields.WIRE_NAME], 2: Version(lambda **args: Metadata({'wrong': True}))}
    monkeypatch.setitem(spec.COMPONENTS, PackFields.WIRE_NAME, versions)
    codec = spec.from_spec(component('pack_observation_fields', layout={'q': 'joints'}))
    assert isinstance(codec, PackFields)
    assert codec.encode({'joints': 7}) == {'q': 7}
    assert codec.to_spec()[VERSION] == 1


@pytest.mark.parametrize('lead', [0.0, 0.25, {'statistic': 'max', 'last': 3, 'max_sec': 0.4}])
def test_scheduler_translation_preserves_explicit_settings(lead):
    old = component(
        ChunkedSchedule.WIRE_NAME, version=2, fps=20, horizon_sec=0.3, record_stats=False, inference_lead=lead
    )
    current = component(ChunkedSchedule.WIRE_NAME, version=3, **old[ARGS], replan_on_error=False)
    assert spec.from_spec(old).to_spec() == spec.from_spec(current).to_spec()
    assert old[VERSION] == 2
    assert ChunkedSchedule.REPLAN_ON_ERROR_ARG not in old[ARGS]


def test_latest_follows_client_upgrades_while_numeric_versions_stay_pinned(monkeypatch):
    name = Metadata.WIRE_NAME
    latest = json.loads(json.dumps(component(name)))
    pinned = component(name, version=1)
    versions = {1: Version(lambda: Metadata({'implementation': 1}))}
    monkeypatch.setitem(spec.COMPONENTS, name, versions)
    assert spec.from_spec(latest).to_spec()[ARGS]['values'] == {'implementation': 1}
    assert spec.from_spec(pinned).to_spec()[ARGS]['values'] == {'implementation': 1}
    versions[2] = Version(lambda: Metadata({'implementation': 2}))
    assert spec.from_spec(latest).to_spec()[ARGS]['values'] == {'implementation': 2}
    assert spec.from_spec(pinned).to_spec()[ARGS]['values'] == {'implementation': 1}


def test_latest_skips_removed_versions_and_fails_when_none_are_supported(monkeypatch):
    name = ChunkedSchedule.WIRE_NAME
    notice = Deprecation(date(2020, 1, 1), date(2020, 7, 1), 'Use another scheduler.')
    versions = {3: spec.COMPONENTS[name][3], 4: Version(None, notice, removed_on=date(2020, 7, 1))}
    monkeypatch.setitem(spec.COMPONENTS, name, versions)
    latest = component(name, fps=20)
    assert spec.from_spec(latest).to_spec()[VERSION] == 3
    with pytest.raises(ValueError, match='has been removed.*Use another scheduler'):
        spec.from_spec(component(name, version=4, fps=20))
    del versions[3]
    with pytest.raises(ValueError, match='No supported versions'):
        spec.from_spec(latest)


def test_unversioned_wire_specs_still_select_v1():
    assert isinstance(spec.from_spec({NAME: Metadata.WIRE_NAME, ARGS: {'values': {}}}), Metadata)
    with pytest.raises(ValueError, match='Unsupported .* version 1'):
        spec.from_spec({NAME: ChunkedSchedule.WIRE_NAME, ARGS: {ChunkedSchedule.FPS_ARG: 20}})


class _EndingModel:
    """Answers ``first``, then chunks of ``length``. Every answer after one with ``END`` carries ``END``.

    ``on`` says which elements of such an answer carry it: ``'first'`` or ``'every'``.
    """

    def __init__(self, runtime, first, on='every', length=10):
        self._runtime = runtime
        self._first = first
        self._on = on
        self._length = length
        self._ended = False
        self.calls: list[int] = []
        self.prefixes: list[list] = []

    def __call__(self, obs):
        prefix = obs[RTC.ACTION_PREFIX][RTC.PREFIX_ACTIONS] if RTC.ACTION_PREFIX in obs else []
        call = len(self.calls)
        self.calls.append(self._runtime.time_ns)
        self.prefixes.append(list(prefix))
        chunk: list[dict[str, Any]] = (
            [dict(e) for e in self._first] if call == 0 else [{MOTOR: (call, i)} for i in range(self._length)]
        )
        if self._ended:
            for i, element in enumerate(chunk):
                if self._on == 'every' or i == 0:
                    element[END] = 'complete'
        self._ended |= any(END in element for element in chunk)
        return chunk


class _DueAnswer(Answer):
    def __init__(self, runtime, value, due_ns):
        self._runtime, self._value, self._due_ns = runtime, value, due_ns

    def done(self):
        return self._runtime.time_ns >= self._due_ns

    def result(self):
        return self._value


def _answer_after(runtime, monkeypatch, *latencies_ns):
    """Make call k answer ``latencies_ns[k]`` after its submission; later calls take the last latency."""
    calls = []

    def submit(function, *args):
        latency_ns = latencies_ns[min(len(calls), len(latencies_ns) - 1)]
        calls.append(latency_ns)
        return _DueAnswer(runtime, function(*args), runtime.time_ns + latency_ns)

    monkeypatch.setattr(runtime, 'submit', submit)


def _play(execution, schedule, model, times_ns, observations=None):
    """The motor commands sent at each time, and the ``PolicyEnd`` the run returned, or ``None``."""
    runtime, clock = execution
    run = runtime.start(Sequential(EndOnField(), schedule), model)
    sent = []
    try:
        for time_ns in times_ns:
            clock.advance_to_ns(time_ns)
            try:
                step = run.send((observations or {}).get(time_ns, {}))
            except StopIteration as stop:
                return sent, stop.value
            sent.append(step.commands.get(MOTOR))
    finally:
        run.close()
    return sent, None


def _ending(motor):
    return {MOTOR: motor, END: 'complete'}


def test_end_on_field_sends_the_other_commands_and_returns_at_the_next_observation(execution):
    runtime, clock = execution
    run = runtime.start(Sequential(EndOnField(), ChunkedSchedule(fps=10)), lambda obs: [{MOTOR: 1}, _ending(2)])
    try:
        assert run.send({}).commands == {MOTOR: 1}
        clock.advance_to_ns(PERIOD_NS)
        assert run.send({}) == Step({MOTOR: 2}, PERIOD_NS)
        with pytest.raises(StopIteration) as end:
            run.send({})
        assert end.value.value is PolicyEnd.COMPLETE
    finally:
        run.close()


def test_a_schedule_passes_end_up_as_a_command(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [{MOTOR: 1}, {MOTOR: 2, END: 'complete'}])
    try:
        run.send({})
        clock.advance_to_ns(PERIOD_NS)
        assert run.send({}).commands == {MOTOR: 2, END: 'complete'}
        clock.advance_to_ns(2 * PERIOD_NS)
        run.send({})
    finally:
        run.close()


def test_a_declared_end_on_field_ends_a_declared_stack(execution):
    runtime, clock = execution
    pose = np.array([0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5])
    declared = sequence(
        component(EndOnField.WIRE_NAME),
        component(ChunkedSchedule.WIRE_NAME, fps=10),
        parallel(
            component(AbsolutePositionAction.WIRE_NAME, tgt_ee_pose_key='pose', tgt_grip_key='grip'),
            component(EndCommand.WIRE_NAME),
        ),
    )
    model = lambda obs: [{ACTION: pose}, {ACTION: pose, END: 'give_up'}]  # noqa: E731
    run = runtime.start(cast(Policy, spec.from_spec(declared)), model)
    try:
        assert run.send({}).commands.keys() == {keys.ROBOT_COMMAND, keys.TARGET_GRIP}
        clock.advance_to_ns(PERIOD_NS)
        assert run.send({}).commands.keys() == {keys.ROBOT_COMMAND, keys.TARGET_GRIP}
        with pytest.raises(StopIteration) as end:
            run.send({})
        assert end.value.value is PolicyEnd.GIVE_UP
    finally:
        run.close()
    assert EndOnField().to_spec() == {NAME: EndOnField.WIRE_NAME, VERSION: EndOnField.WIRE_VERSION}


def test_end_on_field_refuses_a_malformed_end_when_its_element_plays(execution):
    runtime, _ = execution
    run = runtime.start(Sequential(EndOnField(), ChunkedSchedule(fps=10)), lambda obs: [{MOTOR: 1, END: 'done'}])
    try:
        with pytest.raises(ValueError):
            run.send({})
    finally:
        run.close()


# A ready answer ends on its first element; a pending one lets the round play one element past the end.
@pytest.mark.parametrize(('latency_ns', 'sent'), [(0, [1, (1, 0)]), (PERIOD_NS, [1, 5])])
def test_a_late_round_ends_the_run_on_the_next_end_that_plays(execution, monkeypatch, latency_ns, sent):
    runtime, _ = execution
    model = _EndingModel(runtime, [{MOTOR: 1}, {MOTOR: 2}, {MOTOR: 3}, _ending(4), {MOTOR: 5}])
    _answer_after(runtime, monkeypatch, 0, latency_ns)
    times_ns = [0, 5 * PERIOD_NS, 6 * PERIOD_NS]
    assert _play(execution, ChunkedSchedule(fps=10, record_stats=False), model, times_ns) == (sent, PolicyEnd.COMPLETE)
    assert model.calls == [0, 5 * PERIOD_NS]


# Each schedule asks for the next chunk before the end plays: one more call, and no more motion.
@pytest.mark.parametrize(
    ('schedule', 'calls'),
    [
        (lambda: ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.1), record_stats=False), [0, PERIOD_NS]),
        (lambda: RTCSchedule(fps=10, call_after_sec=0.1, latency_estimate=ConstantDuration()), [0, 2 * PERIOD_NS]),
    ],
    ids=['chunked', 'rtc'],
)
def test_a_queued_end_plays_on_its_own_element(execution, schedule, calls):
    runtime, _ = execution
    model = _EndingModel(runtime, [{MOTOR: 1}, _ending(2)])
    sent, end = _play(execution, schedule(), model, [i * PERIOD_NS for i in range(5)])
    assert (sent[-1], end) == (2, PolicyEnd.COMPLETE)
    assert model.calls == calls


def test_an_end_past_the_horizon_plays_on_the_next_chunk(execution):
    runtime, _ = execution
    model = _EndingModel(runtime, [{MOTOR: 1}, _ending(2)], on='first')
    schedule = ChunkedSchedule(fps=10, horizon_sec=0.1, record_stats=False)
    assert _play(execution, schedule, model, [i * PERIOD_NS for i in range(5)]) == ([1, (1, 0)], PolicyEnd.COMPLETE)


def test_a_replan_after_an_arm_error_ends_on_the_replanned_chunk(execution):
    runtime, _ = execution
    model = _EndingModel(runtime, [{MOTOR: i} for i in range(3)] + [_ending(3), {MOTOR: 4}], on='first')
    error = {2 * PERIOD_NS: {keys.ROBOT_STATUS: RobotStatus.ERROR}}
    sent, end = _play(execution, ChunkedSchedule(fps=10), model, [i * PERIOD_NS for i in range(6)], error)
    assert (sent, end) == ([0, 1, (1, 0)], PolicyEnd.COMPLETE)


# Answers arrive 2.5 periods late, so RTC skips each new chunk's first two elements.
@pytest.mark.parametrize(('on', 'end'), [('every', PolicyEnd.COMPLETE), ('first', None)])
def test_rtc_plays_an_end_only_from_an_element_that_it_does_not_skip(execution, monkeypatch, on, end):
    runtime, _ = execution
    model = _EndingModel(runtime, [_ending(i) if i == 6 else {MOTOR: i} for i in range(10)], on=on)
    _answer_after(runtime, monkeypatch, 5 * PERIOD_NS // 2)
    schedule = RTCSchedule(fps=10, call_after_sec=0.1, latency_estimate=ConstantDuration(0.25))
    sent, ended = _play(execution, schedule, model, [i * PERIOD_NS // 2 for i in range(60)])
    assert ended is end
    assert 6 not in sent


def test_the_rtc_prefix_carries_end_like_any_other_command(execution):
    runtime, clock = execution
    model = _EndingModel(runtime, [{MOTOR: 1}, _ending(2), {MOTOR: 3}])
    run = runtime.start(RTCSchedule(fps=10, call_after_sec=0, latency_estimate=ConstantDuration(0.3)), model)
    try:
        run.send({})
        clock.advance_to_ns(PERIOD_NS // 2)
        run.send({})
    finally:
        run.close()
    assert model.prefixes == [[], [{MOTOR: 1}, _ending(2), {MOTOR: 3}]]


def test_rtc_sends_measured_and_expected_latency_with_the_remaining_plan(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=ConstantDuration(0.2))
    run = runtime.start(schedule, Mock())
    try:
        run.send({})
        model.answer('old')
        clock.advance_to_ns(100_000_000)
        run.send({})
        clock.advance_to_ns(600_000_000)
        run.send({})
        first = model.requests[0][RTC.ACTION_PREFIX]
        second = model.requests[1][RTC.ACTION_PREFIX]
        assert first[RTC.PREFIX_ACTIONS] == []
        assert second[RTC.CHUNK_OFFSET_SEC] == 0.5
        assert second[RTC.OBSERVATION_OFFSET_SEC] == 0.0
        assert model.requests[0][RTC.MEASURED_LATENCY_SEC] is None
        assert model.requests[1][RTC.MEASURED_LATENCY_SEC] == 0.1
        assert model.requests[0][RTC.ESTIMATED_LATENCY_SEC] == 0.0
        assert model.requests[1][RTC.ESTIMATED_LATENCY_SEC] == 0.2
        assert [action[MOTOR] for action in second[RTC.PREFIX_ACTIONS]] == [f'old{i}' for i in range(5, 10)]
    finally:
        run.close()


def test_rtc_description_preserves_commands_without_a_model_codec(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    description = RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=ConstantDuration(0.2)).to_spec()
    restored = spec.from_spec(description)
    assert restored.to_spec() == description
    run = runtime.start(restored, Mock())
    try:
        run.send({})
        model.answer_with([
            {keys.ROBOT_COMMAND: JointPosition(np.array([i, i + 1])), keys.TARGET_GRIP: float(i % 2)} for i in range(10)
        ])
        clock.advance_to_ns(100_000_000)
        run.send({})
        clock.advance_to_ns(600_000_000)
        run.send({})
        prefix = model.requests[-1][RTC.ACTION_PREFIX]
        np.testing.assert_array_equal(
            [action[keys.ROBOT_COMMAND].positions for action in prefix[RTC.PREFIX_ACTIONS]],
            [[i, i + 1] for i in range(5, 10)],
        )
        assert [action[keys.TARGET_GRIP] for action in prefix[RTC.PREFIX_ACTIONS]] == [1.0, 0.0, 1.0, 0.0, 1.0]
    finally:
        run.close()


def test_rtc_keeps_camera_history_lazy_until_inference(execution, monkeypatch):
    runtime, _ = execution
    submitted = []

    def submit(function, observation):
        submitted.append(observation)
        return _UnchargedAnswer(Future())

    monkeypatch.setattr(runtime, 'submit', submit)
    stack = Mock(wraps=np.stack)
    monkeypatch.setattr(np, 'stack', stack)
    schedule = Sequential(
        TemporalStack(('camera',), (-0.1, 0.0)),
        RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=ConstantDuration()),
    )
    run = runtime.start(schedule, Mock())
    try:
        run.send({'camera': np.zeros((2, 3, 3), dtype=np.uint8)})
        stack.assert_not_called()
        assert submitted[0]['camera'].shape == (2, 2, 3, 3)
        stack.assert_called_once()
    finally:
        run.close()
