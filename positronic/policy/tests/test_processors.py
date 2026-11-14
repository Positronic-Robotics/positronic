"""Scheduling, fault gating, temporal history, and deliverable stack specifications."""

import json
import threading
from concurrent.futures import Future
from datetime import date
from sys import maxsize
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from attr import asdict, fields
from positronic_model_server import serialization
from positronic_model_server.spec import ARGS, LATEST, NAME, SEQ, VERSION, component, parallel, sequence

import pimm
from pimm.world import VirtualClock
from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import JointPosition, PositionControl
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy import spec
from positronic.policy.base import Answer, NotAnswered, Policy, PolicyEnd, QueueFull, Step
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
    BimanualJointsAction,
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
from positronic.policy.compatibility import CHUNKED_SCHEDULE, ObservationCodecV1, RTCScheduleV1
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.keys import CALL_ID, CALL_TIMINGS, ELAPSED_SEC, END, RTC
from positronic.policy.processors import (
    ChunkedSchedule,
    ConstantDuration,
    DelayEstimate,
    DurationEstimate,
    DurationEstimateRun,
    EndOnField,
    InferenceSchedule,
    Periodic,
    RequestTiming,
    RTCSchedule,
    TemporalStack,
    Trajectory,
    TrajectoryMerge,
    TrajectoryPlayer,
    Waypoint,
)
from positronic.policy.sequential import Sequential
from positronic.policy.tests.fakes import DueAnswer
from positronic.utils.versions import Deprecation, Version

MOTOR = 'motor'
POSITION = 'position'


class Echo(Policy):
    def run(self, runtime):
        obs = yield
        while True:
            obs = yield Step(obs, runtime.time_ns + 100_000_000)


@pytest.fixture
def periodic_execution():
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    try:
        yield runtime, now
    finally:
        runtime.close()


def _sampled_policy(*, action_fps=50, sample_fps=10, obs_time_key=policy_keys.OBS_TIME_NS):
    return Sequential(
        TrajectoryPlayer(),
        Periodic(sample_fps, obs_time_key=obs_time_key),
        InferenceSchedule(
            action_fps,
            request_timing=RequestTiming.every(1 / sample_fps),
            merge=TrajectoryMerge.at_observation(),
            latency_estimate=DelayEstimate.max(),
            obs_time_key=obs_time_key,
        ),
    )


@pytest.mark.parametrize('obs_time_key', [policy_keys.OBS_TIME_NS, 'capture_time_ns'])
def test_periodic_samples_do_not_follow_action_waypoints_or_answer_wakeups(periodic_execution, obs_time_key):
    runtime, now = periodic_execution
    requests = []
    release = threading.Event()

    def infer(obs):
        assert release.wait(timeout=2)
        requests.append(obs)
        return [{MOTOR: i} for i in range(10)] if obs[policy_keys.SAMPLE_INDEX] == 0 else None

    definition = spec.from_spec(_sampled_policy(obs_time_key=obs_time_key).to_spec())
    run = runtime.start(definition, infer)
    image = np.array([0])
    try:
        assert run.send({'image': image}) == Step({}, 100_000_000)
        release.set()
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        image[0] = 99
        now[0] = 30_000_000
        assert run.send({'image': image}) == Step({MOTOR: 0}, 50_000_000)
        for tick, motor in ((50_000_000, 1), (70_000_000, 2), (90_000_000, 3)):
            now[0] = tick
            assert run.send({'image': image}).commands == {MOTOR: motor}
        assert len(requests) == 1
        now[0] = 100_000_000
        run.send({'image': image})
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        run.send({'image': image})
        assert [obs[obs_time_key] for obs in requests] == [0, 100_000_000]
        if obs_time_key != policy_keys.OBS_TIME_NS:
            assert all(policy_keys.OBS_TIME_NS not in obs for obs in requests)
        assert [obs[policy_keys.SAMPLE_INDEX] for obs in requests] == [0, 1]
        assert [obs['image'][0] for obs in requests] == [0, 99]
    finally:
        release.set()
        run.close()


def test_periodic_skips_missed_ticks_without_backdating_or_drifting(periodic_execution):
    runtime, now = periodic_execution
    run = runtime.start(Periodic(fps=10), runtime.start(Echo()))
    try:
        for tick, index, next_tick in (
            (1_000_000_000, 0, 1_100_000_000),
            (1_250_000_000, 2, 1_300_000_000),
            (1_300_000_000, 3, 1_400_000_000),
        ):
            now[0] = tick
            step = run.send({POSITION: tick})
            assert step.commands[policy_keys.SAMPLE_INDEX] == index
            assert step.commands[policy_keys.OBS_TIME_NS] == tick
            assert step.commands[POSITION] == tick
            assert step.resume_at_ns == next_tick
    finally:
        run.close()


def test_answer_wakeup_keeps_sampled_plan_without_replaying_replan_signal(periodic_execution):
    runtime, now = periodic_execution
    run = runtime.start(Sequential(Periodic(10), Echo()))
    command = {MOTOR: 1}
    plan = [Waypoint(command, 0)]
    try:
        first = run.send({policy_keys.PLAN: plan, policy_keys.REPLAN_NOW: True})
        assert first.commands[policy_keys.REPLAN_NOW]
        command[MOTOR] = 99
        plan.clear()
        answer = runtime.submit(lambda: None)
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        now[0] = 50_000_000
        step = run.send({policy_keys.PLAN: plan})
        assert step.commands[policy_keys.PLAN] == [Waypoint({MOTOR: 1}, 0)]
        assert step.commands[policy_keys.SAMPLE_INDEX] == 0
        assert policy_keys.REPLAN_NOW not in step.commands
        answer.result()
    finally:
        run.close()


class _RecordObservedTime(Policy):
    def __init__(self) -> None:
        self.observed_ns: list[int | None] = []

    def run(self, runtime):
        obs = yield
        while True:
            self.observed_ns.append(runtime.observed_ns)
            obs = yield Step(obs, runtime.time_ns + 100_000_000)


def test_answer_wakeup_gives_the_inner_run_the_capture_time_of_the_reused_sample(periodic_execution):
    runtime, now = periodic_execution
    inner = _RecordObservedTime()
    run = runtime.start(Sequential(Periodic(10), inner))
    try:
        runtime.start_tick(5)
        run.send({POSITION: 0})
        answer = runtime.submit(lambda: None)
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        now[0] = 50_000_000
        runtime.start_tick(40_000_000)
        run.send({POSITION: 1})
        assert inner.observed_ns == [5, 5]
        assert runtime.observed_ns == 40_000_000
        answer.result()
    finally:
        run.close()


def test_sampling_tick_gives_the_inner_run_the_capture_time_of_the_new_sample(periodic_execution):
    runtime, now = periodic_execution
    inner = _RecordObservedTime()
    run = runtime.start(Sequential(Periodic(10), inner))
    try:
        runtime.start_tick(5)
        run.send({POSITION: 0})
        now[0] = 100_000_000
        runtime.start_tick(90_000_000)
        run.send({POSITION: 1})
        assert inner.observed_ns == [5, 90_000_000]
    finally:
        run.close()


def test_a_stack_of_past_samples_gives_the_inner_run_the_capture_time_of_its_newest_sample(periodic_execution):
    runtime, now = periodic_execution
    inner = _RecordObservedTime()
    run = runtime.start(Sequential(TemporalStack((POSITION,), (-0.5,)), inner))
    try:
        for tick, time_ns in enumerate((0, 500_000_000, 1_000_000_000)):
            now[0] = time_ns
            runtime.start_tick(time_ns)
            run.send({POSITION: tick})
    finally:
        run.close()
    assert inner.observed_ns == [0, 0, 500_000_000]
    assert runtime.observed_ns == 1_000_000_000


def test_overlapping_samples_preserve_data_and_deliver_trajectories_in_submission_order(periodic_execution):
    runtime, now = periodic_execution
    started, release = threading.Event(), threading.Event()
    requests = []

    def infer(obs):
        if obs[policy_keys.SAMPLE_INDEX] == 0:
            started.set()
            assert release.wait(timeout=2)
        requests.append(obs)
        index = obs[policy_keys.SAMPLE_INDEX]
        return None if index == 1 else [{MOTOR: (index, i)} for i in range(10)]

    run = runtime.start(_sampled_policy(), infer)
    image = np.array([0])
    state = {'position': [0]}
    try:
        run.send({'image': image, 'state': state})
        assert started.wait(timeout=1)
        for index in (1, 2):
            now[0] = index * 100_000_000
            image[0] = state['position'][0] = index
            assert run.send({'image': image, 'state': state}).commands == {}
        image[0] = state['position'][0] = 99
        now[0] = 250_000_000
        release.set()
        completed = 0
        while completed < 3:
            result = runtime.wait(timeout_sec=2)
            assert result.status is WaitStatus.ANSWERS_READY
            completed += len(result.completed)
        commands = []
        while runtime.has_ready_answers():
            step = run.send({'image': image, 'state': state})
            if step.commands:
                commands.append(step.commands[MOTOR])
            if runtime.has_ready_answers():
                assert step.resume_at_ns == now[0]
        assert commands == [(0, 0), (2, 2)]
        requests.sort(key=lambda obs: obs[policy_keys.SAMPLE_INDEX])
        assert [obs['image'][0] for obs in requests] == [0, 1, 2]
        assert [obs['state']['position'][0] for obs in requests] == [0, 1, 2]
        assert [obs[policy_keys.OBS_TIME_NS] for obs in requests] == [0, 100_000_000, 200_000_000]
    finally:
        release.set()
        run.close()


def test_periodic_ordinary_schedule_waits_for_a_new_sample(periodic_execution):
    runtime, now = periodic_execution
    requests = []

    def infer(obs):
        requests.append(obs)
        return [{MOTOR: len(requests)}]

    run = runtime.start(Sequential(TrajectoryPlayer(), Periodic(10), ChunkedSchedule(20)), infer)
    try:
        run.send({POSITION: 0})
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        now[0] = 10_000_000
        run.send({POSITION: 1})
        now[0] = 60_000_000
        assert run.send({POSITION: 2}).resume_at_ns == 100_000_000
        assert len(requests) == 1
        now[0] = 100_000_000
        run.send({POSITION: 3})
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        run.send({POSITION: 4})
        assert [obs[POSITION] for obs in requests] == [0, 3]
        assert [obs[policy_keys.OBS_TIME_NS] for obs in requests] == [0, 100_000_000]
    finally:
        run.close()


@pytest.mark.parametrize('with_index', [False, True])
@pytest.mark.parametrize('obs_time_key', [policy_keys.OBS_TIME_NS, 'capture_time_ns'])
def test_interval_requests_use_observation_time_and_keep_their_grid(periodic_execution, with_index, obs_time_key):
    runtime, now = periodic_execution
    requests = []

    def infer(obs):
        requests.append(obs)

    policy = InferenceSchedule(
        50,
        request_timing=RequestTiming.every(0.1),
        merge=TrajectoryMerge.at_observation(),
        latency_estimate=ConstantDuration(),
        obs_time_key=obs_time_key,
    )
    for origin in (1_000_000_000, 5_000_000_000):
        run = runtime.start(policy, infer)
        try:
            for offset, next_offset in ((0, 100), (250, 300), (300, 400), (650, 700), (700, 800)):
                now[0] = origin + offset * 1_000_000
                obs: dict[str, Any] = {obs_time_key: now[0]}
                if with_index:
                    obs[policy_keys.SAMPLE_INDEX] = 'opaque server identifier'
                run.send(obs)
                assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
                count = len(requests)
                now[0] += 250_000_000 if offset == 0 else 10_000_000
                step = run.send(obs)
                assert step.resume_at_ns == origin + next_offset * 1_000_000
                assert len(requests) == count
                assert not runtime.has_pending
        finally:
            run.close()
    assert [obs[obs_time_key] for obs in requests] == [
        origin + offset * 1_000_000 for origin in (1_000_000_000, 5_000_000_000) for offset in (0, 250, 300, 650, 700)
    ]
    if with_index:
        assert all(obs[policy_keys.SAMPLE_INDEX] == 'opaque server identifier' for obs in requests)


@pytest.mark.parametrize('sample_fps', [None, 20])
def test_interval_requests_can_run_slower_than_observations(periodic_execution, sample_fps):
    runtime, now = periodic_execution
    requests = []
    schedule = InferenceSchedule(
        50,
        request_timing=RequestTiming.every(0.2),
        merge=TrajectoryMerge.at_observation(),
        latency_estimate=ConstantDuration(),
    )
    policy = schedule if sample_fps is None else Sequential(Periodic(sample_fps), schedule)

    def infer(obs):
        requests.append(obs[POSITION])

    run = runtime.start(policy, infer)
    try:
        for tick in range(9):
            now[0] = tick * 50_000_000
            run.send({POSITION: tick})
            if tick % 4 == 0:
                assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
                run.send({POSITION: tick})
            assert not runtime.has_pending
        assert requests == [0, 4, 8]
    finally:
        run.close()


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf'), 1e-12])
def test_interval_requests_require_a_representable_positive_interval(seconds):
    with pytest.raises(ValueError):
        RequestTiming.every(seconds)


@pytest.mark.parametrize(('fps', 'indices'), [(30, range(100)), (60, range(100)), (29.97, [0, 46947, 46948, 46949])])
def test_matching_sampling_and_request_rates_deliver_every_observation(periodic_execution, fps, indices):
    runtime, now = periodic_execution
    requests = []

    def infer(obs):
        requests.append(obs[POSITION])

    run = runtime.start(_sampled_policy(sample_fps=fps), infer)
    next_sample_ns = 0
    previous_index = -1
    try:
        for index in indices:
            now[0] = next_sample_ns if index == previous_index + 1 else round(index * 1e9 / fps)
            run.send({POSITION: index})
            assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
            step = run.send({POSITION: index})
            assert abs(step.resume_at_ns - round((index + 1) * 1e9 / fps)) <= 1
            next_sample_ns, previous_index = step.resume_at_ns, index
        assert requests == list(indices)
    finally:
        run.close()


def test_periodic_capacity_skips_samples_and_preserves_recovery_intent():
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False, max_in_flight=2)
    started, release = threading.Event(), threading.Event()
    calls = []

    def infer(obs):
        calls.append(dict(obs))
        started.set()
        assert release.wait(timeout=2)

    run = runtime.start(_sampled_policy(), infer)
    try:
        run.send({})
        assert started.wait(timeout=1)
        now[0] = 100_000_000
        run.send({})
        now[0] = 200_000_000
        run.send({policy_keys.REPLAN_NOW: True})
        assert runtime.metadata_snapshot()[policy_keys.SKIPPED_SAMPLES] == 1
        release.set()
        completed = 0
        while completed < 2:
            result = runtime.wait(timeout_sec=2)
            assert result.status is WaitStatus.ANSWERS_READY
            completed += len(result.completed)
        while runtime.has_ready_answers():
            run.send({})
        assert len(calls) == 2
        now[0] = 300_000_000
        run.send({})
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        run.send({})
        now[0] = 400_000_000
        run.send({})
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        run.send({})
    finally:
        release.set()
        runtime.close()
        run.close()
    assert [obs[policy_keys.SAMPLE_INDEX] for obs in calls] == [0, 1, 3, 4]
    assert [obs[CALL_ID] for obs in calls] == [0, 1, 2, 3]
    assert calls[2][CALL_TIMINGS] == [{CALL_ID: 0, ELAPSED_SEC: 0.0}, {CALL_ID: 1, ELAPSED_SEC: 0.0}]
    assert calls[3][CALL_TIMINGS] == [{CALL_ID: 2, ELAPSED_SEC: 0.0}]
    assert calls[2][policy_keys.REPLAN_NOW] is True
    assert not calls[3].get(policy_keys.REPLAN_NOW, False)


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf'), 1e9 + 1])
def test_periodic_rejects_unrepresentable_sampling_rates(fps):
    with pytest.raises(ValueError):
        Periodic(fps)


def test_periodic_stack_spec_round_trip():
    stack = _sampled_policy()
    encoded = json.loads(json.dumps(stack.to_spec()))
    assert spec.from_spec(encoded).to_spec() == encoded


@pytest.mark.parametrize('rtc', [False, True])
def test_schedule_retries_after_another_branch_releases_runtime_capacity(rtc):
    now = [0]
    release = threading.Event()
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False, max_in_flight=1)
    occupied = runtime.submit(release.wait, 2)
    schedule = RTCSchedule(10, call_after_sec=0.5, latency_estimate=ConstantDuration()) if rtc else ChunkedSchedule(10)
    calls = []
    run = runtime.start(schedule, lambda obs: calls.append(dict(obs)))
    try:
        run.send({policy_keys.REPLAN_NOW: True})
        release.set()
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        occupied.result()
        now[0] = 100_000_000
        run.send({})
        assert runtime.wait(timeout_sec=2).status is WaitStatus.ANSWERS_READY
        assert len(calls) == 1
        assert calls[0][policy_keys.REPLAN_NOW] is True
    finally:
        release.set()
        runtime.close()
        run.close()


@pytest.mark.parametrize('wrapper', [TemporalStack((MOTOR,), (0.0,)), EndOnField()])
def test_a_wrapper_returns_what_its_inner_run_returned(execution, wrapper):
    runtime, _ = execution

    def inner():
        yield
        return PolicyEnd.COMPLETE

    inner_run = inner()
    next(inner_run)
    run = runtime.start(wrapper, inner_run)
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


class _FutureAnswer(Answer):
    """An externally completed call measured on the test's controlled clock."""

    def __init__(self, future, runtime):
        self._future = future
        self._elapsed_sec = 0.0
        submitted_ns = runtime.time_ns

        def completed(_):
            self._elapsed_sec = (runtime.time_ns - submitted_ns) / 1e9

        future.add_done_callback(completed)

    def done(self):
        return self._future.done()

    @property
    def elapsed_sec(self):
        if not self.done():
            raise NotAnswered()
        return self._elapsed_sec

    @property
    def execution_sec(self):
        return self.elapsed_sec

    def result(self):
        if not self.done():
            raise NotAnswered()
        return self._future.result()

    def cancel(self):
        self._future.cancel()


def test_call_timings_are_batched_without_loss_when_submission_is_full(execution, monkeypatch):
    runtime, clock = execution
    requests, futures = [], []
    full = False

    def submit(function, obs):
        if full:
            raise QueueFull()
        requests.append(dict(obs))
        future = Future()
        futures.append(future)
        return _FutureAnswer(future, runtime)

    monkeypatch.setattr(runtime, 'submit', submit)
    schedule = InferenceSchedule(
        10,
        request_timing=RequestTiming.every(0.1),
        merge=TrajectoryMerge.at_observation(),
        latency_estimate=ConstantDuration(),
    )
    run = runtime.start(schedule, Mock())
    try:
        for now_ns in (0, 100_000_000, 200_000_000):
            clock.advance_to_ns(now_ns)
            run.send({})
        for index, result in enumerate(([{MOTOR: 1}], None, [])):
            clock.advance_to_ns(210_000_000 + index * 10_000_000)
            futures[index].set_result(result)
            run.send({})
        full = True
        clock.advance_to_ns(300_000_000)
        run.send({})
        full = False
        clock.advance_to_ns(400_000_000)
        run.send({})
        assert [request[CALL_ID] for request in requests] == [0, 1, 2, 3]
        assert requests[3][CALL_TIMINGS] == [
            {CALL_ID: 0, ELAPSED_SEC: 0.21},
            {CALL_ID: 1, ELAPSED_SEC: 0.12},
            {CALL_ID: 2, ELAPSED_SEC: 0.03},
        ]
        assert all(request[CALL_TIMINGS] == [] for request in requests[:3])
        clock.advance_to_ns(450_000_000)
        futures[3].set_result(None)
        run.send({})
        clock.advance_to_ns(500_000_000)
        run.send({})
        assert requests[4][CALL_TIMINGS] == [{CALL_ID: 3, ELAPSED_SEC: 0.05}]
        assert len(requests[3][CALL_TIMINGS]) == 3
    finally:
        run.close()


def test_replan_preserves_all_submitted_observations_but_discards_old_predictions(execution, monkeypatch):
    runtime, clock = execution
    requests, futures = [], []

    def submit(function, obs):
        requests.append(dict(obs))
        future = Future()
        futures.append(future)
        return _FutureAnswer(future, runtime)

    monkeypatch.setattr(runtime, 'submit', submit)
    schedule = InferenceSchedule(
        10,
        request_timing=RequestTiming.every(0.1),
        merge=TrajectoryMerge.at_observation(),
        latency_estimate=ConstantDuration(),
    )
    run = runtime.start(schedule, Mock())
    try:
        run.send({POSITION: 0})
        clock.advance_to_ns(100_000_000)
        run.send({POSITION: 1})
        clock.advance_to_ns(150_000_000)
        run.send({POSITION: 2, policy_keys.REPLAN_NOW: True})
        assert [request[POSITION] for request in requests] == [0, 1, 2]
        assert all(not future.cancelled() for future in futures)
        for index in range(2):
            futures[index].set_result([{MOTOR: 'obsolete'}])
            assert run.send({POSITION: 2}).commands == {}
        futures[2].set_result([{MOTOR: 'fresh'}])
        update = Trajectory(**run.send({POSITION: 2}).commands)
        assert update.commands == [{MOTOR: 'fresh'}]
        assert update.start_ns == 150_000_000
        clock.advance_to_ns(200_000_000)
        run.send({POSITION: 3})
        assert [entry[CALL_ID] for entry in requests[-1][CALL_TIMINGS]] == [0, 1, 2]
    finally:
        run.close()


def test_temporal_stack_builds_a_stack_on_the_first_read_of_its_key(execution, monkeypatch):
    runtime, clock = execution
    requests = []
    monkeypatch.setattr(
        runtime, 'submit', lambda function, obs: requests.append(obs) or _FutureAnswer(Future(), runtime)
    )
    stacks = []
    stack = np.stack
    monkeypatch.setattr(np, 'stack', lambda arrays: stacks.append(1) or stack(arrays))
    run = runtime.start(
        Sequential(
            TemporalStack((POSITION,), (-0.1, 0.0)), TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10)
        ),
        Mock(),
    )
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
        assert dict(requests[0]).keys() >= {POSITION, keys.TASK} and requests[0][keys.TASK] == 'tick 0'
    finally:
        run.close()


@pytest.fixture
def execution(monkeypatch):
    with pimm.World(virtual_time=True) as world:
        runtime = Executor(world.clock.now_ns, simulated=True, charge_inference_time=False)

        def submit(function, *args, **kwargs):
            future = Future()
            future.set_result(function(*args, **kwargs))
            return _FutureAnswer(future, runtime)

        monkeypatch.setattr(runtime, 'submit', submit)
        try:
            yield runtime, cast(VirtualClock, world.clock)
        finally:
            runtime.close()


def test_pending_inference_is_not_resubmitted_on_each_tick(execution, monkeypatch):
    runtime, clock = execution
    future = Future()
    submit = Mock(return_value=_FutureAnswer(future, runtime))
    monkeypatch.setattr(runtime, 'submit', submit)
    run = runtime.start(Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10)), lambda obs: [])
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


def test_last_action_keeps_its_full_period(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(Sequential(TrajectoryPlayer(), ChunkedSchedule(fps=10)), Mock())
    try:
        run.send({})
        model.answer_with([{MOTOR: i} for i in (1, 2, 3)])
        for index in range(3):
            clock.advance_to_ns(index * PERIOD_NS)
            assert run.send({}) == Step({MOTOR: index + 1}, (index + 1) * PERIOD_NS)
        assert len(model.calls) == 1
        clock.advance_to_ns(3 * PERIOD_NS)
        assert run.send({}).commands == {}
        assert [time for time, _ in model.calls] == [0, 3 * PERIOD_NS]
        clock.advance_to_ns(350_000_000)
        model.answer_with([{MOTOR: 4}])
        assert run.send({}) == Step({MOTOR: 4}, 450_000_000)
    finally:
        run.close()


def test_late_wake_merges_due_channels_and_keeps_absolute_deadlines(execution):
    runtime, clock = execution
    chunk = [{MOTOR: 1}, {POSITION: 2}, {MOTOR: 3}, {MOTOR: 4}]
    inner = Mock(
        send=Mock(return_value=Step(asdict(Trajectory(chunk, 0, 10, 400_000_000), recurse=False), 400_000_000))
    )
    run = runtime.start(TrajectoryPlayer(), inner)
    run.send({})
    clock.advance_to_ns(225_000_000)
    assert run.send({}) == Step({POSITION: 2, MOTOR: 3}, 300_000_000)
    run.close()
    assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 1


def test_an_overrun_skips_all_but_the_last_due_waypoint_and_counts_the_skip(execution):
    runtime, clock = execution
    inner = Mock(send=Mock(return_value=Step(_trajectory(range(5)), 500_000_000)))
    run = runtime.start(TrajectoryPlayer(), inner)
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
        f'{prefix}.{eval_keys.HOLD_SUM_MS}': 0.0,
        f'{prefix}.{eval_keys.HOLD_MAX_MS}': 0.0,
    }
    run.close()


def test_a_schedule_without_stats_writes_no_metadata(execution):
    runtime, clock = execution
    inner = Mock(send=Mock(return_value=Step(_trajectory(range(5)), 500_000_000)))
    run = runtime.start(TrajectoryPlayer(record_stats=False), inner)
    for now in (0, 100_000_000, 350_000_000):
        clock.advance_to_ns(now)
        run.send({})
    run.close()
    assert runtime.metadata_snapshot() == {}


def test_a_new_chunk_counts_the_due_waypoints_it_replaces_as_dropped(execution):
    runtime, clock = execution
    inner = Mock(
        send=Mock(
            side_effect=[
                Step(_trajectory(range(5)), 600_000_000),
                Step(_trajectory(range(5), 600_000_000), 1_100_000_000),
            ]
        )
    )
    run = runtime.start(TrajectoryPlayer(), inner)
    emitted = []
    try:
        for now in (0, 600_000_000):
            clock.advance_to_ns(now)
            emitted.append(run.send({}).commands[MOTOR])
    finally:
        run.close()
    assert emitted == [0, 0]
    prefix = eval_keys.SCHEDULE
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.SCHEDULED}'] == 10
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.EMITTED}'] == 2
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.DROPPED}'] == 4


def test_a_new_chunk_excludes_the_gap_from_the_previous_chunk(execution):
    runtime, clock = execution
    inner = Mock(
        send=Mock(
            side_effect=[
                Step(_trajectory(range(3)), 300_000_000),
                Step(_trajectory(range(3), 300_000_000), 600_000_000),
            ]
        )
    )
    run = runtime.start(TrajectoryPlayer(), inner)
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


def test_a_chunk_that_starts_after_the_plan_end_counts_the_hold_from_that_end(execution):
    runtime, clock = execution
    inner = Mock(
        send=Mock(
            side_effect=[
                Step(_trajectory(range(3)), 450_000_000),
                Step(_trajectory(range(3), 450_000_000), 600_000_000),
                Step(_trajectory(range(3), 600_000_000), 1_000_000_000),
                Step(_trajectory(range(3), 1_000_000_000), 2_000_000_000),
            ]
        )
    )
    run = runtime.start(TrajectoryPlayer(), inner)
    try:
        for now_ms in (0, 100, 200, 300, 450, 550, 600, 700, 800, 1000):
            clock.advance_to_ns(now_ms * 1_000_000)
            run.send({})
    finally:
        run.close()
    # The plan ends at 300 ms and 900 ms; the chunk that arrives at 600 ms replaces a plan that ends at 750 ms.
    metadata = runtime.metadata_snapshot()
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.HOLD_SUM_MS}'] == 250.0
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.HOLD_MAX_MS}'] == 150.0


def test_a_queued_chunk_counts_the_hold_from_the_end_of_the_chunk_before_it(execution):
    runtime, clock = execution
    inner = Mock(
        send=Mock(
            side_effect=[
                Step(_trajectory(range(3)), 100_000_000),
                Step(_trajectory(range(3), 300_000_000), 200_000_000),
                Step(_trajectory(range(2), 700_000_000), 2_000_000_000),
            ]
        )
    )
    run = runtime.start(TrajectoryPlayer(), inner)
    try:
        for now_ms in (0, 100, 200, 300, 400, 500, 700, 800):
            clock.advance_to_ns(now_ms * 1_000_000)
            run.send({})
    finally:
        run.close()
    # Both later chunks are queued at 200 ms; the arm holds from 600 ms until the third chunk starts at 700 ms.
    metadata = runtime.metadata_snapshot()
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.HOLD_SUM_MS}'] == 100.0
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.HOLD_MAX_MS}'] == 100.0


@pytest.mark.parametrize('late_ms', [[0], [0, 10], [0, 5, 30, 10, 20], list(range(0, 1000, 7))])
def test_late_percentiles_match_numpy(execution, late_ms):
    runtime, clock = execution
    inner = Mock(send=Mock(return_value=Step(_trajectory(range(len(late_ms)), fps=1), len(late_ms) * 1_000_000_000)))
    run = runtime.start(TrajectoryPlayer(), inner)
    for index, late in enumerate(late_ms):
        clock.advance_to_ns(index * 1_000_000_000 + late * 1_000_000)
        run.send({})
    run.close()
    prefix = eval_keys.SCHEDULE
    p50, p90 = np.percentile(late_ms, (50, 90))
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.LATE_P50_MS}'] == pytest.approx(p50)
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.LATE_P90_MS}'] == pytest.approx(p90)
    assert runtime.metadata_snapshot()[f'{prefix}.{eval_keys.LATE_MAX_MS}'] == max(late_ms)


def test_horizon_cuts_commands_but_preserves_boundary(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(Sequential(TrajectoryPlayer(), ChunkedSchedule(fps=10, horizon_sec=0.15)), Mock())
    try:
        run.send({})
        model.answer_with([{MOTOR: i} for i in range(5)])
        assert run.send({}) == Step({MOTOR: 0}, 100_000_000)
        clock.advance_to_ns(100_000_000)
        assert run.send({}) == Step({MOTOR: 1}, 150_000_000)
        clock.advance_to_ns(150_000_000)
        assert run.send({}).commands == {}
        assert [time for time, _ in model.calls] == [0, 150_000_000]
        model.answer_with([{MOTOR: i} for i in range(5)])
        assert run.send({}) == Step({MOTOR: 0}, 250_000_000)
    finally:
        run.close()


def test_empty_chunks_request_an_immediate_retry(execution):
    runtime, clock = execution
    requests = []

    def infer(obs):
        requests.append(runtime.time_ns)
        return []

    run = runtime.start(Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10)), infer)
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
        return _FutureAnswer(self._pending, self._runtime)

    def answer(self, name, length=10):
        self.answer_with([{MOTOR: f'{name}{i}'} for i in range(length)])

    def answer_with(self, actions):
        assert self._pending is not None
        self._pending.set_result(actions)
        self._pending = None


def _trajectory(values, start_ns=0, *, fps=10):
    commands = [{MOTOR: value} for value in values]
    return asdict(Trajectory(commands, start_ns, fps, start_ns + round(len(commands) / fps * 1e9)), recurse=False)


def test_player_waypoint_wakeups_do_not_advance_the_inner_run(execution):
    runtime, clock = execution
    inner_run = Mock(send=Mock(return_value=Step(_trajectory([0, 1, 2]), 500_000_000, outputs={'sample': 42})))
    run = runtime.start(TrajectoryPlayer(), inner_run)
    try:
        assert run.send({}) == Step({MOTOR: 0}, 100_000_000, outputs={'sample': 42})
        for now, commands, deadline in [
            (50_000_000, {}, 100_000_000),
            (100_000_000, {MOTOR: 1}, 200_000_000),
            (200_000_000, {MOTOR: 2}, 500_000_000),
            (300_000_000, {}, 500_000_000),
        ]:
            clock.advance_to_ns(now)
            assert run.send({}) == Step(commands, deadline)
        inner_run.send.assert_called_once_with({})
        clock.advance_to_ns(500_000_000)
        inner_run.send.return_value = Step({}, 600_000_000)
        assert run.send({}) == Step({}, 600_000_000)
        assert inner_run.send.call_count == 2
    finally:
        run.close()
    inner_run.close.assert_not_called()


def test_player_checks_global_readiness_once_without_spinning(execution, monkeypatch):
    runtime, clock = execution
    inner_run = Mock(send=Mock(return_value=Step({}, 500_000_000)))
    run = runtime.start(TrajectoryPlayer(), inner_run)
    try:
        run.send({})
        clock.advance_to_ns(50_000_000)
        monkeypatch.setattr(runtime, 'has_ready_answers', lambda: True)
        inner_run.send.return_value = Step(_trajectory([7], 50_000_000), 500_000_000)
        assert run.send({}) == Step({MOTOR: 7}, 500_000_000)
        assert inner_run.send.call_count == 2
    finally:
        run.close()


def test_player_preserves_earlier_commands_when_a_future_update_replaces_the_tail(execution):
    runtime, clock = execution
    first = _trajectory(['a0', 'a1', 'a2', 'a3'])
    second = _trajectory(['b0', 'b1'], 200_000_000, fps=20)
    inner_run = Mock(
        send=Mock(side_effect=[Step(first, 50_000_000), Step(second, 250_000_000), Step({}, 1_000_000_000)])
    )
    run = runtime.start(TrajectoryPlayer(plan_key=policy_keys.PLAN), inner_run)
    try:
        assert run.send({}).commands == {MOTOR: 'a0'}
        clock.advance_to_ns(50_000_000)
        assert run.send({}).commands == {}
        clock.advance_to_ns(100_000_000)
        assert run.send({}).commands == {MOTOR: 'a1'}
        clock.advance_to_ns(200_000_000)
        assert run.send({}).commands == {MOTOR: 'b0'}
        clock.advance_to_ns(250_000_000)
        assert run.send({}) == Step({MOTOR: 'b1'}, 1_000_000_000)
        prefix = inner_run.send.call_args.args[0][policy_keys.PLAN]
        assert [point.commands[MOTOR] for point in prefix] == ['b1']
        assert [point.due_ns for point in prefix] == [250_000_000]
        assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 2
    finally:
        run.close()


@pytest.mark.parametrize('start_ns', [100_000_000, 0])
def test_player_starts_late_trajectories_at_the_latest_due_command(execution, start_ns):
    runtime, clock = execution
    inner_run = Mock(
        send=Mock(
            side_effect=[
                Step(_trajectory(['a0', 'a1', 'a2', 'a3']), 350_000_000),
                Step(_trajectory(['b0', 'b1', 'b2'], start_ns), 1_000_000_000),
            ]
        )
    )
    run = runtime.start(TrajectoryPlayer(), inner_run)
    try:
        run.send({})
        clock.advance_to_ns(350_000_000)
        assert run.send({}) == Step({MOTOR: 'b2'}, 1_000_000_000)
    finally:
        run.close()


@pytest.mark.parametrize('update', [{}, _trajectory([], 200_000_000)])
def test_player_distinguishes_no_update_from_removing_the_future_plan(execution, update):
    runtime, clock = execution
    inner_run = Mock(send=Mock(side_effect=[Step(_trajectory([0, 1, 2]), 50_000_000), Step(update, 1_000_000_000)]))
    run = runtime.start(TrajectoryPlayer(), inner_run)
    try:
        run.send({})
        clock.advance_to_ns(50_000_000)
        run.send({})
        clock.advance_to_ns(100_000_000)
        assert run.send({}).commands == {MOTOR: 1}
        clock.advance_to_ns(200_000_000)
        assert run.send({}) == Step({MOTOR: 2} if not update else {}, 1_000_000_000)
    finally:
        run.close()


def test_player_context_is_one_remaining_plan_with_exact_replacement_times(execution):
    runtime, clock = execution
    first = _trajectory([0, 1, 2, 3])
    second = _trajectory([4, 5], 250_000_000)
    inner_run = Mock(
        send=Mock(side_effect=[Step(first, 50_000_000), Step(second, 100_000_000), Step({}, 1_000_000_000)])
    )
    player = TrajectoryPlayer(plan_key='remaining', record_stats=False)
    run = runtime.start(spec.from_spec(player.to_spec()), inner_run)
    try:
        run.send({POSITION: 42})
        assert inner_run.send.call_args.args[0]['remaining'] == []
        clock.advance_to_ns(50_000_000)
        run.send({POSITION: 43})
        source = inner_run.send.call_args.args[0]['remaining'][0]
        assert type(source) is Waypoint
        assert {field.name for field in fields(Waypoint)} == {'commands', 'due_ns'}
        assert source.commands is first['commands'][0]
        assert source.due_ns == 0
        clock.advance_to_ns(100_000_000)
        run.send({POSITION: 44})
        context = inner_run.send.call_args.args[0]
        assert context[POSITION] == 44
        assert [point.commands[MOTOR] for point in context['remaining']] == [1, 2, 4, 5]
        assert [point.due_ns for point in context['remaining']] == [100_000_000, 200_000_000, 250_000_000, 350_000_000]
        assert runtime.metadata_snapshot() == {}
    finally:
        run.close()


def test_player_context_includes_the_current_action_without_reissuing_it(execution):
    runtime, clock = execution
    inner_run = Mock(
        send=Mock(
            side_effect=[
                Step(_trajectory([0, 1, 2]), 50_000_000),
                Step({}, 150_000_000),
                Step({}, 300_000_000),
                Step({}, 1_000_000_000),
            ]
        )
    )
    run = runtime.start(TrajectoryPlayer(plan_key=policy_keys.PLAN), inner_run)
    try:
        assert run.send({}).commands == {MOTOR: 0}
        for now_ns, commands, context in [
            (50_000_000, {}, [0, 1, 2]),
            (100_000_000, {MOTOR: 1}, None),
            (150_000_000, {}, [1, 2]),
            (200_000_000, {MOTOR: 2}, None),
            (300_000_000, {}, [2]),
        ]:
            clock.advance_to_ns(now_ns)
            assert run.send({}).commands == commands
            if context is not None:
                prefix = inner_run.send.call_args.args[0][policy_keys.PLAN]
                assert [point.commands[MOTOR] for point in prefix] == context
                if prefix:
                    assert prefix[0].due_ns <= now_ns
                    assert len(prefix) == 1 or now_ns < prefix[1].due_ns
    finally:
        run.close()


def test_player_context_keeps_the_plan_at_observation_time(execution):
    runtime, clock = execution
    inner_run = Mock(
        send=Mock(
            side_effect=[Step(_trajectory([0, 1, 2]), 50_000_000), Step(_trajectory([9], 100_000_000), 500_000_000)]
        )
    )
    run = runtime.start(TrajectoryPlayer(plan_key=policy_keys.PLAN), inner_run)
    try:
        run.send({})
        first_obs = inner_run.send.call_args.args[0]
        clock.advance_to_ns(50_000_000)
        run.send({policy_keys.PLAN: 'overridden'})
        second_obs = inner_run.send.call_args.args[0]
        clock.advance_to_ns(100_000_000)
        assert run.send({}).commands == {MOTOR: 9}
        assert first_obs[policy_keys.PLAN] == []
        assert [point.commands[MOTOR] for point in second_obs[policy_keys.PLAN]] == [0, 1, 2]
    finally:
        run.close()


def test_player_context_holds_the_previous_command_across_a_gap(execution):
    runtime, clock = execution
    inner_run = Mock(
        send=Mock(
            side_effect=[
                Step(_trajectory([1]), 50_000_000),
                Step(_trajectory([2], 300_000_000), 200_000_000),
                Step({}, 400_000_000),
                Step({}, 1_000_000_000),
            ]
        )
    )
    run = runtime.start(TrajectoryPlayer(plan_key=policy_keys.PLAN), inner_run)
    try:
        assert run.send({}).commands == {MOTOR: 1}
        clock.advance_to_ns(50_000_000)
        assert run.send({}).commands == {}
        clock.advance_to_ns(200_000_000)
        assert run.send({}) == Step({}, 300_000_000)
        prefix = inner_run.send.call_args.args[0][policy_keys.PLAN]
        assert [point.commands[MOTOR] for point in prefix] == [1, 2]
        assert [point.due_ns for point in prefix] == [0, 300_000_000]
        clock.advance_to_ns(300_000_000)
        assert run.send({}).commands == {MOTOR: 2}
        clock.advance_to_ns(400_000_000)
        assert run.send({}).commands == {}
        prefix = inner_run.send.call_args.args[0][policy_keys.PLAN]
        assert [point.commands[MOTOR] for point in prefix] == [2]
    finally:
        run.close()


def test_player_interprets_inner_commands_and_returns_its_end(execution):
    runtime, _ = execution
    inner_run = Mock(send=Mock(side_effect=[Step(_trajectory([42]), 0), StopIteration(PolicyEnd.COMPLETE)]))
    run = runtime.start(TrajectoryPlayer(), inner_run)
    assert run.send({}) == Step({MOTOR: 42}, 0)
    with pytest.raises(StopIteration) as end:
        run.send({})
    assert end.value.value is PolicyEnd.COMPLETE


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf'), -float('inf')])
def test_trajectory_rejects_invalid_timing(execution, fps):
    runtime, _ = execution
    update: dict[str, Any] = {'commands': [{MOTOR: 1}], 'start_ns': 0, 'fps': fps, 'end_ns': PERIOD_NS}
    with pytest.raises(ValueError):
        Trajectory(**update)
    inner_run = Mock(send=Mock(return_value=Step(update, 0)))
    run = runtime.start(TrajectoryPlayer(), inner_run)
    with pytest.raises(ValueError):
        run.send({})


@pytest.mark.parametrize('end_ns', [-1, 200_000_001])
def test_trajectory_rejects_an_end_outside_its_duration(end_ns):
    with pytest.raises(ValueError, match='end_ns'):
        Trajectory([{MOTOR: 1}, {MOTOR: 2}], 0, 10, end_ns)


def test_chunked_history_only_answer_preserves_playback_and_updates_duration(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    estimate_run = Mock(update=Mock(return_value=0.1))
    estimate = Mock(run=Mock(return_value=estimate_run))
    run = runtime.start(
        Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=estimate)), Mock()
    )
    try:
        run.send({})
        model.answer('a', length=3)
        run.send({})
        clock.advance_to_ns(200_000_000)
        assert run.send({}).commands == {MOTOR: 'a2'}
        clock.advance_to_ns(250_000_000)
        model.answer_with(None)
        assert run.send({}).commands == {}
        assert estimate_run.update.call_args.args[0] == pytest.approx(0.05)
        assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.SCHEDULED}'] == 3
    finally:
        run.close()


def test_existing_codecs_decode_wire_arrays_and_timing_for_the_player(execution):
    runtime, clock = execution
    commands = (
        PackFields(decode_layout={'commands': []})
        | JointPositionAction('joints', 'grip', num_joints=2)
        | SplitActionChunk()
        | PackFields(decode_layout={'joints': ['joints'], 'grip': ['grip']})
    )
    timing_keys = ('start_ns', 'fps', 'end_ns')
    assert {field.name for field in fields(Trajectory)} == {'commands', *timing_keys}
    codec = spec.from_spec((commands & PackFields(decode_layout={key: [key] for key in timing_keys})).to_spec())
    assert isinstance(codec, Codec)
    response = {
        'joints': np.array([[1, 2], [3, 4]]),
        'grip': np.array([0.1, 0.2]),
        'start_ns': 50_000_000,
        'fps': 10,
        'end_ns': 200_000_000,
    }
    outputs = {'next_inference_ns': 1_000_000_000}

    def predictions():
        yield
        yield Step(serialization.deserialise(serialization.serialise(response)), 1_000_000_000, outputs=outputs)

    inner_run = predictions()
    next(inner_run)
    decoded_run = codec.wrap(inner_run)
    run = runtime.start(TrajectoryPlayer(), decoded_run)
    try:
        first_step = run.send({})
        assert first_step == Step({}, 50_000_000, outputs=outputs)
        assert first_step.outputs is outputs
        clock.advance_to_ns(50_000_000)
        first = run.send({})
        np.testing.assert_array_equal(first.commands[keys.ROBOT_COMMAND].positions, [1, 2])
        assert first.commands[keys.TARGET_GRIP] == 0.1
        clock.advance_to_ns(150_000_000)
        second = run.send({})
        np.testing.assert_array_equal(second.commands[keys.ROBOT_COMMAND].positions, [3, 4])
        assert second.commands[keys.TARGET_GRIP] == 0.2
        assert second.resume_at_ns == 1_000_000_000
    finally:
        run.close()
        decoded_run.close()
        inner_run.close()
    assert codec.wrap(lambda obs: None)({}) is None


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
def test_errors_bring_inference_and_playback_forward_when_enabled(execution, monkeypatch, status, arm, replan_on_error):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    channel = f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.STATUS_SUFFIX}'

    run = runtime.start(Sequential(TrajectoryPlayer(replan_on_error=replan_on_error), ChunkedSchedule(fps=10)), Mock())
    try:
        run.send({})
        model.answer_with([{MOTOR: (1, index)} for index in range(5)])
        assert run.send({}).commands == {MOTOR: (1, 0)}
        clock.advance_to_ns(PERIOD_NS)
        obs = {keys.ROBOT_STATUS: RobotStatus.AVAILABLE.value, channel: status.value, POSITION: 42}
        step = run.send(obs)
        initial = {CALL_ID: 0, CALL_TIMINGS: []}
        if replan_on_error and status is RobotStatus.ERROR:
            model.answer_with([{MOTOR: (2, index)} for index in range(5)])
            step = run.send(obs)
            assert step == Step({MOTOR: (2, 0)}, 2 * PERIOD_NS)
            assert model.requests == [
                initial,
                {**obs, policy_keys.REPLAN_NOW: True, CALL_ID: 1, CALL_TIMINGS: [{CALL_ID: 0, ELAPSED_SEC: 0.0}]},
            ]
            assert runtime.metadata_snapshot()[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 4
        else:
            assert step == Step({MOTOR: (1, 1)}, 2 * PERIOD_NS)
            assert model.requests == [initial]
    finally:
        run.close()


@pytest.mark.parametrize('initial_status', [RobotStatus.AVAILABLE, RobotStatus.ERROR])
def test_repeated_errors_keep_one_recovery_request_and_play_its_answer(execution, monkeypatch, initial_status):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10)), Mock())
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
def test_errors_keep_an_existing_request_but_discard_its_prediction(execution, monkeypatch, ready_at_error):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.3))
        ),
        Mock(),
    )
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
        assert step.commands == {}
        assert model.seen == [0, 2, 25]
        assert model.requests[-1][policy_keys.REPLAN_NOW] is True
        assert [entry[CALL_ID] for entry in model.requests[-1][CALL_TIMINGS]] == [1]
        clock.advance_to_ns(4 * PERIOD_NS)
        model.answer('fresh')
        assert run.send(fault).commands == {MOTOR: 'fresh0'}
        assert len(model.calls) == 3
    finally:
        run.close()


def test_errors_replace_a_prefetched_chunk_before_its_playback_starts(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.3))
        ),
        Mock(),
    )
    try:
        run.send({})
        model.answer('a', length=5)
        run.send({})
        clock.advance_to_ns(2 * PERIOD_NS)
        assert run.send({}).commands == {MOTOR: 'a2'}
        model.answer('prefetched', length=5)
        run.send({})
        clock.advance_to_ns(250_000_000)
        fault = {keys.ROBOT_STATUS: RobotStatus.ERROR}
        assert run.send(fault).commands == {}
        assert len(model.calls) == 3
        clock.advance_to_ns(3 * PERIOD_NS)
        model.answer('fresh', length=5)
        assert run.send(fault).commands == {MOTOR: 'fresh0'}
    finally:
        run.close()


def test_a_failed_request_before_the_error_still_raises(execution, monkeypatch):
    runtime, _ = execution
    future = Future()
    monkeypatch.setattr(runtime, 'submit', Mock(return_value=_FutureAnswer(future, runtime)))
    run = runtime.start(Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10)), Mock())
    run.send({})
    future.set_exception(RuntimeError('inference failed'))
    with pytest.raises(RuntimeError, match='inference failed'):
        run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})


@pytest.mark.parametrize('rtc', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_schedulers_keep_one_pending_call_and_close_it(execution, monkeypatch, rtc, fail):
    runtime, clock = execution
    future = Future()
    submit = Mock(return_value=_FutureAnswer(future, runtime))
    monkeypatch.setattr(runtime, 'submit', submit)
    schedule = (
        RTCSchedule(fps=10, call_after_sec=0, latency_estimate=ConstantDuration()) if rtc else ChunkedSchedule(fps=10)
    )
    run = runtime.start(schedule, Mock())
    try:
        for now_ns in (0, PERIOD_NS, 5 * PERIOD_NS):
            clock.advance_to_ns(now_ns)
            assert run.send({policy_keys.PLAN: []}) == Step({}, now_ns)
        assert submit.call_count == 1
        if fail:
            future.set_exception(RuntimeError('inference failed'))
            with pytest.raises(RuntimeError, match='inference failed'):
                run.send({policy_keys.PLAN: []})
            assert submit.call_count == 1
    finally:
        run.close()
    assert future.cancelled() is not fail


def test_chunked_schedule_starts_the_first_trajectory_when_the_answer_is_read(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    clock.advance_to_ns(250_000_000)
    obs = {POSITION: 42}
    chunk = [{MOTOR: 1}, {MOTOR: 2}]
    run = runtime.start(ChunkedSchedule(fps=10), Mock())
    try:
        run.send(obs)
        clock.advance_to_ns(300_000_000)
        model.answer_with(chunk)
        clock.advance_to_ns(350_000_000)
        step = run.send({POSITION: 43})
        assert Trajectory(**step.commands) == Trajectory(chunk, 350_000_000, 10, 550_000_000)
        assert step.resume_at_ns == 550_000_000
        assert model.requests == [{**obs, CALL_ID: 0, CALL_TIMINGS: []}]
    finally:
        run.close()


@pytest.mark.parametrize('override', [{}, {policy_keys.REPLAN_NOW: False}, {policy_keys.REPLAN_NOW: True}])
@pytest.mark.parametrize('rtc', [False, True])
def test_scheduler_replans_only_when_requested_by_the_observation(execution, monkeypatch, override, rtc):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = (
        RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=ConstantDuration()) if rtc else ChunkedSchedule(fps=10)
    )
    run = runtime.start(schedule, Mock())
    try:
        run.send({})
        model.answer('a', length=5)
        run.send({})
        clock.advance_to_ns(PERIOD_NS)
        fault = {keys.ROBOT_STATUS: RobotStatus.ERROR}
        step = run.send({**fault, **override})
        initial = {CALL_ID: 0, CALL_TIMINGS: []}
        if override.get(policy_keys.REPLAN_NOW, False):
            assert model.requests == [
                initial,
                {**fault, policy_keys.REPLAN_NOW: True, CALL_ID: 1, CALL_TIMINGS: [{CALL_ID: 0, ELAPSED_SEC: 0.0}]},
            ]
            assert step.commands == {}
            clock.advance_to_ns(2 * PERIOD_NS)
            model.answer('b', length=5)
            step = run.send(fault)
            assert Trajectory(**step.commands).start_ns == 2 * PERIOD_NS
        else:
            assert model.requests == [initial]
            assert step == Step({}, 5 * PERIOD_NS)
    finally:
        run.close()


def test_player_requests_replanning_only_at_the_start_of_recovery(execution):
    runtime, clock = execution
    inner = Mock(send=Mock(return_value=Step({}, PERIOD_NS)))
    run = runtime.start(TrajectoryPlayer(replan_on_error=True), inner)
    fault = {keys.ROBOT_STATUS: RobotStatus.ERROR}
    try:
        run.send(fault)
        assert inner.send.call_args.args[0][policy_keys.REPLAN_NOW] is True
        clock.advance_to_ns(PERIOD_NS // 2)
        run.send(fault)
        assert inner.send.call_count == 1
        clock.advance_to_ns(PERIOD_NS)
        run.send(fault)
        assert inner.send.call_count == 2
        assert policy_keys.REPLAN_NOW not in inner.send.call_args.args[0]
        assert policy_keys.REPLAN_NOW not in fault
    finally:
        run.close()


@pytest.mark.parametrize('replan_on_error', [False, True])
def test_serialized_error_replanning_preserves_the_behavior(execution, monkeypatch, replan_on_error):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    definition = Sequential(TrajectoryPlayer(replan_on_error=replan_on_error), ChunkedSchedule(fps=10))
    encoded = json.loads(json.dumps(definition.to_spec()))
    assert encoded[SEQ][1][NAME] == InferenceSchedule.WIRE_NAME
    assert encoded[SEQ][1][VERSION] == 1
    rebuilt = spec.from_spec(encoded)
    assert rebuilt.to_spec() == encoded
    run = runtime.start(rebuilt, Mock())
    try:
        run.send({})
        model.answer('a', length=5)
        run.send({})
        clock.advance_to_ns(PERIOD_NS)
        run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})
        assert len(model.calls) == (2 if replan_on_error else 1)
    finally:
        run.close()


def test_v2_keeps_its_chunk_and_inference_deadline_on_error(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    declared = {NAME: CHUNKED_SCHEDULE, VERSION: 2, ARGS: {ChunkedSchedule.FPS_ARG: 10}}
    schedule = spec.from_spec(declared)
    assert schedule.to_spec() == Sequential(TrajectoryPlayer(replan_on_error=False), ChunkedSchedule(fps=10)).to_spec()
    run = runtime.start(schedule, Mock())
    try:
        run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})
        model.answer_with([{MOTOR: index} for index in range(5)])
        for tick in range(6):
            clock.advance_to_ns(tick * PERIOD_NS)
            if tick == 5:
                run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})
                model.answer_with([{MOTOR: index} for index in range(5)])
            assert run.send({keys.ROBOT_STATUS: RobotStatus.ERROR}).commands == {MOTOR: tick % 5}
        assert len(model.calls) == 2
    finally:
        run.close()


@pytest.mark.parametrize(
    'options',
    [{}, {'record_stats': False}, {'replan_on_error': False}, {'record_stats': False, 'replan_on_error': False}],
)
@pytest.mark.parametrize('version', [3, LATEST])
def test_v3_specs_preserve_playback_and_recovery_options(execution, monkeypatch, options, version):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    declared = component(CHUNKED_SCHEDULE, version=version, fps=10, horizon_sec=0.15, inference_lead=0.05, **options)
    schedule = spec.from_spec(declared)
    expected = Sequential(
        TrajectoryPlayer(
            record_stats=options.get('record_stats', True), replan_on_error=options.get('replan_on_error', True)
        ),
        ChunkedSchedule(fps=10, horizon_sec=0.15, inference_lead=ConstantDuration(0.05)),
    )
    assert schedule.to_spec() == expected.to_spec()
    run = runtime.start(schedule, Mock())
    try:
        run.send({})
        model.answer_with([{MOTOR: index} for index in range(5)])
        assert run.send({}) == Step({MOTOR: 0}, PERIOD_NS)
        clock.advance_to_ns(PERIOD_NS // 2)
        step = run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})
        replans = options.get('replan_on_error', True)
        if replans:
            model.answer_with([{MOTOR: index} for index in range(5)])
            step = run.send({keys.ROBOT_STATUS: RobotStatus.ERROR})
        assert step.commands == ({MOTOR: 0} if replans else {})
        assert len(model.calls) == (2 if replans else 1)
        assert (f'{eval_keys.SCHEDULE}.{eval_keys.SCHEDULED}' in runtime.metadata_snapshot()) == options.get(
            'record_stats', True
        )
    finally:
        run.close()


def test_legacy_rtc_reports_measured_delays_and_resets_its_estimate_between_runs(execution, monkeypatch):
    runtime, clock = execution
    pending = []
    seen = []

    def submit(function, obs):
        future = Future()
        pending.append((future, function, obs))
        return _FutureAnswer(future, runtime)

    def infer(obs):
        seen.append(dict(obs))
        return [{MOTOR: i} for i in range(10)]

    monkeypatch.setattr(runtime, 'submit', submit)
    stack = spec.from_spec(
        component(
            'rtc_schedule',
            fps=10,
            call_after_sec=0.0,
            latency_estimate=DelayEstimate.mean(last=2, max_sec=0.25).to_spec(),
        )
    )
    for _ in range(2):
        seen.clear()
        run = runtime.start(stack, infer)
        try:
            run.send({})
            for delay in (0.1, 0.6, 0.2, 0.1, 0.1):
                future, function, obs = pending.pop(0)
                commands = function(obs)
                clock.advance_to_ns(runtime.time_ns + round(delay * 1e9))
                future.set_result(commands)
                run.send({})
            assert [obs[RTCScheduleV1.MEASURED_LATENCY_SEC] for obs in seen] == [None, 0.1, 0.6, 0.2, 0.1]
            assert [obs[RTCScheduleV1.ESTIMATED_LATENCY_SEC] for obs in seen] == pytest.approx([
                0.0,
                0.1,
                0.25,
                0.25,
                0.15,
            ])
        finally:
            run.close()
            pending.clear()


def test_a_slow_answer_is_asked_for_when_the_last_waypoint_is_due(execution, monkeypatch):
    runtime, clock = execution
    model = ObservingModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    rig = Rig(
        runtime,
        clock,
        Sequential(
            TrajectoryPlayer(replan_on_error=True),
            ChunkedSchedule(fps=10, inference_lead=DelayEstimate.mean(last=1, max_sec=0.1)),
        ),
    )
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
    rig = Rig(
        runtime,
        clock,
        Sequential(
            TrajectoryPlayer(replan_on_error=True),
            ChunkedSchedule(fps=10, inference_lead=DelayEstimate.mean(last=1, max_sec=0.1)),
        ),
    )
    try:
        rig.tick(0)
        clock.advance_to_ns(50_000_000)
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
    rig = Rig(
        runtime,
        clock,
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=100, inference_lead=ConstantDuration(0.01))
        ),
    )
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
        return _FutureAnswer(pending, runtime)

    monkeypatch.setattr(runtime, 'submit', submit)
    rig = Rig(
        runtime,
        clock,
        Sequential(
            TrajectoryPlayer(replan_on_error=True),
            ChunkedSchedule(fps=10, inference_lead=DelayEstimate.mean(last=1, max_sec=0.1)),
        ),
    )
    try:
        rig.tick(0)
        clock.advance_to_ns(100_000_000)
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
def test_prefetched_chunk_resets_gap_when_execution_crosses_its_boundary(
    execution, monkeypatch, first_new_ms, new_index
):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.2))
        ),
        Mock(),
    )
    try:
        run.send({})
        model.answer_with([{MOTOR: (1, i)} for i in range(5)])
        for index in range(5):
            clock.advance_to_ns(index * PERIOD_NS)
            assert run.send({}).commands == {MOTOR: (1, index)}
            if index == 3:
                model.answer_with([{MOTOR: (2, i)} for i in range(5)])
        clock.advance_to_ns(first_new_ms * 1_000_000)
        assert run.send({}).commands == {MOTOR: (2, new_index)}
    finally:
        run.close()
    assert [time for time, _ in model.calls] == [0, 300_000_000]
    metadata = runtime.metadata_snapshot()
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.GAP_MAX_MS}'] == 100.0
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.SCHEDULED}'] == 10
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.EMITTED}'] == 6
    assert metadata[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == new_index


def test_answer_between_emits_keeps_the_old_chunk_gap_history(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.15))
        ),
        Mock(),
    )
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
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.25))
        ),
        Mock(),
    )
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
def test_a_lead_cannot_queue_more_than_the_next_chunk(execution, monkeypatch, lead_sec):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(lead_sec))
        ),
        Mock(),
    )
    try:
        run.send({})
        model.answer_with([{MOTOR: (1, i)} for i in range(3)])
        assert run.send({}) == Step({MOTOR: (1, 0)}, 0)
        model.answer_with([{MOTOR: (2, i)} for i in range(3)])
        assert run.send({}) == Step({}, PERIOD_NS)
        assert run.send({}) == Step({}, PERIOD_NS)
        assert [time for time, _ in model.calls] == [0, 0]
        for index in range(1, 9):
            clock.advance_to_ns(index * PERIOD_NS)
            assert run.send({}).commands == {MOTOR: (index // 3 + 1, index % 3)}
            if index in (3, 6):
                model.answer_with([{MOTOR: (len(model.calls), i)} for i in range(3)])
        assert [time for time, _ in model.calls] == [0, 0, 3 * PERIOD_NS, 6 * PERIOD_NS]
    finally:
        run.close()


@pytest.mark.parametrize(
    'lead, call_ns', [(DelayEstimate.mean(last=2), 1_050_000_000), (DelayEstimate.max(last=2), 1_000_000_000)]
)
def test_delay_based_leads_use_completed_calls(execution, monkeypatch, lead, call_ns):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(
        Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=lead)), Mock()
    )
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
        Sequential(
            TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
            RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=estimate),
        )
        if rtc
        else Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=estimate))
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
            assert model.requests[-1][CALL_TIMINGS] == [{CALL_ID: 0, ELAPSED_SEC: pytest.approx(delay_ns / 1e9)}]
            if rtc:
                assert len(model.calls[-1][1]) == 5
        finally:
            run.close()


def test_an_empty_chunk_resumes_inference_without_using_the_lead_as_a_delay(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    run = runtime.start(
        Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(1))),
        Mock(),
    )
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
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.25))
        ),
        Mock(),
    )
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


def test_lead_and_handoff_use_the_horizon_truncated_end(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = Sequential(
        TrajectoryPlayer(replan_on_error=True),
        ChunkedSchedule(fps=10, horizon_sec=0.15, inference_lead=ConstantDuration(0.1)),
    )
    run = runtime.start(schedule, Mock())
    try:
        run.send({})
        model.answer_with([{MOTOR: i} for i in range(5)])
        assert run.send({}) == Step({MOTOR: 0}, 50_000_000)
        clock.advance_to_ns(50_000_000)
        run.send({})
        model.answer_with([{MOTOR: i} for i in range(5)])
        assert run.send({}) == Step({}, 100_000_000)
        clock.advance_to_ns(100_000_000)
        assert run.send({}) == Step({MOTOR: 1}, 150_000_000)
        clock.advance_to_ns(150_000_000)
        assert run.send({}) == Step({MOTOR: 0}, 200_000_000)
        assert [time for time, _ in model.calls] == [0, 50_000_000]
    finally:
        run.close()


def _rtc(execution, monkeypatch, call_after_sec, latency_estimate):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = Sequential(
        TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
        RTCSchedule(fps=10, call_after_sec=call_after_sec, latency_estimate=latency_estimate),
    )
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


@pytest.mark.parametrize('length', [0, 1, 3])
def test_rtc_discards_an_answer_whose_entire_prediction_has_elapsed(execution, monkeypatch, length):
    model, run, step = _rtc(execution, monkeypatch, 0.2, ConstantDuration())
    try:
        step(0)
        model.answer('old')
        step(1)
        step(3)
        model.answer('expired', length=length)
        assert step(6).commands == {}
        assert step(7).commands == {}
        prefix = model.requests[-1][RTC.ACTION_PREFIX]
        assert prefix[RTC.PREFIX_ACTIONS] == []
        assert prefix[RTC.CHUNK_OFFSET_SEC] == pytest.approx(length / 10)
        assert prefix[RTC.OBSERVATION_OFFSET_SEC] == pytest.approx((length - 3) / 10)
    finally:
        run.close()


def test_rtc_history_only_answer_keeps_playback_and_reports_timing(execution, monkeypatch):
    model, run, step = _rtc(execution, monkeypatch, 0.2, DelayEstimate.mean(last=1))
    _, clock = execution
    try:
        step(0)
        model.answer_with(None)
        assert step(1).commands == {}
        model.answer('old')
        assert step(2).commands == {MOTOR: 'old0'}
        assert step(4).commands == {MOTOR: 'old2'}
        clock.advance_to_ns(5 * PERIOD_NS)
        model.answer_with(None)
        assert step(5).commands == {MOTOR: 'old3'}
        assert model.calls == [
            (0, []),
            (PERIOD_NS, []),
            (4 * PERIOD_NS, [f'old{i}' for i in range(2, 10)]),
            (5 * PERIOD_NS, [f'old{i}' for i in range(3, 10)]),
        ]
        assert model.requests[-1][CALL_TIMINGS] == [{CALL_ID: 2, ELAPSED_SEC: pytest.approx(0.1)}]
    finally:
        run.close()


@pytest.mark.parametrize('rtc', [False, True])
def test_scheduler_accepts_shared_waypoints_without_a_player(execution, monkeypatch, rtc):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = (
        RTCSchedule(fps=10, call_after_sec=0.2, latency_estimate=ConstantDuration())
        if rtc
        else ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.25))
    )
    run = runtime.start(schedule, Mock())
    try:
        run.send({policy_keys.PLAN: []})
        clock.advance_to_ns(100_000_000)
        model.answer('a', length=5)
        trajectory = Trajectory(**run.send({policy_keys.PLAN: []}).commands)
        assert trajectory.waypoints() == [Waypoint({MOTOR: f'a{i}'}, 100_000_000 + i * PERIOD_NS) for i in range(5)]
        clock.advance_to_ns(350_000_000)
        plan = [Waypoint({MOTOR: 'b0'}, 300_000_000), Waypoint({MOTOR: 'b1'}, 400_000_000)]
        run.send({policy_keys.PLAN: plan})
        request = model.requests[-1]
        assert request[RTC.ACTION_PREFIX] == {
            RTC.PREFIX_ACTIONS: [{MOTOR: 'b0'}, {MOTOR: 'b1'}],
            RTC.CHUNK_OFFSET_SEC: 0.2,
            RTC.OBSERVATION_OFFSET_SEC: -0.05,
        }
        assert policy_keys.PLAN not in request
    finally:
        run.close()


def test_rtc_uses_the_players_plan_and_consumes_its_key(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    definition = Sequential(
        TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
        RTCSchedule(fps=10, call_after_sec=0.2, latency_estimate=ConstantDuration()),
    )
    run = runtime.start(spec.from_spec(definition.to_spec()), Mock())
    try:
        clock.advance_to_ns(1_000_000_000)
        run.send({POSITION: 42})
        assert model.requests[0][RTC.ACTION_PREFIX] == {
            RTC.PREFIX_ACTIONS: [],
            RTC.CHUNK_OFFSET_SEC: 0.0,
            RTC.OBSERVATION_OFFSET_SEC: 0.0,
        }
        model.answer('old')
        run.send({POSITION: 42})
        clock.advance_to_ns(1_200_000_000)
        run.send({POSITION: 43})
        assert model.calls[-1] == (1_200_000_000, [f'old{i}' for i in range(2, 10)])
        assert all(policy_keys.PLAN not in request for request in model.requests)
        assert model.requests[-1][POSITION] == 43
        assert runtime.metadata_snapshot() == {}
    finally:
        run.close()


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


@pytest.mark.parametrize('seconds', [-0.1, float('nan'), float('inf')])
@pytest.mark.parametrize('chunk', [None, []])
def test_invalid_lead_is_rejected_even_without_a_new_trajectory(execution, monkeypatch, seconds, chunk):
    runtime, _ = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    estimate = Mock(spec=DurationEstimate)
    estimate.run.return_value.update.return_value = seconds
    run = runtime.start(ChunkedSchedule(fps=10, inference_lead=estimate), lambda obs: chunk)
    try:
        run.send({})
        model.answer_with(chunk)
        with pytest.raises(ValueError):
            run.send({})
    finally:
        run.close()


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
    schedule = Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=lead))
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
    schedule = Sequential(
        TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=PeakDuration())
    )
    with pytest.raises(NotImplementedError, match='PeakDuration has no wire spec'):
        schedule.to_spec()


def test_a_spec_without_a_lead_waits_until_the_chunk_end_even_after_a_slow_answer(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    declared = ChunkedSchedule(fps=10).to_spec()
    schedule = spec.from_spec(declared)
    assert schedule.to_spec() == declared
    run = runtime.start(Sequential(TrajectoryPlayer(), schedule), Mock())
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
        Sequential(
            TemporalStack(('a', 'b'), (-0.5, 0.0), pad_start=False),
            TrajectoryPlayer(replan_on_error=True),
            ChunkedSchedule(fps=10, horizon_sec=0.5),
        ),
        Sequential(
            TrajectoryPlayer(replan_on_error=True),
            ChunkedSchedule(fps=10),
            TransformImages(64, 48, mode=ResizeMode.BOUND),
        ),
        Sequential(
            TrajectoryPlayer(replan_on_error=True),
            ChunkedSchedule(fps=10),
            TransformImages(64, 48, resample=Resample.AREA, mode=ResizeMode.BOUND),
        ),
        Sequential(TrajectoryPlayer(replan_on_error=True, record_stats=False), ChunkedSchedule(fps=10)),
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
        'periodic': Periodic(fps=10),
        'inference_schedule': InferenceSchedule(
            fps=50,
            request_timing=RequestTiming.every(0.1),
            merge=TrajectoryMerge.at_observation(),
            latency_estimate=DelayEstimate.max(),
        ),
        'trajectory_player': TrajectoryPlayer(),
        'encode_images': EncodeImages([['camera']]),
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
        'rtc_schedule': RTCScheduleV1(fps=10, call_after_sec=0.5, latency_estimate=DelayEstimate.max().to_spec()),
        'state_observation': StateObservationCodec(state={}),
        'select_observation_fields': SelectObservationFields(mapping={}),
        'rename_observation_fields': RenameObservationFields({'state': 'input'}),
        'absolute_position_action': AbsolutePositionAction(keys.TARGET_EE_POSE, keys.TARGET_GRIP),
        'cartesian_delta_action': CartesianDeltaAction(),
        'absolute_joints_action': AbsoluteJointsAction(keys.TARGET_JOINTS, keys.TARGET_GRIP),
        'bimanual_joints_action': BimanualJointsAction(),
        'joint_delta_action': JointDeltaAction(),
        'joint_position_action': JointPositionAction('joints', 'grip'),
        'set_control_mode': SetControlMode(PositionControl()),
        'change_ee_frame': ChangeEEFrame(EEFrame(site=GRIPPER_SITE)),
        'convert_pose': ConvertPose('rot6d'),
    }
    registered = spec.COMPONENTS
    assert set(instances) | {
        CHUNKED_SCHEDULE,
        'stop_on_fault',
        'restrict_image_size',
        'pack_observation_fields',
        'unpack_action_chunk',
    } == set(registered)
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
    old = component(CHUNKED_SCHEDULE, version=2, fps=20, horizon_sec=0.3, record_stats=False, inference_lead=lead)
    current = sequence(
        component(TrajectoryPlayer.WIRE_NAME, record_stats=False, replan_on_error=False),
        ChunkedSchedule(fps=20, horizon_sec=0.3, inference_lead=DurationEstimate.from_spec(lead)).to_spec(),
    )
    assert spec.from_spec(old).to_spec() == spec.from_spec(current).to_spec()
    assert old[VERSION] == 2


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
    name = Metadata.WIRE_NAME
    notice = Deprecation(date(2020, 1, 1), date(2020, 7, 1), 'Use another scheduler.')
    versions = {1: spec.COMPONENTS[name][1], 2: Version(None, notice, removed_on=date(2020, 7, 1))}
    monkeypatch.setitem(spec.COMPONENTS, name, versions)
    latest = component(name, values={})
    assert spec.from_spec(latest).to_spec()[VERSION] == 1
    with pytest.raises(ValueError, match='has been removed.*Use another scheduler'):
        spec.from_spec(component(name, version=2, values={}))
    del versions[1]
    with pytest.raises(ValueError, match='No supported versions'):
        spec.from_spec(latest)


def test_unversioned_wire_specs_still_select_v1():
    assert isinstance(spec.from_spec({NAME: Metadata.WIRE_NAME, ARGS: {'values': {}}}), Metadata)
    with pytest.raises(ValueError, match='Unsupported .* version 1'):
        spec.from_spec({NAME: CHUNKED_SCHEDULE, ARGS: {ChunkedSchedule.FPS_ARG: 20}})


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


def _answer_after(runtime, monkeypatch, *latencies_ns):
    """Make call k answer ``latencies_ns[k]`` after its submission; later calls take the last latency."""
    calls = []

    def submit(function, *args):
        latency_ns = latencies_ns[min(len(calls), len(latencies_ns) - 1)]
        calls.append(latency_ns)
        return DueAnswer(runtime, function(*args), runtime.time_ns + latency_ns)

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
    trajectory = Trajectory([{MOTOR: 1}, _ending(2)], 0, 10, 2 * PERIOD_NS)
    inner = Mock(send=Mock(return_value=Step(asdict(trajectory, recurse=False), 2 * PERIOD_NS)))
    run = runtime.start(Sequential(EndOnField(), TrajectoryPlayer()), inner)
    try:
        assert run.send({}).commands == {MOTOR: 1}
        clock.advance_to_ns(PERIOD_NS)
        assert run.send({}) == Step({MOTOR: 2}, PERIOD_NS)
        with pytest.raises(StopIteration) as end:
            run.send({})
        assert end.value.value is PolicyEnd.COMPLETE
    finally:
        run.close()


def test_end_on_field_preserves_auxiliary_outputs_on_the_final_step(execution):
    runtime, _ = execution
    outputs = {'sample': object()}

    def inner():
        yield
        yield Step(_ending(42), 123, outputs=outputs)

    inner_run = inner()
    next(inner_run)
    run = runtime.start(EndOnField(), inner_run)
    try:
        assert run.send({}) == Step({MOTOR: 42}, runtime.time_ns, outputs=outputs)
        with pytest.raises(StopIteration) as end:
            run.send({})
        assert end.value.value is PolicyEnd.COMPLETE
    finally:
        run.close()
        inner_run.close()


def test_a_schedule_passes_end_up_as_a_command(execution, monkeypatch):
    runtime, clock = execution
    _answer_after(runtime, monkeypatch, PERIOD_NS)
    run = runtime.start(
        Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10)),
        lambda obs: [{MOTOR: 1}, {MOTOR: 2, END: 'complete'}],
    )
    try:
        run.send({})
        clock.advance_to_ns(PERIOD_NS)
        run.send({})
        clock.advance_to_ns(2 * PERIOD_NS)
        assert run.send({}).commands == {MOTOR: 2, END: 'complete'}
        clock.advance_to_ns(3 * PERIOD_NS)
        run.send({})
    finally:
        run.close()


def test_a_declared_end_on_field_ends_a_declared_stack(execution, monkeypatch):
    runtime, clock = execution
    _answer_after(runtime, monkeypatch, PERIOD_NS)
    pose = np.array([0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5])
    declared = sequence(
        component(EndOnField.WIRE_NAME),
        component(TrajectoryPlayer.WIRE_NAME, replan_on_error=True),
        ChunkedSchedule(fps=10).to_spec(),
        parallel(
            component(AbsolutePositionAction.WIRE_NAME, tgt_ee_pose_key='pose', tgt_grip_key='grip'),
            component(EndCommand.WIRE_NAME),
        ),
    )
    model = lambda obs: [{ACTION: pose}, {ACTION: pose, END: 'give_up'}]  # noqa: E731
    run = runtime.start(cast(Policy, spec.from_spec(declared)), model)
    try:
        run.send({})
        clock.advance_to_ns(PERIOD_NS)
        assert run.send({}).commands.keys() == {keys.ROBOT_COMMAND, keys.TARGET_GRIP}
        clock.advance_to_ns(2 * PERIOD_NS)
        assert run.send({}).commands.keys() == {keys.ROBOT_COMMAND, keys.TARGET_GRIP}
        with pytest.raises(StopIteration) as end:
            run.send({})
        assert end.value.value is PolicyEnd.GIVE_UP
    finally:
        run.close()
    assert EndOnField().to_spec() == {NAME: EndOnField.WIRE_NAME, VERSION: EndOnField.WIRE_VERSION}


def test_end_on_field_refuses_a_malformed_end_when_its_element_plays(execution):
    runtime, _ = execution
    inner = Mock(send=Mock(return_value=Step({MOTOR: 1, END: 'done'}, 0)))
    run = runtime.start(EndOnField(), inner)
    try:
        with pytest.raises(ValueError):
            run.send({})
    finally:
        run.close()


@pytest.mark.parametrize(('latency_ns', 'sent'), [(PERIOD_NS, [None, 1, 3, (1, 0)]), (4 * PERIOD_NS, [None, 1, 3, 5])])
def test_a_late_round_ends_the_run_on_the_next_end_that_plays(execution, monkeypatch, latency_ns, sent):
    runtime, _ = execution
    model = _EndingModel(runtime, [{MOTOR: 1}, {MOTOR: 2}, {MOTOR: 3}, _ending(4), {MOTOR: 5}])
    _answer_after(runtime, monkeypatch, PERIOD_NS, latency_ns)
    times_ns = [0, PERIOD_NS, 3 * PERIOD_NS, 6 * PERIOD_NS, 7 * PERIOD_NS]
    assert _play(
        execution,
        Sequential(
            TrajectoryPlayer(replan_on_error=True, record_stats=False),
            ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.3)),
        ),
        model,
        times_ns,
    ) == (sent, PolicyEnd.COMPLETE)
    assert model.calls == [0, 3 * PERIOD_NS]


# Each schedule asks for the next chunk before the end plays: one more call, and no more motion.
@pytest.mark.parametrize(
    ('schedule', 'calls'),
    [
        (
            lambda: Sequential(
                TrajectoryPlayer(replan_on_error=True, record_stats=False),
                ChunkedSchedule(fps=10, inference_lead=ConstantDuration(0.1)),
            ),
            [0, 2 * PERIOD_NS],
        ),
        (
            lambda: Sequential(
                TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
                RTCSchedule(fps=10, call_after_sec=0.1, latency_estimate=ConstantDuration()),
            ),
            [0, 2 * PERIOD_NS],
        ),
    ],
    ids=['chunked', 'rtc'],
)
def test_a_queued_end_plays_on_its_own_element(execution, monkeypatch, schedule, calls):
    runtime, _ = execution
    _answer_after(runtime, monkeypatch, PERIOD_NS)
    model = _EndingModel(runtime, [{MOTOR: 1}, _ending(2)])
    sent, end = _play(execution, schedule(), model, [i * PERIOD_NS for i in range(5)])
    assert (sent[-1], end) == (2, PolicyEnd.COMPLETE)
    assert model.calls == calls


def test_an_end_past_the_horizon_plays_on_the_next_chunk(execution, monkeypatch):
    runtime, _ = execution
    _answer_after(runtime, monkeypatch, PERIOD_NS)
    model = _EndingModel(runtime, [{MOTOR: 1}, _ending(2)], on='first')
    schedule = Sequential(
        TrajectoryPlayer(replan_on_error=True, record_stats=False), ChunkedSchedule(fps=10, horizon_sec=0.1)
    )
    assert _play(execution, schedule, model, [i * PERIOD_NS for i in range(5)]) == (
        [None, 1, None, (1, 0)],
        PolicyEnd.COMPLETE,
    )


def test_a_replan_after_an_arm_error_ends_on_the_replanned_chunk(execution, monkeypatch):
    runtime, _ = execution
    _answer_after(runtime, monkeypatch, PERIOD_NS)
    model = _EndingModel(runtime, [{MOTOR: i} for i in range(3)] + [_ending(3), {MOTOR: 4}], on='first')
    error = {2 * PERIOD_NS: {keys.ROBOT_STATUS: RobotStatus.ERROR}}
    sent, end = _play(
        execution,
        Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10)),
        model,
        [i * PERIOD_NS for i in range(6)],
        error,
    )
    assert (sent, end) == ([None, 0, None, (1, 0)], PolicyEnd.COMPLETE)


# Answers arrive 2.5 periods late, so RTC skips each new chunk's first two elements.
@pytest.mark.parametrize(('on', 'end'), [('every', PolicyEnd.COMPLETE), ('first', None)])
def test_rtc_plays_an_end_only_from_an_element_that_it_does_not_skip(execution, monkeypatch, on, end):
    runtime, _ = execution
    model = _EndingModel(runtime, [_ending(i) if i == 6 else {MOTOR: i} for i in range(10)], on=on)
    _answer_after(runtime, monkeypatch, 5 * PERIOD_NS // 2)
    schedule = Sequential(
        TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
        RTCSchedule(fps=10, call_after_sec=0.1, latency_estimate=ConstantDuration(0.25)),
    )
    sent, ended = _play(execution, schedule, model, [i * PERIOD_NS // 2 for i in range(60)])
    assert ended is end
    assert 6 not in sent


def test_the_rtc_prefix_carries_end_like_any_other_command(execution):
    runtime, clock = execution
    model = _EndingModel(runtime, [{MOTOR: 1}, _ending(2), {MOTOR: 3}])
    run = runtime.start(
        Sequential(
            TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
            RTCSchedule(fps=10, call_after_sec=0, latency_estimate=ConstantDuration(0.3)),
        ),
        model,
    )
    try:
        run.send({})
        clock.advance_to_ns(PERIOD_NS // 2)
        run.send({})
    finally:
        run.close()
    assert model.prefixes == [[], [{MOTOR: 1}, _ending(2), {MOTOR: 3}]]


def test_rtc_sends_completed_call_timings_with_the_remaining_plan(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = Sequential(
        TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
        RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=ConstantDuration(0.2)),
    )
    run = runtime.start(schedule, Mock())
    try:
        run.send({})
        clock.advance_to_ns(100_000_000)
        model.answer('old')
        run.send({})
        clock.advance_to_ns(600_000_000)
        run.send({})
        first = model.requests[0][RTC.ACTION_PREFIX]
        second = model.requests[1][RTC.ACTION_PREFIX]
        assert first[RTC.PREFIX_ACTIONS] == []
        assert second[RTC.CHUNK_OFFSET_SEC] == 0.5
        assert second[RTC.OBSERVATION_OFFSET_SEC] == 0.0
        assert model.requests[0][CALL_TIMINGS] == []
        assert model.requests[1][CALL_TIMINGS] == [{CALL_ID: 0, ELAPSED_SEC: 0.1}]
        assert [action[MOTOR] for action in second[RTC.PREFIX_ACTIONS]] == [f'old{i}' for i in range(5, 10)]
    finally:
        run.close()


@pytest.mark.parametrize('rtc', [False, True])
def test_late_answer_reads_do_not_inflate_duration_estimates(execution, monkeypatch, rtc):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    estimate = DelayEstimate.mean(last=1, max_sec=1.0)
    schedule = (
        Sequential(
            TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
            RTCSchedule(fps=10, call_after_sec=0.2, latency_estimate=estimate),
        )
        if rtc
        else Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=10, inference_lead=estimate))
    )
    run = runtime.start(schedule, Mock())
    try:
        run.send({})
        clock.advance_to_ns(100_000_000)
        model.answer('old')
        clock.advance_to_ns(300_000_000)
        run.send({})
        next_call_ns = 500_000_000 if rtc else 1_200_000_000
        clock.advance_to_ns(next_call_ns - 1)
        run.send({})
        assert len(model.calls) == 1
        clock.advance_to_ns(next_call_ns)
        run.send({})
        assert [time_ns for time_ns, _ in model.calls] == [0, next_call_ns]
        assert model.requests[0][CALL_TIMINGS] == []
        assert model.requests[1][CALL_TIMINGS] == [{CALL_ID: 0, ELAPSED_SEC: 0.1}]
        if not rtc:
            assert RTC.ACTION_PREFIX not in model.requests[1]
    finally:
        run.close()


def test_rtc_description_preserves_commands_without_a_model_codec(execution, monkeypatch):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    description = Sequential(
        TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
        RTCSchedule(fps=10, call_after_sec=0.5, latency_estimate=ConstantDuration(0.2)),
    ).to_spec()
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
        return _FutureAnswer(Future(), runtime)

    monkeypatch.setattr(runtime, 'submit', submit)
    stack = Mock(wraps=np.stack)
    monkeypatch.setattr(np, 'stack', stack)
    schedule = Sequential(
        TemporalStack(('camera',), (-0.1, 0.0)),
        TrajectoryPlayer(plan_key=policy_keys.PLAN, record_stats=False),
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
