"""Scheduling, fault gating, temporal history, and deliverable stack specifications."""

from concurrent.futures import Future
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

import pimm
from pimm.world import VirtualClock
from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import CartesianPosition, JointPosition, PositionControl
from positronic.eval import keys as eval_keys
from positronic.geom import Rotation, Transform3D
from positronic.policy import spec
from positronic.policy.base import Policy, Step
from positronic.policy.codecs import (
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    EncodeImages,
    FlipGrip,
    Metadata,
    RestrictImageSize,
)
from positronic.policy.codecs.action import (
    AbsoluteJointsAction,
    AbsolutePositionAction,
    JointDeltaAction,
    JointPositionAction,
    SetControlMode,
    UnpackActionChunk,
)
from positronic.policy.codecs.geometry import ConvertPose
from positronic.policy.codecs.observation import ObservationCodec, PackObservationFields, RenameObservationFields
from positronic.policy.executor import Executor, _UnchargedAnswer
from positronic.policy.processors import (
    ChunkedSchedule,
    PauseOnUnavailable,
    PrefixSampling,
    RTCSchedule,
    TemporalStack,
    max_delay,
    mean_delay,
)
from positronic.policy.sequential import Sequential

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
    assert runtime.metadata[f'{eval_keys.SCHEDULE}.{eval_keys.DROPPED}'] == 1


def test_an_overrun_skips_all_but_the_last_due_waypoint_and_counts_the_skip(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [{MOTOR: i} for i in range(5)])
    emitted = []
    for now in (0, 100_000_000, 350_000_000, 400_000_000):
        clock.advance_to_ns(now)
        emitted.append(run.send({}).commands[MOTOR])
    assert emitted == [0, 1, 3, 4]
    prefix = eval_keys.SCHEDULE
    assert runtime.metadata == {
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
    assert runtime.metadata == {}


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
    assert runtime.metadata[f'{prefix}.{eval_keys.SCHEDULED}'] == 10
    assert runtime.metadata[f'{prefix}.{eval_keys.EMITTED}'] == 2
    assert runtime.metadata[f'{prefix}.{eval_keys.DROPPED}'] == 4


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
    assert runtime.metadata[f'{prefix}.{eval_keys.LATE_P50_MS}'] == pytest.approx(p50)
    assert runtime.metadata[f'{prefix}.{eval_keys.LATE_P90_MS}'] == pytest.approx(p90)
    assert runtime.metadata[f'{prefix}.{eval_keys.LATE_MAX_MS}'] == max(late_ms)


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
        self._pending: Future | None = None

    def submit(self, function, obs, prefix):
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


def _rtc(execution, monkeypatch, call_after_sec, prefix_duration, prefix_sampling=PrefixSampling.PREVIOUS):
    runtime, clock = execution
    model = SlowModel(runtime)
    monkeypatch.setattr(runtime, 'submit', model.submit)
    schedule = RTCSchedule(
        fps=10, call_after_sec=call_after_sec, prefix_duration=prefix_duration, prefix_sampling=prefix_sampling
    )
    run = runtime.start(schedule, Mock())

    def step(index):
        clock.advance_to_ns(index * PERIOD_NS)
        return run.send({})

    return model, run, step


def test_rtc_follows_the_drawing(execution, monkeypatch):
    seen_delays = []

    def prefix_duration(delays):
        seen_delays.append(list(delays))
        return mean_delay()(delays)

    model, run, step = _rtc(execution, monkeypatch, 0.5, prefix_duration)
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
    assert model.calls == [(0, []), (8 * PERIOD_NS, ['o5', 'o6', 'o7']), (13 * PERIOD_NS, ['n5', 'n6', 'n7'])]
    assert seen_delays == [[0.3], [0.3, 0.3]]


def test_rtc_executes_nothing_between_the_old_chunk_end_and_a_slow_answer(execution, monkeypatch):
    model, run, step = _rtc(execution, monkeypatch, 0.2, mean_delay())
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
    assert model.calls == [(0, []), (3 * PERIOD_NS, ['o2']), (8 * PERIOD_NS, ['n5', 'n6', 'n7'])]


@pytest.mark.parametrize(
    ('prefix_duration', 'prefix'),
    [(mean_delay(), ['n3', 'n4']), (max_delay(), ['n3', 'n4', 'n5']), (max_delay(max_sec=0.1), ['n3'])],
)
def test_rtc_calls_at_once_after_a_late_answer(execution, monkeypatch, prefix_duration, prefix):
    model, run, step = _rtc(execution, monkeypatch, 0.2, prefix_duration)
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
    assert model.calls == [(0, []), (3 * PERIOD_NS, ['o2']), (6 * PERIOD_NS, prefix)]


@pytest.mark.parametrize(
    ('call_after_sec', 'sampling', 'prefix'),
    [
        (0.4, PrefixSampling.PREVIOUS, ['o4', 'o5']),
        (0.4, PrefixSampling.NEXT, ['o4', 'o5']),
        (0.52, PrefixSampling.PREVIOUS, ['o5']),
        (0.52, PrefixSampling.NEAREST, ['o5']),
        (0.52, PrefixSampling.NEXT, []),
    ],
)
def test_rtc_prefix_stops_at_the_end_of_the_old_chunk(execution, monkeypatch, call_after_sec, sampling, prefix):
    model, run, step = _rtc(execution, monkeypatch, call_after_sec, lambda delays: 0.5, sampling)
    _, clock = execution
    call_ns = round((0.1 + call_after_sec) * 1e9)
    try:
        for index in range(6):
            if index == 1:
                model.answer('o', 6)
            step(index)
        clock.advance_to_ns(call_ns)
        run.send({})
    finally:
        run.close()
    assert model.calls == [(0, []), (call_ns, prefix)]


@pytest.mark.parametrize(
    ('call_after_sec', 'sampling', 'prefix'),
    [
        (0.42, PrefixSampling.PREVIOUS, ['o4', 'o5', 'o6']),
        (0.42, PrefixSampling.NEAREST, ['o4', 'o5', 'o6']),
        (0.42, PrefixSampling.NEXT, ['o5', 'o6', 'o7']),
        (0.45, PrefixSampling.NEAREST, ['o4', 'o5', 'o6']),
        (0.47, PrefixSampling.PREVIOUS, ['o4', 'o5', 'o6']),
        (0.47, PrefixSampling.NEAREST, ['o5', 'o6', 'o7']),
        (0.47, PrefixSampling.NEXT, ['o5', 'o6', 'o7']),
    ],
)
def test_rtc_prefix_reads_the_old_chunk_at_the_new_due_times(execution, monkeypatch, call_after_sec, sampling, prefix):
    model, run, step = _rtc(execution, monkeypatch, call_after_sec, lambda delays: 0.3, sampling)
    _, clock = execution
    call_ns = round((0.1 + call_after_sec) * 1e9)
    try:
        for index in range(6):
            if index == 1:
                model.answer('o')
            step(index)
        clock.advance_to_ns(call_ns)
        run.send({})
    finally:
        run.close()
    assert model.calls == [(0, []), (call_ns, prefix)]


def _interpolation_action(i):
    return {
        MOTOR: 10.0 * i,
        'joints': JointPosition(positions=np.array([i, 2.0 * i])),
        'pose': CartesianPosition(
            pose=Transform3D(np.array([i, 0.0, 0.0]), Rotation.from_rotvec(np.array([0.0, 0.0, 0.1 * i])))
        ),
        'label': f'x{i}',
    }


def test_rtc_interpolates_each_command_type(execution, monkeypatch):
    model, run, step = _rtc(execution, monkeypatch, 0.42, lambda delays: 0.1, PrefixSampling.INTERPOLATE)
    _, clock = execution
    try:
        for index in range(6):
            if index == 1:
                model.answer_with([_interpolation_action(i) for i in range(10)])
            step(index)
        clock.advance_to_ns(520_000_000)
        run.send({})
    finally:
        run.close()
    [action] = model.prefixes[1]
    assert action[MOTOR] == pytest.approx(42.0)
    np.testing.assert_allclose(action['joints'].positions, [4.2, 8.4])
    np.testing.assert_allclose(action['pose'].pose.translation, [4.2, 0.0, 0.0])
    np.testing.assert_allclose(action['pose'].pose.rotation.as_rotvec, [0.0, 0.0, 0.42])
    assert action['label'] == 'x4'


def test_rtc_interpolation_holds_the_last_old_action_after_its_due_time(execution, monkeypatch):
    model, run, step = _rtc(execution, monkeypatch, 0.52, lambda delays: 0.1, PrefixSampling.INTERPOLATE)
    _, clock = execution
    try:
        for index in range(6):
            if index == 1:
                model.answer_with([_interpolation_action(i) for i in range(6)])
            step(index)
        clock.advance_to_ns(620_000_000)
        run.send({})
    finally:
        run.close()
    [action] = model.prefixes[1]
    assert action[MOTOR] == 50.0


@pytest.mark.parametrize('duration', [mean_delay, max_delay])
def test_prefix_durations_refuse_a_negative_cap(duration):
    with pytest.raises(ValueError, match='max_sec'):
        duration(max_sec=-0.1)


@pytest.mark.parametrize(
    ('prefix_duration', 'expected'),
    [
        (mean_delay(last=2), 0.2),
        (max_delay(last=2), 0.3),
        (mean_delay(last=2, max_sec=0.15), 0.15),
        (max_delay(max_sec=0.4), 0.4),
    ],
)
def test_prefix_durations_read_the_last_delays_and_cap_them(prefix_duration, expected):
    assert prefix_duration([0.9, 0.1, 0.3]) == pytest.approx(expected)


@pytest.mark.parametrize(
    'definition',
    [
        Sequential(TemporalStack(('a', 'b'), (-0.5, 0.0), pad_start=False), ChunkedSchedule(fps=10, horizon_sec=0.5)),
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps=10), RestrictImageSize(64, 48)),
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
        'temporal_stack': TemporalStack(('v',), (0.0,)),
        'binarize_grip_training': BinarizeGripTraining(('grip',)),
        'binarize_grip_inference': BinarizeGripInference(),
        'flip_grip': FlipGrip(),
        'metadata': Metadata({'action_fps': 15}),
        'restrict_image_size': RestrictImageSize(),
        'observation_codec': ObservationCodec(state={}, images={}),
        'pack_observation_fields': PackObservationFields({'state': 'input'}),
        'rename_observation_fields': RenameObservationFields({'state': 'input'}),
        'absolute_position_action': AbsolutePositionAction(keys.TARGET_EE_POSE, keys.TARGET_GRIP),
        'absolute_joints_action': AbsoluteJointsAction(keys.TARGET_JOINTS, keys.TARGET_GRIP),
        'joint_delta_action': JointDeltaAction(),
        'joint_position_action': JointPositionAction('joints', 'grip'),
        'set_control_mode': SetControlMode(PositionControl()),
        'unpack_action_chunk': UnpackActionChunk({'action': ['prediction']}),
        'change_ee_frame': ChangeEEFrame(Transform3D.identity),
        'convert_pose': ConvertPose('rot6d'),
    }
    registered = spec.COMPONENTS
    assert set(instances) == set(registered)
    for name, instance in instances.items():
        assert instance.to_spec()['name'] == name
        assert type(instance) is registered[name][instance.WIRE_VERSION].implementation
        assert instance.to_spec()['version'] == instance.WIRE_VERSION
