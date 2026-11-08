"""Completion delivery and time accounting without a control-system dependency."""

import contextvars
import threading
from contextlib import closing
from math import inf, nan
from typing import cast

import pytest

from positronic import telemetry, telemetry_keys
from positronic.policy import executor as module
from positronic.policy.base import NotAnswered, Policy, PolicyEnd, QueueFull, Step
from positronic.policy.executor import ExecutionRuntime, Executor, WaitResult, WaitStatus, _UnchargedAnswer


@pytest.fixture
def executors():
    created = []

    def make(*, simulated=True, charged=False, fixed_inference_sec=None, max_in_flight=16):
        now = [0]
        runtime = Executor(
            lambda: now[0],
            simulated=simulated,
            inference_charge_sec=fixed_inference_sec if charged else 0,
            max_in_flight=max_in_flight,
        )
        created.append(runtime)
        return runtime, now

    yield make
    for runtime in created:
        runtime.close()


def test_set_metadata_copies_nested_values_and_replaces_previous_value(executors):
    runtime, _ = executors()
    config = {'labels': ['original']}
    runtime.set_metadata('config', config)
    config['labels'].append('changed')
    assert runtime.metadata_snapshot() == {'config': {'labels': ['original']}}

    runtime.set_metadata('config', {'enabled': False})
    assert runtime.metadata_snapshot() == {'config': {'enabled': False}}


def test_metadata_changes_only_on_explicit_writes(executors):
    runtime, _ = executors()
    events = []
    runtime.set_metadata('events', events)
    events.append({'labels': ['first']})
    assert runtime.metadata_snapshot() == {'events': []}

    runtime.set_metadata('events', events)
    events[0]['labels'].append('changed')
    assert runtime.metadata_snapshot() == {'events': [{'labels': ['first']}]}


def test_metadata_is_owned_by_each_runtime(executors):
    first, _ = executors()
    second, _ = executors()
    first.set_metadata('status', 'running')
    assert second.metadata_snapshot() == {}


def test_metadata_snapshot_is_independent_of_runtime_writes_and_caller_mutations(executors):
    runtime, _ = executors()
    runtime.set_metadata('config', {'labels': ['original']})
    snapshot = runtime.metadata_snapshot()
    snapshot['config']['labels'].append('changed')
    snapshot['local'] = True
    assert runtime.metadata_snapshot() == {'config': {'labels': ['original']}}

    runtime.set_metadata('config', {'enabled': False})
    runtime.set_metadata('status', 'running')
    assert snapshot == {'config': {'labels': ['original', 'changed']}, 'local': True}


def test_worker_span_keeps_parent_after_processor_yields(executors, tmp_path):
    runtime, _ = executors()
    release = threading.Event()

    def work():
        assert release.wait(timeout=5)
        with telemetry.span('worker_child'):
            return 42

    class Submit(Policy):
        def run(self, runtime):
            yield
            answer = runtime.submit(work)
            yield Step({}, 1)
            assert answer.result() == 42
            yield Step({}, 2)
            return PolicyEnd.COMPLETE

    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'async-stack'):
        run = runtime.start(Submit())
        try:
            assert run.send({}) == Step({}, 1)
            release.set()
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            assert run.send({}) == Step({}, 2)
        finally:
            release.set()
            runtime.close()
            run.close()
    spans = list(telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS)))
    calls = sorted((s for s in spans if s.name == 'submit'), key=lambda s: s.start_ns)
    [worker] = [s for s in spans if s.name == telemetry_keys.SPAN_POLICY_SUBMIT]
    [child] = [s for s in spans if s.name == 'worker_child']
    assert len(calls) == 2
    assert all(s.parent_id is None for s in calls)
    assert worker.parent_id == calls[0].span_id
    assert child.parent_id == worker.span_id
    assert calls[0].end_ns <= child.start_ns <= child.end_ns <= worker.end_ns <= calls[1].start_ns


@pytest.mark.parametrize('read_first', [False, True])
def test_completion_is_delivered_once_independently_of_result_reads(executors, read_first):
    runtime, _ = executors()
    answer = cast(_UnchargedAnswer[int], runtime.submit(lambda: 42))
    assert answer.call.result(timeout=1) == 42
    assert runtime.has_ready_answers()
    if read_first:
        assert answer.result() == 42
    assert runtime.has_ready_answers() is not read_first
    assert runtime.has_pending
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert not runtime.has_pending
    assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)
    assert runtime.has_ready_answers() is not read_first
    assert answer.result() == 42
    assert not runtime.has_ready_answers()
    assert runtime.take_completed() == ()


def test_submission_and_bounded_wait_do_not_wait_out_a_worker(executors):
    runtime, now = executors()
    release = threading.Event()
    try:
        answer = runtime.submit(lambda: release.wait(timeout=2))
        assert not answer.done()
        assert not runtime.has_ready_answers()
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.TIMED_OUT)
        assert runtime.take_completed() == ()
        assert now == [0]
        with pytest.raises(NotAnswered):
            answer.result()
        with pytest.raises(NotAnswered):
            _ = answer.elapsed_sec
        with pytest.raises(NotAnswered):
            _ = answer.execution_sec
    finally:
        release.set()
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)


def test_fast_call_completes_while_another_worker_is_pending(executors):
    runtime, _ = executors(max_in_flight=2)
    release = threading.Event()
    slow = runtime.submit(lambda: release.wait(timeout=2))
    try:
        fast = cast(_UnchargedAnswer[int], runtime.submit(lambda: 42))
        assert fast.call.result(timeout=1) == 42
        assert runtime.has_ready_answers()
        assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (fast,))
        assert runtime.has_ready_answers()
        with pytest.raises(QueueFull):
            runtime.submit(lambda: None)
        assert fast.result() == 42
        assert not runtime.has_ready_answers()
        assert runtime.has_pending
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.TIMED_OUT)
    finally:
        release.set()
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (slow,))


def test_charged_completion_is_hidden_until_its_episode_time(executors, monkeypatch):
    wall = [0]
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: wall[0])
    runtime, now = executors(charged=True)

    def work():
        wall[0] = 20_000_000
        return 42

    answer = cast(_UnchargedAnswer[int], runtime.submit(work))
    assert answer.call.result(timeout=1) == 42
    for instant in (0, 19_999_999):
        now[0] = instant
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)
        assert runtime.has_pending
        assert not answer.done()
        assert not runtime.has_ready_answers()
        with pytest.raises(NotAnswered):
            _ = answer.elapsed_sec
        with pytest.raises(NotAnswered):
            _ = answer.execution_sec
    now[0] = 20_000_000
    assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert runtime.has_ready_answers()
    assert answer.elapsed_sec == answer.execution_sec == 0.02
    assert answer.result() == 42
    assert not runtime.has_ready_answers()


@pytest.mark.parametrize('simulated, charged', [(False, False), (False, True), (True, True), (True, False)])
def test_durations_measure_the_call_and_ignore_late_reads(executors, monkeypatch, simulated, charged):
    wall = [0]
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: wall[0])
    runtime, now = executors(simulated=simulated, charged=charged)
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        assert release.wait(timeout=2)
        wall[0] = 50_000_000
        return 42

    answer = cast(_UnchargedAnswer[int], runtime.submit(work))
    try:
        assert started.wait(timeout=1)
    finally:
        release.set()
    assert answer.call.result(timeout=1) == 42
    now[0] = 50_000_000
    expected = 0.0 if simulated and not charged else 0.05
    assert answer.elapsed_sec == answer.execution_sec == expected
    runtime.take_completed()
    assert runtime.has_ready_answers()
    wall[0] = now[0] = 1_000_000_000
    assert answer.elapsed_sec == answer.execution_sec == expected
    assert answer.result() == 42
    assert not runtime.has_ready_answers()
    assert answer.elapsed_sec == answer.execution_sec == expected


def test_charged_wait_is_bounded_by_the_time_still_owed(executors, monkeypatch):
    wall = [0]
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: wall[0])
    runtime, now = executors(charged=True)
    release = threading.Event()
    runtime.submit(lambda: release.wait(timeout=2))
    waits = []

    def wait(futures, *, timeout, return_when):
        waits.append(timeout)
        wall[0] += round(timeout * 1e9)

    monkeypatch.setattr(module.concurrent.futures, 'wait', wait)
    try:
        now[0] = 10_000_000
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.TIMED_OUT)
        assert runtime.wait(timeout_sec=0.003) == WaitResult(WaitStatus.TIMED_OUT)
        assert runtime.wait(timeout_sec=0.1) == WaitResult(WaitStatus.CAN_ADVANCE)
        assert waits == pytest.approx([0.003, 0.007])
        assert now == [10_000_000]
    finally:
        release.set()


def test_an_early_fixed_charge_answer_shows_at_the_first_tick_after_the_charge(executors):
    runtime, now = executors(charged=True, fixed_inference_sec=0.1)
    answer = cast(_UnchargedAnswer[int], runtime.submit(lambda: 42))
    assert answer.call.result(timeout=1) == 42
    # A 15 Hz control tick: the charge ends at 100 ms, between the ticks at 66.7 ms and 133.3 ms.
    for instant in (0, 66_666_667):
        now[0] = instant
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)
        assert not answer.done()
    now[0] = 133_333_333
    assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert answer.elapsed_sec == answer.execution_sec == 0.1
    assert answer.result() == 42


def test_a_late_fixed_charge_answer_holds_the_simulator_until_the_worker_finishes(executors, monkeypatch):
    runtime, now = executors(charged=True, fixed_inference_sec=0.1)
    release = threading.Event()
    answer = cast(_UnchargedAnswer[int], runtime.submit(lambda: (release.wait(timeout=2), 42)[1]))
    try:
        now[0] = 66_666_667
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)
        now[0] = 133_333_333
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.TIMED_OUT)
        assert runtime.wait(timeout_sec=0.01) == WaitResult(WaitStatus.TIMED_OUT)
        assert not answer.done()
    finally:
        release.set()
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert now == [133_333_333]
    assert answer.elapsed_sec == answer.execution_sec == 0.1
    assert answer.result() == 42


@pytest.mark.parametrize('seconds', [-0.1, inf, nan])
def test_an_inference_charge_is_a_finite_time(seconds):
    with pytest.raises(ValueError, match='inference_charge_sec'):
        Executor(lambda: 0, simulated=True, inference_charge_sec=seconds)


@pytest.mark.parametrize('factor', [0.5, 0.0, inf, nan])
def test_slow_motion_is_a_finite_factor_of_at_least_one(factor):
    with pytest.raises(ValueError, match='slow_motion'):
        Executor(lambda: 0, simulated=True, inference_charge_sec=None, slow_motion=factor)


def test_an_execution_reads_the_slow_motion_of_its_runtime():
    runtime = Executor(lambda: 0, simulated=True, inference_charge_sec=None, slow_motion=2.5)
    with closing(ExecutionRuntime(runtime, max_in_flight=1)) as execution:
        assert execution.slow_motion == runtime.slow_motion == 2.5
    runtime.close()


@pytest.mark.parametrize('charged', [False, True])
def test_real_execution_never_waits_for_simulated_time(executors, charged, monkeypatch):
    runtime, now = executors(simulated=False, charged=charged)

    def forbidden_wait(*args, **kwargs):
        pytest.fail('Real execution waited for simulated time')

    monkeypatch.setattr(module.concurrent.futures, 'wait', forbidden_wait)
    release = threading.Event()
    try:
        answer = cast(_UnchargedAnswer[int], runtime.submit(lambda: (release.wait(timeout=2), 42)[1]))
        assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.CAN_ADVANCE)
        assert not answer.done()
    finally:
        release.set()
    assert answer.call.result(timeout=1) == 42
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert now == [0]
    assert answer.result() == 42


def test_cancelling_running_work_releases_capacity_only_when_it_finishes(executors):
    runtime, _ = executors(max_in_flight=1)
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        assert release.wait(timeout=2)

    answer = runtime.submit(work)
    try:
        assert started.wait(timeout=1)
        answer.cancel()
        assert not answer.done()
        with pytest.raises(QueueFull):
            runtime.submit(lambda: None)
    finally:
        release.set()
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert not runtime.has_ready_answers()
    runtime.submit(lambda: None)


@pytest.mark.parametrize('completed_first', [False, True])
def test_cancelled_failure_is_reported_once(executors, caplog, completed_first):
    runtime, _ = executors(max_in_flight=1)
    started, release = threading.Event(), threading.Event()

    def work():
        started.set()
        assert release.wait(timeout=2)
        raise ValueError('cancelled failure')

    answer = runtime.submit(work)
    try:
        assert started.wait(timeout=1)
        if completed_first:
            release.set()
            assert runtime.wait(timeout_sec=1).status is WaitStatus.ANSWERS_READY
        answer.cancel()
    finally:
        release.set()
    if not completed_first:
        assert runtime.wait(timeout_sec=1).status is WaitStatus.ANSWERS_READY
    assert not runtime.has_ready_answers()
    runtime.take_completed()
    runtime.close()
    assert caplog.text.count('cancelled failure') == 1


def test_failure_notifies_and_is_raised_by_result(executors, caplog):
    runtime, _ = executors()

    def fail():
        raise ValueError('model failed')

    answer = runtime.submit(fail)
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert runtime.has_ready_answers()
    assert answer.elapsed_sec == answer.execution_sec == 0.0
    assert runtime.has_ready_answers()
    with pytest.raises(ValueError, match='model failed'):
        answer.result()
    assert not runtime.has_ready_answers()
    runtime.close()
    assert 'model failed' not in caplog.text


def test_unread_failure_is_reported_on_close(executors, caplog):
    runtime, _ = executors()

    def fail():
        raise ValueError('unread failure')

    answer = runtime.submit(fail)
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    runtime.close()
    runtime.close()
    assert caplog.text.count('unread failure') == 1


def test_submission_preserves_context_and_keyword_arguments(executors):
    runtime, _ = executors()
    context = contextvars.ContextVar('test_context', default='missing')
    token = context.set('episode')
    try:
        answer = runtime.submit(lambda *, value: (context.get(), value), value=42)
    finally:
        context.reset(token)
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert answer.result() == ('episode', 42)


def test_close_waits_for_running_work_and_refuses_new_calls(executors):
    runtime, _ = executors()
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        assert release.wait(timeout=2)

    answer = runtime.submit(work)
    assert started.wait(timeout=1)
    closer = threading.Thread(target=lambda: (runtime.close(), closed.set()))
    closer.start()
    try:
        assert not closed.wait(timeout=0.01)
    finally:
        release.set()
        closer.join(timeout=2)
    assert closed.is_set()
    assert answer.result() is None
    with pytest.raises(RuntimeError):
        runtime.submit(lambda: None)


def test_close_before_submission_prevents_opening_workers(executors):
    runtime, _ = executors()
    runtime.close()
    runtime.close()
    with pytest.raises(RuntimeError):
        runtime.submit(lambda: None)


@pytest.mark.parametrize('capacity', [1, 3, 20])
def test_all_admitted_calls_can_start_and_unread_answers_hold_capacity(executors, capacity):
    runtime, _ = executors(max_in_flight=capacity)
    started = [threading.Event() for _ in range(capacity)]
    release = threading.Event()

    def work(index):
        started[index].set()
        assert release.wait(timeout=5)
        return index

    answers = [runtime.submit(work, index) for index in range(capacity)]
    try:
        assert all(event.wait(timeout=2) for event in started)
        with pytest.raises(QueueFull):
            runtime.submit(lambda: pytest.fail('Rejected work ran'))
    finally:
        release.set()
    completed = 0
    while completed < capacity:
        result = runtime.wait(timeout_sec=2)
        assert result.status is WaitStatus.ANSWERS_READY
        completed += len(result.completed)
    with pytest.raises(QueueFull):
        runtime.submit(lambda: None)
    assert answers[0].result() == 0
    runtime.submit(lambda: None)
    with pytest.raises(QueueFull):
        runtime.submit(lambda: None)


def test_invisible_or_unread_answer_keeps_capacity(executors, monkeypatch):
    wall = [0]
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: wall[0])
    runtime, now = executors(charged=True, max_in_flight=1)

    def work():
        wall[0] += 100_000_000
        return 42

    first = cast(_UnchargedAnswer, runtime.submit(work))
    assert first.call.result(timeout=1) == 42
    assert not first.done()
    with pytest.raises(QueueFull):
        runtime.submit(work)
    now[0] = 100_000_000
    runtime.take_completed()
    assert first.done()
    with pytest.raises(QueueFull):
        runtime.submit(work)
    assert first.result() == 42
    runtime.submit(work)


@pytest.mark.parametrize('fails', [False, True])
def test_consuming_none_or_failure_frees_capacity(executors, fails):
    runtime, _ = executors(max_in_flight=1)

    def work():
        if fails:
            raise ValueError('failed')

    answer = runtime.submit(work)
    assert runtime.wait(timeout_sec=1).status is WaitStatus.ANSWERS_READY
    with pytest.raises(QueueFull):
        runtime.submit(lambda: None)
    if fails:
        with pytest.raises(ValueError, match='failed'):
            answer.result()
    else:
        assert answer.result() is None
    runtime.submit(lambda: None)


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5])
def test_capacity_must_be_a_positive_integer(executors, capacity):
    with pytest.raises(ValueError, match='max_in_flight'):
        executors(max_in_flight=capacity)
    runtime, _ = executors()
    with pytest.raises(ValueError, match='max_in_flight'):
        ExecutionRuntime(runtime, max_in_flight=capacity)


def test_execution_limits_are_independent_and_share_completions(executors):
    runtime, now = executors(max_in_flight=1)
    started = [threading.Event() for _ in range(4)]
    release = threading.Event()

    def work(index):
        started[index].set()
        assert release.wait(timeout=5)
        return index

    with closing(ExecutionRuntime(runtime, max_in_flight=1)) as first:
        with closing(ExecutionRuntime(runtime, max_in_flight=2)) as second:
            answers = []
            try:
                answers.append(runtime.submit(work, 0))
                answers.append(first.submit(work, 1))
                answers.extend(second.submit(work, index) for index in (2, 3))
                assert all(event.wait(timeout=1) for event in started)
                for execution in (runtime, first, second):
                    with pytest.raises(QueueFull):
                        execution.submit(lambda: None)
                now[0] = 123
                runtime.start_tick()
                assert first.time_ns == second.time_ns == runtime.time_ns == 123
                assert first.tick == second.tick == runtime.tick
                second.set_metadata('value', 42)
                assert runtime.metadata_snapshot()['value'] == 42
            finally:
                release.set()
            completed = 0
            while completed < 4:
                result = runtime.wait(timeout_sec=2)
                assert result.status is WaitStatus.ANSWERS_READY
                completed += len(result.completed)
            assert runtime.has_ready_answers() and first.has_ready_answers() and second.has_ready_answers()
            assert answers[1].result() == 1
            first.submit(lambda: None)
            with pytest.raises(QueueFull):
                second.submit(lambda: None)


@pytest.mark.parametrize('simulated, charged', [(False, False), (True, False), (True, True)])
def test_execution_inherits_episode_time_accounting(executors, monkeypatch, simulated, charged):
    wall = [0]
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: wall[0])
    runtime, now = executors(simulated=simulated, charged=charged, max_in_flight=1)

    def work():
        wall[0] = 20_000_000
        return 42

    with closing(ExecutionRuntime(runtime, max_in_flight=1)) as execution:
        answer = cast(_UnchargedAnswer[int], execution.submit(work))
        assert answer.call.result(timeout=1) == 42
        if simulated and charged:
            assert not answer.done()
            assert not runtime.has_ready_answers()
        now[0] = 20_000_000
        assert answer.done()
        assert runtime.has_ready_answers()
        expected = 0.0 if simulated and not charged else 0.02
        assert answer.elapsed_sec == answer.execution_sec == expected
        assert answer.result() == 42
        assert not runtime.has_ready_answers()


def test_closing_runtime_drains_every_execution(executors):
    runtime, _ = executors()
    started, release, closed = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        assert release.wait(timeout=5)

    with closing(ExecutionRuntime(runtime, max_in_flight=1)) as execution:
        answer = execution.submit(work)
        assert started.wait(timeout=1)
        closer = threading.Thread(target=lambda: (runtime.close(), closed.set()))
        closer.start()
        try:
            assert not closed.wait(timeout=0.01)
        finally:
            release.set()
            closer.join(timeout=2)
        assert closed.is_set()
        assert answer.result() is None
        with pytest.raises(RuntimeError):
            execution.submit(lambda: None)


def test_closing_an_execution_leaves_other_executions_usable(executors):
    runtime, _ = executors()
    first = ExecutionRuntime(runtime, max_in_flight=1)
    with closing(ExecutionRuntime(runtime, max_in_flight=1)) as second:
        first.close()
        with pytest.raises(RuntimeError):
            first.submit(lambda: None)
        answer = second.submit(lambda: 42)
        assert runtime.wait(timeout_sec=1).status is WaitStatus.ANSWERS_READY
        assert answer.result() == 42
