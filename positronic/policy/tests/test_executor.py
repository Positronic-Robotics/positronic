"""Completion delivery and time accounting without a control-system dependency."""

import contextvars
import threading
from concurrent.futures import CancelledError
from typing import cast

import pytest

from positronic import telemetry, telemetry_keys
from positronic.policy import executor as module
from positronic.policy.base import NotAnswered, Policy, Step
from positronic.policy.executor import Executor, WaitResult, WaitStatus, _UnchargedAnswer


@pytest.fixture
def executors():
    created = []

    def make(*, simulated=True, charged=False, workers=1):
        now = [0]
        runtime = Executor(lambda: now[0], simulated=simulated, charge_inference_time=charged, max_workers=workers)
        created.append(runtime)
        return runtime, now

    yield make
    for runtime in created:
        runtime.close()


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
    if read_first:
        assert answer.result() == 42
    assert runtime.has_pending
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert not runtime.has_pending
    assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)
    assert answer.result() == 42
    assert runtime.take_completed() == ()


def test_submission_and_bounded_wait_do_not_wait_out_a_worker(executors):
    runtime, now = executors()
    release = threading.Event()
    try:
        answer = runtime.submit(lambda: release.wait(timeout=2))
        assert not answer.done()
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.TIMED_OUT)
        assert runtime.take_completed() == ()
        assert now == [0]
        with pytest.raises(NotAnswered):
            answer.result()
    finally:
        release.set()
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)


def test_fast_call_completes_while_another_worker_is_pending(executors):
    runtime, _ = executors(workers=2)
    release = threading.Event()
    slow = runtime.submit(lambda: release.wait(timeout=2))
    try:
        fast = cast(_UnchargedAnswer[int], runtime.submit(lambda: 42))
        assert fast.call.result(timeout=1) == 42
        assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (fast,))
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
    now[0] = 20_000_000
    assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    assert answer.result() == 42


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


def test_cancellation_notifies_once(executors):
    runtime, _ = executors()
    release = threading.Event()
    running = runtime.submit(lambda: release.wait(timeout=2))
    try:
        queued = runtime.submit(lambda: 42)
        queued.cancel()
        assert runtime.take_completed() == (queued,)
        assert runtime.take_completed() == ()
        with pytest.raises(CancelledError):
            queued.result()
    finally:
        release.set()
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (running,))


def test_failure_notifies_and_is_raised_by_result(executors, caplog):
    runtime, _ = executors()

    def fail():
        raise ValueError('model failed')

    answer = runtime.submit(fail)
    assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (answer,))
    with pytest.raises(ValueError, match='model failed'):
        answer.result()
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


def test_at_close_callbacks_run_after_running_work_finishes_and_at_once_after_close(executors):
    runtime, _ = executors()
    started, release = threading.Event(), threading.Event()
    order = []

    def work():
        started.set()
        assert release.wait(timeout=2)
        order.append('work')

    runtime.submit(work)
    assert started.wait(timeout=1)
    runtime.at_close(lambda: order.append('first registered'))
    runtime.at_close(lambda: order.append('second registered'))
    releaser = threading.Timer(0.01, release.set)
    releaser.start()
    try:
        runtime.close()
    finally:
        release.set()
        releaser.join()
    assert order == ['work', 'second registered', 'first registered']
    runtime.at_close(lambda: order.append('after close'))
    assert order[-1] == 'after close'


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


def test_close_waits_for_running_work_and_cancels_queued_work(executors):
    runtime, _ = executors()
    started, release, closed = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def first():
        started.set()
        assert release.wait(timeout=2)
        calls.append('first')

    runtime.submit(first)
    assert started.wait(timeout=1)
    queued = runtime.submit(lambda: calls.append('second'))
    assert not queued.done()
    closer = threading.Thread(target=lambda: (runtime.close(), closed.set()))
    closer.start()
    try:
        with pytest.raises(CancelledError):
            cast(_UnchargedAnswer, queued).call.result(timeout=1)
        assert not closed.is_set()
        assert calls == []
    finally:
        release.set()
        closer.join(timeout=2)
    assert closed.is_set()
    assert calls == ['first']
    with pytest.raises(CancelledError):
        queued.result()
    with pytest.raises(RuntimeError):
        runtime.submit(lambda: None)
