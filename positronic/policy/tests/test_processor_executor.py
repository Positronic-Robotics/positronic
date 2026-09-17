"""Completion delivery and time accounting without a control-system dependency."""

import contextvars
import threading
from concurrent.futures import CancelledError
from typing import cast

import pytest

from positronic.policy import executor as module
from positronic.policy.base import NotAnswered
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
    assert not runtime._answers


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
