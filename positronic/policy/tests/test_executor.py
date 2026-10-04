"""Completion delivery and time accounting without a control-system dependency."""

import concurrent.futures
import contextvars
import gc
import threading
import weakref
from concurrent.futures import CancelledError
from contextlib import closing, contextmanager
from typing import cast

import numpy as np
import pytest

from positronic import telemetry, telemetry_keys
from positronic.eval import keys as eval_keys
from positronic.policy import executor as module
from positronic.policy.base import NotAnswered, Policy, Step
from positronic.policy.executor import Executor, JournaledExecutor, WaitResult, WaitStatus, _UnchargedAnswer
from positronic.policy.journal import (
    PLAIN_DATA,
    Activity,
    ActivityFailed,
    CancelRequested,
    Closing,
    Ended,
    Finished,
    Journal,
    PlainData,
    Primed,
    Published,
    Raised,
    Started,
    Startup,
    Stopped,
    Submitted,
    TurnStarted,
    UnrecordableResult,
    Wake,
)
from positronic.policy.processors import ChunkedSchedule
from positronic.policy.replay import verify


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


def test_plain_turns_count_calls_apart_from_ticks_and_keep_the_live_clock(executors):
    runtime, now = executors()
    obs = {'x': 1}
    assert runtime.begin_turn(obs, Wake.FIRST) is obs
    now[0] = 5
    assert runtime.time_ns == 5
    runtime.end_turn(Step({}, 5), 5)
    runtime.begin_turn(obs, Wake.DUE)
    runtime.begin_turn(obs, Wake.COMPLETION)
    assert (runtime.tick, runtime.invocation) == (1, 2)


def journaled_runtime(tmp_path, *, observations=PLAIN_DATA, charged=False, max_workers=1):
    now = [0]
    journal = Journal(tmp_path / 'journal', observations)
    started = Started.create(journal, ChunkedSchedule(fps=10), simulated=True, charge_inference_time=charged)
    runtime = JournaledExecutor(
        lambda: now[0], journal, started, simulated=True, charge_inference_time=charged, max_workers=max_workers
    )
    return runtime, now, journal


@contextmanager
def journaled(tmp_path, **options):
    runtime, now, journal = journaled_runtime(tmp_path, **options)
    try:
        yield runtime, now, journal
    finally:
        runtime.close()


class Hooked(PlainData):
    """Plain data that calls ``on_encode`` before each encode and ``on_decode`` before each frozen decode."""

    NAME = 'hooked'

    def __init__(self):
        self.on_encode = lambda: None
        self.on_decode = lambda: None

    def encode(self, value):
        self.on_encode()
        return super().encode(value)

    def decode_frozen(self, payload):
        self.on_decode()
        return super().decode_frozen(payload)


def finish(runtime: JournaledExecutor, submission: int) -> None:
    """Wait for the worker without publishing its outcome."""
    runtime._unpublished[submission].call.exception(timeout=1)


def test_journaled_turn_holds_time_and_answers_while_work_completes(tmp_path):
    with journaled(tmp_path) as (runtime, now, _):
        runtime.begin_turn({}, Wake.FIRST)
        answer = runtime.submit(Activity('add', 1, lambda value: value + 1), 41)
        finish(runtime, 0)
        now[0] = 5
        for _ in range(2):
            assert runtime.time_ns == 0
            assert not answer.done()
            with pytest.raises(NotAnswered):
                answer.result()
        runtime.end_turn(Step({}, 0), 5_000_000)
        assert runtime.time_ns == 5
        assert runtime.wait(timeout_sec=1) == WaitResult(WaitStatus.ANSWERS_READY, (runtime._unpublished[0],))

        runtime.begin_turn({}, Wake.COMPLETION)
        assert (runtime.time_ns, runtime.tick, runtime.invocation) == (5, 1, 1)
        assert answer.result() == 42
        runtime.end_turn(Step({}, 5), 5_000_005)
        runtime.begin_turn({}, Wake.DUE)
        assert (runtime.tick, runtime.invocation) == (1, 2)
        assert answer.result() == 42
        runtime.end_turn(Step({}, 5), 5_000_005)
        assert runtime.wait(timeout_sec=0) == WaitResult(WaitStatus.CAN_ADVANCE)


def test_journaled_turn_publishes_only_work_that_returned_before_its_cutoff(tmp_path):
    release = threading.Event()
    codec = Hooked()
    with journaled(tmp_path, observations=codec) as (runtime, _, journal):
        runtime.begin_turn({}, Wake.FIRST)
        answer = runtime.submit(Activity('wait', 1, lambda: release.wait(timeout=2)))
        call = runtime._unpublished[0].call
        runtime.end_turn(Step({}, 0), 5_000_000)

        def return_work():
            """Let the work return while the turn encodes its observation."""
            release.set()
            concurrent.futures.wait([call], timeout=2)

        codec.on_encode = return_work
        runtime.begin_turn({}, Wake.DUE)
        codec.on_encode = lambda: None
        assert call.done()
        assert not answer.done()
        runtime.end_turn(Step({}, 0), 5_000_000)

        runtime.begin_turn({}, Wake.COMPLETION)
        assert answer.result() is True
        runtime.end_turn(Step({}, 0), 5_000_000)
    assert [e.invocation for e in journal.read().events if isinstance(e, Published)] == [2]


def test_journaled_charged_work_stays_hidden_when_the_clock_moves_during_turn_entry(tmp_path, monkeypatch):
    wall_ns = [0]
    monkeypatch.setattr(module.time, 'monotonic_ns', lambda: wall_ns[0])
    codec = Hooked()
    runtime, now, _ = journaled_runtime(tmp_path, observations=codec, charged=True)

    def work():
        wall_ns[0] = 20_000_000
        return 42

    try:
        runtime.begin_turn({}, Wake.FIRST)
        answer = runtime.submit(Activity('charged', 1, work))
        finish(runtime, 0)
        runtime.end_turn(Step({}, 0), 5_000_000)

        # The world clock reaches the time the work becomes visible while the turn encodes its observation.
        codec.on_encode = lambda: now.__setitem__(0, 20_000_000)
        runtime.begin_turn({}, Wake.DUE)
        assert runtime.time_ns == 0
        assert not answer.done()
        runtime.end_turn(Step({}, 0), 5_000_000)

        runtime.begin_turn({}, Wake.DUE)
        assert (runtime.time_ns, answer.result()) == (20_000_000, 42)
        runtime.end_turn(Step({}, 20_000_000), 25_000_000)
    finally:
        runtime.close()


def test_journaled_work_that_returns_while_a_result_decodes_waits_for_the_next_turn(tmp_path):
    running, release = threading.Event(), threading.Event()
    codec = Hooked()
    runtime, _, _ = journaled_runtime(tmp_path, max_workers=2)

    def block():
        running.set()
        assert release.wait(timeout=2)
        return 42

    try:
        runtime.begin_turn({}, Wake.FIRST)
        first = runtime.submit(Activity('first', 1, lambda: 41, codec=codec))
        second = runtime.submit(Activity('second', 1, block))
        second_call = runtime._unpublished[1].call
        finish(runtime, 0)
        assert running.wait(timeout=2)
        runtime.end_turn(Step({}, 0), 5_000_000)

        def return_second():
            """Let the second work return while the turn decodes the result of the first."""
            release.set()
            concurrent.futures.wait([second_call], timeout=2)

        codec.on_decode = return_second
        runtime.begin_turn({}, Wake.COMPLETION)
        codec.on_decode = lambda: None
        assert second_call.done()
        assert first.result() == 41
        assert not second.done()
        runtime.end_turn(Step({}, 0), 5_000_000)

        runtime.begin_turn({}, Wake.COMPLETION)
        assert (runtime.time_ns, second.result()) == (0, 42)
        runtime.end_turn(Step({}, 0), 5_000_000)
    finally:
        release.set()
        runtime.close()


def test_journaled_policy_values_are_frozen_and_work_owns_its_input(tmp_path):
    frame = np.zeros(2)
    source = {'frame': frame, 'nested': {'values': [1]}}
    returned = {'chunk': [np.zeros(2)]}
    received = []

    def work(values):
        received.append(values)
        values['values'].append(2)
        return returned

    with journaled(tmp_path) as (runtime, _, _):
        obs = runtime.begin_turn(source, Wake.FIRST)
        answer = runtime.submit(Activity('copy', 1, work), obs['nested'])
        frame[:] = 7
        source['nested']['values'].append(7)
        finish(runtime, 0)
        returned['chunk'][0][:] = 9
        runtime.end_turn(Step({}, 0), 5_000_000)
        runtime.begin_turn({}, Wake.COMPLETION)
        result = answer.result()
        with pytest.raises(TypeError):
            cast(dict, result)['chunk'] = []
        with pytest.raises(TypeError):
            cast(list, result['chunk'])[0] = np.ones(2)
        assert answer.result() is result
        runtime.end_turn(Step({}, 0), 5_000_000)
    assert received == [{'values': [1, 2]}]
    assert obs['nested']['values'] == (1,)
    with pytest.raises(TypeError):
        cast(dict, obs['nested'])['values'] = [3]
    for value in (obs['frame'], result['chunk'][0]):
        np.testing.assert_array_equal(value, [0, 0])
        assert not value.flags.writeable


def test_journaled_outcomes_and_cancellation_requests(tmp_path):
    started, release = threading.Event(), threading.Event()

    def block():
        started.set()
        assert release.wait(timeout=2)
        return 1

    def fail():
        raise ValueError('model failed')

    with journaled(tmp_path) as (runtime, _, journal):
        runtime.begin_turn({}, Wake.FIRST)
        running = runtime.submit(Activity('block', 1, block))
        assert started.wait(timeout=1)
        queued = runtime.submit(Activity('queued', 1, lambda: 0))
        failing = runtime.submit(Activity('fail', 1, fail))
        queued.cancel()
        running.cancel()
        release.set()
        finish(runtime, 2)
        assert not any(answer.done() for answer in (running, queued, failing))
        runtime.end_turn(Step({}, 0), 5_000_000)

        runtime.begin_turn({}, Wake.COMPLETION)
        assert running.result() == 1
        with pytest.raises(CancelledError):
            queued.result()
        with pytest.raises(ActivityFailed, match='ValueError: model failed') as failed:
            failing.result()
        assert (failed.value.args, failed.value.__cause__) == (('ValueError: model failed',), None)
        runtime.end_turn(Step({}, 0), 5_000_000)
    events = journal.read().events
    published = [e for e in events if isinstance(e, Published)]
    assert [(e.submission, e.outcome.kind) for e in published] == [(0, 'returned'), (1, 'cancelled'), (2, 'raised')]
    assert isinstance(raised := published[2].outcome, Raised)
    assert "raise ValueError('model failed')" in raised.traceback
    assert [e.submission for e in events if isinstance(e, CancelRequested)] == [1, 0]


def test_journaled_results_live_as_long_as_the_policy_holds_their_answers(tmp_path, caplog):
    def fail():
        raise ValueError('model failed')

    with journaled(tmp_path) as (runtime, _, journal):
        runtime.begin_turn({}, Wake.FIRST)
        kept = runtime.submit(Activity('kept', 1, lambda: np.arange(3)))
        failing = runtime.submit(Activity('fail', 1, fail))
        finish(runtime, 0)
        finish(runtime, 1)
        frames = []
        for submission in range(2, 10):
            answer = runtime.submit(Activity('frame', 1, lambda: np.zeros(1 << 20, dtype=np.uint8)))
            finish(runtime, submission)
            runtime.end_turn(Step({}, 0), 5_000_000)
            assert runtime.wait(timeout_sec=0).status is WaitStatus.ANSWERS_READY
            runtime.begin_turn({}, Wake.COMPLETION)
            frames.append(weakref.ref(answer.result()))
            del answer
        assert kept.result() is kept.result()
        np.testing.assert_array_equal(kept.result(), [0, 1, 2])
        del failing
        runtime.end_turn(Step({}, 0), 5_000_000)
        gc.collect()
        assert [frame() is None for frame in frames] == [True] * len(frames)
    assert [e.submission for e in journal.read().events if isinstance(e, Submitted)] == list(range(10))
    assert 'failed without its result being read: ValueError: model failed' in caplog.text


def test_journaled_executor_refuses_work_it_cannot_record(tmp_path):
    with journaled(tmp_path) as (runtime, _, journal):
        with pytest.raises(RuntimeError, match='inside a turn'):
            runtime.submit(Activity('outside', 1, lambda: 0))
        runtime.begin_turn({}, Wake.FIRST)
        with pytest.raises(TypeError, match='only an Activity'):
            runtime.submit(lambda: 0)
        with pytest.raises(TypeError):
            runtime.submit(Activity('argument', 1, lambda value: value), object())
        runtime.submit(Activity('result', 1, object))
        finish(runtime, 0)
        runtime.end_turn(Step({}, 0), 5_000_000)
        with pytest.raises(UnrecordableResult, match='result v1 returned object, which plain_data cannot encode'):
            runtime.begin_turn({}, Wake.COMPLETION)
        with pytest.raises(FileExistsError):
            JournaledExecutor(lambda: 0, journal, journal.read().started, simulated=True, charge_inference_time=False)


def test_journaled_startup_holds_its_time_for_nested_processors_and_publishes_at_the_first_turn(tmp_path):
    seen = []

    class Inner(Policy):
        def run(self, runtime):
            seen.append(('inner', runtime.time_ns, runtime.invocation))
            yield
            while True:
                yield Step({}, 0)

    class Outer(Policy):
        def run(self, runtime):
            seen.append(('outer', runtime.time_ns, runtime.invocation))
            now[0] += 1
            answer = runtime.submit(Activity('warm_up', 1, lambda: 'ready'))
            with closing(runtime.start(Inner())) as inner:
                obs = yield
                while True:
                    inner.send(obs)
                    obs = yield Step({'ready': answer.result() if answer.done() else None}, 0)

    runtime, now, journal = journaled_runtime(tmp_path)
    now[0] = 3
    run = runtime.start(Outer())
    try:
        assert runtime.time_ns == 4
        with pytest.raises(RuntimeError, match='starts one policy'):
            runtime.start(Outer())
        finish(runtime, 0)
        step = run.send(runtime.begin_turn({}, Wake.FIRST))
        assert step is not None
        runtime.end_turn(step, 5_000_000)
    finally:
        runtime.close(run)
    assert seen == [('outer', 3, -1), ('inner', 3, -1)]
    assert step.commands == {'ready': 'ready'}
    startup, submitted, primed, turn = journal.read().events[1:5]
    assert isinstance(startup, Startup) and startup.time_ns == 3
    assert isinstance(submitted, Submitted) and submitted.invocation == -1
    assert isinstance(primed, Primed)
    assert isinstance(turn, TurnStarted) and (turn.time_ns, turn.invocation) == (4, 0)


class Reads(Policy):
    """Submit one activity at the first turn, and command its result once it is published."""

    def run(self, runtime):
        yield
        answer = runtime.submit(Activity('value', 1, lambda: 7))
        while True:
            yield Step({'value': answer.result()} if answer.done() else {}, runtime.time_ns)


def test_a_rejected_observation_takes_no_turn_and_publishes_nothing(tmp_path):
    def reject():
        raise ValueError('rejected observation')

    codec = Hooked()
    journal = Journal(tmp_path / 'journal', codec)
    now = [0]
    started = Started.create(journal, Reads(), simulated=True, charge_inference_time=False)
    runtime = JournaledExecutor(lambda: now[0], journal, started, simulated=True, charge_inference_time=False)
    run = runtime.start(Reads())
    try:
        step = run.send(runtime.begin_turn({}, Wake.FIRST))
        assert step is not None
        runtime.end_turn(step, 5_000_000)
        finish(runtime, 0)
        now[0] = 5_000_000
        codec.on_decode = reject
        with pytest.raises(ValueError, match='rejected observation'):
            runtime.begin_turn({}, Wake.COMPLETION)
        codec.on_decode = lambda: None
        assert (runtime.tick, runtime.invocation) == (0, 0)
        step = run.send(runtime.begin_turn({}, Wake.COMPLETION))
        assert step is not None
        assert (runtime.tick, runtime.invocation, step.commands) == (1, 1, {'value': 7})
        runtime.end_turn(step, 10_000_000)
    finally:
        runtime.close(run)
    events = journal.read().events
    assert [(e.invocation, e.tick, e.time_ns) for e in events if isinstance(e, TurnStarted)] == [
        (0, 0, 0),
        (1, 1, 5_000_000),
    ]
    assert [e.invocation for e in events if isinstance(e, Published)] == [1]
    verified = verify(Reads(), journal)
    assert (verified.turns, verified.complete) == (2, True)


def test_journaled_close_drains_work_then_journals_the_policy_finalizers(tmp_path):
    finished = []

    class Finalizing(Policy):
        def run(self, runtime):
            yield
            answer = runtime.submit(Activity('slow', 1, lambda: finished.append(True)))
            try:
                while True:
                    yield Step({}, 0)
            finally:
                answer.cancel()
                runtime.metadata['drained'] = len(finished)
                runtime.metadata['closed_at'] = runtime.time_ns

    runtime, now, journal = journaled_runtime(tmp_path)
    run = runtime.start(Finalizing())
    try:
        step = run.send(runtime.begin_turn({}, Wake.FIRST))
        assert step is not None
        runtime.end_turn(step, 5_000_000)
        now[0] = 7
    finally:
        runtime.close(run)
    recording = journal.read()
    assert [e.kind for e in recording.events[-3:]] == ['closing', 'cancel', 'ended']
    closing_event, cancel, ended = recording.events[-3:]
    assert isinstance(closing_event, Closing) and closing_event.time_ns == 7
    assert isinstance(cancel, CancelRequested) and (cancel.invocation, cancel.submission) == (0, 0)
    assert isinstance(ended, Ended) and ended.termination == Stopped()
    assert PLAIN_DATA.decode(recording.payload(ended.metadata)) == {'drained': 1, 'closed_at': 7}


@pytest.mark.parametrize(
    'ending, termination',
    [(None, Stopped), ({eval_keys.TERMINATED: True}, Finished), (ValueError('episode failed'), Raised)],
)
def test_journaled_close_records_how_the_episode_ended(tmp_path, ending, termination):
    journal = Journal(tmp_path / 'journal')
    started = Started.create(journal, ChunkedSchedule(fps=10), simulated=True, charge_inference_time=False)
    JournaledExecutor(lambda: 0, journal, started, simulated=True, charge_inference_time=False).close(ending=ending)
    recording = journal.read()
    ended = recording.events[-1]
    assert isinstance(ended, Ended) and isinstance(ended.termination, termination)
    match ended.termination:
        case Finished(payload=payload):
            assert PLAIN_DATA.decode(recording.payload(payload)) == ending
        case Raised(error=error):
            assert error == 'ValueError: episode failed'
