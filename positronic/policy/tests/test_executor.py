"""Unit tests for the executor serving functions off the caller's thread."""

import logging
import operator
import threading
import time
from contextvars import ContextVar
from functools import partial

import pytest

from positronic.policy.base import Answer, NotAnswered, Session
from positronic.policy.executor import Caller, Executor, call_until_answered

# How long a test waits for the worker threads before calling the call lost.
TIMEOUT_SEC = 5.0


def settled(answer: Answer) -> Answer:
    """The answer once the worker has run its call; fails the test if it never lands."""
    deadline = time.monotonic() + TIMEOUT_SEC
    while not answer.done():
        assert time.monotonic() < deadline, 'the call was never answered'
        time.sleep(0.001)
    return answer


@pytest.fixture
def serve():
    """Serves the functions it is called with, and closes every executor it made when the test ends."""
    executors = []

    def make(*, max_workers: int = 1, **functions):
        executors.append(Executor(functions, max_workers=max_workers))
        return executors[-1]

    yield make
    for executor in executors:
        executor.close()


def test_fns_are_the_declared_names(serve):
    assert sorted(serve(add=operator.add, mul=operator.mul).fns) == ['add', 'mul']


def test_call_answers_with_the_functions_result(serve):
    answer = serve(add=operator.add).fns['add'](2, 3)

    assert isinstance(answer, Answer)
    assert settled(answer).result() == 5


def test_keyword_arguments_reach_the_function(serve):
    fns = serve(pose=lambda arm, gripper=0.0: (arm, gripper)).fns

    assert settled(fns['pose']('left', gripper=0.5)).result() == ('left', 0.5)


def test_no_keyword_name_is_reserved(serve):
    fns = serve(apply=lambda fn, self: (fn, self)).fns

    assert settled(fns['apply'](fn='a', self='b')).result() == ('a', 'b')


def test_answer_is_pending_until_the_function_returns(serve):
    release = threading.Event()
    answer = serve(gate=lambda: release.wait(TIMEOUT_SEC)).fns['gate']()

    assert not answer.done()
    with pytest.raises(NotAnswered):
        answer.result()

    release.set()
    assert settled(answer).result() is True


def test_result_raises_what_the_function_raised(serve):
    def fail():
        raise ValueError('inference blew up')

    answer = serve(fail=fail).fns['fail']()

    with pytest.raises(ValueError, match='inference blew up'):
        settled(answer).result()


def test_calls_run_one_at_a_time(serve):
    release = threading.Event()
    fns = serve(gate=lambda: release.wait(TIMEOUT_SEC), add=operator.add).fns

    gated, queued = fns['gate'](), fns['add'](2, 3)
    assert not queued.done()

    release.set()
    assert settled(gated).result() is True
    assert settled(queued).result() == 5


def test_max_workers_calls_run_side_by_side(serve):
    # Neither call passes the barrier unless the other is running too, so a single worker breaks it.
    paired = threading.Barrier(2)
    fns = serve(max_workers=2, gate=lambda: paired.wait(TIMEOUT_SEC)).fns

    first, second = fns['gate'](), fns['gate']()

    assert sorted([settled(first).result(), settled(second).result()]) == [0, 1]


# A ContextVar belongs at module level: every context that sets it holds a strong reference, so one made
# inside a function is never collected.
_marker: ContextVar[str] = ContextVar('test_executor_marker', default='unset')


def test_call_runs_under_a_copy_of_the_context_it_was_made_in(serve):
    fns = serve(marker=_marker.get).fns
    _marker.set('episode-7')

    assert settled(fns['marker']()).result() == 'episode-7'


def test_nothing_is_in_flight_before_a_call(serve):
    assert not serve(add=operator.add).in_flight


def test_a_call_is_in_flight_until_it_answers(serve):
    release = threading.Event()
    executor = serve(gate=lambda: release.wait(TIMEOUT_SEC))

    answer = executor.fns['gate']()
    assert executor.in_flight

    release.set()
    settled(answer)
    assert not executor.in_flight


def test_nothing_is_owed_before_a_call(serve):
    assert not serve(add=operator.add).owes_an_answer


def test_an_answer_stays_owed_after_its_call_lands_until_it_is_read(serve):
    executor = serve(add=operator.add)
    answer = settled(executor.fns['add'](2, 3))

    assert not executor.in_flight
    assert executor.owes_an_answer

    answer.result()
    assert not executor.owes_an_answer


def test_wait_returns_once_every_call_has_answered(serve):
    executor = serve(sleep=partial(time.sleep, 0.05), add=operator.add)

    first, second = executor.fns['sleep'](), executor.fns['add'](2, 3)
    executor.wait(TIMEOUT_SEC)

    assert not executor.in_flight
    assert first.done() and second.result() == 5


def test_wait_gives_up_at_its_timeout(serve):
    release = threading.Event()
    executor = serve(gate=lambda: release.wait(TIMEOUT_SEC))
    executor.fns['gate']()

    started = time.monotonic()
    executor.wait(0.01)

    assert time.monotonic() - started < TIMEOUT_SEC
    assert executor.in_flight
    release.set()


def test_close_waits_out_the_call_in_flight(serve):
    started, finished = threading.Event(), []

    def slow():
        started.set()
        time.sleep(0.05)
        finished.append(True)

    executor = serve(slow=slow)
    executor.fns['slow']()
    assert started.wait(TIMEOUT_SEC)
    executor.close()

    assert finished == [True]


def test_calling_a_closed_executor_raises(serve):
    executor = serve(add=operator.add)
    executor.close()

    with pytest.raises(RuntimeError):
        executor.fns['add'](2, 3)


def _fail():
    raise RuntimeError('the server went away')


def test_close_reports_a_failure_that_nobody_read(serve, caplog):
    executor = serve(infer=_fail)
    settled(executor.fns['infer']())

    with caplog.at_level(logging.ERROR):
        executor.close()

    assert 'the server went away' in caplog.text
    assert 'infer' in caplog.text


def test_close_stays_quiet_about_a_failure_its_caller_read(serve, caplog):
    executor = serve(infer=_fail)
    answer = settled(executor.fns['infer']())
    with pytest.raises(RuntimeError):
        answer.result()

    with caplog.at_level(logging.ERROR):
        executor.close()

    assert caplog.text == ''


def test_close_stays_quiet_about_a_call_that_answered(serve, caplog):
    executor = serve(add=operator.add)
    settled(executor.fns['add'](2, 3))

    with caplog.at_level(logging.ERROR):
        executor.close()

    assert caplog.text == ''


def _raises() -> None:
    raise ValueError('the call failed')


class TestCaller:
    """One session's use of one served function."""

    def test_a_caller_of_an_unserved_function_fails_where_it_is_built(self, serve):
        with pytest.raises(KeyError):
            Caller(serve(add=operator.add), 'infer')

    def test_a_fresh_caller_is_idle(self, serve):
        assert Caller(serve(add=operator.add), 'add').idle

    def test_a_started_call_is_no_longer_idle(self, serve):
        caller = Caller(serve(add=operator.add), 'add')
        caller.start(2, 3)

        assert not caller.idle

    def test_take_answers_none_while_the_call_is_out(self, serve):
        release = threading.Event()
        caller = Caller(serve(gate=release.wait), 'gate')
        caller.start(TIMEOUT_SEC)

        assert caller.take() is None
        assert caller.in_flight
        release.set()

    def test_take_answers_the_result_once_the_call_lands(self, serve):
        rt = serve(add=operator.add)
        caller = Caller(rt, 'add')
        caller.start(2, 3)
        rt.wait(TIMEOUT_SEC)

        assert caller.take() == 5

    def test_a_taken_call_leaves_the_caller_idle(self, serve):
        rt = serve(add=operator.add)
        caller = Caller(rt, 'add')
        caller.start(2, 3)
        rt.wait(TIMEOUT_SEC)
        caller.take()

        assert caller.idle
        assert not caller.in_flight

    def test_take_raises_what_the_call_raised(self, serve):
        rt = serve(fail=_raises)
        caller = Caller(rt, 'fail')
        caller.start()
        rt.wait(TIMEOUT_SEC)

        with pytest.raises(ValueError, match='the call failed'):
            caller.take()

    def test_a_cancelled_call_answers_none(self, serve):
        rt = serve(add=operator.add)
        caller = Caller(rt, 'add')
        caller.start(2, 3)
        caller.cancel()
        rt.wait(TIMEOUT_SEC)

        assert caller.take() is None

    def test_a_cancelled_call_still_raises_what_it_raised(self, serve):
        rt = serve(fail=_raises)
        caller = Caller(rt, 'fail')
        caller.start()
        caller.cancel()
        rt.wait(TIMEOUT_SEC)

        with pytest.raises(ValueError, match='the call failed'):
            caller.take()

    def test_a_cancel_ends_with_the_call_it_was_made_against(self, serve):
        rt = serve(add=operator.add)
        caller = Caller(rt, 'add')
        caller.start(2, 3)
        caller.cancel()
        rt.wait(TIMEOUT_SEC)
        caller.take()

        caller.start(4, 5)
        rt.wait(TIMEOUT_SEC)
        assert caller.take() == 9

    def test_a_cancel_with_nothing_in_flight_drops_no_later_call(self, serve):
        rt = serve(add=operator.add)
        caller = Caller(rt, 'add')
        caller.cancel()

        caller.start(2, 3)
        rt.wait(TIMEOUT_SEC)
        assert caller.take() == 5


class _PlainSession(Session):
    """A session that does its work inside its own call."""

    def __init__(self):
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        return [{'action': obs}]


class _RoundsSession(Session):
    """A session that starts one served call per round, and answers the last round's result."""

    def __init__(self, rt: Executor, rounds: int):
        self._infer = Caller(rt, 'echo')
        self._left = rounds
        self.calls = 0

    def __call__(self, obs):
        self.calls += 1
        result = None if self._infer.idle else self._infer.take()
        if self._infer.idle and self._left > 0:
            self._left -= 1
            self._infer.start(obs)
            return None
        return result


class TestCallUntilAnswered:
    """The way a caller with no control loop of its own reads a session."""

    def test_a_session_that_answers_in_its_own_call_is_called_one_time(self, serve):
        session = _PlainSession()

        assert call_until_answered(session, serve(), {'x': 1}) == [{'action': {'x': 1}}]
        assert session.calls == 1

    def test_a_session_is_called_again_for_the_function_it_started(self, serve):
        rt = serve(echo=lambda obs: obs)
        session = _RoundsSession(rt, rounds=1)

        assert call_until_answered(session, rt, {'x': 1}) == {'x': 1}
        assert session.calls == 2

    def test_a_session_is_called_again_for_every_function_it_starts(self, serve):
        rt = serve(echo=lambda obs: obs)
        session = _RoundsSession(rt, rounds=2)

        assert call_until_answered(session, rt, {'x': 1}) == {'x': 1}
        assert session.calls == 3

    def test_a_session_that_starts_nothing_and_answers_none_is_called_one_time(self, serve):
        rt = serve(echo=lambda obs: obs)
        session = _RoundsSession(rt, rounds=0)

        assert call_until_answered(session, rt, {'x': 1}) is None
        assert session.calls == 1
