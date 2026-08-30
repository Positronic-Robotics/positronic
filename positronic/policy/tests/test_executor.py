"""Unit tests for the executor serving functions off the caller's thread."""

import logging
import operator
import threading
import time
import weakref
from contextvars import ContextVar
from functools import partial

import pytest

from positronic.policy.base import Answer, NotAnswered
from positronic.policy.executor import Executor

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


class _Weights:
    """Stands in for what a function is declared with: model weights, a socket."""


def test_close_frees_what_the_functions_held(serve):
    """A closed runtime drops its functions, so the policy that closes next can free the weights."""
    weights = _Weights()
    gone = weakref.ref(weights)
    executor = serve(infer=partial(operator.is_, weights))
    del weights

    assert gone() is not None
    executor.close()
    assert gone() is None
