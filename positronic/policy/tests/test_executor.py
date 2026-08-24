"""Unit tests for the executor serving functions off the caller's thread."""

import operator
import threading
import time
from contextvars import ContextVar

import pytest

import pimm
from positronic.policy.executor import Executor

# How long a test waits for the worker thread before calling the call lost.
TIMEOUT_SEC = 5.0

_marker: ContextVar[str] = ContextVar('test_executor_marker', default='unset')


def settled(answer: pimm.calls.Answer) -> pimm.calls.Answer:
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

    def make(**functions):
        executors.append(Executor(functions))
        return executors[-1]

    yield make
    for executor in executors:
        executor.close()


def test_fns_are_the_declared_names(serve):
    assert sorted(serve(add=operator.add, mul=operator.mul).fns) == ['add', 'mul']


def test_call_answers_with_the_functions_result(serve):
    answer = serve(add=operator.add).fns['add'](2, 3)

    assert isinstance(answer, pimm.calls.Answer)
    assert settled(answer).result() == 5


def test_keyword_arguments_reach_the_function(serve):
    fns = serve(pose=lambda arm, gripper=0.0: (arm, gripper)).fns

    assert settled(fns['pose']('left', gripper=0.5)).result() == ('left', 0.5)


def test_answer_is_pending_until_the_function_returns(serve):
    release = threading.Event()
    answer = serve(gate=lambda: release.wait(TIMEOUT_SEC)).fns['gate']()

    assert not answer.done()
    with pytest.raises(pimm.NoValueException):
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


def test_call_runs_under_a_copy_of_the_context_it_was_made_in(serve):
    fns = serve(marker=_marker.get).fns
    _marker.set('episode-7')

    assert settled(fns['marker']()).result() == 'episode-7'


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
