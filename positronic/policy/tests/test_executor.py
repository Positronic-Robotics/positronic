"""Unit tests for the executor serving functions off the caller's thread."""

import logging
import operator
import threading
import time
import weakref
from contextvars import ContextVar
from functools import partial

import pytest

from positronic.policy.base import (
    Answer,
    ChunkSession,
    DelegatingChunkSession,
    Done,
    Layer,
    NotAnswered,
    Policy,
    Session,
)
from positronic.policy.executor import Executor, blocking

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


class _PlainPolicy(Policy):
    """Serves nothing: its session answers inside the call that asked."""

    class _Session(ChunkSession):
        def __init__(self):
            self.calls = 0

        def __call__(self, obs, time_ns):
            self.calls += 1
            return Done([{'action': obs}])

    def __init__(self):
        self.session = _PlainPolicy._Session()

    def new_session(self, context=None, rt=None) -> ChunkSession:
        return self.session


_ECHO = 'echo'


class _EchoPolicy(Policy):
    """Serves ``echo``: its session hands the caller the handle on one call of it."""

    class _Session(ChunkSession):
        def __init__(self, rt):
            self._rt = rt
            self.calls = 0

        def __call__(self, obs, time_ns):
            self.calls += 1
            return self._rt.fns[_ECHO](obs)

    def __init__(self):
        self.session: _EchoPolicy._Session

    def new_session(self, context=None, rt=None) -> ChunkSession:
        assert rt is not None
        self.session = _EchoPolicy._Session(rt)
        return self.session

    @property
    def functions(self):
        return {_ECHO: lambda obs: obs}


class _CountingLayer(Layer):
    """Counts the calls that reach the session it wraps."""

    def __init__(self):
        self.calls = 0

    class _Session(DelegatingChunkSession):
        def __init__(self, inner: ChunkSession, layer: '_CountingLayer'):
            super().__init__(inner)
            self._layer = layer

        def __call__(self, obs, time_ns):
            self._layer.calls += 1
            return self._inner(obs, time_ns)

    def make_session(self, inner):
        return self._Session(inner, self)


@pytest.fixture
def opened():
    """Opens the sessions a test asks for, and closes every one at teardown."""
    sessions = []

    def make(policy: Policy) -> ChunkSession:
        sessions.append(policy.new_session())
        return sessions[-1]

    yield make
    for session in sessions:
        session.close()


class TestBlocking:
    """A policy whose sessions answer in the call that asked."""

    def test_a_session_that_answers_commands_is_refused_and_closed(self):
        """Nothing above a ``ChunkPlayer`` has work to wait out, and the session opened to find that out
        holds a connection of its own."""

        class _Commanding(Session):
            def __init__(self):
                self.closed = False

            def __call__(self, obs, time_ns):
                return {}, time_ns

            def close(self):
                self.closed = True

        session = _Commanding()

        class _Commander(Policy):
            def new_session(self, context=None, rt=None):
                return session

        with pytest.raises(AssertionError, match='answers commands'):
            blocking(_Commander()).new_session()
        assert session.closed, 'the session it opened to check was dropped without closing'

    def test_a_session_that_answers_in_its_own_call_is_called_one_time(self, opened):
        policy = _PlainPolicy()

        assert opened(blocking(policy))({'x': 1}, 0.0).result() == [{'action': {'x': 1}}]
        assert policy.session.calls == 1

    def test_the_handle_a_call_answers_is_already_done(self, opened):
        """The wait is what ``blocking`` adds: the caller reads the chunk without asking a second time."""
        policy = _EchoPolicy()

        answer = opened(blocking(policy))({'x': 1}, 0.0)

        assert answer.done()
        assert answer.result() == {'x': 1}
        assert policy.session.calls == 1

    def test_a_layer_above_it_is_called_one_time_for_one_answer(self, opened):
        """A layer above ``blocking`` is called once for one answer. That is why ``blocking`` wraps the
        policy and not the chain: a layer that encodes the observation, or records it, would otherwise do
        that work once per call the answer took."""
        layer, policy = _CountingLayer(), _EchoPolicy()

        assert opened(layer.wrap(blocking(policy)))({'x': 1}, 0.0).result() == {'x': 1}
        assert (layer.calls, policy.session.calls) == (1, 1)

    def test_it_serves_its_functions_itself(self):
        """A blocking policy runs its own functions, so nothing above it builds a runtime for them."""
        assert blocking(_EchoPolicy()).functions == {}

    def test_closing_the_session_closes_the_runtime_it_made(self):
        """The session owns the runtime it was made with, and closing the session is the only way to
        close it."""
        policy = _EchoPolicy()
        session = blocking(policy).new_session()
        session.close()

        # The session's own runtime is closed, so the function it would start is gone.
        with pytest.raises(RuntimeError):
            policy.session({'x': 1}, 0)


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
