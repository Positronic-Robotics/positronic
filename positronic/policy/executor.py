"""The in-process runtime: a call starts the work on a worker thread and returns an ``Answer``."""

import concurrent.futures
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any

from pimm import Clock
from positronic.offboard.protocol import MODEL_CALL
from positronic.policy.base import (
    Answer,
    DelegatingPolicy,
    DelegatingSession,
    Fn,
    NotAnswered,
    Policy,
    Runtime,
    Session,
    TimedSession,
)


class _Charge:
    """What one call costs the world it was made in.

    The answer is withheld until that world's clock has advanced, from the instant the call was made, by the
    wall time the call took. A world on a wall clock reaches that instant as the answer lands, so nothing is
    ever withheld there.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._made_ns, self._made_wall_ns = clock.now_ns(), time.monotonic_ns()
        # Stamped from the thread the call ran on, so ``paid`` reads it across threads.
        self._landed_wall_ns: int | None = None
        self._released = False

    def land(self) -> None:
        self._landed_wall_ns = time.monotonic_ns()

    def release(self) -> None:
        """Charge nothing from here on: whoever was advancing this world has stopped."""
        self._released = True

    def paid(self) -> bool:
        if self._released:
            return True
        if self._landed_wall_ns is None:
            return False
        return self._clock.now_ns() >= self._made_ns + (self._landed_wall_ns - self._made_wall_ns)


class Executor(Runtime):
    """Serves a set of functions on worker threads of its own, ``max_workers`` calls at a time.

    A call runs under a copy of the context it was made in, so telemetry recorded inside it anchors where
    it was asked for.
    """

    class _Answer(Answer):
        def __init__(
            self,
            name: str,
            call: Future[Any],
            read: Callable[['Executor._Answer'], None],
            charge: '_Charge | None' = None,
        ):
            self.name = name
            self.call = call
            self._read = read
            self._charge = charge

        def done(self) -> bool:
            return self.call.done() and (self._charge is None or self._charge.paid())

        def result(self) -> Any:
            if not self.done():
                raise NotAnswered('The call is not answered yet')
            self._read(self)
            return self.call.result()

        def release(self) -> None:
            """Answer as soon as the call lands, whatever the world still owes for it."""
            if self._charge is not None:
                self._charge.release()

        def failure(self) -> BaseException | None:
            """What the call raised, once it has answered. ``None`` when it returned a value or was cancelled."""
            return None if self.call.cancelled() else self.call.exception()

    def __init__(self, functions: Mapping[str, Callable[..., Any]], *, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='policy-fn')
        self._fns: Mapping[str, Fn] = {name: partial(self._start, name, fn) for name, fn in functions.items()}
        # Every answer that no caller has read. A call that has still to answer is one of these, so
        # ``has_unanswered_call`` and ``owes_an_answer`` read this one set.
        self._unread: set[Executor._Answer] = set()
        self._lock = threading.Lock()
        self._charged_clock: Clock | None = None
        self._called = False

    def charge_wall_time_to(self, clock: Clock) -> None:
        """Charge every call to ``clock``'s world, for the wall time the call takes. Refused once a call has
        been made: that call is uncharged, and a runtime charging only some of its calls times the world
        against a debt nobody can read off it.

        The answer stays unanswered until ``clock`` has advanced by that duration from the instant the call
        was made, so whoever reads it keeps the world running in the meantime.
        """
        with self._lock:
            if self._called:
                raise RuntimeError('This runtime has already served a call, which nothing charged')
            self._charged_clock = clock

    @property
    def fns(self) -> Mapping[str, Fn]:
        return self._fns

    @property
    def has_unanswered_call(self) -> bool:
        """Whether any call is still to answer. A charged call that has landed is one of these until the
        world has paid for it: nothing is running, and the caller still cannot read a result."""
        with self._lock:
            return any(not answer.done() for answer in self._unread)

    @property
    def owes_an_answer(self) -> bool:
        """Whether any call's answer has still to be read, whether or not that call has landed."""
        # TODO(#661): a caller polls this because a session cannot say when it wants the next call. Rung 7
        # gives the session ``resume_at``, and the poll goes with it.
        with self._lock:
            return bool(self._unread)

    def wait_until_landed(self, timeout: float | None = None) -> None:
        """Block until every call made so far has finished running, or until ``timeout`` seconds pass.

        A charged call has landed before it has answered: the world still owes it the wall time it took,
        and only the caller can run the world, so a wait for the answer would never return.
        """
        with self._lock:
            pending = [answer.call for answer in self._unread]
        concurrent.futures.wait(pending, timeout=timeout)

    def _start(self, name: str, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Answer:
        context = contextvars.copy_context()
        with self._lock:
            # Read under the same lock the setter takes, so no call runs against a half-installed charge.
            self._called, charged_clock = True, self._charged_clock
        # Opened before the submit, so a call that queues for a worker is charged for that wait too.
        charge = _Charge(charged_clock) if charged_clock is not None else None
        call = self._pool.submit(context.run, fn, *args, **kwargs)
        answer = self._Answer(name, call, self._read, charge)
        if charge is not None:
            call.add_done_callback(lambda _: charge.land())
        with self._lock:
            self._unread.add(answer)
        return answer

    def _read(self, answer: '_Answer') -> None:
        with self._lock:
            self._unread.discard(answer)

    @staticmethod
    def _closed(*args: Any, **kwargs: Any) -> Answer:
        raise RuntimeError('The runtime is closed and serves nothing')

    def close(self) -> None:
        """Drop the queued calls and wait out those in flight, which may still hold their caller's resources.
        A call made after close raises.

        Reports what a call raised that no caller read: the session that asked for it has gone.
        """
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            # A function holds what it was declared with — model weights, a socket. Nothing reaches them
            # through this runtime after it closes.
            unread, self._unread = self._unread, set()
            self._fns = dict.fromkeys(self._fns, self._closed)
        for answer in unread:
            answer.release()
            # rules-allow: swallowed-error — the caller dropped the answer, so there is nobody to raise to,
            # and the log is the only place the failure can go.
            if (exc := answer.failure()) is not None:
                logging.error(f'The function {answer.name} failed and no caller read its answer: {exc}')


class _BlockingPolicy(DelegatingPolicy):
    class _Session(DelegatingSession):
        def __init__(self, inner: Session, rt: Executor):
            super().__init__(inner)
            self._rt = rt

        def __call__(self, obs: Mapping[str, Any], time_ns: int) -> list[dict[str, Any]] | None:
            # The inner session reads an answer only on a later call. A test of ``has_unanswered_call``
            # would exit on a call that lands while the session call runs, leaving its answer unread.
            while (actions := self._inner(obs, time_ns)) is None and self._rt.owes_an_answer:
                self._rt.wait_until_landed()
            return actions

        def close(self):
            # The runtime closes first: a call in flight is still using what the session holds.
            self._rt.close()
            self._inner.close()

    def new_session(self, context=None, rt=None) -> Session:
        assert rt is None, 'a blocking policy serves its own functions; nothing above it runs them'
        own = Executor(self._inner.functions)
        try:
            return TimedSession(_BlockingPolicy._Session(self._inner.new_session(context, own), own), MODEL_CALL)
        except BaseException:
            own.close()
            raise

    @property
    def functions(self) -> Mapping[str, Callable[..., Any]]:
        return {}


def blocking(policy: Policy) -> Policy:
    """``policy`` with its heavy work waited out: a session answers in the call that asked.

    For a caller with no control loop to give the time back to — a server request, a warmup, a probe.
    Layers wrap the result rather than the other way round, so each sees one call per answer. Layers that
    ``policy`` composes itself are inside, so those still run once per call, each with the ``time_ns`` of
    the call that asked.
    """
    return _BlockingPolicy(policy)
