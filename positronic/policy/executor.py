"""Threaded policy execution with answer visibility on the episode's clock.

TODO: Migrate callers of the blocking policy adapter to the Processor API.
"""

import concurrent.futures
import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, ParamSpec, TypeVar

import pimm
from positronic.policy.base import Answer, NotAnswered, Runtime

P = ParamSpec('P')
T = TypeVar('T')

# Check shutdown while waiting for a function that may never return.
STOP_POLL_SEC = 0.1


class _UnchargedAnswer(Answer[T]):
    """A function's result, readable as soon as its worker finishes."""

    def __init__(self, call: Future[T]) -> None:
        self.call = call
        self.result_read = False

    def done(self) -> bool:
        return self.call.done()

    def result(self) -> T:
        if not self.done():
            raise NotAnswered('The call has not answered on the episode clock')
        self.result_read = True
        return self.call.result()

    def cancel(self) -> None:
        """Cancel queued work; a function already running must finish before its resources can close."""
        self.call.cancel()

    def wait(self, until_ns: int, should_stop: pimm.SignalReceiver[bool]) -> None:
        """Wait for completion while simulated time stands still."""
        while not self.call.done() and not should_stop.value:
            concurrent.futures.wait((self.call,), timeout=STOP_POLL_SEC)


class _ChargedAnswer(_UnchargedAnswer[T]):
    """A worker result visible after the episode clock pays for queueing and execution."""

    def __init__(self, pool: ThreadPoolExecutor, clock: pimm.Clock, function: Callable[[], T]) -> None:
        self._clock = clock
        self._submitted_ns = clock.now_ns()
        self._submitted_wall_ns = time.monotonic_ns()
        self._ready_at_ns: int | None = None
        super().__init__(pool.submit(self._run, function))

    def _run(self, function: Callable[[], T]) -> T:
        try:
            return function()
        finally:
            self._ready_at_ns = self._submitted_ns + time.monotonic_ns() - self._submitted_wall_ns

    def done(self) -> bool:
        if not self.call.done():
            return False
        if self.call.cancelled():
            return True
        assert self._ready_at_ns is not None, 'a finished function has a visibility timestamp'
        return self._clock.now_ns() >= self._ready_at_ns

    def wait(self, until_ns: int, should_stop: pimm.SignalReceiver[bool]) -> None:
        """Wait for completion or enough elapsed wall time to permit the requested episode time."""
        while not self.call.done() and not should_stop.value:
            remaining_ns = until_ns - self._submitted_ns - (time.monotonic_ns() - self._submitted_wall_ns)
            if remaining_ns <= 0:
                break
            timeout = min(STOP_POLL_SEC, remaining_ns / 1e9)
            concurrent.futures.wait((self.call,), timeout=timeout)


class Executor(Runtime):
    """Run submitted functions on worker threads and expose their answers on ``clock``.

    Real execution exposes completed futures immediately and never waits for simulated time. In
    simulation, charged calls include queueing and execution time; uncharged calls hold the simulator
    until the worker finishes. Submission and result reads belong to the control thread; submitted
    functions must not mutate episode state.
    """

    def __init__(
        self, clock: pimm.Clock, *, simulated: bool, charge_inference_time: bool, max_workers: int = 1
    ) -> None:
        self._clock = clock
        self._simulated = simulated
        self._charge_inference_time = charge_inference_time
        self._tick = -1
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='policy-fn')
        self._answers: set[_UnchargedAnswer[Any]] = set()
        self._lock = threading.Lock()

    @property
    def time_ns(self) -> int:
        return self._clock.now_ns()

    @property
    def tick(self) -> int:
        """The zero-based policy-call index, or -1 before the first call."""
        return self._tick

    def start_tick(self) -> None:
        """Start one root-policy call on the control thread."""
        self._tick += 1

    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]:
        context = contextvars.copy_context()

        def work() -> T:
            return context.run(function, *args, **kwargs)

        answer = (
            _ChargedAnswer(self._pool, self._clock, work)
            if self._simulated and self._charge_inference_time
            else _UnchargedAnswer(self._pool.submit(work))
        )
        with self._lock:
            self._answers = {pending for pending in self._answers if not pending.result_read}
            self._answers.add(answer)
        return answer

    def wait(self, until_ns: int, should_stop: pimm.SignalReceiver[bool]) -> None:
        """In simulation, let calls reach ``until_ns`` or finish. Real execution returns immediately."""
        if not self._simulated:
            return
        with self._lock:
            pending = tuple(self._answers)
        for answer in pending:
            answer.wait(until_ns, should_stop)

    def close(self) -> None:
        """Cancel queued calls, drain running calls, and report failures whose results were never read."""
        self._pool.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            answers, self._answers = self._answers, set()
        for answer in answers:
            if answer.result_read or answer.call.cancelled():
                continue
            # rules-allow: swallowed-error — the caller dropped this answer; report its failure during cleanup.
            if (exc := answer.call.exception()) is not None:
                logging.error('A submitted function failed without its result being read: %s', exc)
