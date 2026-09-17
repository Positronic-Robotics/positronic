"""Threaded policy execution with answer visibility on the episode's clock.

TODO: Migrate callers of the blocking policy adapter to the Processor API.
"""

import concurrent.futures
import contextvars
import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ParamSpec, TypeVar

from positronic.policy.base import Answer, NotAnswered, Runtime

P = ParamSpec('P')
T = TypeVar('T')


class _UnchargedAnswer(Answer[T]):
    """A function's result, readable as soon as its worker finishes."""

    def __init__(self, call: Future[T]) -> None:
        self.call = call
        self.result_read = False
        self.completion_reported = False

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

    def _delay_sec(self, until_ns: int) -> float:
        """Wall time still owed before advancing the clock; infinity requires worker completion."""
        return 0.0 if self.call.done() else float('inf')


class _ChargedAnswer(_UnchargedAnswer[T]):
    """A worker result visible after the episode clock pays for queueing and execution."""

    def __init__(self, pool: ThreadPoolExecutor, clock: Callable[[], int], function: Callable[[], T]) -> None:
        self._clock = clock
        self._submitted_ns = clock()
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
        return self._clock() >= self._ready_at_ns

    def _delay_sec(self, until_ns: int) -> float:
        if self.call.done():
            return 0.0
        remaining_ns = until_ns - self._submitted_ns - (time.monotonic_ns() - self._submitted_wall_ns)
        return max(remaining_ns / 1e9, 0.0)


class WaitStatus(Enum):
    ANSWERS_READY = auto()
    CAN_ADVANCE = auto()
    TIMED_OUT = auto()


@dataclass(frozen=True)
class WaitResult:
    """Why a wait ended and any answers it reports."""

    status: WaitStatus
    completed: tuple[Answer[Any], ...] = ()


class Executor(Runtime):
    """Run submitted functions on worker threads and expose their answers on ``clock``.

    Real execution exposes completed futures immediately and never waits for simulated time. In
    simulation, charged calls include queueing and execution time; uncharged calls hold the simulator
    until the worker finishes. Submission and result reads belong to the control thread; submitted
    functions must not mutate episode state.
    """

    def __init__(
        self, clock: Callable[[], int], *, simulated: bool, charge_inference_time: bool, max_workers: int = 1
    ) -> None:
        self._clock = clock
        self._simulated = simulated
        self._charge_inference_time = charge_inference_time
        self._tick = -1
        self._tick_time_ns: int | None = None
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='policy-fn')
        self._answers: set[_UnchargedAnswer[Any]] = set()

    @property
    def time_ns(self) -> int:
        return self._clock()

    @property
    def tick(self) -> int:
        """The zero-based control-tick index, unchanged when calls occur at the same clock time."""
        return self._tick

    def start_tick(self) -> None:
        """Count a new tick if the episode clock has advanced since the previous call."""
        now_ns = self.time_ns
        if now_ns != self._tick_time_ns:
            self._tick += 1
            self._tick_time_ns = now_ns

    @property
    def has_pending(self) -> bool:
        """Whether any submitted call still has an unreported completion."""
        return any(not answer.completion_reported for answer in self._answers)

    def take_completed(self) -> tuple[Answer[Any], ...]:
        """Consume each completion once, when its answer becomes visible on the episode clock.

        Reading a result and consuming its completion are independent. Failures and cancellations
        also complete a call; callers observe them through ``Answer.result``.
        """
        completed = tuple(answer for answer in self._answers if not answer.completion_reported and answer.done())
        for answer in completed:
            answer.completion_reported = True
        self._answers = {
            answer
            for answer in self._answers
            if not (answer.completion_reported and (answer.result_read or answer.call.cancelled()))
        }
        return completed

    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]:
        context = contextvars.copy_context()

        def work() -> T:
            return context.run(function, *args, **kwargs)

        answer = (
            _ChargedAnswer(self._pool, self._clock, work)
            if self._simulated and self._charge_inference_time
            else _UnchargedAnswer(self._pool.submit(work))
        )
        self._answers.add(answer)
        return answer

    def wait(self, timeout_sec: float) -> WaitResult:
        """Check for new answers, waiting at most ``timeout_sec`` real seconds. Zero only checks.

        Returns:
            ANSWERS_READY: ``completed`` contains newly available answers, each reported once.
            CAN_ADVANCE: no new answers; time can advance.
            TIMED_OUT: simulation must keep waiting.

        This method never advances the clock. On a real rig, it returns immediately.
        """
        deadline_ns = time.monotonic_ns() + round(timeout_sec * 1e9)
        while True:
            if completed := self.take_completed():
                return WaitResult(WaitStatus.ANSWERS_READY, completed)
            if not self._simulated:
                return WaitResult(WaitStatus.CAN_ADVANCE)
            now_ns = self.time_ns
            delay_sec = max((answer._delay_sec(now_ns) for answer in self._answers), default=0.0)
            if delay_sec == 0:
                return WaitResult(WaitStatus.CAN_ADVANCE)
            remaining_sec = (deadline_ns - time.monotonic_ns()) / 1e9
            if remaining_sec <= 0:
                return WaitResult(WaitStatus.TIMED_OUT)
            pending = tuple(answer.call for answer in self._answers if not answer.call.done())
            concurrent.futures.wait(
                pending, timeout=min(delay_sec, remaining_sec), return_when=concurrent.futures.FIRST_COMPLETED
            )

    def close(self) -> None:
        """Cancel queued calls, drain running calls, and report failures whose results were never read."""
        self._pool.shutdown(wait=True, cancel_futures=True)
        for answer in self._answers:
            if answer.result_read or answer.call.cancelled():
                continue
            # rules-allow: swallowed-error — the caller dropped this answer; report its failure during cleanup.
            if (exc := answer.call.exception()) is not None:
                logging.error('A submitted function failed without its result being read: %s', exc)
        self._answers.clear()
