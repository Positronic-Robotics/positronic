"""Threaded policy execution with answer visibility on the episode's clock."""

import concurrent.futures
import contextvars
import logging
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor
from concurrent.futures import Executor as WorkerExecutor
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, ParamSpec, TypeVar

from positronic import telemetry, telemetry_keys
from positronic.policy.base import Answer, NotAnswered, QueueFull, Runtime
from positronic.utils import check_positive_int

P = ParamSpec('P')
T = TypeVar('T')


class _UnchargedAnswer(Answer[T]):
    """A function's result, readable as soon as its worker finishes."""

    def __init__(self, pool: WorkerExecutor, function: Callable[[], T], *, charge_time: bool = True) -> None:
        self.workers = pool
        self._charge_time = charge_time
        self._submitted_wall_ns = time.monotonic_ns()
        self._elapsed_ns = 0
        self._execution_ns = 0
        self.result_read = False
        self.cancel_requested = False
        self.completion_reported = False
        self.call = pool.submit(self._run, function)

    def _run(self, function: Callable[[], T]) -> T:
        started_ns = time.monotonic_ns()
        try:
            return function()
        finally:
            finished_ns = time.monotonic_ns()
            if self._charge_time:
                self._elapsed_ns = finished_ns - self._submitted_wall_ns
                self._execution_ns = finished_ns - started_ns

    def done(self) -> bool:
        return self.call.done()

    def _require_duration(self) -> None:
        if not self.done():
            raise NotAnswered('The call has not answered on the episode clock')
        if self.call.cancelled():
            raise CancelledError()

    @property
    def elapsed_sec(self) -> float:
        self._require_duration()
        return self._elapsed_ns / 1e9

    @property
    def execution_sec(self) -> float:
        self._require_duration()
        return self._execution_ns / 1e9

    def result(self) -> T:
        if not self.done():
            raise NotAnswered('The call has not answered on the episode clock')
        self.result_read = True
        return self.call.result()

    def cancel(self) -> None:
        """Discard the answer; a function already running must finish before releasing its slot."""
        self.cancel_requested = True
        if not self.call.cancel() and self.call.done():
            self.discard()

    def discard(self) -> None:
        """Consume a completed answer and report an unread failure."""
        if self.result_read or self.call.cancelled():
            return
        self.result_read = True
        if (exc := self.call.exception()) is not None:
            logging.error('A submitted function failed without its result being read: %s', exc)

    def _delay_sec(self, until_ns: int) -> float:
        """Wall time still owed before advancing the clock; infinity requires worker completion."""
        return 0.0 if self.call.done() else float('inf')


class _ChargedAnswer(_UnchargedAnswer[T]):
    """A worker result visible after the episode clock pays for queueing and execution."""

    def __init__(self, pool: WorkerExecutor, clock: Callable[[], int], function: Callable[[], T]) -> None:
        self._clock = clock
        self._submitted_ns = clock()
        super().__init__(pool, function)

    def done(self) -> bool:
        if not self.call.done():
            return False
        if self.call.cancelled():
            return True
        return self._clock() >= self._submitted_ns + self._elapsed_ns

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


class ExecutionRuntime(Runtime):
    """Own workers and a call limit while sharing the enclosing runtime's episode state."""

    def __init__(self, runtime: Runtime, *, max_in_flight: int):
        check_positive_int(max_in_flight, 'max_in_flight')
        self._runtime = runtime
        self._limit = max_in_flight
        self._workers = ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix='policy-fn')
        self._closed = False

    @property
    def time_ns(self) -> int:
        return self._runtime.time_ns

    @property
    def tick(self) -> int:
        return self._runtime.tick

    def set_metadata(self, name: str, value: Any) -> None:
        self._runtime.set_metadata(name, value)

    def has_ready_answers(self) -> bool:
        return self._runtime.has_ready_answers()

    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]:
        if self._closed:
            raise RuntimeError('Policy execution is closed')
        return self._runtime._submit_on(self._workers, self._limit, function, *args, **kwargs)

    def _submit_on(
        self, workers: WorkerExecutor, limit: int, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Answer[T]:
        return self._runtime._submit_on(workers, limit, function, *args, **kwargs)

    def close(self) -> None:
        """Drain this execution's workers before its resources close."""
        self._closed = True
        self._workers.shutdown(wait=True, cancel_futures=True)


class Executor(Runtime):
    """Run submitted functions on worker threads and expose their answers on ``clock``.

    Real execution exposes completed futures immediately and never waits for simulated time. In
    simulation, charged calls include queueing and execution time; uncharged calls hold the simulator
    until the worker finishes. Submission and result reads belong to the control thread; submitted
    functions must not mutate episode state.

    ``max_in_flight`` bounds local calls until their results are read or cancelled calls finish.
    Each ``ExecutionRuntime`` owns an independent limit and workers; every answer participates in
    this runtime's timing, completion delivery and shutdown.
    """

    def __init__(
        self, clock: Callable[[], int], *, simulated: bool, charge_inference_time: bool, max_in_flight: int = 16
    ) -> None:
        self._clock = clock
        self._simulated = simulated
        self._charge_inference_time = charge_inference_time
        self._tick = -1
        self._tick_time_ns: int | None = None
        self._closed = False
        self._execution = ExecutionRuntime(self, max_in_flight=max_in_flight)
        self._answers: set[_UnchargedAnswer[Any]] = set()
        self._metadata: dict[str, Any] = {}
        self._observed_ns: int | None = None

    def set_metadata(self, name: str, value: Any) -> None:
        self._metadata[name] = deepcopy(value)

    def observe(self, captured_ns: int | None) -> None:
        """Set the episode-clock time at which the current observation was captured, or ``None`` when unknown."""
        self._observed_ns = captured_ns

    def metadata_snapshot(self) -> dict[str, Any]:
        """Return an independent copy of the episode metadata."""
        return deepcopy(self._metadata)

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

    def has_ready_answers(self) -> bool:
        return any(not answer.result_read and not answer.cancel_requested and answer.done() for answer in self._answers)

    def take_completed(self) -> tuple[Answer[Any], ...]:
        """Consume each completion once, when its answer becomes visible on the episode clock.

        Reading a result and consuming its completion are independent. Failures and cancellations
        also complete a call; callers observe them through ``Answer.result``.
        """
        completed = tuple(answer for answer in self._answers if not answer.completion_reported and answer.done())
        for answer in completed:
            answer.completion_reported = True
            if answer.cancel_requested:
                answer.discard()
        self._answers = {
            answer
            for answer in self._answers
            if not (answer.completion_reported and (answer.result_read or answer.call.cancelled()))
        }
        return completed

    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]:
        return self._execution.submit(function, *args, **kwargs)

    def _submit_on(
        self, workers: WorkerExecutor, limit: int, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Answer[T]:
        if self._closed:
            raise RuntimeError('Policy execution is closed')
        outstanding = sum(
            not answer.call.done() or not (answer.result_read or answer.cancel_requested)
            for answer in self._answers
            if answer.workers is workers
        )
        if outstanding >= limit:
            raise QueueFull('Policy execution is at capacity')
        context = contextvars.copy_context()

        def invoke() -> T:
            return function(*args, **kwargs)

        if telemetry.enabled():
            invoke = partial(self._traced_call, invoke, self._observed_ns)
        work = partial(context.run, invoke)
        answer = (
            _ChargedAnswer(workers, self._clock, work)
            if self._simulated and self._charge_inference_time
            else _UnchargedAnswer(workers, work, charge_time=not self._simulated)
        )
        self._answers.add(answer)
        return answer

    def _traced_call(self, call: Callable[[], T], observed_ns: int | None) -> T:
        """Run ``call`` in a ``policy.submit`` span that holds the age of the observation when the call starts."""
        attrs = {} if observed_ns is None else {telemetry_keys.ATTR_OBS_AGE_MS: (self._clock() - observed_ns) / 1e6}
        with telemetry.span(telemetry_keys.SPAN_POLICY_SUBMIT, **attrs):
            return call()

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
        self._closed = True
        for answer in self._answers:
            answer.call.cancel()
        self._execution.close()
        concurrent.futures.wait([answer.call for answer in self._answers if not answer.call.done()])
        for answer in self._answers:
            answer.discard()
        self._answers.clear()
