"""Threaded policy execution with answer visibility on the episode's clock."""

import concurrent.futures
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import Any, ParamSpec, TypeVar

from positronic import telemetry, telemetry_keys
from positronic.policy.base import Answer, InputT, NotAnswered, Obs, OutputT, Processor, ProcessorRun, Runtime, Step
from positronic.policy.journal import (
    PLAIN_DATA,
    Activity,
    Finished,
    Journal,
    JournalAnswer,
    JournalWriter,
    Raised,
    Started,
    Stopped,
    TurnLog,
    UnrecordableResult,
    Wake,
)

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

    def visible_at(self, now_ns: int) -> bool:
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
        return self.visible_at(self._clock())

    def visible_at(self, now_ns: int) -> bool:
        if not self.call.done():
            return False
        if self.call.cancelled():
            return True
        assert self._ready_at_ns is not None, 'a finished function has a visibility timestamp'
        return now_ns >= self._ready_at_ns

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
        self._invocation = -1
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='policy-fn')
        self._answers: set[_UnchargedAnswer[Any]] = set()

    @property
    def time_ns(self) -> int:
        return self._clock()

    @property
    def tick(self) -> int:
        """The zero-based control-tick index, unchanged when calls occur at the same clock time."""
        return self._tick

    @property
    def invocation(self) -> int:
        return self._invocation

    def begin_turn(self, obs: Obs, wake: Wake) -> Obs:
        """Count the policy call, and a tick if the clock has moved; return the observation the policy receives."""
        self._tick, self._tick_time_ns, self._invocation = self._next_turn()
        return obs

    def _next_turn(self) -> tuple[int, int, int]:
        """The tick, its clock time and the invocation of a turn that begins now. Nothing is counted yet."""
        now_ns = self._clock()
        tick = self._tick if now_ns == self._tick_time_ns else self._tick + 1
        return tick, now_ns, self._invocation + 1

    def end_turn(self, step: Step, wake_at_ns: int) -> None:
        """End the turn with the policy's step and the harness wake-up time. A plain executor records neither."""

    def fail_turn(self, error: BaseException) -> None:
        """End the turn with the error the policy raised. A plain executor does not record it."""

    def report_emitted(self, command: str) -> None:
        """Note that the harness emitted ``command`` of the last step. A plain executor records nothing."""

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
        return self._submit(function, *args, **kwargs)

    def _submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> _UnchargedAnswer[T]:
        context = contextvars.copy_context()

        def invoke() -> T:
            return function(*args, **kwargs)

        if telemetry.enabled():
            invoke = telemetry.traced(telemetry_keys.SPAN_POLICY_SUBMIT)(invoke)
        work = partial(context.run, invoke)
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

    def close(
        self, run: ProcessorRun[Any, Any] | None = None, ending: Mapping[str, Any] | BaseException | None = None
    ) -> None:
        """Drain the submitted work, then close ``run``, a generator that the caller hands over.

        ``ending`` is the terminal payload, the error that ended the episode, or ``None`` without either.
        A plain executor does not record it.
        """
        self._drain()
        if run is not None:
            run.close()

    def _drain(self) -> None:
        """Cancel queued calls, drain running calls, and report failures whose results were never read."""
        self._pool.shutdown(wait=True, cancel_futures=True)
        for answer in self._answers:
            if answer.result_read or answer.call.cancelled():
                continue
            # rules-allow: swallowed-error — the caller dropped this answer; report its failure during cleanup.
            if (exc := answer.call.exception()) is not None:
                logging.error('A submitted function failed without its result being read: %s', exc)
        self._answers.clear()


class JournaledExecutor(Executor):
    """An ``Executor`` that journals the startup, the turns and the close of a policy.

    The first ``start`` primes the policy at a startup time the journal records; a ``start`` inside a scope
    starts a nested processor. Inside a scope, ``time_ns`` is the scope's time; ``tick`` and ``invocation``
    are those of the last journaled turn. Answers change only at turn entry, which publishes in submission
    order the work that returned before it and is visible on the episode clock. The policy submits only an
    ``Activity``, and the work receives its own decoded copy of the arguments. ``close`` drains the work,
    closes the run it is handed and ends the journal.
    """

    def __init__(
        self,
        clock: Callable[[], int],
        journal: Journal,
        started: Started,
        *,
        simulated: bool,
        charge_inference_time: bool,
        max_workers: int = 1,
    ) -> None:
        super().__init__(
            clock, simulated=simulated, charge_inference_time=charge_inference_time, max_workers=max_workers
        )
        self._log = TurnLog(journal, JournalWriter(journal, started), self._cancel)
        self._unpublished: dict[int, _UnchargedAnswer[bytes]] = {}
        self._returned: set[int] = set()
        self._returned_lock = threading.Lock()

    @property
    def time_ns(self) -> int:
        return self._clock() if self._log.scope is None else self._log.time_ns

    @property
    def tick(self) -> int:
        return self._log.tick

    @property
    def invocation(self) -> int:
        return self._log.invocation

    @property
    def journaled(self) -> bool:
        return True

    def start(
        self, processor: Processor[InputT, OutputT], /, *args: Any, **kwargs: Any
    ) -> ProcessorRun[InputT, OutputT]:
        if self._log.scope is not None:
            return super().start(processor, *args, **kwargs)
        self._log.begin_startup(self._clock())
        try:
            run = super().start(processor, *args, **kwargs)
        except BaseException as exc:
            self._log.start_failed(exc)
            raise
        self._log.primed()
        return run

    def begin_turn(self, obs: Obs, wake: Wake) -> Obs:
        with self._returned_lock:
            returned = set(self._returned)
        tick, now_ns, invocation = self._next_turn()
        completed = self._completed(returned, now_ns)
        for call in completed.values():
            if not call.call.cancelled() and isinstance(error := call.call.exception(), UnrecordableResult):
                raise error
        owned = self._log.begin(invocation, now_ns, tick, wake, self._log.journal.observations.encode(obs))
        # Counted only now: a turn that the journal does not record takes no invocation.
        self._tick, self._tick_time_ns, self._invocation = tick, now_ns, invocation
        try:
            for submission, call in completed.items():
                del self._unpublished[submission]
                self._publish(self._log.pending[submission], call)
        except BaseException as exc:
            self._log.fail(exc)
            raise
        return owned

    def _completed(self, returned: set[int], now_ns: int) -> dict[int, _UnchargedAnswer[bytes]]:
        """The unpublished work that was cancelled or had ``returned``, and is visible at ``now_ns``."""
        completed = {}
        for submission, call in self._unpublished.items():
            if submission in returned:
                # The function has returned, so its future settles at once.
                concurrent.futures.wait([call.call])
            elif not call.call.cancelled():
                continue
            if call.visible_at(now_ns):
                completed[submission] = call
        return completed

    def _publish(self, answer: JournalAnswer[Any], call: _UnchargedAnswer[bytes]) -> None:
        call.completion_reported = True
        call.result_read = True
        if call.call.cancelled():
            self._log.publish_cancelled(answer)
        elif (error := call.call.exception()) is None:
            self._log.publish_result(answer, call.call.result())
        else:
            self._log.publish_failure(answer, Raised.of(error))

    def end_turn(self, step: Step, wake_at_ns: int) -> None:
        self._log.end(step, wake_at_ns)

    def fail_turn(self, error: BaseException) -> None:
        self._log.fail(error)

    def report_emitted(self, command: str) -> None:
        self._log.emitted(command)

    def submit(self, function: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> Answer[T]:
        if not isinstance(function, Activity):
            raise TypeError(f'A journaled episode submits only an Activity, not {function!r}')
        answer, owned_args, owned_kwargs = self._log.submit(function, args, kwargs)
        self._unpublished[answer.submission] = self._submit(
            self._run_activity, answer.submission, function, owned_args, owned_kwargs
        )
        return answer

    def _run_activity(
        self, submission: int, activity: Activity[..., Any], args: tuple, kwargs: dict[str, Any]
    ) -> bytes:
        """Run the work, and mark its submission returned before its future settles."""
        try:
            return self._encoded_result(activity, args, kwargs)
        finally:
            with self._returned_lock:
                self._returned.add(submission)

    @staticmethod
    def _encoded_result(activity: Activity[..., Any], args: tuple, kwargs: dict[str, Any]) -> bytes:
        result = activity.function(*args, **kwargs)
        try:
            return activity.codec.encode(result)
        except Exception as exc:
            raise UnrecordableResult(
                f'{activity.operation} v{activity.version} returned {type(result).__name__}, which '
                f'{activity.codec.NAME} cannot encode'
            ) from exc

    def _cancel(self, submission: int) -> None:
        if (call := self._unpublished.get(submission)) is not None:
            call.cancel()

    def close(
        self, run: ProcessorRun[Any, Any] | None = None, ending: Mapping[str, Any] | BaseException | None = None
    ) -> None:
        self._drain()
        self._log.begin_closing(self._clock())
        if run is not None:
            run.close()
        match ending:
            case None:
                termination = Stopped()
            case BaseException():
                termination = Raised.of(ending)
            case _:
                termination = Finished(payload=self._log.retain(PLAIN_DATA.encode(ending)))
        self._log.finish(termination, self.metadata)
