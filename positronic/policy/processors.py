"""Policy processors for scheduling, fault handling, ending a run, and temporal frame stacking.

Constructors hold configuration. ``run(runtime, *dependencies)`` creates an episode generator whose
locals hold its state. Control processors yield a ``Step`` with commands and the next wake-up time.

Describe a local stack without creating episode state::

    from positronic.policy.sequential import Sequential

    definition = Sequential(
        PauseOnUnavailable(), TemporalStack(keys=('image',), offsets_sec=(-0.2, -0.1, 0.0)), ChunkedSchedule(fps=20)
    )
    episode = runtime.start(definition, infer)
    step = episode.send(obs)
"""

from abc import ABC, abstractmethod
from bisect import insort
from collections import ChainMap, deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from statistics import fmean
from sys import maxsize
from typing import Any, Self, TypeVar

import numpy as np
from positronic_model_server.spec import ARGS, NAME, VERSION

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import (
    Answer,
    Commands,
    Obs,
    Policy,
    PolicyEnd,
    PolicyRun,
    Processor,
    ProcessorRun,
    ReturnT,
    Runtime,
    Step,
)
from positronic.policy.keys import ACTION_FPS, END


# TODO(#638): the arm is found by name because the harness serializes before the stack sees anything. Once
# domain types reach the border, this reads the status off the value.
def _is_robot_status(name: str) -> bool:
    """Whether ``name`` is an arm's status: ``robot_state.status``, or an arm's ``robot_state.{side}.status``."""
    return name.startswith(f'{keys.ROBOT_STATE}.') and name.endswith(keys.STATUS_SUFFIX)


def _arms_available(obs) -> bool:
    """Whether every arm in the observation will take a command; one naming no arm status has none to stop for.

    The wire carries a status as its number, so this is where one becomes a ``RobotStatus`` again.
    """
    return all(RobotStatus(v) is RobotStatus.AVAILABLE for name, v in obs.items() if _is_robot_status(name))


MILLISECOND_NS = 10**6


class PauseOnUnavailable(Policy):
    """Withhold commands and child calls while any arm is unavailable.

    An unavailable arm causes an empty command set and a status check one millisecond later. Once every
    arm is available, calls resume on the same child policy, retaining queued commands and pending answers.

    TODO(#789): Define plan invalidation and recovery after robot unavailability.
    """

    WIRE_NAME = 'stop_on_fault'
    WIRE_VERSION = 2

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        while True:
            if _arms_available(obs):
                try:
                    step = inner.send(obs)
                except StopIteration as end:
                    return end.value
                obs = yield step
            else:
                obs = yield Step({}, runtime.time_ns + MILLISECOND_NS)

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION}


class EndOnField(Policy):
    """End the run when a step of the inner run carries the reserved field ``END``.

    The step's other commands go out, and the run returns the ``END`` value as a ``PolicyEnd`` at the next
    observation. A schedule can drop the element that holds ``END`` before it plays, so a model that ends puts
    ``END``, with the same value, on every element of every later answer.
    """

    WIRE_NAME = 'end_on_field'
    WIRE_VERSION = 1

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        while True:
            try:
                step = inner.send(obs)
            except StopIteration as stop:
                return stop.value
            if step is not None and END in step.commands:
                commands = dict(step.commands)
                end = PolicyEnd(commands.pop(END))
                yield Step(commands, runtime.time_ns)
                return end
            obs = yield step

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION}


class DurationEstimateRun(ABC):
    """One episode's inference-delay history and duration estimates."""

    @abstractmethod
    def update(self, delay_sec: float) -> float:
        """Record a delay from submission to reading the answer, and return the updated estimate."""


class DurationEstimate(ABC):
    """A reusable strategy for estimating durations from completed inference delays."""

    @abstractmethod
    def run(self) -> DurationEstimateRun:
        """Create an estimator with independent delay history for one episode."""

    def to_spec(self) -> float | dict[str, Any]:
        """Serialize configuration without the observed delays."""
        raise NotImplementedError(f'{type(self).__name__} has no wire spec')

    @staticmethod
    def from_spec(spec: float | Mapping[str, Any]) -> 'DurationEstimate':
        if isinstance(spec, Mapping):
            return DelayEstimate(**{
                **spec,
                DelayEstimate.STATISTIC_ARG: DelayEstimate.Statistic(spec[DelayEstimate.STATISTIC_ARG]),
            })
        return ConstantDuration(spec)


class ConstantDuration(DurationEstimate):
    """A fixed duration, independent of inference delays."""

    class _Run(DurationEstimateRun):
        def __init__(self, strategy: 'ConstantDuration') -> None:
            self._strategy = strategy

        def update(self, delay_sec: float) -> float:
            return self._strategy._seconds

    def __init__(self, seconds: float = 0.0) -> None:
        if not isfinite(seconds) or seconds < 0:
            raise ValueError('seconds must be finite and not negative')
        self._seconds = seconds

    def run(self) -> DurationEstimateRun:
        return self._Run(self)

    def to_spec(self) -> float:
        return self._seconds


class DelayEstimate(DurationEstimate):
    """A serializable statistic of recent inference delays, capped in seconds."""

    class Statistic(Enum):
        MEAN = 'mean'
        MAX = 'max'

    STATISTIC_ARG = 'statistic'
    LAST_ARG = 'last'
    MAX_SEC_ARG = 'max_sec'

    class _Run(DurationEstimateRun):
        def __init__(self, strategy: 'DelayEstimate') -> None:
            self._strategy = strategy
            self._delays: deque[float] = deque(maxlen=strategy._last)

        def update(self, delay_sec: float) -> float:
            self._delays.append(delay_sec)
            strategy = self._strategy
            value = fmean(self._delays) if strategy._statistic is strategy.Statistic.MEAN else max(self._delays)
            return min(value, strategy._max_sec)

    def __init__(self, statistic: Statistic, last: int, max_sec: float) -> None:
        if isinstance(last, bool) or not isinstance(last, int) or not 1 <= last <= maxsize:
            raise ValueError(f'last must be an integer between 1 and {maxsize}')
        if not isfinite(max_sec) or max_sec < 0:
            raise ValueError('max_sec must be finite and not negative')
        self._statistic = statistic
        self._last = last
        self._max_sec = max_sec

    def run(self) -> DurationEstimateRun:
        return self._Run(self)

    def to_spec(self) -> dict[str, Any]:
        return {self.STATISTIC_ARG: self._statistic.value, self.LAST_ARG: self._last, self.MAX_SEC_ARG: self._max_sec}

    @classmethod
    def mean(cls, last: int = 5, max_sec: float = 0.4) -> 'DelayEstimate':
        """The mean of the last `last` delays, capped at `max_sec`. Use `last=1` for the latest delay."""
        return cls(cls.Statistic.MEAN, last, max_sec)

    @classmethod
    def max(cls, last: int = 5, max_sec: float = 0.4) -> 'DelayEstimate':
        """The longest of the last `last` delays, capped at `max_sec`. The RTC paper does this."""
        return cls(cls.Statistic.MAX, last, max_sec)


class ChunkedSchedule(Policy):
    """Request action chunks asynchronously and emit their commands at a fixed cadence.

        step                0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
        observation T0      ^
        inference           |--------|
        answer                       ^
        old chunk                    o0 o1 o2 o3 o4 o5 o6 o7 o8 o9
        inference_lead                              |--------------|
        observation T1                              ^
        inference                                   |--------|
        answer                                               ^
        answer waits                                         |-----|
        old chunk ends                                             ^
        new chunk                                                  n0 n1 n2 n3 n4
        robot executes      -- -- -- o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 n0 n1 n2 n3 n4

    The drawing uses ``fps=10`` and ``inference_lead=ConstantDuration(0.5)``. Each step is
    100 ms. The old chunk starts at step 3 and ends at step 13, after o9's full period.
    The next request starts five periods before that end, at step 8. Its answer arrives
    at step 11 and waits until step 13: all old actions run, then the new chunk starts at n0.
    If the answer arrives after the old chunk ends, the last target stays in force until
    the answer is read. A zero lead starts the next request at step 13 instead.

    ``infer`` returns an ordered sequence of command sets and must not mutate episode state. The first
    command is due when the completed answer is read; regular playback waits for the previous chunk to end.
    Subsequent commands are spaced by ``1 / fps``. A chunk of K commands covers K periods, including the
    final command's execution period. ``horizon_sec`` limits that duration and discards commands at or
    beyond the horizon.

    At most one call is pending. ``inference_lead`` estimates how many seconds before a chunk ends to
    request the next one, bounded by the chunk's duration. The default is zero: request at the end. A lead of the
    whole chunk requests the next one in the round that reads the answer.
    The first request starts immediately. Later deadlines use the delays of completed calls in this
    episode. An early answer waits behind the remaining commands; the new chunk starts from its first
    action. The observation is read at the request, before commands emitted in that same round.
    With ``record_stats``, each round that plans or emits a waypoint writes the waypoint counters into the
    episode metadata.

    ``replan_on_error=True`` clears the remaining plan on an arm's ``ERROR`` and makes inference due
    immediately. A pending call is kept and its answer starts playback on arrival. Further resets wait
    until a command is emitted. Set it to ``False`` to keep regular scheduling through errors.
    """

    WIRE_NAME = 'chunked_schedule'
    WIRE_VERSION = 3
    FPS_ARG = 'fps'
    HORIZON_SEC_ARG = 'horizon_sec'
    RECORD_STATS_ARG = 'record_stats'
    INFERENCE_LEAD_ARG = 'inference_lead'
    REPLAN_ON_ERROR_ARG = 'replan_on_error'

    @dataclass(frozen=True)
    class _Waypoint:
        commands: Commands
        due_ns: int
        starts_chunk: bool

    class _Stats:
        """How the schedule played its waypoints.

        A round sends the commands of every due waypoint, and on each channel the last one wins. The due
        waypoints before the last one count as dropped, also when one of their channels went out. The due
        waypoints that a new chunk replaces count as dropped too.
        """

        def __init__(self, runtime: Runtime) -> None:
            self._runtime = runtime
            self._scheduled = 0
            self._sorted_late_ns: list[int] = []
            self._gap_max_ns = 0
            self._last_emit_ns: int | None = None

        def count_round(
            self, planned: int | None, due_ns: int | None, now_ns: int, queued: int, *, starts_chunk: bool
        ) -> None:
            """Count the ``planned`` waypoints of a new chunk and the one emitted at ``now_ns``, if any."""
            if planned is None and due_ns is None:
                return
            if planned is not None:
                self._scheduled += planned
            if starts_chunk:
                self._last_emit_ns = None
            if due_ns is not None:
                insort(self._sorted_late_ns, now_ns - due_ns)
                if self._last_emit_ns is not None:
                    self._gap_max_ns = max(self._gap_max_ns, now_ns - self._last_emit_ns)
                self._last_emit_ns = now_ns
            self._write(queued)

        def _write(self, queued: int) -> None:
            late_ns = self._sorted_late_ns
            values: dict[str, float] = {
                eval_keys.SCHEDULED: self._scheduled,
                eval_keys.EMITTED: len(late_ns),
                eval_keys.DROPPED: self._scheduled - len(late_ns) - queued,
            }
            if late_ns:
                values[eval_keys.LATE_P50_MS] = self._percentile_of_sorted(late_ns, 0.5) / 1e6
                values[eval_keys.LATE_P90_MS] = self._percentile_of_sorted(late_ns, 0.9) / 1e6
                values[eval_keys.LATE_MAX_MS] = late_ns[-1] / 1e6
                values[eval_keys.GAP_MAX_MS] = self._gap_max_ns / 1e6
            for name, value in values.items():
                self._runtime.set_metadata(f'{eval_keys.SCHEDULE}.{name}', value)

        @staticmethod
        def _percentile_of_sorted(values: Sequence[int], fraction: float) -> float:
            """``np.percentile``'s linear interpolation in constant time: the control thread calls it each round."""
            position = fraction * (len(values) - 1)
            low = int(position)
            high = min(low + 1, len(values) - 1)
            return values[low] + (values[high] - values[low]) * (position - low)

    def __init__(
        self,
        fps: float,
        horizon_sec: float | None = None,
        record_stats: bool = True,
        inference_lead: DurationEstimate | None = None,
        replan_on_error: bool = True,
    ) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        if horizon_sec is not None and (not isfinite(horizon_sec) or horizon_sec <= 0):
            raise ValueError('horizon_sec must be finite and positive')
        self._fps = fps
        self._horizon_sec = horizon_sec
        self._record_stats = record_stats
        self._inference_lead = ConstantDuration() if inference_lead is None else inference_lead
        self._replan_on_error = replan_on_error

    @property
    def inference_lead(self) -> DurationEstimate:
        return self._inference_lead

    @staticmethod
    def _next_call_ns(start_ns: int, end_ns: int, lead_sec: float) -> int:
        if not isfinite(lead_sec) or lead_sec < 0:
            raise ValueError('inference_lead must return a finite, nonnegative duration')
        return end_ns - round(min(lead_sec * 1e9, end_ns - start_ns))

    def _plan_chunk(self, chunk: Sequence[Commands], start_ns: int) -> tuple[list[_Waypoint], int]:
        period_sec = 1 / self._fps
        duration_sec = len(chunk) * period_sec
        if self._horizon_sec is not None:
            duration_sec = min(duration_sec, self._horizon_sec)
        waypoints = [
            self._Waypoint(waypoint, start_ns + round(i * period_sec * 1e9), starts_chunk=i == 0)
            for i, waypoint in enumerate(chunk)
            if i * period_sec < duration_sec
        ]
        return waypoints, start_ns + round(duration_sec * 1e9)

    def run(self, runtime: Runtime, infer: Callable[[Obs], Sequence[Commands]]) -> PolicyRun:
        answer: Answer[Sequence[Commands]] | None = None
        recovering = False
        trajectory: deque[ChunkedSchedule._Waypoint] = deque()
        end_ns = 0
        call_ns = 0
        called_at_ns = 0
        lead = self._inference_lead.run()
        stats = self._Stats(runtime) if self._record_stats else None

        obs = yield
        try:
            while True:
                now_ns = runtime.time_ns
                planned = None
                statuses = (RobotStatus(obs[name]) for name in obs if _is_robot_status(name))
                needs_recovery = self._replan_on_error and not recovering and RobotStatus.ERROR in statuses
                if needs_recovery:
                    trajectory.clear()
                    end_ns = call_ns = now_ns
                    planned, recovering = 0, True
                submitted = answer is None and now_ns >= call_ns
                if submitted:
                    answer, called_at_ns = runtime.submit(infer, obs), now_ns
                if answer is not None and answer.done():
                    chunk, answer = answer.result(), None
                    lead_sec = lead.update((now_ns - called_at_ns) / 1e9)
                    start_ns = max(now_ns, end_ns)
                    waypoints, end_ns = self._plan_chunk(chunk, start_ns)
                    call_ns = self._next_call_ns(start_ns, end_ns, lead_sec) if waypoints else now_ns
                    # A regular early answer waits behind the commands that are still due.
                    trajectory = deque(waypoints) if start_ns == now_ns else deque([*trajectory, *waypoints])
                    planned = len(waypoints)
                    # A call due at the start of the new chunk goes out in this round, not one round later. A round
                    # that has already called once sends no second call with the same observation.
                    if not submitted and now_ns >= call_ns:
                        answer, called_at_ns = runtime.submit(infer, obs), now_ns

                commands: dict[str, Any] = {}
                due_ns = None
                starts_chunk = False
                while trajectory and trajectory[0].due_ns <= now_ns:
                    waypoint = trajectory.popleft()
                    due_ns = waypoint.due_ns
                    starts_chunk |= waypoint.starts_chunk
                    commands.update(waypoint.commands)
                recovering = recovering and not commands
                if stats is not None:
                    stats.count_round(planned, due_ns, now_ns, queued=len(trajectory), starts_chunk=starts_chunk)
                resume_at_ns = min(trajectory[0].due_ns, call_ns) if trajectory else call_ns
                # Pending inference asks for the earliest allowed poll; action cadence is independent.
                obs = yield Step(commands, now_ns if answer is not None else resume_at_ns)
        finally:
            if answer is not None:
                answer.cancel()

    def meta(self) -> dict[str, Any]:
        meta = {ACTION_FPS: self._fps}
        if self._horizon_sec is not None:
            meta[policy_keys.ACTION_HORIZON_SEC] = self._horizon_sec
        return meta

    def to_spec(self) -> dict[str, Any]:
        args: dict[str, Any] = {self.FPS_ARG: self._fps}
        if self._horizon_sec is not None:
            args[self.HORIZON_SEC_ARG] = self._horizon_sec
        if not self._record_stats:
            args[self.RECORD_STATS_ARG] = False
        if not self._replan_on_error:
            args[self.REPLAN_ON_ERROR_ARG] = False
        lead = self._inference_lead.to_spec()
        if lead != 0:
            args[self.INFERENCE_LEAD_ARG] = lead
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}

    @classmethod
    def from_spec_args(cls, **args: Any) -> Self:
        args[cls.INFERENCE_LEAD_ARG] = DurationEstimate.from_spec(args.get(cls.INFERENCE_LEAD_ARG, 0.0))
        return cls(**args)


class _TimedChunk:
    """A chunk whose action i is due at ``start_ns + i / fps``, executed in due order."""

    def __init__(self, actions: Sequence[Commands], start_ns: int, fps: float, now_ns: int):
        self._actions = actions
        self._start_ns = start_ns
        self._fps = fps
        self._next_index = self._running_index(now_ns)

    def _due_ns(self, index: int) -> int:
        return self._start_ns + round(index * 1e9 / self._fps)

    def _running_index(self, time_ns: int) -> int:
        """The index of the action that runs at ``time_ns``, or ``len(actions)`` after the chunk ends."""
        return next((i for i in range(len(self._actions)) if self._due_ns(i + 1) > time_ns), len(self._actions))

    def prefix(self, now_ns: int) -> dict[str, Any]:
        running = self._running_index(now_ns)
        return {
            policy_keys.RTC.PREFIX_ACTIONS: [dict(action) for action in self._actions[running:]],
            policy_keys.RTC.CHUNK_OFFSET_SEC: running / self._fps,
            policy_keys.RTC.OBSERVATION_OFFSET_SEC: (self._due_ns(running) - now_ns) / 1e9,
        }

    def take_due(self, now_ns: int) -> dict[str, Any]:
        """The merged commands of every action due by ``now_ns`` and not taken yet."""
        commands: dict[str, Any] = {}
        while self._next_index < len(self._actions) and self._due_ns(self._next_index) <= now_ns:
            commands.update(self._actions[self._next_index])
            self._next_index += 1
        return commands

    def next_due_ns(self) -> int | None:
        """When the next action not taken yet is due, or ``None`` after the last one."""
        return self._due_ns(self._next_index) if self._next_index < len(self._actions) else None


class RTCSchedule(Policy):
    """Execute planned commands while requesting the next chunk in the background.

    One handover at 10 fps, with ``call_after_sec=0.5`` and inference taking 0.3 seconds::

        step              0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
        observation T0    ^
        inference         |--------|
        answer                     ^
        old chunk                  o0 o1 o2 o3 o4 o5 o6 o7 o8 o9
        call_after_sec             |--------------|
        observation T1                            ^
        prefix                                    o5 o6 o7 o8 o9
        inference                                 |--------|
        answer                                             ^
        new chunk                                 n0 n1 n2 n3 n4 n5 n6 n7 n8 n9
        robot executes    -- -- -- o0 o1 o2 o3 o4 o5 o6 o7 n3 n4 n5 n6 n7 n8 n9

    A chunk holds one action per ``1 / fps`` seconds. The first chunk starts when its answer
    arrives: there is no preceding plan to execute while waiting. Later chunks start at their
    observation time, so action i is due at ``observation_time + i / fps``.

    ``call_after_sec`` after the chunk starts, the policy takes observation T1 and requests
    another prediction. The request includes the entire remaining plan, including the action
    currently running: o5 through o9 in the drawing. These are planned commands, with their
    original values and spacing. The server chooses which of them to use for conditioning.

    Two offsets locate the prefix. ``chunk_offset_sec`` is the first action's time within the
    preceding prediction: 0.5 seconds for o5. ``observation_offset_sec`` is that action's due
    time minus T1: zero in the drawing. If T1 falls 0.02 seconds after o5 became due, the prefix
    still starts with o5 and this offset is -0.02. The scheduler does not resample the plan.

    Measured latency is the duration of the preceding completed call; estimated latency comes
    from ``latency_estimate``. Both travel separately from the prefix, in seconds. Before the
    first answer, measured latency is None and the estimate is zero; the prefix is empty.

    Until the answer arrives, the policy executes the old chunk. On arrival, it switches to the
    new chunk at the action due now: n0 through n2 have elapsed, so it starts with n3. If the old
    chunk ends before the answer arrives, the policy emits no commands while waiting.

    Only one call runs at a time. If an answer arrives after the next request is already due,
    that request starts immediately. Composed codecs prepare the observation and planned commands
    for the model and decode its reply into the command sequence this scheduler executes.
    """

    WIRE_NAME = 'rtc_schedule'
    WIRE_VERSION = 1
    CALL_AFTER_ARG = 'call_after_sec'
    LATENCY_ESTIMATE_ARG = 'latency_estimate'

    def __init__(self, fps: float, call_after_sec: float, latency_estimate: DurationEstimate) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        if not isfinite(call_after_sec) or call_after_sec < 0:
            raise ValueError('call_after_sec must be finite and not negative')
        self._fps = fps
        self._call_after_ns = round(call_after_sec * 1e9)
        self._latency_estimate = latency_estimate

    def run(self, runtime: Runtime, infer: Callable[[Obs], Sequence[Commands]]) -> PolicyRun:
        duration = self._latency_estimate.run()
        expected_sec = 0.0
        measured_sec: float | None = None
        answer: Answer[Sequence[Commands]] | None = None
        called_at_ns = 0
        chunk: _TimedChunk | None = None
        next_call_ns = 0
        obs = yield
        try:
            while True:
                now_ns = runtime.time_ns
                if answer is not None and answer.done():
                    measured_sec = (now_ns - called_at_ns) / 1e9
                    expected_sec = duration.update(measured_sec)
                    # The first chunk counts from its answer: the robot did not move while it waited.
                    start_ns = now_ns if chunk is None else called_at_ns
                    chunk = _TimedChunk(answer.result(), start_ns, self._fps, now_ns)
                    answer = None
                    next_call_ns = start_ns + self._call_after_ns

                if answer is None and now_ns >= next_call_ns:
                    prefix = (
                        {
                            policy_keys.RTC.PREFIX_ACTIONS: [],
                            policy_keys.RTC.CHUNK_OFFSET_SEC: 0.0,
                            policy_keys.RTC.OBSERVATION_OFFSET_SEC: 0.0,
                        }
                        if chunk is None
                        else chunk.prefix(now_ns)
                    )
                    called_at_ns = now_ns
                    context = {
                        policy_keys.RTC.ACTION_PREFIX: prefix,
                        policy_keys.RTC.ESTIMATED_LATENCY_SEC: expected_sec,
                        policy_keys.RTC.MEASURED_LATENCY_SEC: measured_sec,
                    }
                    answer = runtime.submit(infer, ChainMap(context, obs))

                commands = {} if chunk is None else chunk.take_due(now_ns)
                next_due_ns = None if chunk is None else chunk.next_due_ns()
                if answer is not None:
                    # Pending inference asks for the earliest allowed poll; action cadence is independent.
                    resume_at_ns = now_ns
                elif next_due_ns is not None:
                    resume_at_ns = min(next_call_ns, next_due_ns)
                else:
                    resume_at_ns = next_call_ns
                obs = yield Step(commands, resume_at_ns)
        finally:
            if answer is not None:
                answer.cancel()

    def meta(self) -> dict[str, Any]:
        return {ACTION_FPS: self._fps}

    def to_spec(self) -> dict[str, Any]:
        args = {
            ChunkedSchedule.FPS_ARG: self._fps,
            self.CALL_AFTER_ARG: self._call_after_ns / 1e9,
            self.LATENCY_ESTIMATE_ARG: self._latency_estimate.to_spec(),
        }
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}

    @classmethod
    def from_spec_args(cls, **args: Any) -> Self:
        args[cls.LATENCY_ESTIMATE_ARG] = DurationEstimate.from_spec(args[cls.LATENCY_ESTIMATE_ARG])
        return cls(**args)


class _StackedObs(Mapping[str, Any]):
    """``obs`` with each buffered key replaced by its stack, built on the first read of that key."""

    def __init__(self, obs: Obs, picked: list[dict[str, np.ndarray]]):
        self._obs = obs
        self._picked = picked
        self._stacks: dict[str, np.ndarray] = {}

    def __getitem__(self, key: str) -> Any:
        if key not in self._picked[0]:
            return self._obs[key]
        if key not in self._stacks:
            self._stacks[key] = np.stack([entry[key] for entry in self._picked])
        return self._stacks[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._obs)

    def __len__(self) -> int:
        return len(self._obs)


class _StackBuffer:
    """Time-ordered history of ``(timestamp, values)`` entries, capped to the sampled window.

    ``values`` is a dict of key → array; every entry holds the same keys. ``append`` copies each new
    entry but skips one byte-identical to the previous — a source slower than the control loop repeats
    its value, and carry-over sampling reuses the stored one — then drops entries before the oldest
    sampled offset, keeping the one at or before it. ``sample`` replaces each key of an observation
    with a stack holding, for each offset, the latest value at or before that time — carry-over, never
    the future. Offsets that precede the first entry either repeat the oldest entry (``pad_start=True``,
    a fixed ``len(offsets_sec)``-long stack) or are dropped (``pad_start=False``, the stack grows from 1
    to ``len(offsets_sec)`` as history accumulates).
    """

    def __init__(self, offsets_sec: tuple[float, ...], pad_start: bool = True):
        self._offsets_sec = offsets_sec
        self._pad_start = pad_start
        self._entries: deque[tuple[float, dict[str, np.ndarray]]] = deque()

    def reset(self):
        self._entries.clear()

    def append(self, now: float, values: dict[str, np.ndarray]):
        if self._entries and all(np.array_equal(self._entries[-1][1][k], v) for k, v in values.items()):
            return
        self._entries.append((now, {k: np.array(v) for k, v in values.items()}))
        cutoff = now + min(self._offsets_sec)
        while len(self._entries) >= 2 and self._entries[1][0] <= cutoff:
            self._entries.popleft()

    def sample(self, now: float, obs: Obs) -> Obs:
        times = np.array([t for t, _ in self._entries])
        targets = [now + off for off in self._offsets_sec]
        if not self._pad_start:
            targets = [t for t in targets if t >= times[0]]
        return _StackedObs(dict(obs), [self._entries[self._at_or_before(times, t)][1] for t in targets])

    @staticmethod
    def _at_or_before(times: np.ndarray, target: float) -> int:
        """Index of the latest entry at or before ``target``; clamps to the oldest when none precedes it."""
        return max(int(np.searchsorted(times, target, side='right')) - 1, 0)


OutputT = TypeVar('OutputT')


class TemporalStack(Processor[Obs, OutputT, ReturnT]):
    """Replaces each named observation entry with a temporal stack of recent samples.

    Every sent observation records the selected channels on the runtime's clock, then passes the stacked
    observations to ``inner`` and yields its result. A stack is built when ``inner`` first reads its key, so
    a call that reads none builds none. Offsets are ascending seconds relative to now.
    Wrap a scheduling policy to collect frames on control ticks while inference is pending.

    With ``pad_start=True``, missing history repeats the oldest sample. Otherwise unavailable offsets
    are omitted, and the stack grows until the full window has been observed.
    """

    WIRE_NAME = 'temporal_stack'
    WIRE_VERSION = 2

    def __init__(self, keys: tuple[str, ...], offsets_sec: tuple[float, ...], pad_start: bool = True) -> None:
        self._keys = tuple(keys)
        self._offsets_sec = tuple(offsets_sec)
        self._pad_start = pad_start
        assert pad_start or 0.0 in self._offsets_sec, (
            'pad_start=False requires 0.0 in offsets_sec: with only past offsets the first observation has no '
            'in-range targets and the stack would be empty'
        )

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    @property
    def offsets_sec(self) -> tuple[float, ...]:
        return self._offsets_sec

    @property
    def pad_start(self) -> bool:
        return self._pad_start

    def run(self, runtime: Runtime, inner: ProcessorRun[Obs, OutputT, ReturnT]) -> ProcessorRun[Obs, OutputT, ReturnT]:
        buffer = _StackBuffer(self._offsets_sec, pad_start=self._pad_start)
        obs = yield
        while True:
            now_sec = runtime.time_ns / 1e9
            buffer.append(now_sec, {k: obs[k] for k in self._keys})
            try:
                output = inner.send(buffer.sample(now_sec, obs))
            except StopIteration as end:
                return end.value
            obs = yield output

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'keys': list(self._keys), 'offsets_sec': list(self._offsets_sec), 'pad_start': self._pad_start},
        }
