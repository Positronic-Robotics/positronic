"""Policy processors for scheduling, fault handling, and temporal frame stacking.

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

import logging
from abc import ABC, abstractmethod
from bisect import insort
from collections import deque
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
from positronic.drivers.roboarm.command import interpolate_commands
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import (
    Answer,
    Commands,
    Infer,
    Obs,
    Policy,
    PolicyRun,
    Processor,
    ProcessorRun,
    ReturnT,
    Runtime,
    Step,
)
from positronic.policy.keys import ACTION_FPS

logger = logging.getLogger(__name__)


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


class _Window:
    """The entries a ``_StackBuffer`` held at one tick, and the offsets it samples them at."""

    def __init__(self, buffer: '_StackBuffer', now: float):
        self.buffer = buffer
        self._times = np.array([t for t, _ in buffer._entries])
        self._values = [values for _, values in buffer._entries]
        self._offsets_sec = buffer._offsets_sec
        self._pad_start = buffer._pad_start
        self._now = now

    def pick(
        self, anchor: float | None = None, settled: float | None = None, *, settled_only: bool = False
    ) -> list[dict[str, np.ndarray]]:
        """For each offset, the latest entry at or before its time.

        An offset is settled when ``anchor`` plus the offset is at or before ``settled``: its time is ``anchor`` plus
        the offset. Every other offset is timed from now, or is left out with ``settled_only``.
        """
        targets = []
        for offset in self._offsets_sec:
            if anchor is not None and settled is not None and anchor + offset <= settled:
                targets.append(anchor + offset)
            elif not settled_only:
                targets.append(self._now + offset)
        if not self._pad_start:
            targets = [t for t in targets if t >= self._times[0]]
        return [self._values[self._at_or_before(t)] for t in targets]

    def _at_or_before(self, target: float) -> int:
        """Index of the latest entry at or before ``target``; clamps to the oldest when none precedes it."""
        return max(int(np.searchsorted(self._times, target, side='right')) - 1, 0)


class _StackedObs(Mapping[str, Any]):
    """``obs`` with each buffered key replaced by its stack, built on the first read of that key.

    ``settled`` and ``anchored`` sample the same window again, with the settled offsets timed from an anchor; see
    ``_Window.pick``.
    """

    def __init__(self, obs: Obs, picked: list[dict[str, np.ndarray]], window: _Window):
        self._obs = obs
        self._picked = picked
        self._window = window
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

    def settled(self, anchor: float, settled: float) -> '_StackedObs | None':
        """The observation with stacks of the settled offsets only, or ``None`` when no offset is settled.

        The buffer pins these entries until ``anchored`` or its retention bound expires.
        """
        picked = self._window.pick(anchor, settled, settled_only=True)
        if not picked:
            return None
        self._window.buffer.pin(anchor + min(self._window.buffer.offsets_sec))
        return _StackedObs(self._obs, picked, self._window)

    @property
    def has_pinned_history(self) -> bool:
        return self._window.buffer._pinned is not None

    def anchored(self, anchor: float, settled: float) -> '_StackedObs':
        """Sample retained history at the anchor for settled offsets and at now for the others."""
        assert self.has_pinned_history, 'Anchored sampling requires retained history'
        self._window.buffer.unpin()
        return _StackedObs(self._obs, self._window.pick(anchor, settled), self._window)


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

    # A pin that falls this far behind the oldest offset is dropped, so a stack that is not called for a long time
    # does not keep every entry since the pin.
    PIN_MAX_SEC = 1.0

    def __init__(self, offsets_sec: tuple[float, ...], pad_start: bool = True):
        self._offsets_sec = offsets_sec
        self._pad_start = pad_start
        self._entries: deque[tuple[float, dict[str, np.ndarray]]] = deque()
        self._pinned: float | None = None

    @property
    def offsets_sec(self) -> tuple[float, ...]:
        return self._offsets_sec

    def reset(self):
        self._entries.clear()
        self._pinned = None

    def pin(self, time: float):
        """Keep the entry at or before ``time``, and every later one, until ``unpin``."""
        self._pinned = time

    def unpin(self):
        self._pinned = None

    def append(self, now: float, values: dict[str, np.ndarray]):
        if self._entries and all(np.array_equal(self._entries[-1][1][k], v) for k, v in values.items()):
            return
        self._entries.append((now, {k: np.array(v) for k, v in values.items()}))
        cutoff = now + min(self._offsets_sec)
        if self._pinned is not None and cutoff - self._pinned > self.PIN_MAX_SEC:
            self._pinned = None
        if self._pinned is not None:
            cutoff = min(cutoff, self._pinned)
        while len(self._entries) >= 2 and self._entries[1][0] <= cutoff:
            self._entries.popleft()

    def sample(self, now: float, obs: Obs) -> _StackedObs:
        window = _Window(self, now)
        return _StackedObs(dict(obs), window.pick(), window)


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
    request the next one, bounded by the chunk's duration. The default is zero: request at the end.
    The first request starts immediately. Later deadlines use the delays of completed calls in this
    episode. An early answer waits behind the remaining commands; the new chunk starts from its first
    action. The observation is read at the request, before commands emitted in that same round.
    With ``record_stats``, each round that plans or emits a waypoint writes the waypoint counters into the
    episode metadata.

    ``replan_on_error=True`` clears the remaining plan on an arm's ``ERROR`` and makes inference due
    immediately. A pending call is kept and its answer starts playback on arrival. Further resets wait
    until a command is emitted. Set it to ``False`` to keep regular scheduling through errors.
    With ``context_lead_sec``, the schedule makes a context push ``context_lead_sec`` before the next inference
    call. It calls ``infer(context, context_only=True)``, which returns ``None``. ``context`` holds the frames
    of the next request that exist already: for each offset of the ``TemporalStack`` around this schedule whose
    time, counted from the next call, has passed. The next request samples these offsets at the same times, so
    it holds the same frames, and samples the other offsets from its own time, so its newest frames are fresh.
    The push needs a ``TemporalStack`` directly around this schedule, and an ``infer`` that takes
    ``context_only``, such as ``RemotePolicy``. Without a stack, or with no offset that has passed, there is no
    push and the request is as without ``context_lead_sec``.
    """

    WIRE_NAME = 'chunked_schedule'
    WIRE_VERSION = 3
    FPS_ARG = 'fps'
    HORIZON_SEC_ARG = 'horizon_sec'
    RECORD_STATS_ARG = 'record_stats'
    INFERENCE_LEAD_ARG = 'inference_lead'
    REPLAN_ON_ERROR_ARG = 'replan_on_error'
    CONTEXT_LEAD_SEC_ARG = 'context_lead_sec'

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
        context_lead_sec: float | None = None,
    ) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        if horizon_sec is not None and (not isfinite(horizon_sec) or horizon_sec <= 0):
            raise ValueError('horizon_sec must be finite and positive')
        if context_lead_sec is not None and (not isfinite(context_lead_sec) or context_lead_sec <= 0):
            raise ValueError('context_lead_sec must be finite and positive')
        self._fps = fps
        self._horizon_sec = horizon_sec
        self._record_stats = record_stats
        self._inference_lead = ConstantDuration() if inference_lead is None else inference_lead
        self._replan_on_error = replan_on_error
        self._context_lead_sec = context_lead_sec

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

    class _ContextPush:
        """One episode's upload, sampling anchor, and context-window deadline."""

        def __init__(self, runtime: Runtime, infer: Infer, lead_sec: float | None, obs: Obs):
            self._runtime = runtime
            self._infer = infer
            self._lead_ns = None if lead_sec is None else round(lead_sec * 1e9)
            if self._lead_ns is not None and not isinstance(obs, _StackedObs):
                logger.warning('ChunkedSchedule makes no context push: no TemporalStack is directly around it')
                self._lead_ns = None
            self.pending: Answer[Any] | None = None
            self.window_handled = True
            self._anchor_sec: float | None = None
            self._settled_sec = 0.0

        def recover(self, obs: Obs) -> None:
            self._anchor_sec = None
            self.window_handled = True
            if isinstance(obs, _StackedObs):
                obs._window.buffer.unpin()

        def update(self, obs: Obs, call_ns: int) -> None:
            if self.pending is not None and self.pending.done():
                pending, self.pending = self.pending, None
                pending.result()
            now_ns = self._runtime.time_ns
            if self._lead_ns is None or self.window_handled or not call_ns - self._lead_ns <= now_ns < call_ns:
                return
            self.window_handled = True
            anchor_sec, settled_sec = call_ns / 1e9, now_ns / 1e9
            context = obs.settled(anchor_sec, settled_sec) if isinstance(obs, _StackedObs) else None
            if context is not None:
                self.pending = self._runtime.submit(self._infer, context, context_only=True)
                self._anchor_sec, self._settled_sec = anchor_sec, settled_sec

        def observation(self, obs: Obs) -> Obs:
            if self._anchor_sec is not None and isinstance(obs, _StackedObs) and obs.has_pinned_history:
                obs = obs.anchored(self._anchor_sec, self._settled_sec)
            self._anchor_sec = None
            return obs

        def next_wake_ns(self, call_ns: int) -> int:
            now_ns = self._runtime.time_ns
            if self._lead_ns is not None and not self.window_handled and now_ns < call_ns:
                return max(now_ns, call_ns - self._lead_ns)
            return call_ns

    def run(self, runtime: Runtime, infer: Infer) -> PolicyRun:
        answer: Answer[Sequence[Commands]] | None = None
        recovering = False
        trajectory: deque[ChunkedSchedule._Waypoint] = deque()
        end_ns = 0
        call_ns = 0
        called_at_ns = 0
        lead = self._inference_lead.run()
        stats = self._Stats(runtime) if self._record_stats else None

        obs = yield
        context = self._ContextPush(runtime, infer, self._context_lead_sec, obs)
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
                    context.recover(obs)
                context.update(obs, call_ns)
                if answer is None and context.pending is None and now_ns >= call_ns:
                    answer, called_at_ns = runtime.submit(infer, context.observation(obs)), now_ns
                if answer is not None and answer.done():
                    chunk, answer = answer.result(), None
                    lead_sec = lead.update((now_ns - called_at_ns) / 1e9)
                    start_ns = max(now_ns, end_ns)
                    waypoints, end_ns = self._plan_chunk(chunk, start_ns)
                    call_ns = self._next_call_ns(start_ns, end_ns, lead_sec) if waypoints else now_ns
                    # A regular early answer waits behind the commands that are still due.
                    trajectory = deque(waypoints) if start_ns == now_ns else deque([*trajectory, *waypoints])
                    planned = len(waypoints)
                    context.window_handled = False

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
                resume_at_ns = context.next_wake_ns(call_ns)
                if trajectory:
                    resume_at_ns = min(resume_at_ns, trajectory[0].due_ns)
                # Pending inference asks for the earliest allowed poll; action cadence is independent.
                obs = yield Step(commands, now_ns if answer is not None else resume_at_ns)
        finally:
            if answer is not None:
                answer.cancel()
            if context.pending is not None:
                context.pending.cancel()

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
        if self._context_lead_sec is not None:
            args[self.CONTEXT_LEAD_SEC_ARG] = self._context_lead_sec
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}

    @classmethod
    def from_spec_args(cls, **args: Any) -> Self:
        args[cls.INFERENCE_LEAD_ARG] = DurationEstimate.from_spec(args.get(cls.INFERENCE_LEAD_ARG, 0.0))
        return cls(**args)


class PrefixSampling(Enum):
    """How `RTCSchedule` reads an old action for a time of the new chunk.

    The new chunk is timed from its call, so its times can fall between two old due times.
    At 0.52 s, with old actions o5 due at 0.5 s and o6 due at 0.6 s:

    * `PREVIOUS` gives o5, the action that the robot executes at that time.
    * `NEAREST` gives o5, the action with the closest due time. At 0.57 s it gives o6.
      A tie, at 0.55 s, gives the earlier one.
    * `NEXT` gives o6, the first action due at or after that time.
    * `INTERPOLATE` gives the point 20% of the way from o5 to o6: see `interpolate_commands`.
    """

    PREVIOUS = 'previous'
    NEAREST = 'nearest'
    NEXT = 'next'
    INTERPOLATE = 'interpolate'


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

    def prefix(self, now_ns: int, duration_sec: float, sampling: PrefixSampling) -> list[Commands]:
        """This chunk sampled at ``now_ns + k / fps`` for each k that falls before ``now_ns + duration_sec``.

        Stops at the first time that ``sampling`` gives no action for.
        """
        end_ns = now_ns + round(duration_sec * 1e9)
        prefix: list[Commands] = []
        while (slot_ns := now_ns + round(len(prefix) * 1e9 / self._fps)) < end_ns:
            action = self._sample(slot_ns, sampling)
            if action is None:
                break
            prefix.append(action)
        return prefix

    def _sample(self, time_ns: int, sampling: PrefixSampling) -> Commands | None:
        """The action ``sampling`` gives at ``time_ns``, or ``None`` after the chunk ends."""
        running = self._running_index(time_ns)
        last = len(self._actions) - 1
        if running > last:
            return None
        if sampling is PrefixSampling.PREVIOUS or (running == last and sampling is not PrefixSampling.NEXT):
            return self._actions[running]
        if sampling is PrefixSampling.NEXT:
            following = running if self._due_ns(running) == time_ns else running + 1
            return self._actions[following] if following <= last else None
        since_ns, until_ns = time_ns - self._due_ns(running), self._due_ns(running + 1) - time_ns
        if sampling is PrefixSampling.NEAREST:
            return self._actions[running + 1 if until_ns < since_ns else running]
        fraction = since_ns / (since_ns + until_ns)
        return interpolate_commands(self._actions[running], self._actions[running + 1], fraction)

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
    """Run action chunks from a model, and ask for the next chunk while the current one runs.

        step              0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
        observation T0    ^
        inference         |--------|
        answer                     ^
        old chunk                  o0 o1 o2 o3 o4 o5 o6 o7 o8 o9
        call_after_sec             |--------------|
        observation T1                            ^
        prefix                                    o5 o6 o7
        inference                                 |--------|
        answer                                             ^
        new chunk                                 n0 n1 n2 n3 n4 n5 n6 n7 n8 n9
        robot executes    -- -- -- o0 o1 o2 o3 o4 o5 o6 o7 n3 n4 n5 n6 n7 n8 n9

    A chunk holds one action per 1 / `fps` seconds. Its action i is due at the time of
    its observation + i / `fps`. The policy executes each action when its due time comes.

    `call_after_sec` after T0, the policy takes the observation T1 and calls `infer`.
    The call also gets a prefix: the old chunk read at the new chunk's due times T1,
    T1 + 1 / `fps`, ..., for the duration estimated by `prefix_duration`, and not past
    the end of the old chunk. The prefix tells the model what the robot does while the model
    computes. Its length is the delay estimate: the server reads it as the number of
    prefix actions / `fps`. The server decides how to use the prefix.

    When T1 falls between two old due times, each new due time falls between two old
    actions. `prefix_sampling` decides which value the prefix holds there; see
    `PrefixSampling`. On the grid, as in the drawing, every choice gives the same prefix.

    Until the answer arrives, the policy executes the old chunk. When the answer arrives,
    the policy executes the new chunk, from the action that is due now. If the old chunk
    ends before the answer arrives, the policy executes no action until the answer arrives.

    Only one call runs at a time. If an answer arrives more than `call_after_sec` after
    its observation, the next call starts at once.

    The first call has no old chunk, so its prefix is empty, and the robot executes no
    action until its answer arrives (-- in the drawing).

    `infer` gets the observation and the prefix, in the command format that `infer`
    returns, and converts both to the model's format.

    Only a model with absolute actions works with this policy. No codec converts a
    command back to a relative model action, so the prefix cannot be given to a
    model with relative actions.
    """

    def __init__(
        self,
        fps: float,
        call_after_sec: float,
        prefix_duration: DurationEstimate,
        prefix_sampling: PrefixSampling = PrefixSampling.PREVIOUS,
    ) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        if not isfinite(call_after_sec) or call_after_sec < 0:
            raise ValueError('call_after_sec must be finite and not negative')
        self._fps = fps
        self._call_after_ns = round(call_after_sec * 1e9)
        self._prefix_duration = prefix_duration
        self._prefix_sampling = prefix_sampling

    def run(self, runtime: Runtime, infer: Callable[[Obs, Sequence[Commands]], Sequence[Commands]]) -> PolicyRun:
        duration = self._prefix_duration.run()
        prefix_sec = 0.0
        answer: Answer[Sequence[Commands]] | None = None
        called_at_ns = 0
        chunk: _TimedChunk | None = None
        next_call_ns = 0
        obs = yield
        try:
            while True:
                now_ns = runtime.time_ns
                if answer is not None and answer.done():
                    prefix_sec = duration.update((now_ns - called_at_ns) / 1e9)
                    # The first chunk counts from its answer: the robot did not move while it waited.
                    start_ns = now_ns if chunk is None else called_at_ns
                    chunk = _TimedChunk(answer.result(), start_ns, self._fps, now_ns)
                    answer = None
                    next_call_ns = start_ns + self._call_after_ns

                if answer is None and now_ns >= next_call_ns:
                    prefix = []
                    if chunk is not None:
                        prefix = chunk.prefix(now_ns, prefix_sec, self._prefix_sampling)
                    called_at_ns = now_ns
                    answer = runtime.submit(infer, obs, prefix)

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
