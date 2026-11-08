"""Policy processors for scheduling, fault handling, ending a run, and temporal frame stacking.

Constructors hold configuration. ``run(runtime, *dependencies)`` creates an episode generator whose
locals hold its state. Control processors yield a ``Step`` with commands and the next wake-up time.

Describe a local stack without creating episode state::

    from positronic.policy.sequential import Sequential

    definition = Sequential(
        TemporalStack(keys=('image',), offsets_sec=(-0.2, -0.1, 0.0)),
        TrajectoryPlayer(replan_on_error=True),
        ChunkedSchedule(fps=20),
    )
    episode = runtime.start(definition, infer)
    step = episode.send(obs)
"""

from abc import ABC, abstractmethod
from bisect import bisect_right, insort
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from math import inf
from statistics import fmean
from sys import maxsize
from typing import Any, Self, TypeVar

import numpy as np
from attr import asdict, dataclass
from positronic_model_server.serialization import check_int_range
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
    QueueFull,
    ReturnT,
    Runtime,
    Step,
)
from positronic.policy.keys import ACTION_FPS, END
from positronic.utils import check_finite_nonnegative, check_finite_positive
from positronic.utils.lazy import LazyDict, OverlayMapping


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
                yield Step(commands, runtime.time_ns, outputs=step.outputs)
                return end
            obs = yield step

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION}


class DurationEstimateRun(ABC):
    """One episode's inference-delay history and duration estimates."""

    @abstractmethod
    def update(self, delay_sec: float) -> float:
        """Record a completed call's elapsed duration and return the updated estimate."""


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
        check_finite_nonnegative(seconds, 'seconds')
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
        check_int_range(last, 1, maxsize, 'last')
        check_finite_nonnegative(max_sec, 'max_sec')
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


@dataclass(frozen=True)
class Waypoint:
    """Commands held from ``due_ns`` until the next waypoint."""

    commands: Commands
    due_ns: int


@dataclass
class Trajectory:
    """Commands spaced by ``1 / fps`` from ``start_ns``, with an effective scheduling end.

    ``end_ns`` can shorten the final command's period: at 10 Hz, commands at 0 and 100 ms
    can cover a 150 ms horizon, rather than the 200 ms implied by their count and rate.
    The robot keeps holding its last issued command after this scheduling boundary.
    """

    commands: Sequence[Commands]
    start_ns: int
    fps: float
    end_ns: int

    @property
    def period_ns(self) -> float:
        """Time between commands in nanoseconds."""
        return (1 / self.fps) * 1e9

    def due_ns(self, index: int) -> int:
        """Scheduled time of the command at ``index``."""
        return self.start_ns + round(index * self.period_ns)

    def waypoints(self) -> list[Waypoint]:
        """Commands paired with their scheduled times, without copying command values."""
        return [Waypoint(command, self.due_ns(i)) for i, command in enumerate(self.commands)]

    def __attrs_post_init__(self) -> None:
        check_finite_positive(self.fps, 'fps')
        if not self.start_ns <= self.end_ns <= self.start_ns + round(len(self.commands) / self.fps * 1e9):
            raise ValueError('end_ns must lie within the trajectory duration')


class TrajectoryPlayer(Policy):
    """Play an inner run's timed trajectories independently of its inference cadence.

    The inner run yields a ``Step`` whose command dictionary contains the fields of ``Trajectory``.
    The player converts it to a ``Trajectory`` and emits due commands. It advances the inner run
    at its requested deadline or when any runtime answer is ready. A waypoint deadline alone
    does not advance the inner run.

    A new trajectory replaces the plan from its start time, skipping elapsed actions.
    Empty step commands keep the plan; a trajectory with no commands removes its future
    from that start time. Auxiliary outputs pass through unchanged.

    Args:
        plan_key: Observation field containing the remaining plan as a list of ``Waypoint`` values.
            It includes the command held now, followed by future commands. The last emitted
            command stays in context until replaced or recovery clears it. ``None`` omits this context.
        record_stats: Record scheduled, emitted and dropped waypoint counts, command lateness,
            and gaps between emissions in episode metadata.
        replan_on_error: Clear the plan on an arm's ``ERROR`` status and advance the inner run
            immediately with ``replan_now=True``. Pending calls from the inner run are not cancelled.
            Further resets are suppressed until commands are emitted; ``False`` leaves playback unchanged by arm errors.
    """

    WIRE_NAME = 'trajectory_player'
    WIRE_VERSION = 1

    def __init__(
        self, *, plan_key: str | None = None, record_stats: bool = True, replan_on_error: bool = False
    ) -> None:
        self._plan_key = plan_key
        self._record_stats = record_stats
        self._replan_on_error = replan_on_error

    @dataclass(frozen=True)
    class _Waypoint(Waypoint):
        """A queued waypoint; ``starts_chunk`` resets gap statistics on emission."""

        starts_chunk: bool

    @classmethod
    def _update_plan(cls, plan: deque[_Waypoint], update: Trajectory, now_ns: int) -> int:
        incoming = [
            cls._Waypoint(command, update.due_ns(i), starts_chunk=i == 0) for i, command in enumerate(update.commands)
        ]
        planned = len(incoming)
        if update.start_ns <= now_ns:
            plan.clear()
            running = max(0, bisect_right(incoming, now_ns, key=lambda point: point.due_ns) - 1)
            incoming = incoming[running:]
            if incoming:
                first = incoming[0]
                incoming[0] = cls._Waypoint(first.commands, first.due_ns, starts_chunk=True)
        while plan and plan[-1].due_ns >= update.start_ns:
            plan.pop()
        plan.extend(incoming)
        return planned

    @staticmethod
    def _remaining_plan(plan: Sequence[Waypoint], current: Waypoint | None, now_ns: int) -> list[Waypoint]:
        running = bisect_right(plan, now_ns, key=lambda point: point.due_ns) - 1
        if running >= 0:
            remaining = plan[running:]
        else:
            remaining = [current, *plan] if current is not None else plan
        return [Waypoint(point.commands, point.due_ns) for point in remaining]

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
            self._round_due_ns: int | None = None

        def record_waypoint(self, point: 'TrajectoryPlayer._Waypoint') -> None:
            self._round_due_ns = point.due_ns
            if point.starts_chunk:
                self._last_emit_ns = None

        def count_round(self, planned: int | None, now_ns: int, queued: int) -> None:
            """Count new waypoints and the due commands emitted together at ``now_ns``."""
            if planned is None and self._round_due_ns is None:
                return
            if planned is not None:
                self._scheduled += planned
            if self._round_due_ns is not None:
                insort(self._sorted_late_ns, now_ns - self._round_due_ns)
                if self._last_emit_ns is not None:
                    self._gap_max_ns = max(self._gap_max_ns, now_ns - self._last_emit_ns)
                self._last_emit_ns = now_ns
                self._round_due_ns = None
            self._write(queued)

        def _write(self, queued: int) -> None:
            values: dict[str, float] = {
                eval_keys.SCHEDULED: self._scheduled,
                eval_keys.EMITTED: len(self._sorted_late_ns),
                eval_keys.DROPPED: self._scheduled - len(self._sorted_late_ns) - queued,
            }
            if self._sorted_late_ns:
                values[eval_keys.LATE_P50_MS] = self._percentile_of_sorted(self._sorted_late_ns, 0.5) / 1e6
                values[eval_keys.LATE_P90_MS] = self._percentile_of_sorted(self._sorted_late_ns, 0.9) / 1e6
                values[eval_keys.LATE_MAX_MS] = self._sorted_late_ns[-1] / 1e6
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

    def _need_recover(self, obs: Obs) -> bool:
        # TODO(#638): Read status from domain values when the harness preserves their types.
        return self._replan_on_error and any(
            RobotStatus(obs[name]) is RobotStatus.ERROR
            for name in obs
            if name.startswith(f'{keys.ROBOT_STATE}.') and name.endswith(keys.STATUS_SUFFIX)
        )

    def _with_plan(self, obs: Obs, plan: deque[_Waypoint], current: _Waypoint | None, now_ns: int) -> Obs:
        if self._plan_key is None:
            return obs
        return OverlayMapping(
            obs, add=LazyDict({}, {self._plan_key: partial(self._remaining_plan, tuple(plan), current, now_ns)})
        )

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        plan: deque[TrajectoryPlayer._Waypoint] = deque()
        current_wp: TrajectoryPlayer._Waypoint | None = None
        inner_resume_at_ns = 0
        recovering = False
        stats = self._Stats(runtime) if self._record_stats else None
        obs = yield
        while True:
            now_ns = runtime.time_ns
            planned = None
            if not recovering and self._need_recover(obs):
                plan.clear()
                current_wp = None
                inner_resume_at_ns = now_ns
                planned, recovering = 0, True
                obs = OverlayMapping(obs, add={policy_keys.REPLAN_NOW: True})

            outputs: Mapping[str, Any] = {}
            if now_ns >= inner_resume_at_ns or obs.get(policy_keys.REPLAN_NOW, False) or runtime.has_ready_answers():
                try:
                    step = inner.send(self._with_plan(obs, plan, current_wp, now_ns))
                except StopIteration as end:
                    return end.value
                if step is None:
                    raise TypeError('TrajectoryPlayer requires its inner run to yield a Step')
                inner_resume_at_ns, outputs = step.resume_at_ns, step.outputs
                if step.commands:
                    update = Trajectory(**step.commands)
                    planned = self._update_plan(plan, update, now_ns)

            commands: dict[str, Any] = {}
            while plan and plan[0].due_ns <= now_ns:
                current_wp = plan.popleft()
                commands.update(current_wp.commands)
                if stats is not None:
                    stats.record_waypoint(current_wp)
            recovering = recovering and not commands
            if stats is not None:
                stats.count_round(planned, now_ns, queued=len(plan))
            resume_at_ns = min(inner_resume_at_ns, plan[0].due_ns) if plan else inner_resume_at_ns
            obs = yield Step(commands, resume_at_ns, outputs=outputs)

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'plan_key': self._plan_key,
                'record_stats': self._record_stats,
                'replan_on_error': self._replan_on_error,
            },
        }


class Periodic(Policy):
    """Sample observations on a fixed grid while allowing answers to be handled between samples.

    ``fps`` sets the sampling rate, independently of action playback. Each sample copies the latest
    observation and adds ``SAMPLE_INDEX`` and its actual sampling time under ``obs_time_key`` (nanoseconds).
    Late ticks skip grid positions; unchanged camera frames are still sampled. Answer wakeups reuse
    the last sample. Inner request deadlines are serviced on sampling ticks; ``REPLAN_NOW`` is
    forwarded immediately.
    """

    WIRE_NAME = 'periodic'
    WIRE_VERSION = 1

    def __init__(self, fps: float, *, obs_time_key: str = policy_keys.OBS_TIME_NS) -> None:
        check_finite_positive(fps, 'fps')
        if fps > 1e9:
            raise ValueError('sampling fps must be at most 1 GHz')
        self._fps = fps
        self._obs_time_key = obs_time_key

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        origin_ns = next_sample_ns = runtime.time_ns
        period_ns = (1 / self._fps) * 1e9
        next_index = 0
        sample: Obs = {}
        while True:
            now_ns = runtime.time_ns
            sampling = now_ns >= next_sample_ns
            if sampling:
                index = max(next_index, int((now_ns - origin_ns) / period_ns))
                sample = {
                    **deepcopy(dict(OverlayMapping(obs, remove=(policy_keys.REPLAN_NOW,)))),
                    policy_keys.SAMPLE_INDEX: index,
                    self._obs_time_key: now_ns,
                }
                next_index = index + 1
                next_sample_ns = origin_ns + round(next_index * period_ns)
            step = Step({}, next_sample_ns)
            if sampling or runtime.has_ready_answers() or obs.get(policy_keys.REPLAN_NOW, False):
                inner_obs = sample
                if policy_keys.REPLAN_NOW in obs:
                    inner_obs = OverlayMapping(sample, add={policy_keys.REPLAN_NOW: obs[policy_keys.REPLAN_NOW]})
                try:
                    result = inner.send(inner_obs)
                except StopIteration as end:
                    return end.value
                if result is None:
                    raise TypeError('Periodic requires its inner run to yield a Step')
                resume_at_ns = (
                    min(next_sample_ns, result.resume_at_ns) if runtime.has_ready_answers() else next_sample_ns
                )
                step = Step(result.commands, resume_at_ns, outputs=result.outputs)
            obs = yield step

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'fps': self._fps, 'obs_time_key': self._obs_time_key},
        }


class RequestTiming:
    """Choose when to request inference: periodically, after a trajectory starts, or before it ends.

    ``every(seconds)`` keeps a fixed rhythm from the first submission. ``after_start(seconds)`` uses
    the returned trajectory's start; ``before_end()`` uses its end minus the estimated inference time.
    """

    class Reference(Enum):
        BEFORE_END = 'before_end'
        AFTER_START = 'after_start'
        EVERY = 'every'

    def __init__(self, reference: Reference, seconds: float = 0.0) -> None:
        check_finite_nonnegative(seconds, 'seconds')
        self._reference = reference
        self.duration_ns = seconds * 1e9
        if reference is self.Reference.EVERY and self.duration_ns < 1:
            raise ValueError('request interval must be at least one nanosecond')

    @classmethod
    def before_end(cls) -> Self:
        """Request before the trajectory ends, using the estimated duration as the lead."""
        return cls(cls.Reference.BEFORE_END)

    @classmethod
    def after_start(cls, seconds: float) -> Self:
        """Request a fixed interval after the trajectory starts."""
        return cls(cls.Reference.AFTER_START, seconds)

    @classmethod
    def every(cls, seconds: float) -> Self:
        """Request on a fixed grid from the first submission, skipping missed slots."""
        return cls(cls.Reference.EVERY, seconds)

    class _Run:
        def __init__(self, strategy: 'RequestTiming') -> None:
            self._strategy = strategy
            self._first_call_ns: int | None = None
            self._last_obs_ns: int | None = None
            self.next_call_ns: int | None = 0

        def is_due(self, obs_time_ns: int) -> bool:
            inference_due = self.next_call_ns is not None and obs_time_ns >= self.next_call_ns
            new_obs = self._last_obs_ns is None or obs_time_ns > self._last_obs_ns
            return inference_due and new_obs

        def update_after_attempt(self, obs_time_ns: int | None, now_ns: int, *, pending: bool) -> None:
            """Advance timing after a submission attempt, including a skipped sample."""
            self._last_obs_ns = obs_time_ns
            self.next_call_ns = None if pending else now_ns
            if self._strategy._reference is not RequestTiming.Reference.EVERY:
                return
            sample_time_ns = now_ns if obs_time_ns is None else obs_time_ns
            if self._first_call_ns is None:
                self._first_call_ns = sample_time_ns
            periods = int((sample_time_ns - self._first_call_ns) / self._strategy.duration_ns) + 1
            self.next_call_ns = self._first_call_ns + round(periods * self._strategy.duration_ns)
            if self.next_call_ns <= sample_time_ns:
                self.next_call_ns = self._first_call_ns + round((periods + 1) * self._strategy.duration_ns)

        def update_after_answer(self, trajectory: Trajectory | None, estimated_latency_sec: float, now_ns: int) -> None:
            if self.next_call_ns is not None:
                return
            self.next_call_ns = now_ns
            if trajectory is None:
                return
            if self._strategy._reference is RequestTiming.Reference.AFTER_START:
                self.next_call_ns = trajectory.start_ns + round(self._strategy.duration_ns)
            elif self._strategy._reference is RequestTiming.Reference.BEFORE_END and trajectory.commands:
                lead_ns = min(estimated_latency_sec * 1e9, trajectory.end_ns - trajectory.start_ns)
                self.next_call_ns = trajectory.end_ns - round(lead_ns)

        def request_now(self, now_ns: int) -> None:
            if self.next_call_ns is not None:
                self.next_call_ns = now_ns

    def run(self) -> _Run:
        """Create timing for one episode, anchoring periodic calls at the first submission."""
        return self._Run(self)

    def to_spec(self) -> dict[str, Any]:
        return {'reference': self._reference.value, 'seconds': self.duration_ns / 1e9}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Self:
        return cls(cls.Reference(spec['reference']), spec.get('seconds', 0.0))


class TrajectoryMerge:
    """Place predictions on the playback timeline and limit their duration."""

    class Start(Enum):
        AFTER_PREVIOUS = 'after_previous'
        OBSERVATION = 'observation'

    def __init__(self, start: Start, horizon_sec: float | None = None) -> None:
        if horizon_sec is not None:
            check_finite_positive(horizon_sec, 'horizon_sec')
        self._start = start
        self.horizon_sec = horizon_sec

    @classmethod
    def after_previous(cls, horizon_sec: float | None = None) -> Self:
        """Start after the preceding trajectory, or now if its end has passed."""
        return cls(cls.Start.AFTER_PREVIOUS, horizon_sec)

    @classmethod
    def at_observation(cls) -> Self:
        """Start at the request's observation time, except the first prediction starts now."""
        return cls(cls.Start.OBSERVATION)

    def to_spec(self) -> dict[str, Any]:
        return {'start': self._start.value, 'horizon_sec': self.horizon_sec}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Self:
        return cls(cls.Start(spec['start']), spec.get('horizon_sec'))

    def plan(
        self, chunk: Sequence[Commands], fps: float, previous: Trajectory | None, now_ns: int, obs_time_ns: int
    ) -> Trajectory:
        if previous is None:
            start_ns = now_ns
        elif self._start is self.Start.OBSERVATION:
            start_ns = obs_time_ns
        else:
            start_ns = max(now_ns, previous.end_ns)
        duration_sec = min(len(chunk) / fps, self.horizon_sec or inf)
        trajectory = Trajectory(chunk, start_ns, fps, start_ns + round(duration_sec * 1e9))
        if self.horizon_sec is not None:
            trajectory.commands = [
                command for i, command in enumerate(chunk) if trajectory.due_ns(i) < trajectory.end_ns
            ]
        return trajectory


class InferenceSchedule(Policy):
    """Schedule inference calls and publish completed predictions as trajectories.

    Optional observation fields ``PLAN`` and ``REPLAN_NOW`` provide playback context and request
    replanning. ``PLAN`` becomes RTC context; ``REPLAN_NOW`` remains available to inference codecs.
    ``RequestTiming.every(seconds)`` queues calls at a fixed interval. Other timing strategies keep at
    most one call outstanding. Answers are consumed in submission order, one per resumption.
    ``obs_time_key`` selects the observation timestamp in nanoseconds for request eligibility and
    trajectory placement; without that field, the current runtime time is used.
    Runtime capacity bounds outstanding calls. Samples at capacity are skipped and counted in
    episode metadata; recovery intent is retained until a call can carry it.
    Replanning ignores predictions from earlier calls while letting those calls finish.
    Each accepted call carries a per-run ``call_id`` and unreported completions in ``call_timings``.
    """

    WIRE_NAME = 'inference_schedule'
    WIRE_VERSION = 1

    def __init__(
        self,
        fps: float,
        *,
        request_timing: RequestTiming,
        merge: TrajectoryMerge,
        latency_estimate: DurationEstimate,
        obs_time_key: str = policy_keys.OBS_TIME_NS,
    ) -> None:
        check_finite_positive(fps, 'fps')
        self._fps = fps
        self._request_timing = request_timing
        self._merge = merge
        self._latency_estimate = latency_estimate
        self._obs_time_key = obs_time_key

    def _call_obs(
        self, obs: Obs, trajectory: Trajectory | None, obs_time_ns: int, *, update: Trajectory | None, replan: bool
    ) -> Obs:
        if replan:
            obs = OverlayMapping(obs, add={policy_keys.REPLAN_NOW: True})
        if policy_keys.PLAN not in obs:
            return obs
        plan = update.waypoints() if update is not None else obs[policy_keys.PLAN]
        start_ns = obs_time_ns if trajectory is None else trajectory.start_ns
        end_ns = obs_time_ns if trajectory is None else trajectory.end_ns
        running = max(0, bisect_right(plan, obs_time_ns, key=lambda point: point.due_ns) - 1)
        remaining = plan[running:] if obs_time_ns < end_ns else ()
        due_ns = remaining[0].due_ns if remaining else end_ns
        prefix = {
            policy_keys.RTC.PREFIX_ACTIONS: [dict(point.commands) for point in remaining],
            policy_keys.RTC.CHUNK_OFFSET_SEC: round((due_ns - start_ns) * self._fps / 1e9) / self._fps,
            policy_keys.RTC.OBSERVATION_OFFSET_SEC: (due_ns - obs_time_ns) / 1e9,
        }
        return OverlayMapping(obs, add={policy_keys.RTC.ACTION_PREFIX: prefix}, remove=(policy_keys.PLAN,))

    class _CallQueue:
        """Consume calls in submission order and carry completed timings into accepted submissions."""

        @dataclass(frozen=True)
        class _PendingCall:
            answer: Answer[Sequence[Commands] | None]
            observation_time_ns: int
            call_id: int

        def __init__(self, runtime: Runtime):
            self._runtime = runtime
            self._pending: deque[InferenceSchedule._CallQueue._PendingCall] = deque()
            self._timings: list[dict[str, int | float]] = []
            self.next_call_id = 0

        def __bool__(self) -> bool:
            return bool(self._pending)

        def submit(self, infer: Callable[[Obs], Sequence[Commands] | None], obs: Obs, observation_time_ns: int) -> None:
            obs = OverlayMapping(
                obs, add={policy_keys.CALL_ID: self.next_call_id, policy_keys.CALL_TIMINGS: self._timings}
            )
            self._pending.append(
                self._PendingCall(
                    answer=self._runtime.submit(infer, obs),
                    observation_time_ns=observation_time_ns,
                    call_id=self.next_call_id,
                )
            )
            self.next_call_id += 1
            self._timings = []

        def pop_ready(self) -> _PendingCall | None:
            if not self._pending or not self._pending[0].answer.done():
                return None
            call = self._pending.popleft()
            self._timings.append({policy_keys.CALL_ID: call.call_id, policy_keys.ELAPSED_SEC: call.answer.elapsed_sec})
            return call

        def close(self) -> None:
            for call in self._pending:
                call.answer.cancel()

    def run(self, runtime: Runtime, infer: Callable[[Obs], Sequence[Commands] | None]) -> PolicyRun:
        duration = self._latency_estimate.run()
        timing = self._request_timing.run()
        calls = self._CallQueue(runtime)
        first_valid_call_id = 0
        previous: Trajectory | None = None
        skipped = 0
        replan = False
        obs = yield
        try:
            while True:
                now_ns = runtime.time_ns
                update = None
                if obs.get(policy_keys.REPLAN_NOW, False):
                    first_valid_call_id = calls.next_call_id
                    previous = None
                    timing.request_now(now_ns)
                    replan = True
                sample_time_ns: int = obs.get(self._obs_time_key, now_ns)

                if (call := calls.pop_ready()) is not None:
                    estimated_latency_sec = duration.update(call.answer.elapsed_sec)
                    check_finite_nonnegative(estimated_latency_sec, 'duration estimate')
                    chunk = call.answer.result()
                    if chunk is not None and call.call_id >= first_valid_call_id:
                        update = self._merge.plan(chunk, self._fps, previous, now_ns, call.observation_time_ns)
                        previous = update
                    timing.update_after_answer(update, estimated_latency_sec, now_ns)

                if timing.is_due(sample_time_ns):
                    try:
                        call_obs = self._call_obs(obs, previous, sample_time_ns, update=update, replan=replan)
                        calls.submit(infer, call_obs, sample_time_ns)
                        replan = False
                    except QueueFull:
                        skipped += 1
                        runtime.set_metadata(policy_keys.SKIPPED_SAMPLES, skipped)
                    timing.update_after_attempt(obs.get(self._obs_time_key), now_ns, pending=bool(calls))
                if update is not None and now_ns >= update.end_ns:
                    update = Trajectory([], update.start_ns, update.fps, update.start_ns)
                commands = {} if update is None else asdict(update, recurse=False)
                resume_at_ns = timing.next_call_ns
                if calls or resume_at_ns is None:
                    resume_at_ns = now_ns
                obs = yield Step(commands, resume_at_ns)
        finally:
            calls.close()

    def meta(self) -> dict[str, Any]:
        meta = {ACTION_FPS: self._fps}
        if self._merge.horizon_sec is not None:
            meta[policy_keys.ACTION_HORIZON_SEC] = self._merge.horizon_sec
        return meta

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {
                'fps': self._fps,
                'request_timing': self._request_timing.to_spec(),
                'merge': self._merge.to_spec(),
                'latency_estimate': self._latency_estimate.to_spec(),
                'obs_time_key': self._obs_time_key,
            },
        }

    @classmethod
    def from_spec_args(cls, **args: Any) -> Self:
        args['request_timing'] = RequestTiming.from_spec(args['request_timing'])
        args['merge'] = TrajectoryMerge.from_spec(args['merge'])
        args['latency_estimate'] = DurationEstimate.from_spec(args['latency_estimate'])
        return cls(**args)


class ChunkedSchedule(InferenceSchedule):
    """Request action chunks asynchronously and return timed trajectories for playback.

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

    ``infer`` returns command sets or ``None`` to leave playback unchanged, and must not mutate episode state. The first
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

    Inference observations include completed call timings; the lead estimate stays local.

    The optional observation argument ``replan_now`` defaults to ``False``. When true, inference is due
    immediately and its answer can start immediately. An existing pending call is kept. The argument
    remains available to inference codecs.
    """

    FPS_ARG = 'fps'
    HORIZON_SEC_ARG = 'horizon_sec'
    INFERENCE_LEAD_ARG = 'inference_lead'

    def __init__(
        self, fps: float, horizon_sec: float | None = None, inference_lead: DurationEstimate | None = None
    ) -> None:
        super().__init__(
            fps,
            request_timing=RequestTiming.before_end(),
            merge=TrajectoryMerge.after_previous(horizon_sec),
            latency_estimate=ConstantDuration() if inference_lead is None else inference_lead,
        )


class RTCSchedule(InferenceSchedule):
    """Request overlapping trajectories for an outer ``TrajectoryPlayer`` to execute.

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

    Completed call timings travel separately from the prefix, in seconds. The server estimates
    how long to freeze actions. Before the first answer, the timings and prefix are empty.

    Until the answer arrives, the player executes the old chunk. On arrival, it switches to the
    new chunk at the action due now: n0 through n2 have elapsed, so it starts with n3. If the old
    chunk ends before the answer arrives, playback emits no commands while waiting.

    Only one call runs at a time. If an answer arrives after the next request is already due,
    that request starts immediately. Composed codecs prepare the observation and planned commands
    for the model and decode its reply into commands. ``None`` leaves the trajectory unchanged.

    When the observation contains ``PLAN``, the scheduler consumes it to prepare the model prefix,
    excluding the held command once its prediction has expired. Without ``PLAN``, no prefix is added.
    """

    CALL_AFTER_ARG = 'call_after_sec'
    LATENCY_ESTIMATE_ARG = 'latency_estimate'

    def __init__(self, fps: float, call_after_sec: float, latency_estimate: DurationEstimate) -> None:
        super().__init__(
            fps,
            request_timing=RequestTiming.after_start(call_after_sec),
            merge=TrajectoryMerge.at_observation(),
            latency_estimate=latency_estimate,
        )


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
