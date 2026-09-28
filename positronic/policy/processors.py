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

from bisect import insort
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from math import isfinite
from statistics import fmean
from typing import Any, TypeVar

import numpy as np

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.eval import keys as eval_keys
from positronic.policy import keys as policy_keys
from positronic.policy.base import (
    ARGS,
    NAME,
    VERSION,
    Answer,
    Commands,
    Obs,
    Policy,
    PolicyRun,
    Processor,
    ProcessorRun,
    Runtime,
    Step,
)


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

    WIRE_NAME = 'stop_on_fault'  # Stable protocol identifier, independent of the Python class name.
    WIRE_VERSION = 2

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        while True:
            if _arms_available(obs):
                obs = yield inner.send(obs)
            else:
                obs = yield Step({}, runtime.time_ns + MILLISECOND_NS)

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION}


class ChunkedSchedule(Policy):
    """Request action chunks asynchronously and emit their commands at a fixed cadence.

    ``infer`` returns an ordered sequence of command sets and must not mutate episode state. The first
    command is due when the completed answer is read; subsequent commands are spaced by ``1 / fps``.
    A chunk of K commands covers K periods, including the final command's execution period.
    ``horizon_sec`` limits that duration and discards commands at or beyond the horizon.
    At most one call is pending, and another starts when the current chunk's duration ends.
    With ``record_stats``, each round that plans or emits a waypoint writes the waypoint counters into the
    episode metadata.
    """

    WIRE_NAME = 'chunked_schedule'
    WIRE_VERSION = 2
    FPS_ARG = 'fps'
    HORIZON_SEC_ARG = 'horizon_sec'
    RECORD_STATS_ARG = 'record_stats'

    class _Stats:
        """How the schedule played its waypoints.

        A round sends the commands of every due waypoint, and on each channel the last one wins. The due
        waypoints before the last one count as dropped, also when one of their channels went out. The due
        waypoints that a new chunk replaces count as dropped too.
        """

        def __init__(self, metadata: dict[str, Any]) -> None:
            self._metadata = metadata
            self._scheduled = 0
            self._sorted_late_ns: list[int] = []
            self._gap_max_ns = 0
            self._last_emit_ns: int | None = None

        def count_round(self, planned: int | None, due_ns: int | None, now_ns: int, queued: int) -> None:
            """Count the ``planned`` waypoints of a new chunk and the one emitted at ``now_ns``, if any."""
            if planned is None and due_ns is None:
                return
            if planned is not None:
                self._scheduled += planned
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
                self._metadata[f'{eval_keys.SCHEDULE}.{name}'] = value

        @staticmethod
        def _percentile_of_sorted(values: Sequence[int], fraction: float) -> float:
            """``np.percentile``'s linear interpolation in constant time: the control thread calls it each round."""
            position = fraction * (len(values) - 1)
            low = int(position)
            high = min(low + 1, len(values) - 1)
            return values[low] + (values[high] - values[low]) * (position - low)

    def __init__(self, fps: float, horizon_sec: float | None = None, record_stats: bool = True) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        if horizon_sec is not None and (not isfinite(horizon_sec) or horizon_sec <= 0):
            raise ValueError('horizon_sec must be finite and positive')
        self._fps = fps
        self._horizon_sec = horizon_sec
        self._record_stats = record_stats

    def run(self, runtime: Runtime, infer: Callable[[Obs], Sequence[Commands]]) -> PolicyRun:
        period_sec = 1 / self._fps
        answer: Answer[Sequence[Commands]] | None = None
        trajectory: deque[tuple[Commands, int]] = deque()
        end_ns = 0
        stats = self._Stats(runtime.metadata) if self._record_stats else None
        obs = yield
        try:
            while True:
                now_ns = runtime.time_ns
                planned = None
                if answer is None and now_ns >= end_ns:
                    answer = runtime.submit(infer, obs)
                if answer is not None and answer.done():
                    chunk, answer = answer.result(), None
                    duration_sec = len(chunk) * period_sec
                    if self._horizon_sec is not None:
                        duration_sec = min(duration_sec, self._horizon_sec)
                    end_ns = now_ns + round(duration_sec * 1e9)
                    trajectory = deque(
                        (waypoint, now_ns + round(i * period_sec * 1e9))
                        for i, waypoint in enumerate(chunk)
                        if i * period_sec < duration_sec
                    )
                    planned = len(trajectory)

                commands: dict[str, Any] = {}
                due_ns = None
                while trajectory and trajectory[0][1] <= now_ns:
                    waypoint, due_ns = trajectory.popleft()
                    commands.update(waypoint)
                if stats is not None:
                    stats.count_round(planned, due_ns, now_ns, queued=len(trajectory))
                resume_at_ns = trajectory[0][1] if trajectory else end_ns
                # Pending inference asks for the earliest allowed poll; action cadence is independent.
                obs = yield Step(commands, now_ns if answer is not None else resume_at_ns)
        finally:
            if answer is not None:
                answer.cancel()

    def meta(self) -> dict[str, Any]:
        meta = {policy_keys.ACTION_FPS: self._fps}
        if self._horizon_sec is not None:
            meta[policy_keys.ACTION_HORIZON_SEC] = self._horizon_sec
        return meta

    def to_spec(self) -> dict[str, Any]:
        args: dict[str, Any] = {self.FPS_ARG: self._fps}
        if self._horizon_sec is not None:
            args[self.HORIZON_SEC_ARG] = self._horizon_sec
        if not self._record_stats:
            args[self.RECORD_STATS_ARG] = False
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}


PrefixDuration = Callable[[Sequence[float]], float]
"""Gets the delays of earlier calls to `infer` in this episode, in seconds, newest last.
A delay is the time from a call to its answer. Returns the prefix length in seconds."""


def mean_delay(last: int = 5, max_sec: float = 0.4) -> PrefixDuration:
    """The mean of the last `last` delays, but not more than `max_sec`."""
    if last < 1:
        raise ValueError('last must be at least 1')
    if not max_sec >= 0:
        raise ValueError('max_sec must not be negative')
    return lambda delays: min(fmean(delays[-last:]), max_sec)


def max_delay(last: int = 5, max_sec: float = 0.4) -> PrefixDuration:
    """The longest of the last `last` delays, but not more than `max_sec`. The RTC paper does this."""
    if last < 1:
        raise ValueError('last must be at least 1')
    if not max_sec >= 0:
        raise ValueError('max_sec must not be negative')
    return lambda delays: min(max(delays[-last:]), max_sec)


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
    The call also gets a prefix: the old actions that the policy executes from T1 on,
    for `prefix_duration(delays)` seconds, but not past the end of the old chunk. The
    prefix tells the model what the robot does while the model computes. Its length is
    the delay estimate: the server reads it as the number of prefix actions / `fps`.
    The server decides how to use the prefix.

    Until the answer arrives, the policy executes the old chunk. When the answer arrives,
    the policy executes the new chunk, from the action that is due now. If the old chunk
    ends before the answer arrives, the policy executes no action until the answer arrives.

    Only one call runs at a time. If an answer arrives more than `call_after_sec` after
    its observation, the next call starts at once.

    The first call has no old chunk, so its prefix is empty, and the robot executes no
    action until its answer arrives (-- in the drawing).

    `infer` gets the observation and the prefix, in the command format that `infer`
    returns, and converts both to the model's format.
    """

    def __init__(self, fps: float, call_after_sec: float, prefix_duration: PrefixDuration) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        if not isfinite(call_after_sec) or call_after_sec < 0:
            raise ValueError('call_after_sec must be finite and not negative')
        self._fps = fps
        self._call_after_ns = round(call_after_sec * 1e9)
        self._prefix_duration = prefix_duration

    def _due_ns(self, start_ns: int, index: int) -> int:
        return start_ns + round(index * 1e9 / self._fps)

    def _running_index(self, start_ns: int, chunk: Sequence[Commands], time_ns: int) -> int:
        """The index of the action that runs at ``time_ns``, or ``len(chunk)`` after the chunk ends."""
        return next((i for i in range(len(chunk)) if self._due_ns(start_ns, i + 1) > time_ns), len(chunk))

    def run(self, runtime: Runtime, infer: Callable[[Obs, Sequence[Commands]], Sequence[Commands]]) -> PolicyRun:
        delays: list[float] = []
        answer: Answer[Sequence[Commands]] | None = None
        called_at_ns = 0
        chunk: Sequence[Commands] | None = None
        start_ns = 0
        next_index = 0
        next_call_ns = 0
        obs = yield
        try:
            while True:
                now_ns = runtime.time_ns
                if answer is not None and answer.done():
                    new_chunk, answer = answer.result(), None
                    delays.append((now_ns - called_at_ns) / 1e9)
                    # The first chunk counts from its answer: the robot did not move while it waited.
                    start_ns = now_ns if chunk is None else called_at_ns
                    chunk = new_chunk
                    next_index = self._running_index(start_ns, chunk, now_ns)
                    next_call_ns = start_ns + self._call_after_ns

                if answer is None and now_ns >= next_call_ns:
                    prefix: list[Commands] = []
                    if chunk is not None:
                        first = self._running_index(start_ns, chunk, now_ns)
                        prefix_end_ns = now_ns + round(self._prefix_duration(delays) * 1e9)
                        end = self._running_index(start_ns, chunk, prefix_end_ns - 1) + 1
                        prefix = list(chunk[first:end]) if prefix_end_ns > now_ns else []
                    called_at_ns = now_ns
                    answer = runtime.submit(infer, obs, prefix)

                commands: dict[str, Any] = {}
                while chunk is not None and next_index < len(chunk) and self._due_ns(start_ns, next_index) <= now_ns:
                    commands.update(chunk[next_index])
                    next_index += 1
                if answer is not None:
                    # Pending inference asks for the earliest allowed poll; action cadence is independent.
                    resume_at_ns = now_ns
                elif chunk is not None and next_index < len(chunk):
                    resume_at_ns = min(next_call_ns, self._due_ns(start_ns, next_index))
                else:
                    resume_at_ns = next_call_ns
                obs = yield Step(commands, resume_at_ns)
        finally:
            if answer is not None:
                answer.cancel()

    def meta(self) -> dict[str, Any]:
        return {policy_keys.ACTION_FPS: self._fps}


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


class TemporalStack(Processor[Obs, OutputT]):
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

    def run(self, runtime: Runtime, inner: ProcessorRun[Obs, OutputT]) -> ProcessorRun[Obs, OutputT]:
        buffer = _StackBuffer(self._offsets_sec, pad_start=self._pad_start)
        obs = yield
        while True:
            now_sec = runtime.time_ns / 1e9
            buffer.append(now_sec, {k: obs[k] for k in self._keys})
            obs = yield inner.send(buffer.sample(now_sec, obs))

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            VERSION: self.WIRE_VERSION,
            ARGS: {'keys': list(self._keys), 'offsets_sec': list(self._offsets_sec), 'pad_start': self._pad_start},
        }
