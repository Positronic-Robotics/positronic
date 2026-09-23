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

from collections import Counter, defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from math import isfinite
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


class ScheduleAccount:
    """Per command channel: rows planned, emitted and skipped, how late each emitted row went out, and the
    largest gap between two emits of one chunk.

    A row is skipped when it came due and went out on no round: a later due row replaced it in the same
    round, or a new chunk replaced it after its due time.
    """

    def __init__(self) -> None:
        self._planned: Counter[str] = Counter()
        self._emitted: Counter[str] = Counter()
        self._skipped: Counter[str] = Counter()
        self._late_ns: defaultdict[str, list[int]] = defaultdict(list)
        self._gap_max_ns: Counter[str] = Counter()
        self._chunk_emit_ns: dict[str, int] = {}

    def plan(self, rows: Iterable[Commands]) -> None:
        """Take a new chunk's rows. Rows of the chunk before it that are still due must go to ``skip`` first."""
        self._chunk_emit_ns.clear()
        for row in rows:
            self._planned.update(row.keys())

    def skip(self, rows: Iterable[Commands]) -> None:
        for row in rows:
            self._skipped.update(row.keys())

    def emit(self, due: Sequence[tuple[Commands, int]], now_ns: int) -> dict[str, Any]:
        """The commands of the due ``(row, due_ns)`` rows: per channel, the last row wins and the rest skip."""
        commands: dict[str, Any] = {}
        due_ns_by_name: dict[str, int] = {}
        for row, due_ns in due:
            self._skipped.update(name for name in row if name in due_ns_by_name)
            due_ns_by_name.update(dict.fromkeys(row, due_ns))
            commands.update(row)
        for name, due_ns in due_ns_by_name.items():
            self._emitted[name] += 1
            self._late_ns[name].append(now_ns - due_ns)
            if name in self._chunk_emit_ns:
                self._gap_max_ns[name] = max(self._gap_max_ns[name], now_ns - self._chunk_emit_ns[name])
            self._chunk_emit_ns[name] = now_ns
        return commands

    def meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for name, planned in self._planned.items():
            prefix = f'{eval_keys.SCHEDULE}.{name}'
            meta[f'{prefix}.{eval_keys.SCHEDULED}'] = planned
            meta[f'{prefix}.{eval_keys.EMITTED}'] = self._emitted[name]
            meta[f'{prefix}.{eval_keys.DROPPED}'] = self._skipped[name]
            if late_ns := self._late_ns[name]:
                p50, p90 = np.percentile(late_ns, (50, 90))
                meta[f'{prefix}.{eval_keys.LATE_P50_MS}'] = float(p50) / 1e6
                meta[f'{prefix}.{eval_keys.LATE_P90_MS}'] = float(p90) / 1e6
                meta[f'{prefix}.{eval_keys.LATE_MAX_MS}'] = max(late_ns) / 1e6
                meta[f'{prefix}.{eval_keys.GAP_MAX_MS}'] = self._gap_max_ns[name] / 1e6
        return meta


class ChunkedSchedule(Policy):
    """Request action chunks asynchronously and emit their commands at a fixed cadence.

    ``infer`` returns an ordered sequence of command sets and must not mutate episode state. The first
    command is due when the completed answer is read; subsequent commands are spaced by ``1 / fps``.
    A chunk of K commands covers K periods, including the final command's execution period.
    ``horizon_sec`` limits that duration and discards commands at or beyond the horizon.
    At most one call is pending, and another starts when the current chunk's duration ends.
    """

    WIRE_NAME = 'chunked_schedule'
    WIRE_VERSION = 2

    def __init__(self, fps: float, horizon_sec: float | None = None) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError('fps must be finite and positive')
        if horizon_sec is not None and (not isfinite(horizon_sec) or horizon_sec <= 0):
            raise ValueError('horizon_sec must be finite and positive')
        self._fps = fps
        self._horizon_sec = horizon_sec

    def run(self, runtime: Runtime, infer: Callable[[Obs], Sequence[Commands]]) -> PolicyRun:
        period_sec = 1 / self._fps
        answer: Answer[Sequence[Commands]] | None = None
        trajectory: deque[tuple[Commands, int]] = deque()
        end_ns = 0
        account = ScheduleAccount()
        runtime.report(account.meta)
        obs = yield
        try:
            while True:
                now_ns = runtime.time_ns
                if answer is None and now_ns >= end_ns:
                    answer = runtime.submit(infer, obs)
                if answer is not None and answer.done():
                    chunk, answer = answer.result(), None
                    duration_sec = len(chunk) * period_sec
                    if self._horizon_sec is not None:
                        duration_sec = min(duration_sec, self._horizon_sec)
                    end_ns = now_ns + round(duration_sec * 1e9)
                    account.skip(row for row, due_ns in trajectory if due_ns <= now_ns)
                    trajectory = deque(
                        (waypoint, now_ns + round(i * period_sec * 1e9))
                        for i, waypoint in enumerate(chunk)
                        if i * period_sec < duration_sec
                    )
                    account.plan(row for row, _ in trajectory)

                due = []
                while trajectory and trajectory[0][1] <= now_ns:
                    due.append(trajectory.popleft())
                commands = account.emit(due, now_ns)
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
        args = {'fps': self._fps}
        if self._horizon_sec is not None:
            args['horizon_sec'] = self._horizon_sec
        return {NAME: self.WIRE_NAME, VERSION: self.WIRE_VERSION, ARGS: args}


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
