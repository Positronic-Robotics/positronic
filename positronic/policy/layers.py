"""Policy processors for scheduling, fault handling, and temporal frame stacking.

Constructors hold configuration. ``run(runtime, *dependencies)`` creates an episode generator whose
locals hold its state. Control processors yield a ``Step`` with commands and the next wake-up time.

Describe a local stack without creating episode state::

    from positronic.policy.sequential import Sequential

    definition = Sequential(
        StopOnFault(), TemporalStack(keys=('image',), offsets_sec=(-0.2, -0.1, 0.0)), ChunkedSchedule(fps=20)
    )
    episode = runtime.start(definition, infer)
    step = episode.send(obs)
"""

from collections import deque
from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any, TypeVar

import numpy as np

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.policy import keys as policy_keys
from positronic.policy.base import (
    ARGS,
    NAME,
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


class StopOnFault(Policy):
    """Withhold commands and child calls while any arm is unavailable.

    An unavailable arm causes an empty command set and a status check one millisecond later. Once every
    arm is available, calls resume on the same child policy.
    """

    WIRE_NAME = 'stop_on_fault'

    def run(self, runtime: Runtime, inner: PolicyRun) -> PolicyRun:
        obs = yield
        while True:
            if _arms_available(obs):
                obs = yield inner.send(obs)
            else:
                obs = yield Step({}, runtime.time_ns + MILLISECOND_NS)

    def to_spec(self) -> dict[str, Any]:
        return {NAME: self.WIRE_NAME}


class ChunkedSchedule(Policy):
    """Request action chunks asynchronously and emit their commands at a fixed cadence.

    ``infer`` returns an ordered sequence of command sets and must not mutate episode state. The first
    command is due when the completed answer is read; subsequent commands are spaced by ``1 / fps``.
    A chunk of K commands covers K periods, including the final command's execution period.
    ``horizon_sec`` limits that duration and discards commands at or beyond the horizon.
    At most one call is pending, and another starts when the current chunk's duration ends.
    """

    WIRE_NAME = 'chunked_schedule'

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
                    trajectory = deque(
                        (waypoint, now_ns + round(i * period_sec * 1e9))
                        for i, waypoint in enumerate(chunk)
                        if i * period_sec < duration_sec
                    )

                commands: dict[str, Any] = {}
                while trajectory and trajectory[0][1] <= now_ns:
                    commands.update(trajectory.popleft()[0])
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
        return {NAME: self.WIRE_NAME, ARGS: args}


class _StackBuffer:
    """Time-ordered history of ``(timestamp, values)`` entries, capped to the sampled window.

    ``values`` is a dict of key → array; every entry holds the same keys. ``append`` copies each new
    entry but skips one byte-identical to the previous — a source slower than the control loop repeats
    its value, and carry-over sampling reuses the stored one — then drops entries before the oldest
    sampled offset, keeping the one at or before it. ``sample`` returns, per key, a stack holding, for
    each offset, the latest value at or before that time — carry-over, never the future. Offsets that
    precede the first entry either repeat the oldest entry (``pad_start=True``, a fixed
    ``len(offsets_sec)``-long stack) or are dropped (``pad_start=False``, the stack grows from 1 to
    ``len(offsets_sec)`` as history accumulates).
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

    def sample(self, now: float) -> dict[str, np.ndarray]:
        times = np.array([t for t, _ in self._entries])
        targets = [now + off for off in self._offsets_sec]
        if not self._pad_start:
            targets = [t for t in targets if t >= times[0]]
        picked = [self._entries[self._at_or_before(times, t)][1] for t in targets]
        return {k: np.stack([entry[k] for entry in picked]) for k in picked[0]}

    @staticmethod
    def _at_or_before(times: np.ndarray, target: float) -> int:
        """Index of the latest entry at or before ``target``; clamps to the oldest when none precedes it."""
        return max(int(np.searchsorted(times, target, side='right')) - 1, 0)


OutputT = TypeVar('OutputT')


class TemporalStack(Processor[Obs, OutputT]):
    """Replaces each named observation entry with a temporal stack of recent samples.

    Every sent observation records the selected channels on the runtime's clock, then passes the stacked
    observations to ``inner`` and yields its result. Offsets are ascending seconds relative to now.
    Wrap a scheduling policy to collect frames on control ticks while inference is pending.

    With ``pad_start=True``, missing history repeats the oldest sample. Otherwise unavailable offsets
    are omitted, and the stack grows until the full window has been observed.
    """

    WIRE_NAME = 'temporal_stack'

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
            obs = yield inner.send({**obs, **buffer.sample(now_sec)})

    def to_spec(self) -> dict[str, Any]:
        return {
            NAME: self.WIRE_NAME,
            ARGS: {'keys': list(self._keys), 'offsets_sec': list(self._offsets_sec), 'pad_start': self._pad_start},
        }
