"""Policy processors for scheduling, fault handling, and temporal frame stacking.

Processors receive their runtime and child callables at construction. Control processors return a
``Step`` containing commands and the next wake-up time on the runtime's clock.

Use factories to describe a local stack without creating episode state::

    from positronic.policy.base import Factory, Sequential

    definition = Sequential(
        Factory(StopOnFault),
        Factory(TemporalStack, keys=('image',), offsets_sec=(-0.2, -0.1, 0.0)),
        Factory(ChunkedSchedule, fps=20),
    )
    policy = definition.build(runtime, infer)
"""

from collections import deque
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.policy.base import Answer, Commands, Obs, Policy, Processor, Runtime, Step


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


MILLISECOND = 10**6  # In nanoseconds


class StopOnFault(Policy):
    """Withhold commands and child calls while any arm is unavailable.

    An unavailable arm causes an empty command set and a status check one millisecond later. Once every
    arm is available, calls resume on the same child policy.
    """

    WIRE_NAME = 'stop_on_fault'

    def __init__(self, runtime: Runtime, inner: Policy) -> None:
        super().__init__(runtime)
        self._inner = inner

    def __call__(self, obs: Obs) -> Step:
        if _arms_available(obs):
            return self._inner(obs)
        return Step({}, self._runtime.time_ns + MILLISECOND)


class ChunkedSchedule(Policy):
    """Request action chunks asynchronously and emit their commands at a fixed cadence.

    ``infer`` returns an ordered sequence of command sets and must not mutate episode state. The first
    command is due when the completed answer is read; subsequent commands are spaced by ``1 / fps``.
    At most one call is pending, and another starts once the current chunk has been emitted.
    """

    WIRE_NAME = 'chunked_schedule'

    def __init__(self, runtime: Runtime, infer: Callable[[Obs], Sequence[Commands]], fps: float) -> None:
        super().__init__(runtime)
        self._infer = infer
        self._trajectory: deque[tuple[Commands, int]] = deque()
        self._answer: Answer[Sequence[Commands]] | None = None
        self._tick = int(1e9 / fps)

    def __call__(self, obs: Obs) -> Step:
        now_ns = self._runtime.time_ns

        if self._answer is not None and self._answer.done():
            chunk = self._answer.result()
            self._trajectory = deque((waypoint, now_ns + i * self._tick) for i, waypoint in enumerate(chunk))
            self._answer = None

        commands: dict[str, Any] = {}
        while self._trajectory:
            waypoint, execute_at_ns = self._trajectory[0]
            if execute_at_ns > now_ns:
                break
            commands.update(waypoint)
            self._trajectory.popleft()

        if not self._trajectory and self._answer is None:
            self._answer = self._runtime.submit(self._infer, obs)

        return Step(commands, now_ns + self._tick)

    def close(self) -> None:
        if self._answer is not None:
            self._answer.cancel()
            self._answer = None


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

    Every call records the selected channels on the runtime's clock, then passes the stacked
    observations to ``inner`` and returns its result. Offsets are ascending seconds relative to now.
    Wrap a scheduling policy to collect frames on control ticks while inference is pending.

    With ``pad_start=True``, missing history repeats the oldest sample. Otherwise unavailable offsets
    are omitted, and the stack grows until the full window has been observed.
    """

    WIRE_NAME = 'temporal_stack'

    def __init__(
        self,
        runtime: Runtime,
        inner: Callable[[Obs], OutputT],
        keys: tuple[str, ...],
        offsets_sec: tuple[float, ...],
        pad_start: bool = True,
    ) -> None:
        super().__init__(runtime)
        self._inner = inner
        self._keys = tuple(keys)
        assert pad_start or 0.0 in offsets_sec, (
            'pad_start=False requires 0.0 in offsets_sec: with only past offsets the first observation has no '
            'in-range targets and the stack would be empty'
        )
        self._buffer = _StackBuffer(tuple(offsets_sec), pad_start=pad_start)

    def __call__(self, obs: Obs) -> OutputT:
        now_sec = self._runtime.time_ns / 1e9
        self._buffer.append(now_sec, {k: obs[k] for k in self._keys})
        return self._inner({**obs, **self._buffer.sample(now_sec)})
