"""Composable policy layers — chunk playback, fault handling and temporal frame stacking.

Layers are serving-time concerns wrapped around a policy with ``|`` (left is outermost). ``ChunkPlayer``
turns the chunk the policy answers into the commands a caller runs; the layers above it read time from the
observation (``obs_time_ns``) or from the call's ``time_ns``, both in nanoseconds.
"""

from collections import deque
from typing import Any, NamedTuple

import numpy as np

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.policy.base import INFER, Answer, DelegatingPolicy, DelegatingSession, Fn, Layer, Policy, Session

# How far from the call a waypoint may sit: past any real chunk, short of the decades a chunk is off by
# when it reaches the player already anchored.
MAX_ACTION_SKEW_SEC = 60.0


def _obs_time(obs) -> float:
    """Observation timestamp in seconds, from the harness's nanosecond stamp."""
    return obs[keys.OBS_TIME_NS] / 1e9


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


class StopOnFault(Layer):
    """Stop the arm while it will not take a command, and plan afresh once it will.

    An arm the driver has taken, or that is faulted, is not tracking the plan it was given: this commands
    nothing and resets the sessions below, which drops the chunk being played. It goes outside the player,
    which would otherwise keep emitting waypoints without seeing the status. Every arm is checked, so a
    bimanual rig stops on either.
    """

    WIRE_NAME = 'stop_on_fault'
    POLL_SEC = 0.01

    class _Session(DelegatingSession):
        def __call__(self, obs, time_ns):
            if _arms_available(obs):
                return self._inner(obs, time_ns)
            self.cancel()
            return {}, time_ns + int(StopOnFault.POLL_SEC * 1e9)

    def make_session(self, inner: Session) -> Session:
        return StopOnFault._Session(inner)

    def to_spec(self):
        return {'name': self.WIRE_NAME}


class ChunkPlayer(Layer):
    """Hold the chunk ``INFER`` answers, anchor it to the call that received it, and emit each waypoint at
    its own time.

    The player is the bottom session of a chunk policy: it turns the work below into the commands a caller
    runs, so the layers under it wrap that work rather than a session.
    """

    WIRE_NAME = 'chunk_player'
    PLAYS_CHUNKS = True
    POLL_SEC = 0.01

    class _Policy(DelegatingPolicy):
        """The player is the session of a chunk policy, so nothing below it opens one."""

        def new_session(self, rt):
            return ChunkPlayer._Session(rt.fns[INFER])

    class _Session(Session):
        class _Waypoint(NamedTuple):
            cmd: dict[str, Any]
            time_ns: int

        def __init__(self, infer: Fn):
            self._infer = infer
            self._waypoints: deque[ChunkPlayer._Session._Waypoint] = deque()
            # The one call this session keeps in flight, and whether a ``cancel`` has orphaned the chunk it
            # will bring back.
            self._answer: Answer | None = None
            self._orphaned = False

        def __call__(self, obs, time_ns):
            """Plays the chunk it holds; asks for the next one in the call that drains it."""
            if not self._waypoints or self._waypoints[-1].time_ns <= time_ns:
                if self._answer is None:
                    self._answer = self._infer(obs)
                if self._answer.done():
                    chunk = self._take()
                    if chunk is not None:
                        self._load(chunk, time_ns)
            commands: dict[str, Any] = {}
            while self._waypoints and self._waypoints[0].time_ns <= time_ns:
                commands.update(self._waypoints.popleft().cmd)
            if not self._waypoints:
                return commands, time_ns + int(ChunkPlayer.POLL_SEC * 1e9)
            return commands, self._waypoints[0].time_ns

        def _take(self) -> list[dict[str, Any]] | dict[str, Any] | None:
            """The chunk the call brought back, and ``None`` for one a ``cancel`` orphaned.

            An orphaned call is read too, so its failure reaches the caller. The state clears before that
            read, so a cancel ends with the call it was made against and never drops the chunk after it.
            """
            assert self._answer is not None, 'only a call this session made brings a chunk back'
            answer, orphaned = self._answer, self._orphaned
            self._answer, self._orphaned = None, False
            chunk = answer.result()
            return None if orphaned else chunk

        def _load(self, chunk: list[dict[str, Any]] | dict[str, Any], time_ns: int) -> None:
            """Anchor ``chunk`` to ``time_ns`` and hold it.

            A single-action policy may answer a bare dict, and a no-codec path may omit ``timestamp``
            (servers can stamp and truncate themselves); both are normalized here so an immediate action
            plays instead of raising.
            """
            if isinstance(chunk, dict):
                chunk = [chunk]
            skew = max((abs(action.get(keys.ACTION_TIMESTAMP, 0.0)) for action in chunk), default=0.0)
            if skew > MAX_ACTION_SKEW_SEC:
                raise ValueError(
                    f'Action scheduled {skew:.0f}s from the call, over the {MAX_ACTION_SKEW_SEC:.0f}s bound: '
                    f'the work below is timing actions against a clock of its own'
                )
            # The single explicit seconds->ns seam: the work below times actions in float seconds, and
            # every pimm channel is in ns. The offset converts, not the sum, so a waypoint at 0.0 lands on the
            # call itself whatever the clock reads. A waypoint naming no channel — the codecs' end-of-chunk
            # sentinel — commands nothing and states where the chunk ends.
            self._waypoints = deque(
                self._Waypoint(
                    {name: value for name, value in action.items() if name != keys.ACTION_TIMESTAMP},
                    time_ns + int(action.get(keys.ACTION_TIMESTAMP, 0.0) * 1e9),
                )
                for action in chunk
            )

        def cancel(self):
            self._waypoints.clear()
            # The chunk in flight describes a world that has gone. The call is still read, for its failure.
            self._orphaned = self._answer is not None

    def wrap(self, policy: Policy) -> Policy:
        return ChunkPlayer._Policy(policy)

    def to_spec(self):
        return {'name': self.WIRE_NAME}


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


class TemporalStack(Layer):
    """Replaces each named observation entry with a temporal stack of recent samples.

    A model that conditions on a short window of history (e.g. DreamZero's video context) needs several
    samples spanning the just-executed chunk at the cadence seen in training, but the harness only
    forwards an observation to the policy at re-query boundaries. This layer sits outside the
    player so it sees every control tick: it records the named ``keys`` and substitutes, for
    each, a ``(len(offsets_sec), ...)`` stack sampled at ``offsets_sec`` (ascending negative seconds
    relative to now), so every stacked step carries its own value at that time rather than the current
    one repeated across history.

    ``pad_start`` controls the stack before a full window of history exists (the first
    ``-min(offsets_sec)`` seconds of a session). ``True`` repeats the oldest sample so the stack always
    has ``len(offsets_sec)`` steps — for servers that require the trained window length. ``False``
    sends only observed samples (the stack grows from 1 to ``len(offsets_sec)``), so the server sees a
    genuine episode start instead of a fabricated static history — a model conditioned on "nothing has
    moved for the whole window" predicts near-zero motion, and servers with a cold-start path (e.g. a
    chunk-0 empty prefix) never engage it on a padded full-length stack.
    """

    WIRE_NAME = 'temporal_stack'

    class _Session(DelegatingSession):
        def __init__(self, inner: Session, keys: tuple[str, ...], offsets_sec: tuple[float, ...], pad_start: bool):
            super().__init__(inner)
            self._keys = keys
            self._buffer = _StackBuffer(offsets_sec, pad_start=pad_start)

        def __call__(self, obs, time_ns):
            now = _obs_time(obs)
            self._buffer.append(now, {k: obs[k] for k in self._keys})
            return self._inner({**obs, **self._buffer.sample(now)}, time_ns)

        def cancel(self):
            self._buffer.reset()
            super().cancel()

    def __init__(self, keys: tuple[str, ...], offsets_sec: tuple[float, ...], pad_start: bool = True):
        self._keys = tuple(keys)
        self._offsets_sec = tuple(offsets_sec)
        self._pad_start = pad_start
        assert pad_start or 0.0 in self._offsets_sec, (
            'pad_start=False requires 0.0 in offsets_sec: with only past offsets the first observation has no '
            'in-range targets and the stack would be empty'
        )

    def make_session(self, inner: Session) -> Session:
        return TemporalStack._Session(inner, self._keys, self._offsets_sec, self._pad_start)

    def to_spec(self):
        return {
            'name': self.WIRE_NAME,
            'args': {'keys': list(self._keys), 'offsets_sec': list(self._offsets_sec), 'pad_start': self._pad_start},
        }
