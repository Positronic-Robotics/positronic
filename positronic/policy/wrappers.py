"""Composable policy wrappers — scheduling, error recovery, and temporal frame stacking.

Wrappers are composable serving-time concerns layered around a policy with ``|`` (left is
outermost), exactly like codecs. Most read time from the observation (``obs_time_ns``); only
``ChunkedSchedule`` needs the live clock — it anchors a chunk to inference *completion*, which the
pre-inference observation stamp cannot give — so the harness passes ``now`` (a ``Callable[[], float]``
in seconds) to ``wrap`` and it reaches that one session.
"""

from collections import deque

import numpy as np

from positronic.drivers import roboarm
from positronic.policy.base import DelegatingSession, Now, PolicyWrapper, Session


def _obs_time(obs) -> float:
    """Observation timestamp in seconds, from the harness's nanosecond stamp."""
    return obs['obs_time_ns'] / 1e9


class ChunkedSchedule(PolicyWrapper):
    """Wait for the current trajectory to finish before calling the inner policy again.

    Owns relative→absolute time conversion: inner layers (codecs, models) emit relative timestamps;
    this wrapper anchors them to ``now()`` *after* inner inference returns, so execution aligns to
    inference-finish (not inference-start). Returns ``None`` ("keep executing the current trajectory")
    until the last action's timestamp is reached, then calls the inner policy.
    """

    class _Session(DelegatingSession):
        """Skips inner calls while the current trajectory plays; stamps absolute on emit."""

        def __init__(self, inner: Session, now: Now):
            super().__init__(inner)
            self._now = now
            self._trajectory_end: float | None = None

        def __call__(self, obs):
            if self._trajectory_end is not None and self._now() < self._trajectory_end:
                return None
            result = self._inner(obs)
            if result is not None:
                # A single-action session may return a bare dict, and a no-codec path may omit
                # ``timestamp`` (servers can stamp/truncate themselves); normalize both so an
                # immediate action executes instead of raising.
                if isinstance(result, dict):
                    result = [result]
                # Anchor to post-inference time so execution starts when inference *finished*.
                # Copy dicts so we don't mutate caller-owned data (sessions may reuse templates).
                now = self._now()
                result = [{**r, 'timestamp': now + r.get('timestamp', 0.0)} for r in result]
                self._trajectory_end = result[-1]['timestamp'] if result else None
            return result

        def cancel(self):
            self._trajectory_end = None
            super().cancel()

    def wrap_session(self, inner: Session, context, now: Now):
        return ChunkedSchedule._Session(inner, now)


class ErrorRecovery(PolicyWrapper):
    """Wraps a policy to handle robot errors by emitting Recover commands.

    On error: emits a single Recover trajectory, then returns None until the robot recovers. On
    recovery: resumes normal inference.

    TODO: this wrapper is not name-free. It hard-codes the ``robot_state.error`` observation and the
    ``robot_command`` channel (with a Franka ``Recover``), so it only fits Franka-named embodiments;
    others must omit it from their ``wrap``. How an embodiment should declare its error signal and
    recovery action is still open.
    """

    class _Session(DelegatingSession):
        """Emits Recover trajectory on robot error, delegates otherwise."""

        def __init__(self, inner: Session):
            super().__init__(inner)
            self._in_error = False

        def __call__(self, obs):
            was_ok = not self._in_error
            self._in_error = obs['robot_state.error'] == 1

            if self._in_error:
                if was_ok:
                    # Reset any inner scheduling state so post-recovery doesn't stall on a stale
                    # trajectory_end from the pre-error chunk.
                    self._inner.cancel()
                    return [{'robot_command': roboarm.command.Recover(), 'timestamp': _obs_time(obs)}]
                return None

            return self._inner(obs)

    def wrap_session(self, inner: Session, context, now: Now):
        return ErrorRecovery._Session(inner)


class _FrameBuffer:
    """Time-ordered history of ``(timestamp, values)`` entries, capped to the sampled window.

    ``values`` is a dict of key → array (cameras, and optionally per-frame proprio); every entry holds
    the same keys. ``append`` copies each new entry (cameras are views into a producer-reused buffer)
    but skips one byte-identical to the previous entry — cameras tick slower than the control loop, and
    carry-over sampling reuses the stored value — then drops entries before the oldest sampled offset,
    keeping the one at or before it. ``sample`` returns, per key, a ``(len(offsets_sec), ...)`` stack
    holding, for each offset, the latest value at or before that time — carry-over, never the future;
    before enough history accumulates it repeats the oldest entry.
    """

    def __init__(self, offsets_sec: tuple[float, ...]):
        self._offsets_sec = offsets_sec
        self._entries: deque[tuple[float, dict[str, np.ndarray]]] = deque()

    def reset(self):
        self._entries.clear()

    def append(self, now: float, frames: dict[str, np.ndarray]):
        if self._entries and all(np.array_equal(self._entries[-1][1][k], v) for k, v in frames.items()):
            return
        self._entries.append((now, {k: np.array(v) for k, v in frames.items()}))
        cutoff = now + min(self._offsets_sec)
        while len(self._entries) >= 2 and self._entries[1][0] <= cutoff:
            self._entries.popleft()

    def sample(self, now: float) -> dict[str, np.ndarray]:
        times = np.array([t for t, _ in self._entries])
        picked = [self._entries[self._at_or_before(times, now + off)][1] for off in self._offsets_sec]
        return {k: np.stack([frames[k] for frames in picked]) for k in picked[0]}

    @staticmethod
    def _at_or_before(times: np.ndarray, target: float) -> int:
        """Index of the latest frame at or before ``target``; clamps to the oldest when none precedes it."""
        return max(int(np.searchsorted(times, target, side='right')) - 1, 0)


class TemporalFrameStack(PolicyWrapper):
    """Replaces each named observation entry with a temporal stack of recent samples.

    Servers whose model conditions on a short video (e.g. DreamZero) need several frames spanning the
    just-executed chunk at the cadence seen in training, but the harness only forwards an observation
    to the policy at re-query boundaries. This wrapper sits outside the scheduling wrapper so it sees
    every control tick: it records the named entries and substitutes a ``(len(offsets_sec), ...)`` stack
    sampled at ``offsets_sec`` (ascending negative seconds relative to now).

    ``image_keys`` are the cameras (``(T, H, W, 3)``). ``proprio_keys`` are optional per-frame
    proprioceptive signals (e.g. ``robot_state.ee_pose`` → ``(T, 7)``, ``grip`` → ``(T,)``) recorded
    on the SAME buffer, so each stacked frame carries its own proprio — the trajectory a model trained
    on real per-frame proprio expects, rather than the current pose repeated across history. Leave it
    empty for models whose codec consumes only the current proprio (stacking would reshape it wrong).
    """

    class _Session(DelegatingSession):
        def __init__(self, inner: Session, stack_keys: tuple[str, ...], offsets_sec: tuple[float, ...]):
            super().__init__(inner)
            self._stack_keys = stack_keys
            self._buffer = _FrameBuffer(offsets_sec)

        def __call__(self, obs):
            now = _obs_time(obs)
            self._buffer.append(now, {k: obs[k] for k in self._stack_keys})
            return self._inner({**obs, **self._buffer.sample(now)})

        def cancel(self):
            self._buffer.reset()
            super().cancel()

    def __init__(self, image_keys: tuple[str, ...], offsets_sec: tuple[float, ...], proprio_keys: tuple[str, ...] = ()):
        self._stack_keys = (*image_keys, *proprio_keys)
        self._offsets_sec = tuple(offsets_sec)

    def wrap_session(self, inner: Session, context, now: Now):
        return TemporalFrameStack._Session(inner, self._stack_keys, self._offsets_sec)
