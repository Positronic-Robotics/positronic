"""Composable policy wrappers — scheduling, error recovery, and temporal frame stacking.

Wrappers are clock-aware concerns layered around a policy at serving time. They take ``now`` — a
``Callable[[], float]`` returning the current time in seconds — rather than a ``pimm.Clock``, so the
policy layer stays free of the control-system runtime. Each wrapper exposes a ``@cfn.config`` that
captures its static arguments and returns a ``now -> PolicyWrapper`` factory; ``compose`` chains those
factories, and ``default_wrappers`` is the standard pipeline.
"""

import functools
import operator
from collections.abc import Callable
from functools import partial

import configuronic as cfn
import numpy as np

from positronic.drivers import roboarm
from positronic.policy.base import DelegatingSession, PolicyWrapper, Session

Now = Callable[[], float]
WrapperFactory = Callable[[Now], PolicyWrapper]


def compose(*factories: WrapperFactory) -> WrapperFactory:
    """Chain wrapper factories into one ``now -> PolicyWrapper`` pipeline (left is outermost)."""
    return lambda now: functools.reduce(operator.or_, (factory(now) for factory in factories))


class ChunkedSchedule(PolicyWrapper):
    """Wait for the current trajectory to finish before calling the inner policy again.

    Owns relative→absolute time conversion: inner layers (codecs, models) emit relative timestamps;
    this wrapper anchors them to ``now()`` *after* inner inference returns, so execution aligns to
    inference-finish (not inference-start). Returns ``None`` ("keep executing the current trajectory")
    until the last action's timestamp is reached, then calls the inner policy.
    """

    class _Session(DelegatingSession):
        """Skips inner calls while the current trajectory plays; stamps absolute on emit."""

        def __init__(self, inner: Session, wrapper: 'ChunkedSchedule'):
            super().__init__(inner)
            self._wrapper = wrapper
            self._trajectory_end: float | None = None

        def __call__(self, obs):
            if self._trajectory_end is not None and self._wrapper._now() < self._trajectory_end:
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
                now = self._wrapper._now()
                result = [{**r, 'timestamp': now + r.get('timestamp', 0.0)} for r in result]
                self._trajectory_end = result[-1]['timestamp'] if result else None
            return result

        def cancel(self):
            self._trajectory_end = None
            super().cancel()

    def __init__(self, now: Now):
        self._now = now

    def wrap_session(self, inner: Session, context):
        return ChunkedSchedule._Session(inner, self)


class ErrorRecovery(PolicyWrapper):
    """Wraps a policy to handle robot errors by emitting Recover commands.

    On error: emits a single Recover trajectory, then returns None until the robot recovers. On
    recovery: resumes normal inference.

    TODO: this wrapper is not name-free. It hard-codes the ``robot_state.error`` observation and the
    ``robot_command`` channel (with a Franka ``Recover``), so it only fits Franka-named embodiments;
    others must disable ``default_wrappers``. How an embodiment should declare its error signal and
    recovery action is still open.
    """

    class _Session(DelegatingSession):
        """Emits Recover trajectory on robot error, delegates otherwise."""

        def __init__(self, inner: Session, wrapper: 'ErrorRecovery'):
            super().__init__(inner)
            self._wrapper = wrapper
            self._in_error = False

        def __call__(self, obs):
            was_ok = not self._in_error
            self._in_error = obs['robot_state.error'] == 1

            if self._in_error:
                if was_ok:
                    # Reset any inner scheduling state so post-recovery doesn't stall on a stale
                    # trajectory_end from the pre-error chunk.
                    self._inner.cancel()
                    return [{'robot_command': roboarm.command.Recover(), 'timestamp': self._wrapper._now()}]
                return None

            return self._inner(obs)

    def __init__(self, now: Now):
        self._now = now

    def wrap_session(self, inner: Session, context):
        return ErrorRecovery._Session(inner, self)


class _FrameBuffer:
    """Per-camera history of ``(timestamp, frame)`` capped to the sampled window.

    ``append`` subsamples to ``min_dt_sec`` and drops frames older than the oldest sampled offset
    (keeping one just beyond it so nearest-neighbor sampling still brackets it), so the buffer stays
    bounded over an episode. ``sample`` returns a ``(len(offsets_sec), H, W, 3)`` stack picked nearest
    to each offset; early on it repeats the oldest frame.
    """

    def __init__(self, image_keys: tuple[str, ...], offsets_sec: tuple[float, ...], min_dt_sec: float):
        self._image_keys = image_keys
        self._offsets_sec = offsets_sec
        self._min_dt_sec = min_dt_sec
        self._buffers: dict[str, list[tuple[float, np.ndarray]]] = {k: [] for k in image_keys}

    def reset(self):
        self._buffers = {k: [] for k in self._buffers}

    def append(self, key: str, now: float, frame: np.ndarray):
        buf = self._buffers[key]
        if not buf or now - buf[-1][0] >= self._min_dt_sec:
            # Copy: camera frames are views into a shared-memory buffer the producer reuses each tick,
            # so storing the view would alias every slot to the latest frame.
            buf.append((now, np.array(frame)))
            cut = next(i for i, (t, _) in enumerate(buf) if t >= now + min(self._offsets_sec))
            del buf[: max(cut - 1, 0)]

    def sample(self, key: str, now: float) -> np.ndarray:
        buf = self._buffers[key]
        times = np.array([t for t, _ in buf])
        idxs = [int(np.argmin(np.abs(times - (now + off)))) for off in self._offsets_sec]
        return np.stack([buf[i][1] for i in idxs])


class TemporalFrameStack(PolicyWrapper):
    """Replaces each named image in the observation with a temporal stack of recent frames.

    Servers whose model conditions on a short video (e.g. DreamZero) need several frames spanning the
    just-executed chunk at the cadence seen in training, but the harness only forwards an observation
    to the policy at re-query boundaries. This wrapper sits outside the scheduling wrapper so it sees
    every control tick: it records each camera frame and substitutes a ``(len(offsets_sec), H, W, 3)``
    stack sampled at ``offsets_sec`` (negative seconds relative to now).
    """

    class _Session(DelegatingSession):
        def __init__(self, inner: Session, wrapper: 'TemporalFrameStack'):
            super().__init__(inner)
            self._wrapper = wrapper
            self._buffer = _FrameBuffer(wrapper._image_keys, wrapper._offsets_sec, wrapper._min_dt_sec)

        def __call__(self, obs):
            now = self._wrapper._now()
            stacked = dict(obs)
            for key in self._wrapper._image_keys:
                self._buffer.append(key, now, obs[key])
                stacked[key] = self._buffer.sample(key, now)
            return self._inner(stacked)

        def cancel(self):
            self._buffer.reset()
            super().cancel()

    def __init__(self, now: Now, image_keys: tuple[str, ...], offsets_sec: tuple[float, ...], min_dt_sec: float = 0.18):
        self._now = now
        self._image_keys = tuple(image_keys)
        self._offsets_sec = tuple(offsets_sec)
        self._min_dt_sec = min_dt_sec

    def wrap_session(self, inner: Session, context):
        return TemporalFrameStack._Session(inner, self)


@cfn.config()
def chunked_schedule():
    """Factory config for ``ChunkedSchedule``."""
    return ChunkedSchedule


@cfn.config()
def error_recovery():
    """Factory config for ``ErrorRecovery``."""
    return ErrorRecovery


@cfn.config(min_dt_sec=0.18)
def temporal_frame_stack(image_keys: tuple[str, ...], offsets_sec: tuple[float, ...], min_dt_sec: float):
    """Factory config for ``TemporalFrameStack`` — binds the static args, leaving ``now`` open."""
    return partial(
        TemporalFrameStack, image_keys=tuple(image_keys), offsets_sec=tuple(offsets_sec), min_dt_sec=min_dt_sec
    )


def default_wrappers(now: Now) -> PolicyWrapper:
    """Default wrapper pipeline: error recovery + chunked scheduling bound to the harness clock."""
    return ErrorRecovery(now) | ChunkedSchedule(now)
