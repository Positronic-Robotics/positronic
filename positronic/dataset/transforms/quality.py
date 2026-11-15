"""Per-frame quality signals for dataset debugging.

Quality signals are composed from general-purpose signal transforms (diff, norm,
view) — nothing expensive happens until values are accessed.
Derivative timelines must measure nanoseconds.
"""

import numpy as np

from positronic import keys

from .signals import Elementwise, Join, diff, norm, view

_TRANSLATION = slice(0, 3)
_DT_SEC = 1 / 15


def idle_mask(episode, signal=keys.JOINTS, velocity_threshold=0.015, dt_sec=_DT_SEC, *, timelines: tuple[str, ...]):
    """Per-frame bool: True where joint speed < threshold (rad/s)."""
    speed = norm(diff(episode.signals[signal], dt_sec, timelines=timelines))

    def fn(vals):
        return np.array(vals) < velocity_threshold

    return Elementwise(speed, fn)


def jerk(episode, signal=keys.JOINTS, dt_sec=_DT_SEC, *, timelines: tuple[str, ...]):
    """Per-frame joint acceleration magnitude (rad/s^2)."""
    return norm(diff(episode.signals[signal], dt_sec, order=2, timelines=timelines))


def cmd_lag(
    episode,
    cmd_signal=keys.TARGET_EE_POSE,
    state_signal=keys.EE_POSE,
    components=_TRANSLATION,
    *,
    timelines: tuple[str, ...],
):
    """Per-frame distance between commanded and actual pose (meters)."""
    cmd = episode.signals[cmd_signal]
    ee = episode.signals[state_signal]

    def fn(pairs):
        arr = np.array(pairs)  # (batch, 2, dim)
        return np.linalg.norm(arr[:, 0, components] - arr[:, 1, components], axis=-1)

    return Elementwise(Join(cmd, ee, timelines=timelines), fn)


def cmd_velocity(
    episode, signal=keys.TARGET_EE_POSE, components=_TRANSLATION, dt_sec=_DT_SEC, *, timelines: tuple[str, ...]
):
    """Per-frame command translation velocity (m/s). Spikes = tracking glitches."""
    return norm(diff(view(episode.signals[signal], components), dt_sec, timelines=timelines))


def cmd_delay_ms(
    episode,
    cmd_signal=keys.TARGET_EE_POSE,
    state_signal=keys.EE_POSE,
    components=_TRANSLATION,
    max_delay_ms: int = 500,
    min_travel: float = 0.01,
    *,
    timeline: str,
) -> int | None:
    """Milliseconds by which the measured pose follows the commanded pose in one episode.

    The shift from 0 to ``max_delay_ms`` with the smallest mean distance between each measured pose and the command
    that was in force that long before it. ``None`` when the commands travel less than
    ``min_travel`` on every component, or when no measured pose is ``max_delay_ms`` past the first command. A tie
    between shifts returns the shortest, so the pose period limits the resolution.

    The delay holds every stage between the points where ``timeline`` stamps a command and a pose.
    """
    cmd = episode.signals[cmd_signal]
    state = episode.signals[state_signal]
    cmd_ns = np.asarray(cmd.timestamps(timeline), dtype=np.int64)
    if len(cmd_ns) == 0:
        return None
    cmd_values = np.asarray(cmd.values())[:, components]
    if np.ptp(cmd_values, axis=0).max() < min_travel:
        return None
    state_ns = np.asarray(state.timestamps(timeline), dtype=np.int64)
    # Each shift compares the same measured poses: those with a command in force at the longest shift.
    measured = state_ns >= cmd_ns[0] + max_delay_ms * 1_000_000
    if not measured.any():
        return None
    state_ns = state_ns[measured]
    state_values = np.asarray(state.values())[measured][:, components]
    errors = [
        np.linalg.norm(
            cmd_values[np.searchsorted(cmd_ns, state_ns - delay_ms * 1_000_000, side='right') - 1] - state_values,
            axis=-1,
        ).mean()
        for delay_ms in range(max_delay_ms + 1)
    ]
    return int(np.argmin(errors))
