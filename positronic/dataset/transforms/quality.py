"""Per-frame quality signals for dataset debugging.

Quality signals are composed from general-purpose signal transforms (diff, norm,
view) — nothing expensive happens until values are accessed.
Derivative timelines must measure nanoseconds.
"""

import numpy as np

from positronic import keys
from positronic.dataset.episode import select_timeline

from .signals import Elementwise, Join, diff, norm, view

_TRANSLATION = slice(0, 3)
_DT_SEC = 1 / 15


def idle_mask(
    episode, signal=keys.JOINTS, velocity_threshold=0.015, dt_sec=_DT_SEC, *, timelines: tuple[str, ...] | None = None
):
    """Per-frame bool: True where joint speed < threshold (rad/s)."""
    if timelines is None:
        timelines = (select_timeline(episode.signals[signal].timelines),)
    speed = norm(diff(episode.signals[signal], dt_sec, timelines=timelines))

    def fn(vals):
        return np.array(vals) < velocity_threshold

    return Elementwise(speed, fn)


def jerk(episode, signal=keys.JOINTS, dt_sec=_DT_SEC, *, timelines: tuple[str, ...] | None = None):
    """Per-frame joint acceleration magnitude (rad/s^2)."""
    if timelines is None:
        timelines = (select_timeline(episode.signals[signal].timelines),)
    return norm(diff(episode.signals[signal], dt_sec, order=2, timelines=timelines))


def cmd_lag(
    episode,
    cmd_signal=keys.TARGET_EE_POSE,
    state_signal=keys.EE_POSE,
    components=_TRANSLATION,
    *,
    timelines: tuple[str, ...] | None = None,
):
    """Per-frame distance between commanded and actual pose (meters)."""
    cmd = episode.signals[cmd_signal]
    ee = episode.signals[state_signal]
    if timelines is None:
        timelines = (select_timeline(set(cmd.timelines) & set(ee.timelines)),)

    def fn(pairs):
        arr = np.array(pairs)  # (batch, 2, dim)
        return np.linalg.norm(arr[:, 0, components] - arr[:, 1, components], axis=-1)

    return Elementwise(Join(cmd, ee, timelines=timelines), fn)


def cmd_velocity(
    episode,
    signal=keys.TARGET_EE_POSE,
    components=_TRANSLATION,
    dt_sec=_DT_SEC,
    *,
    timelines: tuple[str, ...] | None = None,
):
    """Per-frame command translation velocity (m/s). Spikes = tracking glitches."""
    if timelines is None:
        timelines = (select_timeline(episode.signals[signal].timelines),)
    return norm(diff(view(episode.signals[signal], components), dt_sec, timelines=timelines))
