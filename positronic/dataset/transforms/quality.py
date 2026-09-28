"""Per-frame quality signals for dataset debugging.

Quality signals are composed from general-purpose signal transforms (diff, norm,
view) — nothing expensive happens until values are accessed.
"""

import numpy as np

from positronic import keys

from .signals import Elementwise, Join, diff, norm, view

_TRANSLATION = slice(0, 3)
_DT_SEC = 1 / 15


def idle_mask(episode, signal=keys.JOINTS, velocity_threshold=0.015, dt_sec=_DT_SEC, *, timeline: str):
    """Per-frame bool: True where joint speed < threshold (rad/s)."""
    speed = norm(diff(episode.signals[signal], dt_sec, timeline=timeline))

    def fn(vals):
        return np.array(vals) < velocity_threshold

    return Elementwise(speed, fn)


def jerk(episode, signal=keys.JOINTS, dt_sec=_DT_SEC, *, timeline: str):
    """Per-frame joint acceleration magnitude (rad/s^2)."""
    return norm(diff(episode.signals[signal], dt_sec, order=2, timeline=timeline))


def cmd_lag(
    episode, cmd_signal=keys.TARGET_EE_POSE, state_signal=keys.EE_POSE, components=_TRANSLATION, *, timeline: str
):
    """Per-frame distance between commanded and actual pose (meters)."""
    cmd = episode.signals[cmd_signal]
    ee = episode.signals[state_signal]

    def fn(pairs):
        arr = np.array(pairs)  # (batch, 2, dim)
        return np.linalg.norm(arr[:, 0, components] - arr[:, 1, components], axis=-1)

    return Elementwise(Join(cmd, ee, timeline=timeline), fn)


def cmd_velocity(episode, signal=keys.TARGET_EE_POSE, components=_TRANSLATION, dt_sec=_DT_SEC, *, timeline: str):
    """Per-frame command translation velocity (m/s). Spikes = tracking glitches."""
    return norm(diff(view(episode.signals[signal], components), dt_sec, timeline=timeline))
