"""The conversion between positronic's grip and a gripper's aperture, for a device or an env server to apply.

This module must work in an isolated interpreter without Positronic installed.
"""


def flip_grip(value: float) -> float:
    """Between positronic's grip (1 closed) and an aperture (1 open), clipped to [0, 1]. It is its own inverse."""
    return 1.0 - min(max(float(value), 0.0), 1.0)
