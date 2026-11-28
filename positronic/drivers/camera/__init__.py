from typing import NamedTuple

# Two instants of a frame on the camera's own clock: when the sensor took it, and when the driver sent it.
CAPTURE_TIME = 'camera.capture'
HANDOFF_TIME = 'camera.handoff'


class PinholeIntrinsics(NamedTuple):
    """The size of an image with no lens distortion, and its camera matrix, both in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
