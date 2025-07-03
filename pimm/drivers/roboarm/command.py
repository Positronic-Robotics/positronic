"""Collection of commands that can be sent to the robot."""

from dataclasses import dataclass

import geom


@dataclass
class Reset:
    """Reset the robot to the home position."""
    pass


@dataclass
class CartesianMove:
    """Move the robot end-effector to the given pose."""
    pose: geom.Transform3D
