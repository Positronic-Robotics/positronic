"""Collection of commands that can be sent to the robot."""

from dataclasses import dataclass

import numpy as np

import geom


@dataclass
class Reset:
    """Reset the robot to the home position."""
    pass


@dataclass
class CartesianMove:
    """Move the robot end-effector to the given pose."""
    pose: geom.Transform3D


@dataclass
class JointMove:
    """Move the robot joints to the given positions."""
    positions: np.ndarray