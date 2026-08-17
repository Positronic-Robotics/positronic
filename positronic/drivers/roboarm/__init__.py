"""Robot arm drivers package.

This package provides drivers for various robot arms including Franka, Kinova, and SO-101.
"""

from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np

from positronic import geom

# Import command submodule to make it accessible as roboarm.command
from . import command


class RobotStatus(IntEnum):
    """Different statuses that the robot can be in.

    The exact meaning of this statuses currently is defined by the robot driver. But in general:

    - AVAILABLE: The robot is available to accept new commands.
    - RESETTING: The robot is resetting.
    - MOVING: The robot is moving to a new position, but is not yet at the new position.
    - ERROR: The robot is in an error state.
    """

    AVAILABLE = 0
    RESETTING = 1
    MOVING = 2
    ERROR = 3


def is_sound(status: int) -> bool:
    """Whether the arm is tracking the commands it was given and its pose is worth reading. Takes the
    status's number as readily as the member, that being what a dataset and the offboard wire give back."""
    return status in (RobotStatus.AVAILABLE, RobotStatus.MOVING)


class State(ABC):
    """
    Abstract state of the robot. Each robot must have its own implementation of this class.
    """

    @property
    @abstractmethod
    def q(self) -> np.ndarray:
        """Joints positions of the robot."""
        pass

    @property
    @abstractmethod
    def dq(self) -> np.ndarray:
        """Joints velocities of the robot."""
        pass

    @property
    @abstractmethod
    def ee_pose(self) -> geom.Transform3D:
        """Position of the robot's end-effector."""
        pass

    @property
    @abstractmethod
    def status(self) -> RobotStatus:
        """Robot status."""
        pass

    @property
    def ee_wrench(self) -> np.ndarray | None:
        """Wrench of the robot's end-effector in its own coordinate frame."""
        return None


__all__ = ['RobotStatus', 'State', 'command', 'is_sound']
