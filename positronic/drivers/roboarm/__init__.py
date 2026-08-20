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
    """Whether a command sent to the robot now will reach it.

    - AVAILABLE: The robot is tracking whatever it was last commanded, and will take another command.
    - BUSY: The driver is putting the robot somewhere itself — homing, or serving a synchronous move — and
      leaves the command stream unread until it arrives.
    - ERROR: The robot is in an error state.

    State is published every tick whatever the status; only who is driving the robot changes.
    """

    # These numbers are written into recorded datasets
    AVAILABLE = 0
    BUSY = 2
    ERROR = 3


def is_sound(status: RobotStatus) -> bool:
    """Whether the arm is under the commander's control, rather than the driver's or a fault's."""
    return status == RobotStatus.AVAILABLE


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
