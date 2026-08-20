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

    - AVAILABLE: tracking what it was last commanded, and will take another command.
    - BUSY: the driver is putting it somewhere itself, and leaves the command stream unread until it arrives.
    - ERROR: the robot is in an error state.

    State is published every tick whatever the status; only who is driving the robot changes.
    """

    # These numbers are written into recorded datasets and sent over the wire
    AVAILABLE = 0
    BUSY = 1
    ERROR = 3

    @classmethod
    def _missing_(cls, value: object) -> 'RobotStatus | None':
        # The wire protocol also publishes 2, an arm travelling towards a setpoint while still taking commands
        return cls.AVAILABLE if value == 2 else None


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


__all__ = ['RobotStatus', 'State', 'command']
