from pkgutil import extend_path as _extend_path
__path__ = _extend_path(__path__, __name__)

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from positronic import geom
from positronic.drivers.roboarm import command


class RobotStatus(Enum):
    """Different statuses that the robot can be in.

    The exact meaning of this statuses currently is defined by the robot driver. But in general:

    - AVAILABLE: The robot is available to accept new commands.
    - RESETTING: The robot is resetting.
    - MOVING: The robot is moving to a new position, but is not yet at the new position.
    """
    AVAILABLE = 0
    RESETTING = 1
    MOVING = 2


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
