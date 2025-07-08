from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

import geom

from pimm.drivers.roboarm.command import CartesianMove, Reset



class RobotStatus(Enum):
    """Different statuses that the robot can be in."""
    AVAILABLE = 0
    RESETTING = 1


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
