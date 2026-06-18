"""Robot arm drivers package.

This package provides drivers for various robot arms including Franka, Kinova, and SO-101.
"""

from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from pathlib import Path

import numpy as np

from positronic import geom

# Import command submodule to make it accessible as roboarm.command
from . import command


@lru_cache(maxsize=1)
def bundled_franka_model() -> dict:
    """The bundled Franka arm model for the 3D viewer: the FR3 URDF, its collision meshes, and the
    canonical joint names and control frame.

    Used where no live robot reports its own model — the MuJoCo sim and offline dataset transforms.
    The FR3 arm shares the simulator panda's 7-DOF kinematics exactly, so the rendered robot matches
    the simulated one (verified by the URDF/MJCF kinematics test).
    """
    here = Path(__file__).resolve()
    meshes = (here.parents[2] / 'assets' / 'fr3_collision').iterdir()
    return {
        'urdf': (here.parent / 'fr3.urdf').read_text(),
        'meshes': {f.name: f.read_bytes() for f in sorted(meshes) if f.suffix == '.stl'},
        'joint_names': [f'joint{i}' for i in range(1, 8)],
        'control_frame': 'end_effector',
    }


class RobotStatus(Enum):
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


__all__ = ['RobotStatus', 'State', 'bundled_franka_model', 'command']
