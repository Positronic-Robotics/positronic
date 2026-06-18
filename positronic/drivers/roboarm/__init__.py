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
    """The bundled real Franka arm model for the 3D viewer: the FR3 URDF, its collision meshes, and
    the canonical joint names and control frame.

    Backfills real-robot datasets recorded before they stored their own model. Its ``end_effector``
    is the physical flange frame the driver measures against; the MuJoCo sim measures at a different
    grasp site and supplies its own model via ``bundled_panda_model``.
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
