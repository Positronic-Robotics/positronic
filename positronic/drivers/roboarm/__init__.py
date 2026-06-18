"""Robot arm drivers package.

This package provides drivers for various robot arms including Franka, Kinova, and SO-101.
"""

import xml.etree.ElementTree as ET
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


@lru_cache(maxsize=1)
def bundled_panda_model() -> dict:
    """The bundled simulated Franka panda (arm + hand) for the 3D viewer and offline IK: the panda
    URDF, its collision meshes, the joint names, the ``end_effector`` control frame — the grasp site
    where the sim measures ``robot_state.ee_pose`` — and the ``gripper`` spec that slides the fingers
    from the recorded ``grip`` signal. Supplied to sim datasets by ``SIM_ROBOT_TRANSFORM``, mirroring
    how ``bundled_franka_model`` backfills the real arm; its ``end_effector`` sits at the simulated
    grasp site rather than the FR3 physical flange.
    """
    urdf_path = Path(__file__).resolve().parents[2] / 'assets' / 'mujoco' / 'panda.urdf'
    urdf = urdf_path.read_text()
    mesh_dir = urdf_path.parent / 'assets'
    mesh_files = {mesh.get('filename') for mesh in ET.fromstring(urdf).iter('mesh')}
    return {
        'urdf': urdf,
        'meshes': {name: (mesh_dir / name).read_bytes() for name in sorted(mesh_files)},
        'joint_names': [f'joint{i}' for i in range(1, 8)],
        'control_frame': 'end_effector',
        # ``grip`` is recorded in [0, 1] (closed→open); each finger slides 0..0.04 m along its axis.
        'gripper': {'signal': 'grip', 'joints': ['finger_joint1', 'finger_joint2'], 'travel': 0.04},
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


__all__ = ['RobotStatus', 'State', 'bundled_franka_model', 'bundled_panda_model', 'command']
