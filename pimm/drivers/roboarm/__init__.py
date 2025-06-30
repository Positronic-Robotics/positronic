from dataclasses import dataclass
from enum import Enum

import numpy as np

import geom


class CommandType(Enum):
    """
    Different types of commands that can be sent to the robot.
    Robot implementation can choose to ignore some of them.
    """
    RESET = 0
    CARTESIAN_MOVE = 1


class RobotStatus(Enum):
    """Different statuses that the robot can be in."""
    AVAILABLE = 0
    RESETTING = 1


@dataclass
class Command:
    """Command to be sent to the robot. Robot implementation may support only some of the types."""
    type: CommandType
    # For performance reasons, commands are packed into a single array.
    # All implementations must rely on the same interpretation of the array.
    # RESET:
    #   The values are ignored.
    # CARTESIAN_MOVE:
    #   value[0:3] - xyz translation
    #   value[3:7] - quaternion in wxyz format
    value: np.array

    @staticmethod
    def move_to(pos: geom.Transform3D):
        return Command(CommandType.CARTESIAN_MOVE, np.concatenate([pos.translation, pos.rotation.as_quat]))

    @staticmethod
    def reset():
        return Command(CommandType.RESET, np.zeros(7))


# TODO: Add forces when we are ready
class State:
    """
    State of the robot.

    Robots must emit this as often as possible, hence they most performant connections
    must be used to emit and read this state.
    """
    # For performance reasons, the state is packed into a single array.
    # [0:3] - xyz translation
    # [3:7] - quaternion in wxyz format
    # [7:14] - joints positions
    # [14] - status
    _values: np.array

    def __init__(self):
        # TODO: Support different number of joints.
        POSITION_DIM, JOINTS_DIM, STATUS_DIM = 7, 7, 1
        self._values = np.zeros(POSITION_DIM + JOINTS_DIM + STATUS_DIM)

    @property
    def position(self) -> geom.Transform3D:
        """Position of the robot's end-effector."""
        return geom.Transform3D(self._values[:3], geom.Rotation.from_quat(self._values[3:7]))

    @property
    def joints(self) -> np.array:
        """Joints positions of the robot."""
        return self._values[7:14]

    @property
    def status(self) -> RobotStatus:
        """Robot status."""
        return RobotStatus.AVAILABLE if self._values[14] == 0 else RobotStatus.RESETTING

    def _start_reset(self):
        self._values[14] = 1

    def _finish_reset(self):
        self._values[14] = 0
