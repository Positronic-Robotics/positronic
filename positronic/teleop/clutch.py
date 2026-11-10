import logging
from enum import Enum
from typing import Any

import numpy as np

from positronic import geom

# The axes of a Quest controller in the frame of the headset, as a teleoperator holds it: x to the right, y back to
# the teleoperator, z down. A quarter turn about the x axis; checked by an operator on the CRON stand. The default of
# `Tracker`; a station with another controller or grip sets its own.
CONTROLLER_AXES = geom.Rotation.from_rotvec(np.array([np.pi / 2, 0.0, 0.0]))


class Tracker:
    """Map a hand pose to a robot pose while control is on. Control starts from the robot pose, so it does not jump.

    With no ``operator_position``, control is always on and the hand pose is the robot pose (UMI mode).

    With ``controller_axes``, a shift or a turn of the hand counts in the axes that the controller had when control came
    on, and not in the axes of the headset, so the way the operator faces does not count. ``controller_axes`` puts the
    axes of the controller in the frame of the headset. With None, a shift right in the room is a shift right of the
    robot, whichever way the controller points.
    """

    def __init__(
        self, operator_position: geom.Transform3D | None, controller_axes: geom.Rotation | None = CONTROLLER_AXES
    ):
        self._operator_position = operator_position
        self._controller_axes = controller_axes
        self._offset = geom.Transform3D()
        self._robot_start = geom.Transform3D()
        self._hand_start = geom.Transform3D()
        self.on = self.is_umi_mode

    @property
    def is_umi_mode(self):
        return self._operator_position is None

    def turn_on(self, robot_pos: geom.Transform3D, hand_pos: geom.Transform3D):
        operator = self._operator_position
        if operator is None:
            logging.info('Ignoring tracking on/off in UMI mode')
            return

        self.on = True
        logging.info('Starting tracking')
        teleop_t = operator * hand_pos * operator.inv
        self._offset = geom.Transform3D(
            -teleop_t.translation + robot_pos.translation, teleop_t.rotation.inv * robot_pos.rotation
        )
        # Copies: a driver can hand out views of the state that it writes again at each tick
        self._robot_start, self._hand_start = robot_pos.copy(), hand_pos.copy()

    def turn_off(self):
        if self.is_umi_mode:
            logging.info('Ignoring tracking on/off in UMI mode')
            return
        self.on = False
        logging.info('Stopped tracking')

    def update(self, tracker_pos: geom.Transform3D):
        operator = self._operator_position
        if operator is None:
            return tracker_pos

        if self._controller_axes is not None:
            moved = self._hand_start.inv * tracker_pos
            axes = operator.rotation * self._controller_axes
            turn = axes * moved.rotation * axes.inv
            return geom.Transform3D(
                self._robot_start.translation + axes(moved.translation), turn * self._robot_start.rotation
            )
        teleop_t = operator * tracker_pos * operator.inv
        return geom.Transform3D(
            teleop_t.translation + self._offset.translation, teleop_t.rotation * self._offset.rotation
        )


class OperatorPosition(Enum):
    # map xyz -> zxy
    FRONT = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
    # map xyz -> zxy + flip x and y
    BACK = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))

    def __reduce_ex__(self, proto: object) -> tuple[Any, ...]:
        # A member pickles by its name: a ``Transform3D`` value has no equality to look the member up by.
        return getattr, (type(self), self.name)
