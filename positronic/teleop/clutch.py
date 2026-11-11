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

    With ``controller_axes``, a shift or a turn of the hand counts in the axes of the headset turned by the heading of
    the controller when control came on: its turn about the vertical. So the way the operator faces does not count,
    and a tilt of the controller does not count either: up stays up. ``controller_axes`` puts the axes of the
    controller in the frame of the headset, as the operator holds it. With None, a shift right in the room is a shift
    right of the robot, whichever way the controller points.
    """

    def __init__(
        self, operator_position: geom.Transform3D | None, controller_axes: geom.Rotation | None = CONTROLLER_AXES
    ):
        self._operator_position = operator_position
        self._controller_axes = controller_axes
        # The axes of the moves of the hand in the robot frame
        self._axes = geom.Rotation.identity
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
        self._axes = operator.rotation
        if self._controller_axes is not None:
            # How far the controller is turned from the way the operator holds it facing ahead, in the robot frame
            turned = operator.rotation * self._controller_axes * hand_pos.rotation.inv * operator.rotation.inv
            self._axes = geom.Rotation.from_rotvec(np.array([0.0, 0.0, _angle_about_z(turned)])) * operator.rotation
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

        axes, start = self._axes, self._hand_start
        shift = axes(tracker_pos.translation - start.translation)
        turn = axes * tracker_pos.rotation * start.rotation.inv * axes.inv
        return geom.Transform3D(self._robot_start.translation + shift, turn * self._robot_start.rotation)


def _angle_about_z(turn: geom.Rotation) -> float:
    """The angle of ``turn`` about the z axis, when ``turn`` is a tilt after a turn about z."""
    w, _, _, z = turn.as_quat
    # A half turn about an axis in the xy plane has no turn about z
    return 0.0 if np.hypot(w, z) < 1e-9 else float(2.0 * np.arctan2(z, w))


class OperatorPosition(Enum):
    # map xyz -> zxy
    FRONT = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
    # map xyz -> zxy + flip x and y
    BACK = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))

    def __reduce_ex__(self, proto: object) -> tuple[Any, ...]:
        # A member pickles by its name: a ``Transform3D`` value has no equality to look the member up by.
        return getattr, (type(self), self.name)
