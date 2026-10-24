import logging
from enum import Enum
from typing import Any

from positronic import geom


class Tracker:
    """Map a hand pose to a robot pose while control is on. Control starts from the robot pose, so it does not jump.

    With no ``operator_position``, control is always on and the hand pose is the robot pose (UMI mode).
    """

    def __init__(self, operator_position: geom.Transform3D | None):
        self._operator_position = operator_position
        self._offset = geom.Transform3D()
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
