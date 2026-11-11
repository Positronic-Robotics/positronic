import logging
from enum import Enum
from typing import Any

import numpy as np

from positronic import geom


class Tracker:
    """Map a hand pose to a robot pose while control is on. Control starts from the robot pose, so it does not jump.

    With no ``operator_position``, control is always on and the hand pose is the robot pose (UMI mode).
    The robot turns about its own z axis ``twist_gain`` times as far as the hand turns about that axis.

    With ``hand_axes``, a shift or a turn of the hand counts in the axes the hand had when control came on, not in
    the axes of the headset, and ``hand_axes`` turns those axes into the robot frame. So the way the operator faces
    does not count.

    With ``hand_axes`` and ``heading_only``, only the heading of the hand when control came on counts: its turn about
    the robot z axis. The tilt of the hand does not count, so up stays up.
    """

    def __init__(
        self,
        operator_position: geom.Transform3D | None,
        twist_gain: float = 1.0,
        hand_axes: geom.Rotation | None = None,
        heading_only: bool = False,
    ):
        self._operator_position = operator_position
        self._twist_gain = twist_gain
        self._hand_axes = hand_axes
        self._heading_only = heading_only
        self._offset = geom.Transform3D()
        self._robot_start = geom.Transform3D()
        self._hand_start = geom.Transform3D()
        # The axes of the headset in the robot frame, turned by the heading of the hand when control came on
        self._heading_axes = geom.Rotation.identity
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
        if self._hand_axes is not None:
            # The turn that the hand axes add to the axes of the headset, in the robot frame
            added = self._hand_axes * hand_pos.rotation.inv * operator.rotation.inv
            heading = geom.Rotation.from_rotvec(np.array([0.0, 0.0, _angle_about_z(added)]))
            self._heading_axes = heading * operator.rotation

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

        start = self._robot_start
        if self._hand_axes is not None and self._heading_only:
            axes = self._heading_axes
            shift = axes(tracker_pos.translation - self._hand_start.translation)
            world_turn = axes * tracker_pos.rotation * self._hand_start.rotation.inv * axes.inv
            translation = start.translation + shift
        elif self._hand_axes is not None:
            moved = self._hand_start.inv * tracker_pos
            # The shift and the turn of the hand in its start axes, as a shift and a turn in the robot frame
            axes = self._hand_axes
            shift = axes(moved.translation)
            world_turn = axes * moved.rotation * axes.inv
            translation = start.translation + shift
        else:
            teleop_t = operator * tracker_pos * operator.inv
            world_turn = teleop_t.rotation * self._offset.rotation * start.rotation.inv
            translation = teleop_t.translation + self._offset.translation
        # The turn in the frame of the robot pose when control came on
        turn = start.rotation.inv * world_turn * start.rotation
        return geom.Transform3D(translation, start.rotation * _scaled_twist(turn, self._twist_gain))


def _scaled_twist(turn: geom.Rotation, gain: float) -> geom.Rotation:
    """``turn`` with its part about the z axis multiplied by ``gain``."""
    if gain == 1.0:
        return turn
    return turn * geom.Rotation.from_rotvec(np.array([0.0, 0.0, (gain - 1.0) * _angle_about_z(turn)]))


def _angle_about_z(turn: geom.Rotation) -> float:
    """The angle of the part of ``turn`` about the z axis, from -pi to pi."""
    w, _, _, z = turn.as_quat
    if np.hypot(w, z) < 1e-9:  # a half turn about an axis in the xy plane has no part about z
        return 0.0
    # q and -q are one turn: wrap to the short way round
    return float((2.0 * np.arctan2(z, w) + np.pi) % (2.0 * np.pi) - np.pi)


class OperatorPosition(Enum):
    # map xyz -> zxy
    FRONT = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
    # map xyz -> zxy + flip x and y
    BACK = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))

    def __reduce_ex__(self, proto: object) -> tuple[Any, ...]:
        # A member pickles by its name: a ``Transform3D`` value has no equality to look the member up by.
        return getattr, (type(self), self.name)
