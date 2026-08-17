"""Fakes for the ``roboarm`` interfaces, for tests that drive an arm without a driver."""

import numpy as np

from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State


class FakeRobotState(State):
    """A ``State`` over the four values it is given, copied on read so a caller cannot reach the arrays
    a producer keeps emitting from."""

    def __init__(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D, status: RobotStatus) -> None:
        self._q = q
        self._dq = dq
        self._ee_pose = ee_pose
        self._status = status

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    @property
    def dq(self) -> np.ndarray:
        return self._dq.copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        return geom.Transform3D(translation=self._ee_pose.translation.copy(), rotation=self._ee_pose.rotation)

    @property
    def status(self) -> RobotStatus:
        return self._status


def make_robot_state(translation, joints, status: RobotStatus = RobotStatus.AVAILABLE) -> FakeRobotState:
    """A stationary arm at ``translation`` with identity rotation: zero joint velocity throughout."""
    translation = np.asarray(translation, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32)
    pose = geom.Transform3D(translation=translation, rotation=geom.Rotation.identity)
    return FakeRobotState(joints, np.zeros_like(joints), pose, status)
