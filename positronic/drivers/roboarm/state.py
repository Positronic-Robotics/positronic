"""The arm state a driver ships to the rest of a run."""

from typing import Any

import numpy as np

import pimm
from positronic import geom

from . import RobotStatus, State


class PackedState(State, pimm.shared_memory.NumpySMAdapter):
    """An arm's joints, their velocities, the end effector pose and the arm's status, in one float32 array.

    Shared memory carries a fixed-size payload, so the state is packed rather than shipped as fields. Every
    arm this drives states the same four things and differs only in how many joints it has.
    """

    def __init__(self, n_joints: int):
        self.n_joints = n_joints
        self._q = slice(0, n_joints)
        self._dq = slice(n_joints, 2 * n_joints)
        self._ee = slice(2 * n_joints, 2 * n_joints + 7)
        self._status = 2 * n_joints + 7
        super().__init__(shape=(self._status + 1,), dtype=np.dtype(np.float32))

    def instantiation_params(self) -> tuple[Any, ...]:
        return (self.n_joints,)

    @property
    def q(self) -> np.ndarray:
        return self.array[self._q].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[self._dq].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        pose = self.array[self._ee].copy()
        return geom.Transform3D(pose[:3], geom.Rotation.from_quat(pose[3:7]))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[self._status]))

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D, status: RobotStatus) -> None:
        self.array[self._q] = q
        self.array[self._dq] = dq
        self.array[self._ee.start : self._ee.start + 3] = ee_pose.translation
        self.array[self._ee.start + 3 : self._ee.stop] = ee_pose.rotation.as_quat
        self.array[self._status] = status.value
