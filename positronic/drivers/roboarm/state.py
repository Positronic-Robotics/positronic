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
        self._q_slice = slice(0, n_joints)
        self._dq_slice = slice(n_joints, 2 * n_joints)
        self._ee_slice = slice(2 * n_joints, 2 * n_joints + 7)
        self._status_index = 2 * n_joints + 7
        super().__init__(shape=(self._status_index + 1,), dtype=np.dtype(np.float32))

    @property
    def n_joints(self) -> int:
        """How many joints the layout carries. Read off the layout, so the two cannot disagree."""
        return self._q_slice.stop - self._q_slice.start

    def instantiation_params(self) -> tuple[Any, ...]:
        return (self.n_joints,)

    @property
    def q(self) -> np.ndarray:
        return self.array[self._q_slice].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[self._dq_slice].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        pose = self.array[self._ee_slice].copy()
        return geom.Transform3D(pose[:3], geom.Rotation.from_quat(pose[3:7]))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[self._status_index]))

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D, status: RobotStatus) -> None:
        self.array[self._q_slice] = q
        self.array[self._dq_slice] = dq
        self.array[self._ee_slice.start : self._ee_slice.start + 3] = ee_pose.translation
        self.array[self._ee_slice.start + 3 : self._ee_slice.stop] = ee_pose.rotation.as_quat
        self.array[self._status_index] = status.value
