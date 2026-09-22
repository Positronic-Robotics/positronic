"""The state of one i2rt YAM chain: six joints, the end-effector pose and the status.

The real driver and every simulator of the arm publish it, so it must not import i2rt.
"""

from typing import Any

import numpy as np

import pimm
from positronic import geom

from . import RobotStatus, State


class YamState(State, pimm.shared_memory.NumpySMAdapter):
    Q_OFFSET = 0
    DQ_OFFSET = Q_OFFSET + 6
    EE_POSE_OFFSET = DQ_OFFSET + 6
    STATUS_OFFSET = EE_POSE_OFFSET + 7
    TOTAL = STATUS_OFFSET + 1

    def __init__(self):
        super().__init__(shape=(YamState.TOTAL,), dtype=np.dtype(np.float32))

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[YamState.Q_OFFSET : YamState.Q_OFFSET + 6].copy()

    @property
    def dq(self) -> np.ndarray:
        return self.array[YamState.DQ_OFFSET : YamState.DQ_OFFSET + 6].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        pose = self.array[YamState.EE_POSE_OFFSET : YamState.EE_POSE_OFFSET + 7].copy()
        return geom.Transform3D(pose[:3], geom.Rotation.from_quat(pose[3:7]))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[YamState.STATUS_OFFSET]))

    def encode(self, q: np.ndarray, dq: np.ndarray, ee_pose: geom.Transform3D, status: RobotStatus):
        self.array[YamState.Q_OFFSET : YamState.Q_OFFSET + 6] = q
        self.array[YamState.DQ_OFFSET : YamState.DQ_OFFSET + 6] = dq
        self.array[YamState.EE_POSE_OFFSET : YamState.EE_POSE_OFFSET + 3] = ee_pose.translation
        self.array[YamState.EE_POSE_OFFSET + 3 : YamState.EE_POSE_OFFSET + 7] = ee_pose.rotation.as_quat
        self.array[YamState.STATUS_OFFSET] = status.value
