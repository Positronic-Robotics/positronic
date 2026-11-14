"""Checks that a hand can follow its part of a path before the path plays."""

from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol

import configuronic as cfn
import numpy as np

from positronic import geom
from positronic.drivers.roboarm.yam.kinematics import YamKinematics
from positronic.utils import check_finite_positive


class Box(NamedTuple):
    """An axis-aligned box of hand positions, in metres."""

    low: tuple[float, ...]
    high: tuple[float, ...]

    def holds(self, position: np.ndarray) -> bool:
        return bool(np.all(position >= self.low) and np.all(position <= self.high))


class Reach(Protocol):
    """Why a hand cannot follow its poses."""

    def refusal(
        self, joints: np.ndarray, hand: geom.Transform3D, poses: Sequence[geom.Transform3D]
    ) -> tuple[int, str] | None:
        """The index of the first pose of ``poses`` that a hand at ``hand`` with its arm at ``joints`` cannot follow
        in order, and why; or ``None``."""
        ...

    def meta(self) -> dict[str, Any]: ...


class YamReach:
    """The IK of the YAM driver on the hand poses of an arm.

    The arm base is where the measured hand and joints put it, since the station reports the hand at its base pose
    times the FK of its joints. A pose is out of reach when the driver's IK cannot meet it inside the joint ranges. A
    pose is also refused when a joint turns more than ``max_joint_step`` (rad) from the pose before it: the driver
    would then swing the arm to another posture.
    """

    # TODO: the station owns the joint ranges (its chain reports them), and they reach no policy. The check uses the
    # MJCF ranges, which equal the ranges of a chain on i2rt's own limits. Read them from the station at the next pin
    # roll of the platform.
    def __init__(self, max_joint_step: float = 0.5):
        check_finite_positive(max_joint_step, 'max_joint_step')
        self._max_joint_step = max_joint_step
        self._kinematics = YamKinematics()

    def meta(self) -> dict[str, Any]:
        return {'type': 'yam', 'max_joint_step': self._max_joint_step}

    def _limit_refusal(self, target: geom.Transform3D, joints: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> str:
        free = self._kinematics.ik(target, joints, np.full(len(joints), -np.inf), np.full(len(joints), np.inf))
        if free is None:
            return 'the pose is beyond the arm. Reach it with a nearer position'
        # A joint a full turn away is the same pose, so each joint counts at the turn nearest to its range.
        turns = free[:, None] + np.array([-2 * np.pi, 0.0, 2 * np.pi])
        excess = np.maximum(lower[:, None] - turns, turns - upper[:, None])
        nearest = turns[np.arange(len(free)), np.argmin(excess, axis=1)]
        joint = int(np.argmax(excess.min(axis=1)))
        if excess[joint].min() <= 0:
            return 'the IK finds no joints inside the ranges from this posture. Reach it through a nearer pose'
        side, limit = ('lower', lower[joint]) if nearest[joint] < lower[joint] else ('upper', upper[joint])
        return (
            f'joint {joint + 1} is at its {side} limit ({limit:.2f} rad). '
            'Reach it with another orientation or a nearer position'
        )

    def refusal(
        self, joints: np.ndarray, hand: geom.Transform3D, poses: Sequence[geom.Transform3D]
    ) -> tuple[int, str] | None:
        kinematics = self._kinematics
        base = (hand * kinematics.fk(joints).inv).inv
        # A joint that sags a little past its range may stay there: the driver holds the arm where it is.
        lower, upper = np.minimum(kinematics.lower, joints), np.maximum(kinematics.upper, joints)
        for index, pose in enumerate(poses):
            target = base * pose
            solution = kinematics.ik(target, joints, lower, upper)
            if solution is None:
                return index, self._limit_refusal(target, joints, lower, upper)
            step = np.abs(solution - joints)
            joint = int(np.argmax(step))
            if step[joint] > self._max_joint_step:
                return index, (
                    f'joint {joint + 1} would turn {step[joint]:.2f} rad at once. '
                    'Reach it with another orientation or through a nearer pose'
                )
            joints = solution
        return None


yam_reach = cfn.Config(YamReach)
