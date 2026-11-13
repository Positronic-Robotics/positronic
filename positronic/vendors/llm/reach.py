"""Checks that a hand can follow its part of a path before the path plays."""

from collections.abc import Mapping, Sequence
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
    """The arms whose hands it checks, and why a hand cannot follow its poses."""

    arms: frozenset[str | None]

    def refusal(self, arm: str | None, joints: np.ndarray, poses: Sequence[geom.Transform3D]) -> tuple[int, str] | None:
        """The index of the first pose of ``poses`` that the hand of ``arm`` cannot follow in order from ``joints``,
        and why; or ``None``."""
        ...

    def meta(self) -> dict[str, Any]: ...


class YamReach:
    """The IK of the YAM driver on the hand poses of each arm, with the base of each arm at ``mounts[arm]``.

    A pose is out of reach when the driver's IK cannot meet it inside the joint ranges. A pose is also refused when a
    joint turns more than ``max_joint_step`` (rad) from the pose before it: the driver would then swing the arm to
    another posture.
    """

    # TODO: the station owns the mounts (static meta) and the joint ranges (its chain reports them). They reach no
    # policy, so the box config repeats the mounts and the check uses the MJCF ranges, which equal the ranges of a
    # chain on i2rt's own limits. Read both from the station at the next pin roll of the platform.
    def __init__(self, mounts: Mapping[str | None, Sequence[float]], max_joint_step: float = 0.5):
        check_finite_positive(max_joint_step, 'max_joint_step')
        for arm, mount in mounts.items():
            if len(mount) != 3 or not np.all(np.isfinite(mount)):
                raise ValueError(f'The mount of arm {arm} must be a finite xyz position')
        self.arms = frozenset(mounts)
        self._bases = {arm: geom.Transform3D(np.asarray(mount, dtype=float)) for arm, mount in mounts.items()}
        self._max_joint_step = max_joint_step
        self._kinematics = YamKinematics()

    def meta(self) -> dict[str, Any]:
        mounts = {str(arm): base.translation.tolist() for arm, base in self._bases.items()}
        return {'type': 'yam', 'mounts': mounts, 'max_joint_step': self._max_joint_step}

    def _limit_refusal(self, target: geom.Transform3D, joints: np.ndarray) -> str:
        kinematics = self._kinematics
        free = kinematics.ik(target, joints, np.full(len(joints), -np.inf), np.full(len(joints), np.inf))
        if free is None:
            return 'the pose is beyond the arm. Reach it with a nearer position'
        lower, upper = kinematics.lower, kinematics.upper
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

    def refusal(self, arm: str | None, joints: np.ndarray, poses: Sequence[geom.Transform3D]) -> tuple[int, str] | None:
        base = self._bases[arm].inv
        kinematics = self._kinematics
        for index, pose in enumerate(poses):
            target = base * pose
            solution = kinematics.ik(target, joints, kinematics.lower, kinematics.upper)
            if solution is None:
                return index, self._limit_refusal(target, joints)
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
