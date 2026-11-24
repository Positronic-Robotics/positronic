"""Checks that a hand can follow its part of a path before the path plays."""

from collections.abc import Sequence
from typing import Any, NamedTuple, Protocol

import configuronic as cfn
import numpy as np
from pydantic import BaseModel

from positronic import geom
from positronic.drivers.roboarm.yam.kinematics import YamKinematics, within_ik_tolerance
from positronic.utils import check_finite_positive


class Box(NamedTuple):
    """An axis-aligned box of hand positions, in metres."""

    low: tuple[float, ...]
    high: tuple[float, ...]

    def holds(self, position: np.ndarray) -> bool:
        return bool(np.all(position >= self.low) and np.all(position <= self.high))

    def crosses(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Whether the straight segment from ``start`` to ``end`` has a point in the box."""
        low, high, step = np.asarray(self.low), np.asarray(self.high), end - start
        moving = step != 0
        if np.any(~moving & ((start < low) | (start > high))):
            return False
        near = (low[moving] - start[moving]) / step[moving]
        far = (high[moving] - start[moving]) / step[moving]
        enter = max(0.0, *np.minimum(near, far))
        leave = min(1.0, *np.maximum(near, far))
        return enter <= leave


class ReachReport(BaseModel):
    """What the IK of an arm says about its measured joints and about the previous target of its hand.

    ``target_refusal`` says why the arm cannot reach that target, as a refusal of the target does.
    """

    joints_at_limit: list[str]
    target_in_reach: bool | None = None
    target_refusal: str | None = None


class Reach(Protocol):
    """Why a hand cannot follow its poses, and where the arm base of a hand is."""

    def refusal(
        self,
        joints: np.ndarray,
        hand: geom.Transform3D,
        held: geom.Transform3D | None,
        poses: Sequence[geom.Transform3D],
    ) -> tuple[int, str] | None:
        """The index of the first pose of ``poses`` that a hand at ``hand`` with its arm at ``joints`` cannot follow
        in order, and why; or ``None``. ``held`` is the target that the hand holds when the hand is within the
        tracking error of it, else ``None``."""
        ...

    def base(self, joints: np.ndarray, hand: geom.Transform3D) -> geom.Transform3D:
        """The pose of the arm base, in the frame of ``hand``, for an arm at ``joints``."""
        ...

    def fact(self) -> str:
        """What the prompt states about where a hand can go, in the frame of its arm base."""
        ...

    def report(self, joints: np.ndarray, hand: geom.Transform3D, target: geom.Transform3D | None) -> ReachReport:
        """The joints of an arm at ``joints`` that sit at a limit, and whether the arm can reach ``target``."""
        ...

    def meta(self) -> dict[str, Any]: ...


class YamReach:
    """The IK of the YAM driver on the hand poses of an arm.

    The arm base is where the measured hand and joints put it, since the station reports the hand at its base pose
    times the FK of its joints. A pose is out of reach when the driver's IK cannot meet it inside the joint ranges,
    unless the hand is already there and has not moved yet. A pose is also refused when a joint turns more than
    ``max_joint_step`` (rad) from the pose before it: the driver would then swing the arm to another posture.
    """

    # Offsets from the base, in metres, where the IK finds joints inside the ranges at every y from the base to `INWARD`
    # toward the other arm. A hand that points straight down reaches up to each height, as far as each distance
    # forward; joint 4, the wrist pitch, sets it.
    TOP_DOWN_CEILING = ((0.10, 0.19), (0.25, 0.15), (0.30, 0.11), (0.35, 0.05))
    # A hand 45 degrees down, between these distances forward and up to this height.
    TILTED_FROM, TILTED_TO, TILTED_CEILING = 0.15, 0.45, 0.20
    # A horizontal approach needs at least this distance forward and this height.
    SIDE_FROM, SIDE_ABOVE = 0.25, 0.14
    # The farthest a horizontal hand reaches, at each height: in line with the base, and `INWARD` toward the other arm.
    SIDE_TO = ((0.29, 0.70, 0.62), (0.39, 0.64, 0.54))
    INWARD = 0.30
    # A measured joint this close to a range bound, in radians, sits at the bound.
    LIMIT_MARGIN = 0.02

    # TODO: the station owns the joint ranges (its chain reports them), and they reach no policy. The check uses the
    # MJCF ranges, which equal the ranges of a chain on i2rt's own limits. Read them from the station at the next pin
    # roll of the platform.
    def __init__(self, max_joint_step: float = 0.5):
        check_finite_positive(max_joint_step, 'max_joint_step')
        self._max_joint_step = max_joint_step
        self._kinematics = YamKinematics()

    def meta(self) -> dict[str, Any]:
        return {'type': 'yam', 'max_joint_step': self._max_joint_step}

    def base(self, joints: np.ndarray, hand: geom.Transform3D) -> geom.Transform3D:
        return hand * self._kinematics.fk(joints).inv

    def fact(self) -> str:
        ceilings = ', '.join(f'z {up:.2f} up to x {ahead:.2f}' for ahead, up in self.TOP_DOWN_CEILING)
        (up, inline, inward), *higher = self.SIDE_TO
        far = (
            f'A horizontal hand reaches at most x {inline:.2f} in line with its base and x {inward:.2f} at '
            f'{self.INWARD:.2f} m toward the other arm, up to z {up:.2f}'
        )
        far += ''.join(f'; at z {up:.2f}, at most x {inline:.2f} and x {inward:.2f}' for up, inline, inward in higher)
        return (
            f'In the frame of its arm base, a hand that points straight down (roll 3.1416, pitch 0) reaches at most '
            f'{ceilings}, and lower further out. Above that the wrist cannot bend far enough, so hover lower or tilt '
            'the hand. Its yaw works from 0.52 rad below to 3.67 rad above the direction from the base to the hand; '
            'yaw 1.5708 works for every arm. A hand 45 degrees down (roll 2.3562, pitch 0, yaw 1.5708) works from '
            f'x {self.TILTED_FROM:.2f} to x {self.TILTED_TO:.2f}, up to z {self.TILTED_CEILING:.2f}. A horizontal '
            f'approach (roll 1.5708) needs x of at least {self.SIDE_FROM:.2f} and z of at least {self.SIDE_ABOVE:.2f}. '
            f'{far}. Each of these bounds holds up to {self.INWARD:.2f} m toward the other arm; past that the hand '
            'reaches less far. '
        )

    def report(self, joints: np.ndarray, hand: geom.Transform3D, target: geom.Transform3D | None) -> ReachReport:
        kinematics = self._kinematics
        lower, upper = kinematics.lower, kinematics.upper
        at_limit = [
            f'joint {joint + 1} is at its {side} limit ({round(limit, 2) + 0.0:.2f} rad)'
            for joint in range(len(joints))
            for side, limit, near in (
                ('lower', lower[joint], joints[joint] <= lower[joint] + self.LIMIT_MARGIN),
                ('upper', upper[joint], joints[joint] >= upper[joint] - self.LIMIT_MARGIN),
            )
            if near
        ]
        if target is None:
            return ReachReport(joints_at_limit=at_limit)
        goal = self.base(joints, hand).inv * target
        if within_ik_tolerance(target, hand) or kinematics.ik(goal, joints, lower, upper) is not None:
            return ReachReport(joints_at_limit=at_limit, target_in_reach=True)
        refusal = self._limit_refusal(goal, joints, lower, upper)
        return ReachReport(joints_at_limit=at_limit, target_in_reach=False, target_refusal=refusal)

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
        # Adding 0.0 prints a limit of -1e-15 rad as 0.00, not -0.00.
        return (
            f'joint {joint + 1} is at its {side} limit ({round(limit, 2) + 0.0:.2f} rad). '
            'Reach it with another orientation or a nearer position'
        )

    @staticmethod
    def _already_there(pose: geom.Transform3D, hand: geom.Transform3D, held: geom.Transform3D | None) -> bool:
        """Whether ``pose`` is where the hand already is: at ``hand``, or on the way from ``hand`` to ``held``."""
        if held is None:
            return within_ik_tolerance(pose, hand)
        gap = held.translation - hand.translation
        if (length := float(gap @ gap)) > 0:
            fraction = float(np.clip((pose.translation - hand.translation) @ gap / length, 0, 1))
        elif (turn := (hand.rotation.inv * held.rotation).angle) > 0:
            fraction = float(np.clip((hand.rotation.inv * pose.rotation).angle / turn, 0, 1))
        else:
            fraction = 0.0
        return within_ik_tolerance(pose, hand.interpolate(held, fraction))

    def refusal(
        self,
        joints: np.ndarray,
        hand: geom.Transform3D,
        held: geom.Transform3D | None,
        poses: Sequence[geom.Transform3D],
    ) -> tuple[int, str] | None:
        kinematics = self._kinematics
        base = self.base(joints, hand)
        lower, upper = kinematics.lower, kinematics.upper
        staying = True
        for index, pose in enumerate(poses):
            target = base.inv * pose
            already_there = staying and self._already_there(pose, hand, held)
            solution = kinematics.ik(target, joints, lower, upper)
            if solution is None:
                # The driver keeps its last target when the IK fails. Until the hand moves, that target holds the hand
                # where it is; after, the arm would stop short.
                if already_there:
                    continue
                return index, self._limit_refusal(target, joints, lower, upper)
            step = np.abs(solution - joints)
            joint = int(np.argmax(step))
            if step[joint] > self._max_joint_step:
                return index, (
                    f'joint {joint + 1} would turn {step[joint]:.2f} rad at once. '
                    'Reach it with another orientation or through a nearer pose'
                )
            # The driver solves each sample from the measured joints, which stay put while the hand stays.
            if not already_there:
                staying, joints = False, solution
        return None


yam_reach = cfn.Config(YamReach)
