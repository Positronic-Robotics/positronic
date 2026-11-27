"""The scripted expert of RoboLab's ``BananaInBowlTask``: the helper of a run. It puts the banana in the bowl with one
arm of the two-arm YAM.

The expert reads the pose and the grip of each arm, and the ground truth of the scene under ``GROUND_TRUTH``, as the
RoboLab env server sends it. It sends a pose at ``GRIPPER_SITE`` and a grip on the channels of each arm. At each step it
picks its sub-goal from the state. So it can take over at any pose and grip.

Each element also carries the press fields of a console. With ``takes_over_at_start``, the expert presses Take over at
its second element and drives to the end of the episode, as in round 0 of DAgger. Else it presses Take over when the
source that drives leaves its path, drives to its next sub-goal, and presses Hand back::

    expert = BananaInBowlExpert.for_yam(
        '...', grip_point=grip_point, reach=reach, rest_yaw=rest_yaw, wrist_turn=wrist_turn, park=park
    )
    rollout = Rollout(task, policy, writer, expert)

``positronic.cfg.dagger.yam_banana_in_bowl_expert`` gives the statics of the YAM.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import numpy as np

from positronic import geom, keys
from positronic.drivers.roboarm.models import ee_frames_entry
from positronic.policy.base import Obs, Policy, PolicyRun, Runtime, Step
from positronic.policy.keys import CONTROL, LICENSE, TELEOPERATOR, TYPE
from positronic.policy.takeover import ControlObservation
from positronic.simulator.expert import (
    DOWN,
    ArmStatics,
    Command,
    Console,
    IdleArm,
    apart,
    arm_commands,
    check_truth,
    down,
    frames,
    grip,
    grip_pose,
    object_pose,
    touched,
    wrap,
    yaw,
)
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab import yam_spec

BANANA = 'banana'
BOWL = 'bowl'
# The YCB banana of RoboLab, in its own frame, whose origin is the centre of its bounding box. The grasp point is in the
# thick middle of its body, and the long axis is the direction of the body at that point.
GRASP_POINT = np.array([-0.034, -0.015, 0.0])
LONG_AXIS = np.array([0.11, 0.994, 0.0]) / np.linalg.norm([0.11, 0.994, 0.0])
BOWL_HALF_HEIGHT = 0.0275  # m, of the YCB bowl of RoboLab, whose origin is the centre of its bounding box
# m, the largest distance in the plane of the banana origin from the bowl centre at which the banana is in the bowl. A
# banana that falls on the rim and off it rests farther.
IN_BOWL_M = 0.05
DROP_M = 0.03  # m, the largest distance in the plane of the banana origin from the bowl centre at which it lets go
# m, how far from the bowl centre toward the mount of the arm the expert carries the banana origin
DROP_SHIFT = 0.02
APPROACH_HEIGHT = 0.08  # m, the height of the grip point above the grasp point before the expert goes down
GRASP_HEIGHT = 0.005  # m, the height of the grip point above the grasp point when the grip closes
CARRY_CLEAR = 0.07  # m, the height of the grip point above the bowl rim while it carries the banana
RELEASE_CLEAR = 0.035  # m, the height of the grip point above the bowl rim when it lets go
# The tilt of the approach axis away from the mount, for a goal past the reach of the arm: the tilt per m past it, and
# the largest tilt.
TILT_PER_M = np.radians(60.0) / 0.16
MAX_TILT = np.radians(60.0)
PERIOD_NS = 20_000_000  # the time between two steps of the expert
# How long the grip closes, from the first observed grip past ``OPEN``, before the expert checks that it holds the
# banana.
CLOSE_NS = 600_000_000
OPEN_NS = 600_000_000  # how long the grip opens before the expert checks where the banana is
REACH_M = 0.012  # the largest distance from a sub-goal pose of a grip point that is at it
REACH_RAD = 0.15  # the largest angle from a sub-goal pose of a grip point that is at it
KEEP_M = 0.025  # the largest distance from its path that keeps the arm in ``DOWN`` or ``CLOSE``
KEEP_RAD = 0.3  # the largest angle from its path that keeps the arm in ``DOWN`` or ``CLOSE``
HOLD_M = 0.1  # the largest distance from the grip point of the origin of a banana in the grip
OPEN = 0.1  # the largest observed grip of an open grip
HELD_GRIP = (0.3, 0.9)  # the observed grip of a grip that closes on the banana and stops before its end
STILL_M = 0.003  # m, the largest move of a grip point that stays still
STILL_RAD = 0.03  # rad, the largest turn of a grip point that stays still
STUCK_NS = 500_000_000  # how long a grip point stays still, short of its goal, before the expert gives up on it
CLEAR_M = 0.12  # m, the least distance in y of the grip point of an arm out of the way from the banana
# The poses of the banana and the bowl in the scene file of ``BananaInBowlTask``, from the env origin: the position, and
# the rotation as a quaternion wxyz.
BANANA_AT = ((0.5097271, -0.1158069, 0.0211774), (0.99999917, -0.00061546825, -0.0011157601, -0.0000106499065))
BOWL_AT = ((0.5611013, 0.1526149, 0.0303494), (0.9999998, -0.00038121603, -0.00044255334, 0.00005450022))
LAYOUT_APART_M = 0.2  # m, the least distance in the plane between the banana and the bowl of a layout


class Phase(Enum):
    RISE = 'rise'  # up to the approach height, with the grip open
    ABOVE = 'above'  # to the approach pose above the banana
    DOWN = 'down'  # to the grasp pose
    CLOSE = 'close'  # close the grip at the grasp pose
    LIFT = 'lift'  # up to the carry height with the banana in the grip
    CARRY = 'carry'  # at the carry height, until the banana is above the bowl
    LOWER = 'lower'  # down to the release height above the bowl
    OPEN = 'open'  # open the grip, so that the banana falls into the bowl
    RETREAT = 'retreat'  # up to the carry height with the grip open, while the banana is in the bowl
    PARK = 'park'  # to the park position, while another arm is in the way


# How far each phase is along the path of the expert. A phase of a lower rank after a phase is a step back.
RANK = {
    Phase.PARK: 0,
    Phase.RISE: 0,
    Phase.ABOVE: 1,
    Phase.DOWN: 2,
    Phase.CLOSE: 3,
    Phase.LIFT: 4,
    Phase.CARRY: 5,
    Phase.LOWER: 6,
    Phase.OPEN: 7,
    Phase.RETREAT: 8,
}


@dataclass(frozen=True)
class ReachStatics(ArmStatics):
    """The statics of an arm that say where its grip point can point straight down."""

    reach: float  # m, the distance in the horizontal plane from the mount up to which the grip point can point down
    # The yaw of the grip point, measured from the direction from the mount to it, at which the wrist is in the middle
    # of its range, and the largest turn from it that the wrist can take, in rad.
    rest_yaw: float
    wrist_turn: float
    # m, how far past the approach height ``RISE`` goes, and past the carry height ``LIFT`` goes, so that an arm that
    # sags gets to it. An arm in ``ABOVE`` or ``CARRY`` that sags by more than this goes back to ``RISE`` or ``LIFT``.
    rise_margin: float = 0.03

    def __post_init__(self):
        # Above pi / 2, one of two grasp yaws pi apart is in the range. At pi, the wrist has no limit.
        if not np.pi / 2 < self.wrist_turn <= np.pi:
            raise ValueError(f'The wrist turn is in rad, above pi / 2 and not above pi, not {self.wrist_turn}')


def _holds(pose: geom.Transform3D, grip: float, banana: np.ndarray, contact: bool) -> bool:
    """Whether a grip point at ``pose`` with the observed ``grip`` holds ``banana``, which a finger touches if
    ``contact``."""
    return bool(
        contact and HELD_GRIP[0] <= grip <= HELD_GRIP[1] and np.linalg.norm(pose.translation - banana[:3]) <= HOLD_M
    )


def in_bowl(banana: np.ndarray, bowl: np.ndarray) -> bool:
    """Whether the origin of ``banana`` is near the centre of ``bowl`` and below the middle of the bowl wall. A banana
    that rests on the rim is higher."""
    inside = np.linalg.norm(banana[:2] - bowl[:2]) <= IN_BOWL_M
    return bool(inside and banana[2] <= bowl[2] + BOWL_HALF_HEIGHT / 2)


def grasp_point(banana: np.ndarray) -> np.ndarray:
    """The grasp point of ``banana``, a pose in the layout of ``gt_state``, in the rig's world frame."""
    return geom.Transform3D(banana[:3], geom.Rotation.from_quat(banana[3:]))(GRASP_POINT)


def grasp_yaw(banana: np.ndarray) -> float:
    """A yaw of the grip that closes across the long axis of ``banana``. The yaw plus pi closes the same way."""
    axis = geom.Rotation.from_quat(banana[3:])(LONG_AXIS)
    return float(np.arctan2(axis[1], axis[0]) + np.pi / 2)


def layout(rng: np.random.Generator, move_m: float, turn_rad: float) -> dict[str, Any]:
    """A scene state of ``BananaInBowlTask``, as ``robolab_keys.STATE`` takes it. It moves the banana and the bowl up to
    ``move_m`` in x and in y from the scene file, ``LAYOUT_APART_M`` or more apart, and turns the banana up to
    ``turn_rad`` about z."""
    while True:
        banana = np.add(BANANA_AT[0], [*rng.uniform(-move_m, move_m, 2), 0.0])
        bowl = np.add(BOWL_AT[0], [*rng.uniform(-move_m, move_m, 2), 0.0])
        if np.linalg.norm(banana[:2] - bowl[:2]) > LAYOUT_APART_M:
            break
    spin = geom.Rotation.from_euler([0.0, 0.0, rng.uniform(-turn_rad, turn_rad)])
    quat = (spin * geom.Rotation.from_quat(BANANA_AT[1])).as_quat
    return {
        robolab_keys.STATE_RIGID_OBJECTS: {
            BANANA: {robolab_keys.STATE_ROOT_POSE: [*banana.tolist(), *quat.tolist()]},
            BOWL: {robolab_keys.STATE_ROOT_POSE: [*bowl.tolist(), *BOWL_AT[1]]},
        }
    }


class BananaInBowl:
    """The sub-goal of the arm that puts the banana in the bowl, which it picks from the state at each step, and the
    pose that it commands. The grasp points down, with the grip across the long axis of the banana."""

    def __init__(self, start: geom.Transform3D, statics: ReachStatics, now_ns: int):
        """``start`` is the pose of the grip point."""
        self._statics = statics
        self.phase: Phase | None = None
        self.goal = start  # the pose of the sub-goal of the phase
        self._command = Command(start, now_ns)
        self._phase_ns = now_ns
        # The time at which the observed grip first got past ``OPEN`` in ``CLOSE``, or ``None``.
        self._closing_ns: int | None = None
        self._yaw = wrap(yaw(start.rotation) - self._heading(start.translation))  # from the heading, before the tilt
        self._still = start, now_ns  # a pose of the grip point, and the time from which the grip point stays near it
        self._turned = False  # whether the arm took the other grasp rotation as it stayed still, since the last grasp

    def _heading(self, point: np.ndarray) -> float:
        """The angle about z of the direction from the mount to ``point``."""
        away = point[:2] - self._statics.mount[:2]
        return float(np.arctan2(away[1], away[0]))

    def _tilted(self, turn: float, goal: np.ndarray) -> geom.Rotation:
        """The rotation of a grip point at ``goal`` with the yaw ``turn`` from the heading of ``goal``, tilted away
        from the mount past the reach."""
        away = np.linalg.norm(goal[:2] - self._statics.mount[:2])
        tilt = float(np.clip((away - self._statics.reach) * TILT_PER_M, 0.0, MAX_TILT))
        return (
            geom.Rotation.from_euler([0.0, 0.0, self._heading(goal)])
            * geom.Rotation.from_euler([0.0, -tilt, 0.0])
            * geom.Rotation.from_euler([0.0, 0.0, turn])
            * DOWN
        )

    def _grasp(self, banana: np.ndarray) -> geom.Rotation:
        point = grasp_point(banana)
        across = grasp_yaw(banana) - self._heading(point)
        near, far = sorted((across, across + np.pi), key=lambda turn: abs(wrap(turn - self._yaw)))
        rests = abs(wrap(near - self._statics.rest_yaw)) <= self._statics.wrist_turn
        self._yaw = near if rests or self._turned else far
        return self._tilted(self._yaw, point)

    def _drop(self, pose: geom.Transform3D, banana: np.ndarray, bowl: np.ndarray) -> np.ndarray:
        """Where the grip point carries the banana: the point at the carry height that puts the banana origin above
        the bowl centre, ``DROP_SHIFT`` toward the mount."""
        toward_mount = self._statics.mount[:2] - bowl[:2]
        target = bowl[:2] + DROP_SHIFT * toward_mount / np.linalg.norm(toward_mount)
        xy = target - (banana[:2] - pose.translation[:2])
        return np.append(xy, bowl[2] + BOWL_HALF_HEIGHT + CARRY_CLEAR)

    def _pick(
        self,
        pose: geom.Transform3D,
        grip: float,
        banana: np.ndarray,
        bowl: np.ndarray,
        contact: bool,
        clear: bool,
        now_ns: int,
    ) -> Phase:
        if _holds(pose, grip, banana, contact):
            release_z = bowl[2] + BOWL_HALF_HEIGHT + RELEASE_CLEAR
            if np.linalg.norm(banana[:2] - bowl[:2]) <= DROP_M:
                # An arm that stays still above the release height cannot get lower, most often as the banana touches
                # the bowl.
                low = pose.translation[2] <= release_z + REACH_M
                stuck = self.phase in (Phase.LOWER, Phase.OPEN) and now_ns - self._still[1] >= STUCK_NS
                return Phase.OPEN if low or stuck else Phase.LOWER
            carry_z = bowl[2] + BOWL_HALF_HEIGHT + CARRY_CLEAR
            if pose.translation[2] >= carry_z - (self._statics.rise_margin if self.phase == Phase.CARRY else REACH_M):
                return Phase.CARRY
            return Phase.LIFT
        if self.phase == Phase.OPEN and now_ns - self._phase_ns < OPEN_NS:
            return Phase.OPEN
        if in_bowl(banana, bowl):
            return Phase.RETREAT
        if not clear:
            return Phase.PARK
        offset = pose.translation - grasp_point(banana) - [0.0, 0.0, GRASP_HEIGHT]
        # An arm that stays still near the approach pose cannot get closer to it, most often at the edge of its reach.
        keep = self.phase in (Phase.DOWN, Phase.CLOSE) or now_ns - self._still[1] >= STUCK_NS
        limit_m, limit_rad = (KEEP_M, KEEP_RAD) if keep else (REACH_M, REACH_RAD)
        if self.phase == Phase.CLOSE and grip > OPEN and np.linalg.norm(offset) <= limit_m:
            self._closing_ns = now_ns if self._closing_ns is None else self._closing_ns
            if now_ns - self._closing_ns < CLOSE_NS:
                return Phase.CLOSE
        turn = (pose.rotation.inv * self._grasp(banana)).angle
        if grip <= OPEN and np.linalg.norm(offset[:2]) <= limit_m and turn <= limit_rad:
            if abs(offset[2]) <= REACH_M:
                return Phase.CLOSE
            if 0.0 < offset[2] <= APPROACH_HEIGHT + REACH_M:
                return Phase.DOWN
        if offset[2] < APPROACH_HEIGHT - (self._statics.rise_margin if self.phase == Phase.ABOVE else REACH_M):
            return Phase.RISE
        return Phase.ABOVE

    def _goal(self, pose: geom.Transform3D, banana: np.ndarray, bowl: np.ndarray) -> geom.Transform3D:
        here = self._command.pose
        carry_z = bowl[2] + BOWL_HALF_HEIGHT + CARRY_CLEAR
        match self.phase:
            case Phase.LIFT | Phase.RETREAT:
                return geom.Transform3D(
                    np.append(here.translation[:2], carry_z + self._statics.rise_margin), here.rotation
                )
            case Phase.CARRY:
                drop = self._drop(pose, banana, bowl)
                return geom.Transform3D(drop, self._tilted(self._yaw, drop))
            case Phase.LOWER:
                release = np.append(here.translation[:2], bowl[2] + BOWL_HALF_HEIGHT + RELEASE_CLEAR)
                return geom.Transform3D(release, here.rotation)
            case Phase.OPEN:
                return here
            case Phase.PARK:
                return geom.Transform3D(self._statics.mount + self._statics.park, down(here.rotation))
        point = grasp_point(banana) + [0.0, 0.0, GRASP_HEIGHT]
        if self.phase == Phase.RISE:
            height = point[2] + APPROACH_HEIGHT + self._statics.rise_margin
            return geom.Transform3D(np.append(here.translation[:2], height), self._grasp(banana))
        height = APPROACH_HEIGHT if self.phase == Phase.ABOVE else 0.0
        return geom.Transform3D(point + [0.0, 0.0, height], self._grasp(banana))

    def command(
        self,
        pose: geom.Transform3D,
        grip: float,
        banana: np.ndarray,
        bowl: np.ndarray,
        contact: bool,
        clear: bool,
        now_ns: int,
        drives: bool,
    ) -> tuple[geom.Transform3D, float]:
        """The pose and the grip that the arm takes at ``now_ns``. ``pose`` and ``grip`` are what the arm is at.
        ``banana`` and ``bowl`` are poses in the layout of ``gt_state``: [x, y, z, qw, qx, qy, qz]. ``contact`` is
        whether a finger touches the banana. ``clear`` is whether each other arm is out of the way. While the expert
        does not drive, the commanded pose stays at ``pose``, so that a takeover starts there."""
        phase = self._pick(pose, grip, banana, bowl, contact, clear, now_ns)
        if phase != self.phase:
            self.phase, self._phase_ns = phase, now_ns
            self._turned = self._turned and phase in (Phase.RISE, Phase.ABOVE, Phase.DOWN, Phase.PARK)
        if phase != Phase.CLOSE:
            self._closing_ns = None
        if not drives:
            self._command.restart(pose, now_ns)
            self._still = pose, now_ns
        if apart(pose, self._still[0], STILL_M, STILL_RAD):
            self._still = pose, now_ns
        goal = self._goal(pose, banana, bowl)
        short = (pose.rotation.inv * goal.rotation).angle > REACH_RAD
        far = np.linalg.norm(goal.translation - pose.translation) > (REACH_M if phase == Phase.DOWN else KEEP_M)
        stuck = now_ns - max(self._still[1], self._phase_ns) >= STUCK_NS  # still since the phase began, or longer
        if (short or far) and stuck and phase in (Phase.RISE, Phase.ABOVE, Phase.DOWN):
            self._yaw, self._turned = self._yaw + np.pi, True
            self._command.pose, self._still = pose, (pose, now_ns)
            goal = self._goal(pose, banana, bowl)
        self._command.follow(pose)
        self.goal = goal
        closed = phase in (Phase.CLOSE, Phase.LIFT, Phase.CARRY, Phase.LOWER)
        return self._command.advance(goal, now_ns), 1.0 if closed else 0.0


class BananaInBowlExpert(Policy):
    """The scripted expert of ``BananaInBowlTask``, a helper. ``arms`` gives the statics of each arm as its channels
    name it. ``takes_over_at_start`` says that the expert takes over at once and never hands back.

    At the first step it picks the arm that holds the banana, or else the arm whose mount is nearer the banana. It
    moves the grip point of each other arm to its park position, with the grip open.
    """

    WIRE_NAME = 'robolab_banana_in_bowl_expert'

    def __init__(self, arms: Mapping[str, ReachStatics], license: str, *, takes_over_at_start: bool = False):
        self._arms = dict(arms)
        self._license = license
        self._takes_over_at_start = takes_over_at_start

    @classmethod
    def for_yam(
        cls,
        license: str,
        *,
        grip_point: geom.Transform3D,
        reach: float,
        rest_yaw: float,
        wrist_turn: float,
        park: np.ndarray,
        rise_margin: float = 0.03,
        takes_over_at_start: bool = False,
    ) -> Self:
        """The expert of the two-arm YAM of the RoboLab env server (``yam_spec``). Each arm has the statics
        ``grip_point``, ``park``, ``reach``, ``rest_yaw``, ``wrist_turn`` and ``rise_margin`` of ``ReachStatics``."""
        arms = {
            side: ReachStatics(
                grip_point,
                np.add(yam_spec.ROOT_IN_RIG_WORLD, yam_spec.mount_xyz(side)),
                np.asarray(park, dtype=np.float64),
                reach=reach,
                rest_yaw=rest_yaw,
                wrist_turn=wrist_turn,
                rise_margin=rise_margin,
            )
            for side in keys.BIMANUAL_ARMS
        }
        return cls(arms, license, takes_over_at_start=takes_over_at_start)

    @property
    def arms(self) -> Mapping[str, ReachStatics]:
        return self._arms

    def meta(self) -> dict[str, Any]:
        return {TYPE: self.WIRE_NAME, LICENSE: self._license, **ee_frames_entry(frames(self._arms))}

    def _grip_pose(self, obs: Obs, arm: str) -> geom.Transform3D:
        return grip_pose(obs, arm, self._arms[arm].grip_point)

    def _clear(self, obs: Obs, arm: str, banana: np.ndarray) -> bool:
        """Whether the grip point of ``arm`` is ``CLEAR_M`` or more from the banana, to the side of its mount."""
        side = np.sign(self._arms[arm].mount[1] - banana[1])
        return bool((self._grip_pose(obs, arm).translation[1] - banana[1]) * side >= CLEAR_M)

    def _pick_arm(self, obs: Obs) -> str:
        banana = object_pose(obs, BANANA)
        for arm in self._arms:
            if _holds(self._grip_pose(obs, arm), grip(obs, arm), banana, touched(obs, BANANA)):
                return arm
        return min(self._arms, key=lambda arm: float(np.linalg.norm(self._arms[arm].mount[:2] - banana[:2])))

    def run(self, runtime: Runtime) -> PolicyRun:
        obs = yield
        check_truth(obs, BANANA)
        arm = self._pick_arm(obs)
        statics = self._arms[arm]
        task = BananaInBowl(self._grip_pose(obs, arm), statics, runtime.time_ns)
        idle = [IdleArm(other, self._arms[other], obs, runtime.time_ns) for other in self._arms if other != arm]
        console = Console(RANK, self._takes_over_at_start)
        while True:
            now, control = runtime.time_ns, ControlObservation.model_validate(obs[CONTROL])
            drives = control.source == TELEOPERATOR
            commands = {}
            for each in idle:
                commands.update(each.commands(obs, now, drives))
            banana, bowl = (object_pose(obs, name) for name in (BANANA, BOWL))
            clear = all(self._clear(obs, other, banana) for other in self._arms if other != arm)
            here = self._grip_pose(obs, arm)
            contact = touched(obs, BANANA)
            pose, closed = task.command(here, grip(obs, arm), banana, bowl, contact, clear, now, drives)
            commands.update(arm_commands(pose, closed, arm, statics.grip_point))
            assert task.phase is not None, 'a command picks a phase'
            distance = float(np.linalg.norm(task.goal.translation - here.translation))
            element = console.element(control, task.phase, distance, commands, now)
            obs = yield Step(element, now + PERIOD_NS)
