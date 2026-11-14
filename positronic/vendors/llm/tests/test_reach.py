import numpy as np
import pytest

from positronic import geom
from positronic.drivers.roboarm.yam.kinematics import YamKinematics
from positronic.vendors.llm.reach import YamReach

MOUNT = geom.Transform3D(np.array([0.30, -0.305, 0.3125]))
KINEMATICS = YamKinematics()
# The fingers point down, and joint 4 is 0.09 rad above its lower limit.
TOP_DOWN = np.array([0.0, 1.9, 1.9, -1.6, 0.0, 0.0])
SIDE_ON = np.array([0.0, 1.2, 1.0, 0.0, 0.3, 0.0])


def hand_at(joints: np.ndarray, mount: geom.Transform3D = MOUNT) -> geom.Transform3D:
    return mount * KINEMATICS.fk(joints)


def refusal(joints: np.ndarray, poses: list[geom.Transform3D], reach: YamReach | None = None, mount=MOUNT):
    """The refusal of ``poses`` for an arm at ``joints`` whose hand holds no target near it."""
    return (reach or YamReach()).refusal(joints, hand_at(joints, mount), None, poses)


def lift(start: geom.Transform3D, height: float, steps: int) -> list[geom.Transform3D]:
    return [
        geom.Transform3D(start.translation + [0, 0, height * i / steps], start.rotation) for i in range(1, steps + 1)
    ]


def test_a_hand_that_its_arm_can_take_along_the_path_is_not_refused():
    assert refusal(SIDE_ON, lift(hand_at(SIDE_ON), 0.1, 17)) is None


def test_a_top_down_lift_is_refused_where_joint_4_reaches_its_lower_limit():
    index, reason = refusal(TOP_DOWN, lift(hand_at(TOP_DOWN), 0.3, 50))
    assert 0 < index < 50
    assert reason.startswith('joint 4 is at its lower limit (-1.69 rad)')


def test_a_pose_past_an_upper_limit_names_that_limit():
    beyond = SIDE_ON + [0, 0, 0, 1.82, 0, 0]
    _, reason = refusal(SIDE_ON, [hand_at(beyond)])
    assert reason.startswith('joint 4 is at its upper limit (1.57 rad)')


def test_a_joint_a_full_turn_from_its_range_counts_inside_it():
    # The unbounded IK puts joint 1 at 3.78 rad, past its upper limit; a turn back, -2.50 rad, is inside the range.
    start = np.array([3.0, 1.2, 1.0, 0.0, 0.3, 0.0])
    beyond = np.array([3.78, 1.2, 1.0, 1.82, 0.3, 0.0])
    _, reason = refusal(start, [hand_at(beyond)])
    assert reason.startswith('joint 4 is at its upper limit (1.57 rad)')


def test_a_pose_beyond_the_arm_is_refused():
    far = geom.Transform3D(np.array([2.0, -0.305, 0.5]), hand_at(SIDE_ON).rotation)
    assert refusal(SIDE_ON, [far]) == (0, 'the pose is beyond the arm. Reach it with a nearer position')


def test_a_pose_where_a_joint_turns_more_than_the_step_limit_is_refused():
    other = hand_at(SIDE_ON + [0.6, 0, 0, 0, 0, 0])
    index, reason = refusal(SIDE_ON, [other])
    assert index == 0
    assert reason.startswith('joint 1 would turn 0.60 rad at once')
    assert refusal(SIDE_ON, [other], YamReach(max_joint_step=0.7)) is None


def test_each_pose_is_solved_from_the_joints_of_the_pose_before_it():
    # Each pose turns joint 6 by 0.4 rad, under the step limit; the last one is 1.2 rad from the measured joints.
    path = [hand_at(SIDE_ON + [0, 0, 0, 0, 0, 0.4 * i]) for i in range(1, 4)]
    assert refusal(SIDE_ON, path) is None


@pytest.mark.parametrize(
    'mount',
    [
        geom.Transform3D(np.array([0.30, -0.305, 0.3125])),
        geom.Transform3D(np.array([-1.0, 2.0, 0.7]), geom.Rotation.from_euler([0.0, 0.0, 1.0])),
    ],
)
def test_the_arm_base_is_where_the_measured_hand_and_joints_put_it(mount):
    assert refusal(SIDE_ON, lift(hand_at(SIDE_ON, mount), 0.1, 17), mount=mount) is None
    _, reason = refusal(TOP_DOWN, lift(hand_at(TOP_DOWN, mount), 0.3, 50), mount=mount)
    assert reason.startswith('joint 4 is at its lower limit')


def sagged(joint: int, past: float) -> np.ndarray:
    """``TOP_DOWN`` with ``joint`` measured ``past`` (rad) beyond its lower limit (< 0) or its upper limit (> 0)."""
    joints = TOP_DOWN.copy()
    joints[joint] = (KINEMATICS.lower if past < 0 else KINEMATICS.upper)[joint] + past
    return joints


@pytest.mark.parametrize('joint,past', [(3, -0.007), (3, -0.01), (4, 0.01), (1, -0.012)])
def test_an_arm_a_little_past_its_range_can_hold_its_measured_pose(joint, past):
    # The driver's IK fails there and the driver keeps its last target, so the arm stays.
    at = sagged(joint, past)
    assert KINEMATICS.ik(MOUNT.inv * hand_at(at), at, KINEMATICS.lower, KINEMATICS.upper) is None
    assert refusal(at, [hand_at(at)] * 10) is None


def test_an_arm_past_its_range_cannot_go_further_past_it():
    # The driver's IK refuses each sample of the lift, so the arm would stall below the waypoint.
    at = sagged(3, -0.01)
    index, reason = refusal(at, lift(hand_at(at), 0.05, 10))
    assert index == 0
    assert reason.startswith('joint 4 is at its lower limit (-1.69 rad)')


def test_a_hold_that_the_driver_would_turn_into_another_posture_is_refused():
    # Joint 2 is 8 mrad under its lower limit; from there the driver's IK finds the pose only with joints 1 and 6 half
    # a turn away, and would swing the arm to it.
    at = np.array([1.175, -0.008, 2.439, 0.079, 1.318, -1.12])
    index, reason = refusal(at, [hand_at(at)] * 5)
    assert index == 0
    assert reason.startswith('joint 6 would turn 3.14 rad at once')


def trailing(joint: int, past: float) -> tuple[np.ndarray, geom.Transform3D, geom.Transform3D]:
    """An arm ``past`` its range at ``joint``, its hand, and the target it was sent to: the hand of the range end."""
    at = sagged(joint, past)
    return at, hand_at(at), hand_at(np.clip(at, KINEMATICS.lower, KINEMATICS.upper))


def test_a_hand_holds_its_target_where_the_ik_fails_on_the_way_from_the_hand_to_that_target():
    # The hand is 2.5 mm from its target. Motion samples the way between them for an arm that a waypoint leaves out.
    at, hand, held = trailing(3, -0.01)
    way = [hand.interpolate(held, i / 9) for i in range(1, 10)]
    assert YamReach().refusal(at, hand, held, way) is None
    assert YamReach().refusal(at, hand, None, way) is not None


def test_a_pose_beside_the_way_from_the_hand_to_its_target_is_refused_where_the_ik_fails():
    at, hand, held = trailing(3, -0.01)
    beside = hand.interpolate(held, 0.5)
    beside = geom.Transform3D(beside.translation + [0, 0.002, 0], beside.rotation)
    assert YamReach().refusal(at, hand, held, [beside]) is not None


def test_a_hand_that_leaves_and_comes_back_where_the_ik_fails_is_refused():
    # Going down turns joint 4 away from its limit. Back at the start, the driver would keep the last pose it could
    # solve, short of the start.
    at, hand, held = trailing(3, -0.01)
    down = geom.Transform3D(hand.translation + [0, 0, -0.02], hand.rotation)
    away = [hand.interpolate(down, i / 5) for i in range(1, 6)]
    back = [down.interpolate(hand, i / 5) for i in range(1, 6)]
    assert YamReach().refusal(at, hand, held, away) is None
    index, reason = YamReach().refusal(at, hand, held, away + back)
    assert index == len(away + back) - 1
    assert reason.startswith('joint 4 is at its lower limit (-1.69 rad)')
