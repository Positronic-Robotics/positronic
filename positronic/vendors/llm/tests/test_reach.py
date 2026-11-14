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


def lift(start: geom.Transform3D, height: float, steps: int) -> list[geom.Transform3D]:
    return [
        geom.Transform3D(start.translation + [0, 0, height * i / steps], start.rotation) for i in range(1, steps + 1)
    ]


def test_a_hand_that_its_arm_can_take_along_the_path_is_not_refused():
    assert YamReach().refusal(SIDE_ON, hand_at(SIDE_ON), lift(hand_at(SIDE_ON), 0.1, 17)) is None


def test_a_top_down_lift_is_refused_where_joint_4_reaches_its_lower_limit():
    index, reason = YamReach().refusal(TOP_DOWN, hand_at(TOP_DOWN), lift(hand_at(TOP_DOWN), 0.3, 50))
    assert 0 < index < 50
    assert reason.startswith('joint 4 is at its lower limit (-1.69 rad)')


def test_a_pose_past_an_upper_limit_names_that_limit():
    beyond = SIDE_ON + [0, 0, 0, 1.82, 0, 0]
    _, reason = YamReach().refusal(SIDE_ON, hand_at(SIDE_ON), [hand_at(beyond)])
    assert reason.startswith('joint 4 is at its upper limit (1.57 rad)')


def test_a_joint_a_full_turn_from_its_range_counts_inside_it():
    # The unbounded IK puts joint 1 at 3.78 rad, past its upper limit; a turn back, -2.50 rad, is inside the range.
    start = np.array([3.0, 1.2, 1.0, 0.0, 0.3, 0.0])
    beyond = np.array([3.78, 1.2, 1.0, 1.82, 0.3, 0.0])
    _, reason = YamReach().refusal(start, hand_at(start), [hand_at(beyond)])
    assert reason.startswith('joint 4 is at its upper limit (1.57 rad)')


def test_a_pose_beyond_the_arm_is_refused():
    far = geom.Transform3D(np.array([2.0, -0.305, 0.5]), hand_at(SIDE_ON).rotation)
    assert YamReach().refusal(SIDE_ON, hand_at(SIDE_ON), [far]) == (
        0,
        'the pose is beyond the arm. Reach it with a nearer position',
    )


def test_a_pose_where_a_joint_turns_more_than_the_step_limit_is_refused():
    other = hand_at(SIDE_ON + [0.6, 0, 0, 0, 0, 0])
    index, reason = YamReach().refusal(SIDE_ON, hand_at(SIDE_ON), [other])
    assert index == 0
    assert reason.startswith('joint 1 would turn 0.60 rad at once')
    assert YamReach(max_joint_step=0.7).refusal(SIDE_ON, hand_at(SIDE_ON), [other]) is None


def test_each_pose_is_solved_from_the_joints_of_the_pose_before_it():
    # Each pose turns joint 6 by 0.4 rad, under the step limit; the last one is 1.2 rad from the measured joints.
    path = [hand_at(SIDE_ON + [0, 0, 0, 0, 0, 0.4 * i]) for i in range(1, 4)]
    assert YamReach().refusal(SIDE_ON, hand_at(SIDE_ON), path) is None


@pytest.mark.parametrize(
    'mount',
    [
        geom.Transform3D(np.array([0.30, -0.305, 0.3125])),
        geom.Transform3D(np.array([-1.0, 2.0, 0.7]), geom.Rotation.from_euler([0.0, 0.0, 1.0])),
    ],
)
def test_the_arm_base_is_where_the_measured_hand_and_joints_put_it(mount):
    assert YamReach().refusal(SIDE_ON, hand_at(SIDE_ON, mount), lift(hand_at(SIDE_ON, mount), 0.1, 17)) is None
    _, reason = YamReach().refusal(TOP_DOWN, hand_at(TOP_DOWN, mount), lift(hand_at(TOP_DOWN, mount), 0.3, 50))
    assert reason.startswith('joint 4 is at its lower limit')


@pytest.mark.parametrize('joint,past', [(3, -0.007), (4, 0.004), (1, -0.012)])
def test_an_arm_a_little_past_its_range_can_hold_and_leave_the_limit(joint, past):
    sagged = TOP_DOWN.copy()
    sagged[joint] = (KINEMATICS.lower if past < 0 else KINEMATICS.upper)[joint] + past
    on_limit = np.clip(sagged, KINEMATICS.lower, KINEMATICS.upper)
    reach = YamReach()
    assert reach.refusal(sagged, hand_at(sagged), [hand_at(sagged)] * 10) is None
    assert reach.refusal(sagged, hand_at(sagged), [hand_at(on_limit)] * 10) is None
