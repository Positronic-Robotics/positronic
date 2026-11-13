import numpy as np
import pytest

from positronic import geom
from positronic.drivers.roboarm.yam.kinematics import YamKinematics
from positronic.vendors.llm.reach import YamReach

MOUNT = [0.30, -0.305, 0.3125]
# The fingers point down, and joint 4 is 0.09 rad above its lower limit.
TOP_DOWN = np.array([0.0, 1.9, 1.9, -1.6, 0.0, 0.0])
SIDE_ON = np.array([0.0, 1.2, 1.0, 0.0, 0.3, 0.0])


def hand_at(joints: np.ndarray) -> geom.Transform3D:
    return geom.Transform3D(np.array(MOUNT)) * YamKinematics().fk(joints)


def lift(start: geom.Transform3D, height: float, steps: int) -> list[geom.Transform3D]:
    return [
        geom.Transform3D(start.translation + [0, 0, height * i / steps], start.rotation) for i in range(1, steps + 1)
    ]


def test_a_hand_that_its_arm_can_take_along_the_path_is_not_refused():
    reach = YamReach({'right': MOUNT})
    assert reach.refusal('right', SIDE_ON, lift(hand_at(SIDE_ON), 0.1, 17)) is None


def test_a_top_down_lift_is_refused_where_joint_4_reaches_its_lower_limit():
    reach = YamReach({'right': MOUNT})
    index, reason = reach.refusal('right', TOP_DOWN, lift(hand_at(TOP_DOWN), 0.3, 50))
    assert 0 < index < 50
    assert reason.startswith('joint 4 is at its lower limit (-1.69 rad)')


def test_a_pose_beyond_the_arm_is_refused():
    far = geom.Transform3D(np.array([2.0, -0.305, 0.5]), hand_at(SIDE_ON).rotation)
    assert YamReach({'right': MOUNT}).refusal('right', SIDE_ON, [far]) == (
        0,
        'the pose is beyond the arm. Reach it with a nearer position',
    )


def test_a_pose_where_a_joint_turns_more_than_the_step_limit_is_refused():
    other = hand_at(SIDE_ON + [0.6, 0, 0, 0, 0, 0])
    index, reason = YamReach({'right': MOUNT}).refusal('right', SIDE_ON, [other])
    assert index == 0
    assert reason.startswith('joint 1 would turn 0.60 rad at once')
    assert YamReach({'right': MOUNT}, max_joint_step=0.7).refusal('right', SIDE_ON, [other]) is None


def test_the_hand_poses_are_in_the_world_and_the_arm_base_is_at_its_mount():
    pose = hand_at(SIDE_ON)
    assert YamReach({'right': MOUNT}).refusal('right', SIDE_ON, [pose]) is None
    moved = YamReach({'right': [MOUNT[0] - 1.0, MOUNT[1], MOUNT[2]]}).refusal('right', SIDE_ON, [pose])
    assert moved is not None and moved[0] == 0


@pytest.mark.parametrize('mount', [[0.3, 0.0], [0.3, 0.0, np.inf]])
def test_a_mount_must_be_a_finite_position(mount):
    with pytest.raises(ValueError, match='must be a finite xyz position'):
        YamReach({'right': mount})
