import pickle

import numpy as np

from positronic import geom
from positronic.teleop.clutch import OperatorPosition, Tracker


def test_an_operator_position_pickles_by_its_name():
    assert pickle.loads(pickle.dumps(OperatorPosition.BACK)) is OperatorPosition.BACK


def test_control_starts_at_the_robot_pose_from_the_hand_pose_at_turn_on():
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    hand = geom.Transform3D([1.0, -2.0, 0.5], geom.Rotation.from_euler([-0.4, 0.5, 1.2]))
    tracker = Tracker(OperatorPosition.BACK.value)
    tracker.turn_on(robot, hand)
    np.testing.assert_allclose(tracker.update(hand).as_matrix, robot.as_matrix, atol=1e-9)
