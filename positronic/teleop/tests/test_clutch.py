import pickle

import numpy as np
import pytest

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


_QUARTER_ABOUT_Z = geom.Rotation.from_rotvec(np.array([0.0, 0.0, np.pi / 2]))


def test_in_the_hand_axes_a_shift_along_an_axis_of_the_hand_moves_the_robot_the_way_that_axis_maps_to():
    """The axes of the headset do not count: a hand held any way that moves along its own x moves the robot along
    the robot axis that ``hand_axes`` maps x to, and a turn about its own x turns the robot about that axis too."""
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    hand = geom.Transform3D([1.0, -2.0, 0.5], geom.Rotation.from_euler([-0.4, 0.5, 1.2]))
    tracker = Tracker(OperatorPosition.FRONT.value, hand_axes=_QUARTER_ABOUT_Z)
    tracker.turn_on(robot, hand)
    moved = hand * geom.Transform3D([0.05, 0.0, 0.0], geom.Rotation.from_rotvec(np.array([0.3, 0.0, 0.0])))

    pose = tracker.update(moved)

    # A quarter turn about z maps the x of the hand to the y of the robot
    np.testing.assert_allclose(pose.translation, robot.translation + [0.0, 0.05, 0.0], atol=1e-9)
    expected = geom.Rotation.from_rotvec(np.array([0.0, 0.3, 0.0])) * robot.rotation
    np.testing.assert_allclose(pose.rotation.as_rotation_matrix, expected.as_rotation_matrix, atol=1e-9)


def test_in_the_hand_axes_control_starts_at_the_robot_pose():
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    hand = geom.Transform3D([1.0, -2.0, 0.5], geom.Rotation.from_euler([-0.4, 0.5, 1.2]))
    tracker = Tracker(OperatorPosition.FRONT.value, hand_axes=_QUARTER_ABOUT_Z)
    tracker.turn_on(robot, hand)
    np.testing.assert_allclose(tracker.update(hand).as_matrix, robot.as_matrix, atol=1e-9)


def _heading_tracker_at(hand_turn: geom.Rotation, robot: geom.Transform3D) -> tuple[Tracker, geom.Transform3D]:
    """A tracker in the heading of the hand, on with the hand held as ``hand_axes`` expects and then turned by
    ``hand_turn``, given in the axes of the headset."""
    operator = OperatorPosition.FRONT.value
    held = operator.rotation.inv * _QUARTER_ABOUT_Z
    hand = geom.Transform3D([1.0, -2.0, 0.5], hand_turn * held)
    tracker = Tracker(operator, hand_axes=_QUARTER_ABOUT_Z, heading_only=True)
    tracker.turn_on(robot, hand)
    return tracker, hand


def test_in_the_heading_of_the_hand_a_tilt_of_the_hand_does_not_turn_up():
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    # 45 degrees about the x axis of the headset: the hand points down
    tracker, hand = _heading_tracker_at(geom.Rotation.from_rotvec(np.array([np.pi / 4, 0.0, 0.0])), robot)
    up = geom.Transform3D(hand.translation + [0.0, 0.05, 0.0], hand.rotation)  # y is up in the headset

    pose = tracker.update(up)

    np.testing.assert_allclose(pose.translation, robot.translation + [0.0, 0.0, 0.05], atol=1e-9)
    np.testing.assert_allclose(pose.rotation.as_rotation_matrix, robot.rotation.as_rotation_matrix, atol=1e-9)


def test_in_the_heading_of_the_hand_a_turn_of_the_operator_does_not_count():
    """A hand turned about up moves the robot along its own x as a hand held as ``hand_axes`` expects does."""
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    tracker, hand = _heading_tracker_at(geom.Rotation.from_rotvec(np.array([0.0, 1.2, 0.0])), robot)

    pose = tracker.update(hand * geom.Transform3D([0.05, 0.0, 0.0]))

    np.testing.assert_allclose(pose.translation, robot.translation + _QUARTER_ABOUT_Z([0.05, 0.0, 0.0]), atol=1e-9)


def test_in_the_heading_of_the_hand_control_starts_at_the_robot_pose():
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    tracker, hand = _heading_tracker_at(geom.Rotation.from_euler([-0.4, 0.5, 1.2]), robot)
    np.testing.assert_allclose(tracker.update(hand).as_matrix, robot.as_matrix, atol=1e-9)


def _hand_turned(operator: geom.Transform3D, robot: geom.Transform3D, hand: geom.Transform3D, turn: geom.Rotation):
    """``hand`` turned by ``turn``, given in the frame of ``robot``."""
    teleop = operator * hand * operator.inv
    rotation = robot.rotation * turn * robot.rotation.inv * teleop.rotation
    return operator.inv * geom.Transform3D(teleop.translation, rotation) * operator


@pytest.mark.parametrize(
    ('turn', 'robot_turn'),
    [
        (geom.Rotation.from_rotvec([0.0, 0.0, 0.3]), geom.Rotation.from_rotvec([0.0, 0.0, 0.6])),
        (geom.Rotation.from_rotvec([0.0, 0.0, -2.0]), geom.Rotation.from_rotvec([0.0, 0.0, -4.0])),
        (geom.Rotation.from_rotvec([0.3, 0.0, 0.0]), geom.Rotation.from_rotvec([0.3, 0.0, 0.0])),
    ],
    ids=['about z', 'about z the long way', 'about x'],
)
def test_the_robot_turns_about_its_z_axis_twist_gain_times_as_far_as_the_hand(turn, robot_turn):
    operator = OperatorPosition.FRONT.value
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    hand = geom.Transform3D([1.0, -2.0, 0.5], geom.Rotation.from_euler([-0.4, 0.5, 1.2]))
    tracker = Tracker(operator, twist_gain=2.0)
    tracker.turn_on(robot, hand)

    pose = tracker.update(_hand_turned(operator, robot, hand, turn))

    expected = robot.rotation * robot_turn
    np.testing.assert_allclose(pose.rotation.as_rotation_matrix, expected.as_rotation_matrix, atol=1e-9)
    np.testing.assert_allclose(pose.translation, robot.translation, atol=1e-9)
