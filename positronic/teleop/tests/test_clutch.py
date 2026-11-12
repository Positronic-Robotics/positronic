import pickle

import numpy as np

from positronic import geom
from positronic.teleop.clutch import CONTROLLER_AXES, OperatorPosition, Tracker


def test_an_operator_position_pickles_by_its_name():
    assert pickle.loads(pickle.dumps(OperatorPosition.BACK)) is OperatorPosition.BACK


def test_control_starts_at_the_robot_pose_from_the_hand_pose_at_turn_on():
    robot = geom.Transform3D([0.4, 0.1, 0.3], geom.Rotation.from_euler([0.1, 0.2, 0.3]))
    hand = geom.Transform3D([1.0, -2.0, 0.5], geom.Rotation.from_euler([-0.4, 0.5, 1.2]))
    tracker = Tracker(OperatorPosition.BACK.value)
    tracker.turn_on(robot, hand)
    np.testing.assert_allclose(tracker.update(hand).as_matrix, robot.as_matrix, atol=1e-9)


def _moved(tracker: Tracker, hand: geom.Transform3D, move: geom.Transform3D) -> geom.Transform3D:
    """The robot pose after the hand makes ``move`` in its own axes from ``hand``, with control on at the origin."""
    tracker.turn_on(geom.Transform3D(), hand)
    return tracker.update(hand * move)


def test_with_controller_axes_a_move_along_the_controller_is_the_same_whichever_way_the_operator_faces():
    move = geom.Transform3D([0.1, -0.05, 0.2], geom.Rotation.from_euler([0.2, 0.1, -0.3]))
    facing_ahead = geom.Transform3D([0.3, 1.2, -0.4], geom.Rotation.from_euler([0.1, -0.2, 0.4]))
    # The operator turns round a vertical axis of the headset: the controller turns with the operator
    turned = geom.Transform3D(rotation=geom.Rotation.from_rotvec(np.array([0.0, np.pi / 2, 0.0]))) * facing_ahead
    ahead = _moved(Tracker(OperatorPosition.FRONT.value), facing_ahead, move)
    aside = _moved(Tracker(OperatorPosition.FRONT.value), turned, move)
    np.testing.assert_allclose(aside.as_matrix, ahead.as_matrix, atol=1e-9)


def test_with_controller_axes_the_operator_in_front_moves_the_robot_as_the_controller_axes_point():
    """In front of the robot, x of the robot points to the operator: red (x) right, green (y) to the operator, blue (z)
    down, as an operator checked on the CRON stand."""
    tracker = Tracker(OperatorPosition.FRONT.value)
    hand = geom.Transform3D([0.2, 1.0, -0.3], CONTROLLER_AXES)
    for along, robot in (([0.1, 0, 0], [0, 0.1, 0]), ([0, 0.1, 0], [0.1, 0, 0]), ([0, 0, 0.1], [0, 0, -0.1])):
        moved = _moved(tracker, hand, geom.Transform3D(np.array(along, dtype=float)))
        np.testing.assert_allclose(moved.translation, robot, atol=1e-9)


def test_with_controller_axes_a_tilt_of_the_controller_does_not_change_up():
    up = geom.Transform3D([0.0, 0.1, 0.0])  # y is up in the headset
    robots = []
    for tilt in ([0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, -0.6]):
        hand = geom.Transform3D([0.2, 1.0, -0.3], geom.Rotation.from_rotvec(np.array(tilt)) * CONTROLLER_AXES)
        tracker = Tracker(OperatorPosition.FRONT.value)
        tracker.turn_on(geom.Transform3D(), hand)
        robots.append(tracker.update(up * hand).translation)
    np.testing.assert_allclose(robots, [[0.0, 0.0, 0.1]] * 3, atol=1e-9)


def test_the_controller_axes_are_the_headset_axes_turned_a_quarter_turn_about_x():
    np.testing.assert_allclose(CONTROLLER_AXES([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(CONTROLLER_AXES([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0], atol=1e-12)


def test_with_the_headset_axes_a_move_in_the_room_moves_the_robot_whichever_way_the_controller_points():
    shift = geom.Transform3D([0.1, 0.0, -0.2])
    straight = geom.Transform3D([0.3, 1.2, -0.4])
    tilted = geom.Transform3D(straight.translation, geom.Rotation.from_euler([0.7, -0.4, 1.1]))
    robots = []
    for hand in (straight, tilted):
        tracker = Tracker(OperatorPosition.BACK.value, controller_axes=None)
        tracker.turn_on(geom.Transform3D(), hand)
        robots.append(tracker.update(shift * hand).translation)
    np.testing.assert_allclose(robots[0], robots[1], atol=1e-9)


def test_control_does_not_follow_a_start_pose_that_its_owner_writes_again():
    """A driver can hand out views of the state that it writes again at each tick."""
    robot = geom.Transform3D(np.array([0.4, 0.1, 0.3]))
    hand = geom.Transform3D(np.array([1.0, -2.0, 0.5]))
    tracker = Tracker(OperatorPosition.FRONT.value)
    tracker.turn_on(robot, hand)
    np.asarray(robot.translation)[:] = 0.0
    np.asarray(hand.translation)[:] = 9.0
    unmoved = tracker.update(geom.Transform3D(np.array([1.0, -2.0, 0.5])))
    np.testing.assert_allclose(unmoved.translation, [0.4, 0.1, 0.3])


def test_a_station_sets_its_own_axes_of_the_controller():
    """Another controller or grip. With the identity, a controller in the axes of the headset moves the robot as the
    axes of the headset do."""
    tracker = Tracker(OperatorPosition.FRONT.value, controller_axes=geom.Rotation.identity)
    hand = geom.Transform3D(np.array([0.2, 1.0, -0.3]))
    moved = _moved(tracker, hand, geom.Transform3D(np.array([0.1, 0.05, -0.2])))
    np.testing.assert_allclose(moved.translation, OperatorPosition.FRONT.value.rotation([0.1, 0.05, -0.2]), atol=1e-9)
