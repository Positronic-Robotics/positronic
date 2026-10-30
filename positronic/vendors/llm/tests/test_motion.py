import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from positronic import geom, keys
from positronic.vendors.llm.motion import Motion, MovePath, Waypoint

# The cases below are written around these bounds, not around the defaults.
SHORT_BOUNDS = {'max_translation': 0.05, 'max_rotation': math.radians(20), 'linear_speed': 0.05}


@pytest.mark.parametrize('quaternion_sign', [1, -1])
def test_motion_uses_short_rotation_path_and_respects_speed(quaternion_sign):
    start = geom.Transform3D([0, 0, 0], geom.Rotation.from_euler([0, 0, math.radians(179)]))
    start.rotation = geom.Rotation.from_quat(quaternion_sign * start.rotation.as_quat)
    target = Waypoint(x=0.05, y=0, z=0, roll=0, pitch=0, yaw=math.radians(-179), gripper=0.7)
    motion = Motion(**SHORT_BOUNDS)
    (bounded,), trajectory = motion.trajectory(start, 0.7, [target])
    assert bounded == target
    previous, previous_time = start, 0
    for index, action in enumerate(trajectory, 1):
        pose = action[keys.ROBOT_COMMAND].pose
        dt = (index / motion.fps) - previous_time
        assert 0 < dt <= 1 / motion.fps + 1e-9
        assert np.linalg.norm(pose.translation - previous.translation) <= motion.linear_speed * dt + 1e-9
        rotation = Rotation.from_matrix(pose.rotation.as_rotation_matrix @ previous.rotation.as_rotation_matrix.T)
        assert rotation.magnitude() <= motion.angular_speed * dt + 1e-9
        assert action[keys.TARGET_GRIP] == 0.7
        previous, previous_time = pose, (index / motion.fps)
    np.testing.assert_allclose(previous.translation, target.pose.translation)
    np.testing.assert_allclose(previous.rotation.as_rotation_matrix, target.pose.rotation.as_rotation_matrix, atol=1e-8)
    assert previous_time == pytest.approx(1.0)


@pytest.mark.parametrize(
    'offset', [(0, 0, 0), (0.01, 0, 0), (0.05, 0, 0), (0.051, 0, 0), (0.08, -0.06, 0.12), (1e200, -1e200, 1e200)]
)
def test_translation_clamps_distance_preserving_direction(offset):
    start = geom.Transform3D([0.3, -0.2, 0.1])
    x, y, z = start.translation + offset
    target = Waypoint(x=x, y=y, z=z, roll=0, pitch=0, yaw=0, gripper=0.7)
    original = target.model_dump()
    motion = Motion(**SHORT_BOUNDS)
    (bounded,), trajectory = motion.trajectory(start, 0.7, [target])
    offset = np.asarray(offset)
    distance = math.hypot(*offset)
    expected = offset / distance * motion.max_translation if distance > motion.max_translation else offset
    np.testing.assert_allclose(bounded.pose.translation, start.translation + expected)
    assert target.model_dump() == original
    assert bounded.gripper == target.gripper
    if distance < motion.max_translation:
        assert bounded == target
    previous, previous_time = start, 0
    for index, action in enumerate(trajectory, 1):
        pose = action[keys.ROBOT_COMMAND].pose
        dt = (index / motion.fps) - previous_time
        assert np.linalg.norm(pose.translation - start.translation) <= motion.max_translation + 1e-9
        assert np.linalg.norm(pose.translation - previous.translation) <= motion.linear_speed * dt + 1e-9
        np.testing.assert_allclose(pose.rotation.as_rotation_matrix, start.rotation.as_rotation_matrix)
        assert action[keys.TARGET_GRIP] == target.gripper
        previous, previous_time = pose, (index / motion.fps)
    np.testing.assert_allclose(previous.translation, bounded.pose.translation)


@pytest.mark.parametrize('quaternion_sign', [1, -1])
@pytest.mark.parametrize('angle_degrees', [0, 5, 20, 21, 179, 181])
def test_rotation_clamps_shortest_arc_independently_of_translation(quaternion_sign, angle_degrees):
    initial = Rotation.from_euler('xyz', [0.2, -0.3, math.radians(179)])
    requested = initial * Rotation.from_rotvec(np.array([1, 2, 3]) / math.sqrt(14) * math.radians(angle_degrees))
    start = geom.Transform3D([0, 0, 0], geom.Rotation.from_rotation_matrix(initial.as_matrix()))
    start.rotation = geom.Rotation.from_quat(quaternion_sign * start.rotation.as_quat)
    roll, pitch, yaw = requested.as_euler('xyz')
    target = Waypoint(x=0.2, y=0, z=0, roll=roll, pitch=pitch, yaw=yaw, gripper=0.3)
    motion = Motion(**SHORT_BOUNDS)
    (bounded,), trajectory = motion.trajectory(start, 0.3, [target])
    relative = initial.inv() * requested
    fraction = min(1, motion.max_rotation / relative.magnitude()) if relative.magnitude() > 0 else 1
    expected = initial * Rotation.from_rotvec(relative.as_rotvec() * fraction)
    np.testing.assert_allclose(bounded.pose.rotation.as_rotation_matrix, expected.as_matrix(), atol=1e-9)
    assert bounded.x == pytest.approx(motion.max_translation)
    previous, previous_time = initial, 0
    for index, action in enumerate(trajectory, 1):
        rotation = Rotation.from_matrix(action[keys.ROBOT_COMMAND].pose.rotation.as_rotation_matrix)
        dt = (index / motion.fps) - previous_time
        assert (initial.inv() * rotation).magnitude() <= motion.max_rotation + 1e-9
        assert (previous.inv() * rotation).magnitude() <= motion.angular_speed * dt + 1e-9
        previous, previous_time = rotation, (index / motion.fps)
    np.testing.assert_allclose(previous.as_matrix(), expected.as_matrix(), atol=1e-9)


def test_each_waypoint_is_clamped_from_the_previous_clamped_waypoint():
    motion = Motion(**SHORT_BOUNDS)
    path = [Waypoint(x=x, y=0, z=0, roll=0, pitch=0, yaw=0, gripper=0) for x in (0.2, 0.02)]
    bounded, trajectory = motion.trajectory(geom.Transform3D([0, 0, 0]), 0, path)
    assert [waypoint.x for waypoint in bounded] == pytest.approx([0.05, 0.02])
    xs = [action[keys.ROBOT_COMMAND].pose.translation[0] for action in trajectory]
    assert max(xs) == pytest.approx(0.05)
    assert xs[-1] == pytest.approx(0.02)
    assert len(trajectory) == round((0.05 + 0.03) / motion.linear_speed * motion.fps)


@pytest.mark.parametrize('grip', [0.0, 1.0])
def test_a_waypoint_that_changes_the_gripper_holds_still_at_it(grip):
    motion = Motion(**SHORT_BOUNDS)
    path = [
        Waypoint(x=0.01, y=0, z=0, roll=0, pitch=0, yaw=0, gripper=0),
        Waypoint(x=0.01, y=0, z=0, roll=0, pitch=0, yaw=0, gripper=1),
        Waypoint(x=0.01, y=0, z=0.01, roll=0, pitch=0, yaw=0, gripper=1),
    ]
    _, trajectory = motion.trajectory(geom.Transform3D([0, 0, 0]), grip, path)
    hold = math.ceil(motion.grip_hold * motion.fps)
    segment = round(0.01 / motion.linear_speed * motion.fps)
    holds = (grip != 0) + 1
    assert len(trajectory) == 2 * segment + 1 + holds * hold
    closing = trajectory[-segment - hold :]
    assert [action[keys.TARGET_GRIP] for action in closing] == [1] * len(closing)
    np.testing.assert_allclose(closing[0][keys.ROBOT_COMMAND].pose.translation, [0.01, 0, 0])
    np.testing.assert_allclose(closing[hold - 1][keys.ROBOT_COMMAND].pose.translation, [0.01, 0, 0])
    np.testing.assert_allclose(trajectory[-1][keys.ROBOT_COMMAND].pose.translation, [0.01, 0, 0.01])


@pytest.mark.parametrize(
    'change', [{'x': float('nan')}, {'yaw': float('inf')}, {'gripper': 2}, {'extra': 1}, {'x': '0'}]
)
def test_invalid_tool_numbers_cannot_reach_motion(change):
    values = {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': 0}
    with pytest.raises(ValidationError):
        Waypoint(**(values | change))


def test_a_path_needs_a_waypoint():
    with pytest.raises(ValidationError):
        MovePath(waypoints=[], note='approach')


@pytest.mark.parametrize(
    'change', [{'fps': 0}, {'max_translation': float('inf')}, {'linear_speed': 0.00001}, {'max_waypoints': 0.5}]
)
def test_invalid_motion_configuration_fails(change):
    with pytest.raises(ValueError):
        Motion(**change)


@pytest.mark.parametrize('change', [{'fps': 0.1}, {'max_translation': 0.5}, {'max_translation': 0.45, 'fps': 1.25}])
def test_motion_duration_bound_includes_sampling_and_expiry(change):
    with pytest.raises(ValueError, match='10 second trajectory'):
        Motion(**SHORT_BOUNDS | change)


@pytest.mark.parametrize(('fps', 'max_translation'), [(0.2, 0.05), (2, 0.475)])
def test_motion_can_expire_at_duration_bound(fps, max_translation):
    motion = Motion(**SHORT_BOUNDS | {'max_translation': max_translation, 'fps': fps})
    start = geom.Transform3D([0, 0, 0])
    target = Waypoint(x=max_translation, y=0, z=0, roll=0, pitch=0, yaw=0, gripper=0)
    _, trajectory = motion.trajectory(start, 0, [target])
    assert (len(trajectory) + 1) / motion.fps == pytest.approx(10)
