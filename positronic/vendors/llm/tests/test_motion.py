import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from positronic import geom, keys
from positronic.vendors.llm.motion import Motion, MoveTo


@pytest.mark.parametrize('quaternion_sign', [1, -1])
def test_motion_uses_short_rotation_path_and_respects_speed(quaternion_sign):
    start = geom.Transform3D([0, 0, 0], geom.Rotation.from_euler([0, 0, math.radians(179)]))
    start.rotation = geom.Rotation.from_quat(quaternion_sign * start.rotation.as_quat)
    target = MoveTo(x=0.05, y=0, z=0, roll=0, pitch=0, yaw=math.radians(-179), gripper=0.7, note='approach')
    motion = Motion()
    bounded, trajectory = motion.trajectory(start, target)
    assert bounded == target
    previous, previous_time = start, 0
    for action in trajectory:
        pose = action[keys.ROBOT_COMMAND].pose
        dt = action[keys.ACTION_TIMESTAMP] - previous_time
        assert 0 < dt <= 1 / motion.fps + 1e-9
        assert np.linalg.norm(pose.translation - previous.translation) <= motion.linear_speed * dt + 1e-9
        rotation = Rotation.from_matrix(pose.rotation.as_rotation_matrix @ previous.rotation.as_rotation_matrix.T)
        assert rotation.magnitude() <= motion.angular_speed * dt + 1e-9
        assert action[keys.TARGET_GRIP] == 0.7
        previous, previous_time = pose, action[keys.ACTION_TIMESTAMP]
    np.testing.assert_allclose(previous.translation, target.pose.translation)
    np.testing.assert_allclose(previous.rotation.as_rotation_matrix, target.pose.rotation.as_rotation_matrix, atol=1e-8)
    assert previous_time == pytest.approx(1.0)


@pytest.mark.parametrize(
    'offset', [(0, 0, 0), (0.01, 0, 0), (0.05, 0, 0), (0.051, 0, 0), (0.08, -0.06, 0.12), (1e200, -1e200, 1e200)]
)
def test_translation_clamps_distance_preserving_direction(offset):
    start = geom.Transform3D([0.3, -0.2, 0.1])
    x, y, z = start.translation + offset
    target = MoveTo(x=x, y=y, z=z, roll=0, pitch=0, yaw=0, gripper=0.7, note='approach')
    original = target.model_dump()
    motion = Motion()
    bounded, trajectory = motion.trajectory(start, target)
    offset = np.asarray(offset)
    distance = math.hypot(*offset)
    expected = offset / distance * motion.max_translation if distance > motion.max_translation else offset
    np.testing.assert_allclose(bounded.pose.translation, start.translation + expected)
    assert target.model_dump() == original
    assert bounded.gripper == target.gripper
    assert bounded.note == target.note
    if distance < motion.max_translation:
        assert bounded == target
    previous, previous_time = start, 0
    for action in trajectory:
        pose = action[keys.ROBOT_COMMAND].pose
        dt = action[keys.ACTION_TIMESTAMP] - previous_time
        assert np.linalg.norm(pose.translation - start.translation) <= motion.max_translation + 1e-9
        assert np.linalg.norm(pose.translation - previous.translation) <= motion.linear_speed * dt + 1e-9
        np.testing.assert_allclose(pose.rotation.as_rotation_matrix, start.rotation.as_rotation_matrix)
        assert action[keys.TARGET_GRIP] == target.gripper
        previous, previous_time = pose, action[keys.ACTION_TIMESTAMP]
    np.testing.assert_allclose(previous.translation, bounded.pose.translation)


@pytest.mark.parametrize('quaternion_sign', [1, -1])
@pytest.mark.parametrize('angle_degrees', [0, 5, 20, 21, 179, 181])
def test_rotation_clamps_shortest_arc_independently_of_translation(quaternion_sign, angle_degrees):
    initial = Rotation.from_euler('xyz', [0.2, -0.3, math.radians(179)])
    requested = initial * Rotation.from_rotvec(np.array([1, 2, 3]) / math.sqrt(14) * math.radians(angle_degrees))
    start = geom.Transform3D([0, 0, 0], geom.Rotation.from_rotation_matrix(initial.as_matrix()))
    start.rotation = geom.Rotation.from_quat(quaternion_sign * start.rotation.as_quat)
    roll, pitch, yaw = requested.as_euler('xyz')
    target = MoveTo(x=0.2, y=0, z=0, roll=roll, pitch=pitch, yaw=yaw, gripper=0.3, note='turn')
    motion = Motion()
    bounded, trajectory = motion.trajectory(start, target)
    relative = initial.inv() * requested
    fraction = min(1, motion.max_rotation / relative.magnitude()) if relative.magnitude() > 0 else 1
    expected = initial * Rotation.from_rotvec(relative.as_rotvec() * fraction)
    np.testing.assert_allclose(bounded.pose.rotation.as_rotation_matrix, expected.as_matrix(), atol=1e-9)
    assert bounded.x == pytest.approx(motion.max_translation)
    previous, previous_time = initial, 0
    for action in trajectory:
        rotation = Rotation.from_matrix(action[keys.ROBOT_COMMAND].pose.rotation.as_rotation_matrix)
        dt = action[keys.ACTION_TIMESTAMP] - previous_time
        assert (initial.inv() * rotation).magnitude() <= motion.max_rotation + 1e-9
        assert (previous.inv() * rotation).magnitude() <= motion.angular_speed * dt + 1e-9
        previous, previous_time = rotation, action[keys.ACTION_TIMESTAMP]
    np.testing.assert_allclose(previous.as_matrix(), expected.as_matrix(), atol=1e-9)


@pytest.mark.parametrize(
    'change', [{'x': float('nan')}, {'yaw': float('inf')}, {'gripper': 2}, {'extra': 1}, {'x': '0'}]
)
def test_invalid_tool_numbers_cannot_reach_motion(change):
    values = {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': 0, 'note': 'approach'}
    with pytest.raises(ValidationError):
        MoveTo(**(values | change))


@pytest.mark.parametrize('change', [{'fps': 0}, {'max_translation': float('inf')}, {'linear_speed': 0.00001}])
def test_invalid_motion_configuration_fails(change):
    with pytest.raises(ValueError):
        Motion(**change)
