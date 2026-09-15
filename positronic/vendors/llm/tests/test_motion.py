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
    trajectory = motion.trajectory(start, target)
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


@pytest.mark.parametrize('change', [{'x': 0.051}, {'roll': math.radians(21)}])
def test_oversized_motion_is_rejected(change):
    values = {'x': 0, 'y': 0, 'z': 0, 'roll': 0, 'pitch': 0, 'yaw': 0, 'gripper': 0, 'note': 'approach'}
    target = MoveTo(**(values | change))
    with pytest.raises(ValueError, match='Split the move'):
        Motion().trajectory(geom.Transform3D.identity, target)


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
