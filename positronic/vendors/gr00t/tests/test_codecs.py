import numpy as np
import pytest

from positronic import keys
from positronic.geom import Rotation
from positronic.vendors.gr00t.codecs import ee_quat, joints_traj

_T0_OBS = {'obs_time_ns': 0}


def test_ee_quat_decodes_modality_keyed_actions():
    """GR00T models return ``{'ee_pose': ..., 'grip': ...}`` per action step, not a flat ``action``
    vector; the codec chain must convert this format into robot commands."""
    codec = ee_quat()

    ee_pose = np.concatenate([Rotation.identity.as_quat, [0.1, 0.2, 0.3]]).astype(np.float32)
    model_output = [{'ee_pose': ee_pose, 'grip': np.float32(0.5)} for _ in range(3)]

    decoded = codec.decode(model_output, context=_T0_OBS)
    assert len(decoded) == 4  # 3 actions + timestamp sentinel
    for d in decoded[:-1]:
        assert keys.ROBOT_COMMAND in d
        assert 'target_grip' in d
        assert 'timestamp' in d
    assert decoded[-1] == {'timestamp': pytest.approx(3 / 15.0)}  # timestamp sentinel


def test_joints_traj_decodes_modality_keyed_actions():
    codec = joints_traj()

    joint_pos = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32)
    model_output = [{'joint_position': joint_pos, 'grip': np.float32(0.8)} for _ in range(3)]

    decoded = codec.decode(model_output, context=_T0_OBS)
    assert len(decoded) == 4  # 3 actions + timestamp sentinel
    for d in decoded[:-1]:
        assert keys.ROBOT_COMMAND in d
        assert 'target_grip' in d
    assert decoded[-1] == {'timestamp': pytest.approx(3 / 15.0)}  # timestamp sentinel
