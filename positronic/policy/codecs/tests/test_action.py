import numpy as np
import pytest

import positronic.drivers.roboarm.command as cmd_module
from positronic import keys as obs_keys
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.drivers.roboarm.command import Impedance, JointDelta
from positronic.geom import Rotation
from positronic.policy.codecs import ACTION, SetControlMode
from positronic.policy.codecs.action import AbsoluteJointsAction, AbsolutePositionAction, IKJointsAction


def test_absolute_position_action_encode_decode_quat():
    ts = [1000, 2000]
    q = [Rotation.identity for _ in ts]
    t = [np.array([0.1, -0.2, 0.3], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    pose = [np.concatenate([t[i], q[i].as_quat]).astype(np.float32) for i in range(len(ts))]

    ep = EpisodeContainer({obs_keys.TARGET_EE_POSE: DummySignal(ts, pose), obs_keys.TARGET_GRIP: DummySignal(ts, g)})

    act = AbsolutePositionAction(obs_keys.TARGET_EE_POSE, obs_keys.TARGET_GRIP, Rotation.Representation.QUAT)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 4 quat + 3 trans + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({ACTION: vec})
    command = decoded[obs_keys.ROBOT_COMMAND]
    target_grip = decoded[obs_keys.TARGET_GRIP]
    assert isinstance(command, cmd_module.CartesianPosition)
    np.testing.assert_allclose(command.pose.translation, t[0], atol=1e-6)
    np.testing.assert_allclose(command.pose.rotation.as_quat, q[0].as_quat, atol=1e-6)
    assert np.isclose(target_grip, g[0])


def test_absolute_joints_action_encode_decode():
    ts = [1000, 2000]
    joints = [np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7], dtype=np.float32) for _ in ts]
    g = [0.5, 0.6]

    ep = EpisodeContainer({obs_keys.TARGET_JOINTS: DummySignal(ts, joints), obs_keys.TARGET_GRIP: DummySignal(ts, g)})

    act = AbsoluteJointsAction(obs_keys.TARGET_JOINTS, obs_keys.TARGET_GRIP, num_joints=7)
    sig = act._encode_episode(ep)
    vec = list(sig)[0][0]
    assert vec.shape == (8,)  # 7 joints + 1 grip
    assert vec.dtype == np.float32

    decoded = act._decode_single({ACTION: vec})
    command = decoded[obs_keys.ROBOT_COMMAND]
    target_grip = decoded[obs_keys.TARGET_GRIP]
    assert isinstance(command, cmd_module.JointPosition)
    np.testing.assert_allclose(command.positions, joints[0], atol=1e-6)
    assert np.isclose(target_grip, g[0])


IMPEDANCE = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)


class TestSetControlMode:
    def test_every_command_in_a_chunk_carries_the_mode(self):
        chunk = [
            {obs_keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7))},
            {obs_keys.ROBOT_COMMAND: JointDelta(velocities=np.ones(7))},
            {obs_keys.TARGET_GRIP: 0.5},
        ]
        decoded = SetControlMode(IMPEDANCE).decode(chunk)
        assert isinstance(decoded, list)
        for action in decoded[:2]:
            assert isinstance(action, dict)
            assert action[obs_keys.ROBOT_COMMAND].mode == IMPEDANCE
        assert obs_keys.ROBOT_COMMAND not in decoded[2]

    def test_a_single_action_carries_the_mode(self):
        decoded = SetControlMode(IMPEDANCE).decode({obs_keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7))})
        assert isinstance(decoded, dict)
        assert decoded[obs_keys.ROBOT_COMMAND].mode == IMPEDANCE

    def test_every_arm_channel_is_stamped(self):
        """A bimanual action names a channel per arm, and both execute under the mode."""
        action = {
            f'{obs_keys.ROBOT_COMMAND}.left': JointDelta(velocities=np.zeros(7)),
            f'{obs_keys.ROBOT_COMMAND}.right': JointDelta(velocities=np.ones(7)),
            obs_keys.TARGET_JOINTS: np.zeros(7),  # in the command family by name, but a vector
            obs_keys.TARGET_GRIP: 0.5,
        }
        decoded = SetControlMode(IMPEDANCE).decode(action)
        assert isinstance(decoded, dict)
        assert decoded[f'{obs_keys.ROBOT_COMMAND}.left'].mode == IMPEDANCE
        assert decoded[f'{obs_keys.ROBOT_COMMAND}.right'].mode == IMPEDANCE
        np.testing.assert_array_equal(decoded[obs_keys.TARGET_JOINTS], np.zeros(7))


def test_non_deliverable_codec_is_rejected():
    with pytest.raises(NotImplementedError, match='IKJointsAction'):
        IKJointsAction(solver_cls=None).to_spec()
