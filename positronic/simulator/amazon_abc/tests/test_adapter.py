"""The commands ``AbcAdapter`` sends for each arm, against the poses its observations measured."""

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import command as roboarm_command
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.amazon_abc.adapter import AbcAdapter
from positronic.simulator.env_server import protocol

_ROTMAT = geom.Rotation.Representation.ROTATION_MATRIX
_LEFT, _RIGHT = mapping.ARMS
_EEF = {
    _LEFT: geom.Transform3D(np.array([0.3, 0.25, 0.4]), geom.Rotation.from_euler([0.1, 0.2, 0.3])),
    _RIGHT: geom.Transform3D(np.array([0.35, -0.2, 0.3]), geom.Rotation.from_euler([-0.3, 0.1, -0.2])),
}


def _raw_obs() -> dict:
    raw: dict = {}
    for arm, eef in _EEF.items():
        raw[protocol.arm_channel(protocol.OBS_EEF_POS, arm)] = eef.translation
        raw[protocol.arm_channel(protocol.OBS_EEF_QUAT, arm)] = eef.rotation.as_quat
        raw[protocol.arm_channel(protocol.OBS_JOINT_POS, arm)] = np.zeros(mapping.ARM_JOINTS)
        raw[protocol.arm_channel(protocol.OBS_JOINT_VEL, arm)] = np.zeros(mapping.ARM_JOINTS)
        raw[protocol.arm_channel(protocol.OBS_GRIP, arm)] = 0.0
    return raw


def test_a_framed_delta_on_each_arm_composes_at_that_arms_measured_pose():
    adapter = AbcAdapter({})
    adapter.observations(_raw_obs())
    frame = geom.Transform3D(np.array([0.0, 0.0, 0.05]), geom.Rotation.from_euler([0.0, np.pi / 2, 0.0]))
    delta = geom.Transform3D(np.array([0.01, -0.02, 0.03]), geom.Rotation.from_euler([0.0, 0.0, 0.2]))
    cmd = roboarm_command.CartesianDelta(delta, frame=frame)
    channels = {arm: keys.arm_channel(keys.ROBOT_COMMAND, arm) for arm in mapping.ARMS}

    action = adapter.action({channel: pimm.Message(cmd) for channel in channels.values()})

    for arm, channel in channels.items():
        wired = action[channel]
        assert wired[protocol.COMMAND_TYPE] == protocol.CARTESIAN
        np.testing.assert_allclose(wired[protocol.COMMAND_POSE], cmd.apply(_EEF[arm]).as_vector(_ROTMAT), atol=1e-12)
