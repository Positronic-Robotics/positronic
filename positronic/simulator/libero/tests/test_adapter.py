"""LiberoAdapter's frame, on synthetic observations; no LIBERO installation is required."""

import numpy as np

import pimm
from positronic import geom, keys
from positronic.cfg import codecs as cfg_codecs
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.models import PANDA_EE_FRAME
from positronic.policy.codecs import ChangeEEFrame
from positronic.simulator.env_server import protocol
from positronic.simulator.libero.adapter import LiberoAdapter

_EEF_POSE = geom.Transform3D(np.array([0.4, 0.1, 0.3]), geom.Rotation.from_euler([0.1, 0.2, 0.3]))
_ROTMAT = geom.Rotation.Representation.ROTATION_MATRIX
_QUAT = geom.Rotation.Representation.QUAT


def _raw_obs(eef_pose: geom.Transform3D) -> dict:
    return {
        'eef_pos': eef_pose.translation,
        'eef_quat': eef_pose.rotation.as_quat_xyzw,
        'joint_pos': np.zeros(7),
        'joint_vel': np.zeros(7),
        'grip': 0.0,
    }


def test_the_eef_pose_is_reported_at_the_gripper_site_of_the_sim_panda():
    """The adapter takes the env's grip site to be the panda's ``EE_LINK``, which sits ``PANDA_EE_FRAME`` off it."""
    reported = LiberoAdapter({}).observations(_raw_obs(_EEF_POSE))[keys.ROBOT_STATE].ee_pose
    np.testing.assert_allclose(reported.as_matrix, (_EEF_POSE * PANDA_EE_FRAME.inv).as_matrix, atol=1e-6)


def test_the_libero_checkpoint_frame_sees_the_pose_the_env_reports():
    reported = LiberoAdapter({}).observations(_raw_obs(_EEF_POSE))[keys.ROBOT_STATE].ee_pose
    frame = cfg_codecs.panda_ee_frame.instantiate()
    seen = ChangeEEFrame(frame.site, frame.offset).encode({keys.EE_POSE: reported.as_vector(_QUAT)})[keys.EE_POSE]
    np.testing.assert_allclose(geom.Transform3D.from_vector(seen, _QUAT).as_matrix, _EEF_POSE.as_matrix, atol=1e-6)


def test_a_reported_pose_commanded_back_reaches_the_env_at_its_grip_site():
    adapter = LiberoAdapter({})
    reported = adapter.observations(_raw_obs(_EEF_POSE))[keys.ROBOT_STATE].ee_pose
    wire = adapter.action({keys.ROBOT_COMMAND: pimm.Message(roboarm_command.CartesianPosition(reported))})
    np.testing.assert_allclose(wire[keys.ROBOT_COMMAND][protocol.COMMAND_POSE], _EEF_POSE.as_vector(_ROTMAT), atol=1e-6)
