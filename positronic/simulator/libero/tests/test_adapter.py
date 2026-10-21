"""The LIBERO adapter's frames: the env measures and drives at its eef site, and the embodiment at the flange."""

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.models import PANDA_EE_FRAME
from positronic.simulator.env_server import protocol
from positronic.simulator.libero.adapter import LiberoAdapter

_EEF_POSE = geom.Transform3D(np.array([0.4, 0.1, 0.3]), geom.Rotation.from_euler([0.1, 0.2, 0.3]))
_ROTMAT = geom.Rotation.Representation.ROTATION_MATRIX


def _observed_flange(adapter: LiberoAdapter) -> geom.Transform3D:
    raw = {
        'eef_pos': _EEF_POSE.translation,
        'eef_quat': _EEF_POSE.rotation.as_quat_xyzw,
        'joint_pos': np.zeros(7),
        'joint_vel': np.zeros(7),
        'grip': 0.0,
    }
    return adapter.observations(raw)[keys.ROBOT_STATE].ee_pose


def test_the_eef_site_sits_at_the_old_panda_point_from_the_reported_flange():
    """The eef site keeps its place on the old panda ``default``, so a LIBERO recording keeps its meaning."""
    flange = _observed_flange(LiberoAdapter({}))
    np.testing.assert_allclose((flange * PANDA_EE_FRAME).as_vector(_ROTMAT), _EEF_POSE.as_vector(_ROTMAT), atol=1e-6)


def test_a_reported_flange_pose_commanded_back_reaches_the_env_at_its_eef_site():
    adapter = LiberoAdapter({})
    flange = _observed_flange(adapter)
    wire = adapter.action({keys.ROBOT_COMMAND: pimm.Message(roboarm_command.CartesianPosition(flange))})
    np.testing.assert_allclose(wire[keys.ROBOT_COMMAND][protocol.COMMAND_POSE], _EEF_POSE.as_vector(_ROTMAT), atol=1e-6)
