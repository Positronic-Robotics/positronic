"""The YAM arm observations of a remote env: the per-arm wire fields decoded into a ``YamState`` and a grip."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from positronic import geom, keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.yam import YamState
from positronic.simulator.env_server import protocol


def arm_observations(
    raw_obs: dict[str, Any],
    arms: Sequence[str],
    measure: Callable[[geom.Transform3D, np.ndarray, str], geom.Transform3D],
) -> dict[str, Any]:
    """The robot state and the grip of each arm in ``arms``.

    ``measure`` takes the env's pose and joints of an arm and gives the pose at the embodiment's grip site.
    """
    obs: dict[str, Any] = {}
    for arm in arms:
        joints = raw_obs[protocol.arm_channel(protocol.OBS_JOINT_POS, arm)]
        pose = measure(
            geom.Transform3D(
                raw_obs[protocol.arm_channel(protocol.OBS_EEF_POS, arm)],
                geom.Rotation.from_quat(raw_obs[protocol.arm_channel(protocol.OBS_EEF_QUAT, arm)]),
            ),
            joints,
            arm,
        )
        state = YamState()
        state.encode(joints, raw_obs[protocol.arm_channel(protocol.OBS_JOINT_VEL, arm)], pose, RobotStatus.AVAILABLE)
        obs[keys.arm_channel(keys.ROBOT_STATE, arm)] = state
        obs[keys.arm_channel(keys.GRIP, arm)] = float(raw_obs[protocol.arm_channel(protocol.OBS_GRIP, arm)])
    return obs
