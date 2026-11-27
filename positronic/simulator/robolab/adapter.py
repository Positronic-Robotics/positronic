"""The client-side RoboLab adapters, between the canonical embodiment contract and the RoboLab wire.

The env server owns the action encoding: joint targets and IK. ``RobolabDroidAdapter`` serves the DROID rig and
``RobolabYamAdapter`` the two-arm YAM.
"""

from abc import abstractmethod
from typing import Any

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm.models import ROBOLAB_EE_FRAME
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol, yam_observations
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.mujoco.sim import MujocoFrankaState
from positronic.simulator.robolab import keys as robolab_keys


class RobolabAdapter(WireCommandAdapter):
    """The task, camera, privileged and terminal mappings that every robot of the RoboLab env server shares."""

    def __init__(self, camera_dict: dict[str, str], env_control_frame: geom.Transform3D | None = None):
        super().__init__(env_control_frame)
        self._camera_dict = camera_dict  # logical observation name -> the RoboLab obs image key

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                eval_keys.TASK: record[robolab_keys.RECORD_NAME],
                robolab_keys.EPISODE_LENGTH: record[robolab_keys.RECORD_EPISODE_LENGTH_S],
                robolab_keys.INSTRUCTION_TYPE: record[robolab_keys.RECORD_INSTRUCTION_TYPE],
            }
            for record in records
        ]

    def _reset_token(self, params: dict[str, Any]) -> Any:
        # No seed rides the token: RoboLab's eval path has no seed hook, so a recorded seed would only mislead.
        return {
            robolab_keys.TOKEN_TASK: params[eval_keys.TASK],
            robolab_keys.TOKEN_INSTRUCTION_TYPE: params[robolab_keys.INSTRUCTION_TYPE],
            robolab_keys.TOKEN_CAMERAS: params[robolab_keys.CAMERAS],
            **(
                {robolab_keys.TOKEN_EPISODE_LENGTH_S: params[robolab_keys.EPISODE_LENGTH_OVERRIDE]}
                if robolab_keys.EPISODE_LENGTH_OVERRIDE in params
                else {}
            ),
        }

    @abstractmethod
    def _arm_observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """The robot state and the grip of each arm, from the env's raw observation payload."""

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        obs = self._arm_observations(raw_obs)
        for logical, env_key in self._camera_dict.items():
            if env_key not in raw_obs:
                rendered = sorted(k for k, v in raw_obs.items() if isinstance(v, np.ndarray) and v.ndim == 3)
                raise ValueError(
                    f'camera_dict maps {logical!r} to {env_key!r}, which the RoboLab env server does not '
                    f'render; it emits {rendered}'
                )
            frame = raw_obs[env_key]  # Isaac tiled cameras render top-down already — no flip
            adapter = pimm.shared_memory.NumpySMAdapter(shape=frame.shape, dtype=frame.dtype)
            adapter.array[:] = frame
            obs[logical] = adapter
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {name: raw_obs[name] for name in (robolab_keys.SUBTASK, robolab_keys.GT_STATE)}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        # ``done`` covers termination and truncation, so the trial ends either way; ``success`` is True only
        # when the task's success condition fired, keeping timeouts honest. A fault adds its value beside the verdict.
        if not result[protocol.FRAME_DONE]:
            return None
        payload = {eval_keys.SUCCESS: bool(result[protocol.FRAME_SUCCESS])}
        if robolab_keys.FRAME_FAULT in result:
            payload[robolab_keys.FAULT] = result[robolab_keys.FRAME_FAULT]
        return payload


class RobolabDroidAdapter(RobolabAdapter):
    """The DROID rig: one arm, measured and driven at ``ROBOLAB_EE_FRAME``."""

    def __init__(self, camera_dict: dict[str, str]):
        # Converting here lets one checkpoint run unchanged on RoboLab and on the rig it stands in for.
        super().__init__(camera_dict, ROBOLAB_EE_FRAME)

    def _arm_observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # The env reports the eef pose in the control frame IK drives; ``eef_quat`` is scalar-first (wxyz),
        # so ``from_quat`` is the matching decode.
        eef_pose = geom.Transform3D(
            raw_obs[protocol.OBS_EEF_POS], geom.Rotation.from_quat(raw_obs[protocol.OBS_EEF_QUAT])
        )
        ee_pose = self._measure(eef_pose, raw_obs[protocol.OBS_JOINT_POS])
        state = MujocoFrankaState()
        state.encode(raw_obs[protocol.OBS_JOINT_POS], raw_obs[protocol.OBS_JOINT_VEL], ee_pose)
        return {keys.ROBOT_STATE: state, keys.GRIP: float(raw_obs[protocol.OBS_GRIP])}


class RobolabYamAdapter(RobolabAdapter):
    """The two-arm YAM: per arm, the env measures at ``GRIPPER_SITE``, so the control frame is the identity."""

    def _arm_observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return yam_observations.arm_observations(raw_obs, keys.BIMANUAL_ARMS, self._measure)
