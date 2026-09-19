"""``RobolabAdapter``: the canonical embodiment contract <-> RoboLab's raw obs/command payloads, client-side.

Runs in positronic's interpreter (the RoboLab env server runs in its own Isaac Lab interpreter). The command
side is ``WireCommandAdapter``'s forwarding; all action encoding — the joint-target conversion and the
differential IK that bridges Cartesian commands — lives server-side where the articulation model is, so the
adapter's only geometry is the constant frame offset between what the env measures and what the rig reports.
"""

from typing import Any

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm.models import DROID_EE_FRAME
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.mujoco.sim import MujocoFrankaState
from positronic.simulator.robolab import keys as robolab_keys


class RobolabAdapter(WireCommandAdapter):
    def __init__(self, camera_dict: dict[str, str]):
        # Converting here is what lets one checkpoint run on RoboLab and on the rig it stands in for unchanged.
        super().__init__(DROID_EE_FRAME)
        self._camera_dict = camera_dict  # logical observation name -> the RoboLab obs image key

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {eval_keys.TASK: record['name'], robolab_keys.EPISODE_LENGTH: record['episode_length_s']}
            for record in records
        ]

    def _reset_token(self, params: dict[str, Any]) -> Any:
        # No seed rides the token: RoboLab's eval path has no seed hook, so a recorded seed would only mislead.
        return {'task': params[eval_keys.TASK], 'instruction_type': params[robolab_keys.INSTRUCTION_TYPE]}

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # The env reports the eef pose in the control frame IK drives; ``eef_quat`` is scalar-first (wxyz),
        # so ``from_quat`` is the matching decode.
        eef_pose = geom.Transform3D(
            raw_obs[robolab_keys.OBS_EEF_POS], geom.Rotation.from_quat(raw_obs[robolab_keys.OBS_EEF_QUAT])
        )
        ee_pose = eef_pose * self.env_control_frame.inv
        state = MujocoFrankaState()
        state.encode(raw_obs[robolab_keys.OBS_JOINT_POS], raw_obs[robolab_keys.OBS_JOINT_VEL], ee_pose)
        obs: dict[str, Any] = {keys.ROBOT_STATE: state, keys.GRIP: float(raw_obs[robolab_keys.OBS_GRIP])}
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
        return {robolab_keys.OBS_SUBTASK: raw_obs[robolab_keys.OBS_SUBTASK]}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        # ``done`` covers termination and truncation, so the trial ends either way; ``success`` is True only
        # when the task's success condition fired, keeping timeouts honest.
        return {eval_keys.SUCCESS: bool(result[protocol.FRAME_SUCCESS])} if result[protocol.FRAME_DONE] else None
