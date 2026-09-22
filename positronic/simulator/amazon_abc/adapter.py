"""``AbcAdapter``: the client side, between the canonical embodiment contract and ABC's raw per-arm payloads."""

from typing import Any

import numpy as np

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.yam_state import YamState
from positronic.eval import keys as eval_keys
from positronic.simulator.amazon_abc import keys as abc_keys
from positronic.simulator.amazon_abc import mapping
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.adapter import WireCommandAdapter

_LEFT, _RIGHT = mapping.ARMS
CAMERAS = {keys.EXTERIOR_IMAGE: 'top', keys.WRIST_LEFT_IMAGE: _LEFT, keys.WRIST_RIGHT_IMAGE: _RIGHT}


class AbcAdapter(WireCommandAdapter):
    def __init__(self, camera_dict: dict[str, str]):
        super().__init__()
        self._camera_dict = camera_dict  # logical observation name -> the ABC camera name

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{eval_keys.TASK: r[mapping.TASK_NAME]} for r in records]

    def _reset_token(self, params: dict[str, Any]) -> Any:
        return {
            mapping.TOKEN_TASK: params[eval_keys.TASK],
            mapping.TOKEN_SEED: params.get(eval_keys.SEED),
            mapping.TOKEN_CAMERA_HEIGHT: params[abc_keys.CAMERA_HEIGHT],
            mapping.TOKEN_CAMERA_WIDTH: params[abc_keys.CAMERA_WIDTH],
        }

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        for arm in mapping.ARMS:
            pose = geom.Transform3D(
                raw_obs[protocol.arm_channel(mapping.OBS_EEF_POS, arm)],
                geom.Rotation.from_quat(raw_obs[protocol.arm_channel(mapping.OBS_EEF_QUAT, arm)]),
            )
            state = YamState()
            state.encode(
                raw_obs[protocol.arm_channel(mapping.OBS_JOINT_POS, arm)],
                raw_obs[protocol.arm_channel(mapping.OBS_JOINT_VEL, arm)],
                pose,
                RobotStatus.AVAILABLE,
            )
            obs[keys.arm_channel(keys.ROBOT_STATE, arm)] = state
            obs[keys.arm_channel(keys.GRIP, arm)] = float(raw_obs[protocol.arm_channel(mapping.OBS_GRIP, arm)])
        for logical, camera in self._camera_dict.items():
            # ABC renders channels-first; the wire carries images as height, width, channels.
            frame = np.ascontiguousarray(raw_obs[camera].transpose(1, 2, 0))
            obs[logical] = pimm.shared_memory.NumpySMAdapter.lazy_init(frame, None)
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {mapping.OBS_SIM_STATE: raw_obs[mapping.OBS_SIM_STATE]}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        return {eval_keys.SUCCESS: bool(result[protocol.FRAME_SUCCESS])} if result[protocol.FRAME_DONE] else None
