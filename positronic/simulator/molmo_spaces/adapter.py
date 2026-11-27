"""Client-side mapping between MolmoSpaces observations and the canonical embodiment contract."""

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import pimm
from positronic import geom, keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.models import bundled_franka_model
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.mujoco.sim import MujocoFrankaState


class ShoulderSide(StrEnum):
    LEFT = 'left'
    RIGHT = 'right'


# Camera aliases in MolmoSpaces Pi-policy preference order, and the shoulder cameras by robot side.
WRIST_CAMERAS = ('wrist_camera_zed_mini', 'wrist_camera')
SHOULDER_CAMERAS = {
    ShoulderSide.LEFT: mapping.LEFT_SHOULDER_CAMERAS,
    ShoulderSide.RIGHT: (mapping.RIGHT_SHOULDER_CAMERA,),
}


_EXTERIOR_SLOTS = (keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2)


def camera_dict(exterior_sides: Sequence[ShoulderSide]) -> dict[str, tuple[str, ...]]:
    """The aliases of each camera the embodiment declares; ``exterior_sides`` names the shoulder of each exterior."""
    slots = zip(_EXTERIOR_SLOTS[: len(exterior_sides)], exterior_sides, strict=True)
    return {keys.WRIST_IMAGE: WRIST_CAMERAS, **{slot: SHOULDER_CAMERAS[side] for slot, side in slots}}


class MolmoAdapter(WireCommandAdapter):
    _DIMENSION_KEYS = dict(zip(mapping.BenchmarkPath._fields, molmo_keys.BENCHMARK_DIMENSIONS, strict=True))

    def __init__(
        self,
        cameras: Mapping[str, tuple[str, ...]],
        with_right_shoulder: bool,
        with_object_image_points: bool = True,
        hook_file: mapping.HookFile | None = None,
    ):
        # The env measures and drives at the 2F-85 grip point of this model, ``GRIPPER_SITE``, as the rig does.
        self.robot_model = bundled_franka_model()
        super().__init__()
        self._cameras = cameras
        self._with_right_shoulder = with_right_shoulder
        self._with_object_image_points = with_object_image_points
        self._hook_file = hook_file

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **{key: record[dimension] for dimension, key in self._DIMENSION_KEYS.items()},
                eval_keys.TASK: record[mapping.TASK_NAME],
                molmo_keys.EPISODE_INDEX: record[mapping.TOKEN_EPISODE_INDEX],
                molmo_keys.TASK_HORIZON: record[mapping.TASK_HORIZON_SEC],
                molmo_keys.ENV_HOOK: record[mapping.TASK_ENV_HOOK],
                **(
                    {molmo_keys.TASK_HORIZON_OVERRIDE: record[mapping.TASK_HORIZON_OVERRIDE_SEC]}
                    if record.get(mapping.TASK_HORIZON_OVERRIDE_SEC) is not None
                    else {}
                ),
            }
            for record in records
        ]

    def _reset_token(self, params: dict[str, Any]) -> Any:
        return {
            **{dimension: params[key] for dimension, key in self._DIMENSION_KEYS.items()},
            mapping.TOKEN_EPISODE_INDEX: params[molmo_keys.EPISODE_INDEX],
            mapping.TOKEN_SEED: params.get(eval_keys.SEED),
            **(
                {mapping.TOKEN_HORIZON_SEC: params[molmo_keys.TASK_HORIZON_OVERRIDE]}
                if molmo_keys.TASK_HORIZON_OVERRIDE in params
                else {}
            ),
            mapping.WITH_RIGHT_SHOULDER: self._with_right_shoulder,
            mapping.WITH_OBJECT_IMAGE_POINTS: self._with_object_image_points,
            mapping.TOKEN_SUCCESS_PROFILE: params[molmo_keys.SUCCESS_PROFILE],
            **(self._hook_file.request() if self._hook_file is not None else {}),
        }

    def check_reset(self, params: dict[str, Any], frame: dict[str, Any]) -> None:
        meta = frame[protocol.FRAME_META]
        asked = params[molmo_keys.SUCCESS_PROFILE]
        stated = meta.get(mapping.META_SUCCESS_PROFILE)
        if stated != asked:
            raise RuntimeError(
                f'The trial asks for success profile {asked!r}, but the env server judges by {stated!r}. '
                'A server that predates success profiles states none.'
            )
        asked_hook = self._hook_file.record if self._hook_file is not None else None
        ran_hook = meta.get(mapping.META_ENV_HOOK)
        if ran_hook != asked_hook:
            raise RuntimeError(
                f'The trial asks for env hook {asked_hook!r}, but the env server ran {ran_hook!r}. '
                'A server that takes no hook from the reset token states none.'
            )

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        grasp_pose = geom.Transform3D(
            raw_obs[protocol.OBS_EEF_POS], geom.Rotation.from_quat(raw_obs[protocol.OBS_EEF_QUAT])
        )
        ee_pose = self._measure(grasp_pose, raw_obs[protocol.OBS_JOINT_POS])
        state = MujocoFrankaState()
        state.encode(raw_obs[protocol.OBS_JOINT_POS], raw_obs[protocol.OBS_JOINT_VEL], ee_pose)
        if raw_obs[protocol.OBS_ARM_ERROR]:
            state.set_status(RobotStatus.ERROR)
        obs: dict[str, Any] = {keys.ROBOT_STATE: state, keys.GRIP: float(raw_obs[protocol.OBS_GRIP])}
        for logical, candidates in self._cameras.items():
            key = next((c for c in candidates if c in raw_obs), candidates[-1])
            obs[logical] = pimm.shared_memory.NumpySMAdapter.lazy_init(raw_obs[key], None)
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {name: raw_obs[name] for name in (mapping.OBS_SIM_STATE, mapping.OBS_TASK_METRICS)}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        return {eval_keys.SUCCESS: bool(result[protocol.FRAME_SUCCESS])} if result[protocol.FRAME_DONE] else None
