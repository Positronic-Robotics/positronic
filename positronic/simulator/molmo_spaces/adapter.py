"""``MolmoAdapter``: the canonical embodiment contract <-> MolmoSpaces' raw obs/command payloads, client-side.

Runs on positronic side.
"""

from typing import Any

import pimm
from positronic import geom, keys
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.molmo_spaces import keys as molmo_keys
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.mujoco.sim import MujocoFrankaState

# Per default MolmoSpaces DROID camera, the benchmark-variant keys the upstream Pi policy falls back to; an
# explicitly configured non-default camera key is read as-is (no variants).
_CAMERA_VARIANTS = {
    mapping.MOLMO_WRIST_CAMERA: mapping.MOLMO_WRIST_CAMERA_VARIANTS,
    mapping.MOLMO_EXTERIOR_CAMERA: mapping.MOLMO_EXTERIOR_CAMERA_VARIANTS,
}

# !!!!!!!!!! Where do _CAMERA_VARIANTS come from?


# Which MolmoSpaces camera each logical observation reads on the DROID rig — the pairing the benchmarks record,
# and the one whose variants the table above resolves.
DEFAULT_CAMERA_DICT = {keys.WRIST_IMAGE: mapping.MOLMO_WRIST_CAMERA, keys.EXTERIOR_IMAGE: mapping.MOLMO_EXTERIOR_CAMERA}


# Each benchmark dimension's trial key: the env's records and the reset token carry the dimension names.
_DIMENSION_KEYS = dict(zip(mapping.BenchmarkPath._fields, molmo_keys.BENCHMARK_DIMENSIONS, strict=True))


class MolmoAdapter(WireCommandAdapter):
    def __init__(self, camera_dict: dict[str, str]) -> None:
        super().__init__()
        self._camera_dict = camera_dict

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **{key: record[dimension] for dimension, key in _DIMENSION_KEYS.items()},
                eval_keys.TASK: record['name'],
                molmo_keys.EPISODE_INDEX: record['episode_index'],
                molmo_keys.TASK_HORIZON: record['task_horizon_sec'],
            }
            for record in records
        ]

    def _reset_token(self, params: dict[str, Any]) -> Any:
        return {
            **{dimension: params[key] for dimension, key in _DIMENSION_KEYS.items()},
            mapping.TOKEN_EPISODE_INDEX: params[molmo_keys.EPISODE_INDEX],
            mapping.TOKEN_SEED: params.get(eval_keys.SEED),
        }

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # Env server reports ``eef_quat`` in wxyz format.
        ee_pose = geom.Transform3D(raw_obs[mapping.OBS_EEF_POS], geom.Rotation.from_quat(raw_obs[mapping.OBS_EEF_QUAT]))
        state = MujocoFrankaState()
        state.encode(raw_obs[mapping.OBS_JOINT_POS], raw_obs[mapping.OBS_JOINT_VEL], ee_pose)
        obs: dict[str, Any] = {keys.ROBOT_STATE: state, keys.GRIP: float(raw_obs[mapping.OBS_GRIP])}
        for logical, molmo_key in self._camera_dict.items():
            env_key = mapping.resolve_camera_key(raw_obs, molmo_key, _CAMERA_VARIANTS.get(molmo_key, ()))
            frame = raw_obs[env_key]
            obs[logical] = pimm.shared_memory.NumpySMAdapter.lazy_init(frame, None)
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {mapping.OBS_SIM_STATE: raw_obs[mapping.OBS_SIM_STATE]}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        return {eval_keys.SUCCESS: bool(result[protocol.FRAME_SUCCESS])} if result[protocol.FRAME_DONE] else None
