"""``MolmoAdapter``: the canonical embodiment contract <-> MolmoSpaces' raw obs/command payloads, client-side.

Runs in positronic's interpreter (the ``MolmoSpacesEnv`` server runs in MolmoSpaces' own). Mirrors the LIBERO
adapter on the observation side; the command side is ``WireCommandAdapter``'s forwarding. All action encoding —
the wire command into MolmoSpaces' per-move-group joint targets — lives server-side in ``env.py`` where the
MuJoCo model is, so the adapter holds no model and stays geometry-only.
"""

from typing import Any

import pimm
from positronic import geom, keys
from positronic.eval import EVAL_EPISODE_INDEX, EVAL_SEED, EVAL_SUCCESS
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.molmo_spaces import mapping
from positronic.simulator.mujoco.sim import MujocoFrankaState

# Per default MolmoSpaces DROID camera, the benchmark-variant keys the upstream Pi policy falls back to; an
# explicitly configured non-default camera key is read as-is (no variants).
_CAMERA_VARIANTS = {
    mapping.MOLMO_WRIST_CAMERA: mapping.MOLMO_WRIST_CAMERA_VARIANTS,
    mapping.MOLMO_EXTERIOR_CAMERA: mapping.MOLMO_EXTERIOR_CAMERA_VARIANTS,
}

# Which MolmoSpaces camera each logical observation reads, for the DROID rig's two default cameras. An eval may
# configure others; this is the pairing the benchmarks record and the variants above resolve.
DEFAULT_CAMERA_DICT = {keys.WRIST_IMAGE: mapping.MOLMO_WRIST_CAMERA, keys.EXTERIOR_IMAGE: mapping.MOLMO_EXTERIOR_CAMERA}


class MolmoAdapter(WireCommandAdapter):
    def __init__(self, camera_dict: dict[str, str]) -> None:
        super().__init__()
        self._camera_dict = camera_dict  # logical observation name -> the MolmoSpaces obs camera key

    def _reset_token(self, context: dict[str, Any]) -> Any:
        # The benchmark episode selector rides the token. An absent seed leaves the episode spec's own in force.
        return {mapping.TOKEN_EPISODE_INDEX: context[EVAL_EPISODE_INDEX], mapping.TOKEN_SEED: context.get(EVAL_SEED)}

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # env.py reports the eef pose in the grasp-site world frame; ``eef_quat`` is scalar-first (wxyz, from
        # ``mju_mat2Quat``), so ``from_quat`` is the matching decode.
        ee_pose = geom.Transform3D(raw_obs[mapping.OBS_EEF_POS], geom.Rotation.from_quat(raw_obs[mapping.OBS_EEF_QUAT]))
        state = MujocoFrankaState()
        state.encode(raw_obs[mapping.OBS_JOINT_POS], raw_obs[mapping.OBS_JOINT_VEL], ee_pose)
        obs: dict[str, Any] = {'robot_state': state, keys.GRIP: float(raw_obs[mapping.OBS_GRIP])}
        for logical, molmo_key in self._camera_dict.items():
            env_key = mapping.resolve_camera_key(raw_obs, molmo_key, _CAMERA_VARIANTS.get(molmo_key, ()))
            frame = raw_obs[env_key]  # MolmoSpaces renders top-down already — no flip
            adapter = pimm.shared_memory.NumpySMAdapter(shape=frame.shape, dtype=frame.dtype)
            adapter.array[:] = frame
            obs[logical] = adapter
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # The env's full MuJoCo state — recorded as ground truth so success can be recomputed offline, never fed
        # to the policy (mirrors the libero adapter).
        return {mapping.OBS_SIM_STATE: raw_obs[mapping.OBS_SIM_STATE]}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        # ``done`` covers termination and timeout; ``success`` is the task's judged success, so a timeout stays
        # honest.
        return {EVAL_SUCCESS: bool(result[protocol.FRAME_SUCCESS])} if result[protocol.FRAME_DONE] else None
