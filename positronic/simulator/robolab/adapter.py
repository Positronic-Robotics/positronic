"""``RobolabAdapter``: the canonical embodiment contract <-> RoboLab's raw obs/command payloads, client-side.

Runs in positronic's interpreter (the RoboLab env server runs in its own Isaac Lab interpreter). The command
side is ``WireCommandAdapter``'s forwarding; all action encoding — the joint-target conversion and the
differential IK that bridges Cartesian commands — lives server-side where the articulation model is, so the
adapter holds no model and stays geometry-only.
"""

from typing import Any

import pimm
from positronic import geom
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.mujoco.sim import MujocoFrankaState


class RobolabAdapter(WireCommandAdapter):
    def __init__(self, camera_dict: dict[str, str]):
        super().__init__()
        self._camera_dict = camera_dict  # logical observation name -> the RoboLab obs image key

    def _reset_token(self, context: dict[str, Any]) -> Any:
        # No seed rides the token: RoboLab's eval path has no seed hook, so a recorded seed would only mislead.
        return {'task': context['eval.task'], 'instruction_type': context['eval.instruction_type']}

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # The env reports the eef pose in the control frame IK drives; ``eef_quat`` is scalar-first (wxyz),
        # so ``from_quat`` is the matching decode.
        ee_pose = geom.Transform3D(raw_obs['eef_pos'], geom.Rotation.from_quat(raw_obs['eef_quat']))
        state = MujocoFrankaState()
        state.encode(raw_obs['joint_pos'], raw_obs['joint_vel'], ee_pose)
        obs: dict[str, Any] = {'robot_state': state, 'grip': float(raw_obs['grip'])}
        for logical, env_key in self._camera_dict.items():
            frame = raw_obs[env_key]  # Isaac tiled cameras render top-down already — no flip
            adapter = pimm.shared_memory.NumpySMAdapter(shape=frame.shape, dtype=frame.dtype)
            adapter.array[:] = frame
            obs[logical] = adapter
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {'subtask': raw_obs['subtask']}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        # ``done`` covers termination and truncation, so the trial ends either way; ``success`` is True only
        # when the task's success condition fired, keeping timeouts honest.
        return {'eval.success': bool(result['success'])} if result['done'] else None
