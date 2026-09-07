"""``LiberoAdapter``: the canonical embodiment contract <-> LIBERO's raw obs/command payloads, client-side.

Runs in positronic's interpreter (the ``LiberoEnv`` server runs in LIBERO's). Mirrors the reference
``StackCubesAdapter`` on the observation side; the command side is ``WireCommandAdapter``'s forwarding. All
action encoding — the OSC pose delta and its normalization, and the FK/IK that bridge pose<->joint commands —
lives server-side in ``LiberoEnv`` where the MuJoCo model is; the adapter holds no model and stays geometry-only.
"""

from typing import Any

import numpy as np

import pimm
from positronic import geom, keys
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.libero import keys as libero_keys
from positronic.simulator.mujoco.sim import MujocoFrankaState


class LiberoAdapter(WireCommandAdapter):
    def __init__(self, camera_dict: dict[str, str]):
        super().__init__()
        self._camera_dict = camera_dict  # logical observation name -> the LIBERO obs image key

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {eval_keys.TASK: r['name'], libero_keys.SUITE: r['suite'], libero_keys.TASK_ID: r['task_id']}
            for r in records
        ]

    def _reset_token(self, params: dict[str, Any]) -> Any:
        # The whole scene spec rides the trial params: the server caches its env by ``(suite, task_id,
        # camera_resolution, control_mode)``, so one adapter + one server serve any mix of suites and tasks.
        # ``seed`` selects a saved init-state (``None`` -> the server draws one at random); ``settle_steps`` is
        # the hold-arm/open-gripper wait the server runs after a seeded reset so dropped objects settle before
        # the first observation (openpi's num_steps_wait dummy-action wait).
        return {
            'suite': params[libero_keys.SUITE],
            'task_id': params[libero_keys.TASK_ID],
            'camera_resolution': params[libero_keys.CAMERA_RESOLUTION],
            'control_mode': params[libero_keys.CONTROL_MODE],
            'seed': params.get(eval_keys.SEED),
            'settle_steps': params[libero_keys.SETTLE_STEPS],
        }

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # The env reports the eef pose in the grip-site frame it controls; ``eef_quat`` is scalar-last (xyzw,
        # from ``mat2quat``), so ``from_quat_xyzw`` is the matching decode.
        ee_pose = geom.Transform3D(raw_obs['eef_pos'], geom.Rotation.from_quat_xyzw(raw_obs['eef_quat']))
        state = MujocoFrankaState()
        state.encode(raw_obs['joint_pos'], raw_obs['joint_vel'], ee_pose)
        obs: dict[str, Any] = {keys.ROBOT_STATE: state, keys.GRIP: float(raw_obs['grip'])}
        for logical, env_key in self._camera_dict.items():
            # robosuite renders bottom-up; flip to standard top-down orientation (LIBERO's own video path
            # flips the same way).
            frame = np.ascontiguousarray(raw_obs[env_key][::-1])
            adapter = pimm.shared_memory.NumpySMAdapter(shape=frame.shape, dtype=frame.dtype)
            adapter.array[:] = frame
            obs[logical] = adapter
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {'sim_state': raw_obs['sim_state']}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        # ``done`` is LIBERO's success check rather than a step limit, so reaching it is the success.
        return {eval_keys.SUCCESS: True} if result['done'] else None
