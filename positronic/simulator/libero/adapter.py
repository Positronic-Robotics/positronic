"""``LiberoAdapter``: the canonical embodiment contract <-> LIBERO's raw obs/command payloads, client-side.

Runs in positronic's interpreter (the ``LiberoEnv`` server runs in LIBERO's). Mirrors the reference
``StackCubesAdapter`` on the observation side; the command side is ``WireCommandAdapter``'s forwarding. All
action encoding — the OSC pose delta and its normalization, and the FK/IK that bridge pose<->joint commands —
lives server-side in ``LiberoEnv`` where the MuJoCo model is; the adapter holds no model and stays geometry-only.
"""

from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.simulator.env_server.adapter import WireCommandAdapter
from positronic.simulator.mujoco.sim import MujocoFrankaState


class LiberoAdapter(WireCommandAdapter):
    def __init__(self, camera_dict: dict[str, str]):
        super().__init__()
        self._camera_dict = camera_dict  # logical observation name -> the LIBERO obs image key

    def _reset_token(self, context: dict[str, Any]) -> Any:
        # The whole scene spec rides the trial context: the server caches its env by ``(suite, task_id,
        # camera_resolution, control_mode)``, so one adapter + one server serve any mix of suites and tasks.
        # ``seed`` selects a saved init-state (``None`` -> the server draws one at random); ``settle_steps`` is
        # the hold-arm/open-gripper wait the server runs after a seeded reset so dropped objects settle before
        # the first observation (openpi's num_steps_wait dummy-action wait).
        return {
            'suite': context['eval.suite'],
            'task_id': context['eval.task_id'],
            'camera_resolution': context['eval.camera_resolution'],
            'control_mode': context['eval.control_mode'],
            'seed': context.get('eval.seed'),
            'settle_steps': context['eval.settle_steps'],
        }

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # The env reports the eef pose in the grip-site frame it controls; ``eef_quat`` is scalar-last (xyzw,
        # from ``mat2quat``), so ``from_quat_xyzw`` is the matching decode.
        ee_pose = geom.Transform3D(raw_obs['eef_pos'], geom.Rotation.from_quat_xyzw(raw_obs['eef_quat']))
        state = MujocoFrankaState()
        state.encode(raw_obs['joint_pos'], raw_obs['joint_vel'], ee_pose)
        obs: dict[str, Any] = {'robot_state': state, 'grip': float(raw_obs['grip'])}
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
        return {'eval.success': True} if result['done'] else None
