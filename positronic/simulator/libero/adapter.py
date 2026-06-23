"""``LiberoAdapter``: the canonical embodiment contract <-> LIBERO's raw obs/command payloads, client-side.

Runs in positronic's interpreter (the ``LiberoEnv`` server runs in LIBERO's). Mirrors the reference
``StackCubesAdapter`` on the observation side; on the command side it is a forwarder: it plays each command
channel's trajectory down to the clock, holds the last waypoint between waypoints, and ships the held command
(an absolute Cartesian pose, joint positions, or joint velocities) plus the gripper opening to the server. All
action encoding — the OSC pose delta and its normalization, and the FK/IK that bridge pose<->joint commands —
lives server-side in ``LiberoEnv`` where the MuJoCo model is; the adapter holds no model and stays geometry-only.

Several LIBERO/robosuite obs conventions can't be verified in positronic's interpreter and are marked TODO: the
gripper observation normalization and the camera image orientation. They need a confirmation pass on a
LIBERO-capable box.
"""

from typing import Any

import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import command as roboarm_command
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.mujoco.sim import MujocoFrankaState


def _wire_command(cmd: Any) -> dict[str, Any]:
    """The held command as a positronic-free payload the server decodes (no ``geom``/``roboarm`` on its side)."""
    match cmd:
        case roboarm_command.CartesianPosition(pose):
            return {'type': 'cartesian', 'pose': pose.as_vector(geom.Rotation.Representation.ROTATION_MATRIX)}
        case roboarm_command.JointPosition(positions):
            return {'type': 'joint_pos', 'q': positions}
        case roboarm_command.JointDelta(velocities):
            return {'type': 'joint_vel', 'dq': velocities}
        case None:
            return {'type': 'hold'}
        case other:
            raise ValueError(f'LiberoAdapter cannot forward robot_command {type(other).__name__}')


class LiberoAdapter(EnvAdapter):
    def __init__(self, camera_dict: dict[str, str]):
        self._camera_dict = camera_dict  # logical observation name -> the LIBERO obs image key
        self._players: dict[str, roboarm_command.TrajectoryPlayer] = {}
        self._held: dict[str, Any] = {}  # last sampled waypoint per channel — re-sent until it changes

    def reset_token(self, seed: int | None) -> Any:
        self._players = {}
        self._held = {}
        return seed if seed is not None else 0  # LIBERO selects a concrete init-state index

    def action(self, commands: dict[str, pimm.Message], now_ns: int) -> dict[str, Any]:
        for name, msg in commands.items():
            player = self._players.setdefault(name, roboarm_command.TrajectoryPlayer())
            if msg.updated and msg.data is not None:
                player.set(msg.data)
                if not msg.data:  # an empty trajectory cancels: stop replaying the held waypoint
                    self._held.pop(name, None)
            for value in player.advance(now_ns):
                self._held[name] = value
        # The server maps the held command into the active controller's action. Reset/Recover have no robosuite
        # action, so they forward as a hold.
        cmd = self._held.get('robot_command')
        if isinstance(cmd, roboarm_command.Reset | roboarm_command.Recover):
            self._held.pop('robot_command')
            cmd = None
        return {'command': _wire_command(cmd), 'grip': float(self._held.get('target_grip', 0.0))}

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        # The env reports the eef pose in the grip-site frame it controls; ``eef_quat`` is scalar-last (xyzw,
        # from ``mat2quat``), so ``from_quat_xyzw`` is the matching decode.
        ee_pose = geom.Transform3D(raw_obs['eef_pos'], geom.Rotation.from_quat_xyzw(raw_obs['eef_quat']))
        state = MujocoFrankaState()
        state.encode(raw_obs['joint_pos'], raw_obs['joint_vel'], ee_pose)
        # TODO (verify on a LIBERO box): map the two Panda finger qpos to a single [0, 1] openness.
        obs: dict[str, Any] = {'robot_state': state, 'grip': float(np.sum(raw_obs['gripper_qpos']))}
        for logical, env_key in self._camera_dict.items():
            # TODO (verify on a LIBERO box): robosuite renders bottom-up; flip to image orientation.
            frame = np.ascontiguousarray(raw_obs[env_key][::-1])
            adapter = pimm.shared_memory.NumpySMAdapter(shape=frame.shape, dtype=frame.dtype)
            adapter.array[:] = frame
            obs[logical] = adapter
        return obs

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {'sim_state': raw_obs['sim_state']}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        return {'eval.success': True} if result['done'] else None
