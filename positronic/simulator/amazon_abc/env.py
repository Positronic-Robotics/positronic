"""The env server for ABC's bimanual YAM tasks.

Runs in ABC's own interpreter, which the launcher builds. A Cartesian command is solved to joints here, by
damped-least-squares IK on the arm's control site.
"""

import argparse
import functools
from typing import Any

import abc_sim  # pyright: ignore[reportMissingImports] -- installed only in ABC's own venv, which the launcher builds
import arm_action
import mapping
import mujoco
import numpy as np
import protocol
from server import EnvProtocol, EnvServer


class _Arm:
    """One YAM chain of the ABC scene: the joints it moves and the site it is measured and driven at."""

    def __init__(self, model: Any, name: str):
        self.name = name
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mapping.JOINT.format(arm=name, index=i + 1))
            for i in range(mapping.ARM_JOINTS)
        ]
        self.qpos_ids = np.array([model.jnt_qposadr[j] for j in joints])
        self.dof_ids = np.array([model.jnt_dofadr[j] for j in joints])
        ranges = np.array([model.jnt_range[j] for j in joints])
        self.lower, self.upper = ranges[:, 0], ranges[:, 1]
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, mapping.CONTROL_SITE.format(arm=name))


class AbcEnv(EnvProtocol):
    """An ABC task behind the ``tasks``/``reset``/``step``/``close`` the env server serves.

    ``reset`` rebuilds the env when the token's task or camera size changes.
    """

    def __init__(self):
        self._key: tuple[Any, ...] | None = None
        self._env: Any = None
        self._arms: list[_Arm] = []

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """ABC's own catalogue, or the tasks ``spec`` names in it; a name, an alias or a prompt resolves."""
        selection = spec.get(mapping.SELECT_TASKS)
        if selection is None:
            specs = abc_sim.list_task_specs()
        else:
            names = [selection] if isinstance(selection, str) else list(selection)
            specs = [abc_sim.get_task_spec(name) for name in names]
        return [{mapping.TASK_NAME: s.name} for s in specs]

    def _build(self, token: dict[str, Any]) -> None:
        key = (token[mapping.TOKEN_TASK], token[mapping.TOKEN_CAMERA_HEIGHT], token[mapping.TOKEN_CAMERA_WIDTH])
        if key == self._key:
            return
        self.close()
        spec = abc_sim.get_task_spec(token[mapping.TOKEN_TASK])
        self._env = abc_sim.make_env(
            task=spec.env_task,
            prompt=spec.prompt,
            render_cameras=True,
            camera_height=token[mapping.TOKEN_CAMERA_HEIGHT],
            camera_width=token[mapping.TOKEN_CAMERA_WIDTH],
            terminate_on_success=True,
        )
        self._key = key

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        self._build(token)
        obs, _info = self._env.reset(seed=token.get(mapping.TOKEN_SEED), randomize=True)
        # A reset recompiles the scene and renumbers every joint and site.
        self._arms = [_Arm(self._env.model, name) for name in self._env.robot_names]
        return {
            protocol.FRAME_OBS: self._observe(obs),
            protocol.FRAME_META: {mapping.META_TASK: obs[mapping.ABC_OBS_PROMPT]},
            protocol.FRAME_ROBOT_META: {},
            protocol.FRAME_CONTROL_DT: self._control_dt(),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        obs, _reward, terminated, truncated, info = self._env.step(self._joint_action(action))
        return {
            protocol.FRAME_OBS: self._observe(obs),
            protocol.FRAME_DONE: bool(terminated or truncated),
            protocol.FRAME_SUCCESS: bool(info[mapping.ABC_INFO_SUCCESS]),
            protocol.FRAME_CONTROL_DT: self._control_dt(),
        }

    def _control_dt(self) -> float:
        # The scene sets the physics step and how many of them one action spans; ABC exposes neither.
        return float(self._env.model.opt.timestep * self._env._control_decimation)

    def _joint_action(self, action: dict[str, Any]) -> np.ndarray:
        """ABC's ``[joints(6), grip]`` per arm, in the order its robots are named."""
        self._sync_sites()
        per_arm = []
        for arm in self._arms:
            command = action[protocol.arm_channel(protocol.ROBOT_COMMAND, arm.name)]
            joints = arm_action.wire_command_to_arm_action(
                command, self._measured_q(arm), ik=functools.partial(self._ik, arm), current_eef=self._measured_eef(arm)
            )
            grip = mapping.invert_grip(action[protocol.arm_channel(protocol.TARGET_GRIP, arm.name)])
            per_arm.append(np.append(joints, grip))
        return np.concatenate(per_arm).astype(np.float32)

    def _sync_sites(self) -> None:
        # A site pose lags ``qpos`` by one forward pass after a reset or a step.
        mujoco.mj_kinematics(self._env.model, self._env.data)

    def _measured_q(self, arm: _Arm) -> np.ndarray:
        return np.asarray(self._env.data.qpos[arm.qpos_ids], dtype=np.float32)

    def _measured_eef(self, arm: _Arm) -> tuple[np.ndarray, np.ndarray]:
        data = self._env.data
        return data.site_xpos[arm.site_id].copy(), data.site_xmat[arm.site_id].reshape(3, 3).copy()

    def _ik(self, arm: _Arm, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        """Damped-least-squares differential IK on the arm's site Jacobian, iterated on a scratch ``MjData``
        seeded from the live scene so the objects standing in it are never perturbed."""
        iterations, damping, tolerance = 100, 0.05, 1e-4
        model = self._env.model
        data = mujoco.MjData(model)
        data.qpos[:] = self._env.data.qpos
        q = self._measured_q(arm).astype(np.float64)
        rotation_error = np.empty(3)
        quat = np.empty(4)
        for _ in range(iterations):
            data.qpos[arm.qpos_ids] = q
            mujoco.mj_forward(model, data)
            reached_rot = data.site_xmat[arm.site_id].reshape(3, 3)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(target_rot @ reached_rot.T).reshape(9))
            mujoco.mju_quat2Vel(rotation_error, quat, 1.0)
            error = np.concatenate([target_pos - data.site_xpos[arm.site_id], rotation_error])
            if np.linalg.norm(error) < tolerance:
                break
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, arm.site_id)
            jac = np.vstack([jacp, jacr])[:, arm.dof_ids]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), error)
            q = np.clip(q + dq, arm.lower, arm.upper)
        return q

    def _observe(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._sync_sites()
        per_arm = np.asarray(obs[mapping.ABC_OBS_STATE]).reshape(len(self._arms), mapping.ARM_JOINTS + 1)
        payload: dict[str, Any] = {mapping.OBS_SIM_STATE: self._physics_state()}
        for index, arm in enumerate(self._arms):
            pos, rot = self._measured_eef(arm)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).reshape(9))
            payload.update({
                protocol.arm_channel(mapping.OBS_JOINT_POS, arm.name): self._measured_q(arm),
                protocol.arm_channel(mapping.OBS_JOINT_VEL, arm.name): np.asarray(
                    self._env.data.qvel[arm.dof_ids], dtype=np.float32
                ),
                protocol.arm_channel(mapping.OBS_EEF_POS, arm.name): pos.astype(np.float32),
                protocol.arm_channel(mapping.OBS_EEF_QUAT, arm.name): quat.astype(np.float32),
                protocol.arm_channel(mapping.OBS_GRIP, arm.name): np.float32(mapping.invert_grip(per_arm[index, -1])),
            })
        payload.update({name: np.ascontiguousarray(frame) for name, frame in obs[mapping.ABC_OBS_IMAGES].items()})
        return payload

    def _physics_state(self) -> np.ndarray:
        """MuJoCo's ``mjSTATE_INTEGRATION`` vector for this scene."""
        model, data = self._env.model, self._env.data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
        return state

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._key = None


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve ABC over the env-server protocol.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    EnvServer(AbcEnv(), args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
