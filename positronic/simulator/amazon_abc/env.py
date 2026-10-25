"""The env server for ABC's bimanual YAM tasks.

Runs in ABC's own interpreter, which the launcher builds.
"""

import argparse
import functools
import logging
from typing import Any

import abc_sim  # pyright: ignore[reportMissingImports] -- installed only in ABC's own venv
import arm_action
import mapping
import mujoco
import mujoco_state
import numpy as np
import protocol
from server import EnvProtocol, EnvServer

logger = logging.getLogger(__name__)

# `make_env` lays out one scene from an unseeded draw, which fails for about 1 draw in 150 on put_relative.
# The seeded reset replaces that scene, so another build leaves the trial's scene as its seed names it.
_BUILD_ATTEMPTS = 3
# The YAM driver's acceptance of an IK solution.
_IK_POSITION_TOLERANCE = 1e-3  # m
_IK_ROTATION_TOLERANCE = 1e-2  # rad


def _joint_delta_width(command: dict[str, Any]) -> int | None:
    """The number of joints a joint-delta command names, or None for any other command."""
    if command[protocol.COMMAND_TYPE] != protocol.JOINT_DELTA:
        return None
    return np.asarray(command[protocol.COMMAND_JOINT_DELTA]).reshape(-1).shape[0]


class _Arm:
    """One YAM chain of the ABC scene."""

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
        self.base_body_id = model.body_parentid[model.jnt_bodyid[joints[0]]]


class AbcEnv(EnvProtocol):
    def __init__(self):
        self._key: tuple[Any, ...] | None = None
        self._env: Any = None
        self._arms: list[_Arm] = []
        self._last_reported_apertures = np.zeros(0)
        self._scored = False

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """ABC's own catalogue, or the tasks ``spec`` names in it."""
        names = mapping.task_names(spec.get(mapping.SELECT_TASKS))
        specs = abc_sim.list_task_specs() if names is None else [abc_sim.get_task_spec(name) for name in names]
        return [{mapping.TASK_NAME: s.name} for s in specs]

    def _build(self, token: dict[str, Any]) -> None:
        key = (token[mapping.TOKEN_TASK], token[mapping.TOKEN_CAMERA_HEIGHT], token[mapping.TOKEN_CAMERA_WIDTH])
        if key == self._key:
            return
        self.close()
        spec = abc_sim.get_task_spec(token[mapping.TOKEN_TASK])
        for attempt in range(1, _BUILD_ATTEMPTS + 1):
            try:
                # ABC attaches a task's evaluator and prompt by its name; several tasks share one `env_task` scene.
                self._env = abc_sim.make_env(
                    task=spec.name,
                    prompt=spec.prompt,
                    render_cameras=True,
                    camera_height=token[mapping.TOKEN_CAMERA_HEIGHT],
                    camera_width=token[mapping.TOKEN_CAMERA_WIDTH],
                    max_episode_steps=mapping.ABC_EVAL_STEPS,
                )
                break
            except RuntimeError as error:
                if attempt == _BUILD_ATTEMPTS:
                    raise
                logger.warning(f'Building the ABC env for {spec.name} failed ({error}); building it again')
        self._key = key

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        self._build(token)
        self._env.forget_arm_state()  # ABC otherwise starts each episode where the last one left the arms
        obs, _info = self._env.reset(seed=token.get(mapping.TOKEN_SEED), randomize=True)
        # A reset recompiles the scene and renumbers every joint and site.
        self._arms = [_Arm(self._env.model, name) for name in self._env.robot_names]
        return {
            protocol.SLOTS: [{protocol.FRAME_OBS: self._observe(obs, self._task_eval())}],
            protocol.FRAME_META: self._meta(obs),
            protocol.FRAME_ROBOT_META: {protocol.MOUNTS: self._mounts()},
            protocol.FRAME_CONTROL_DT: self._control_dt(),
        }

    def _task_eval(self) -> dict[str, Any]:
        """ABC's task metrics now, or none for a task that ships no evaluator."""
        result = self._env.evaluate_task()
        self._scored = result is not None
        return {} if result is None else result.to_info(squeeze=True)

    def _step_task_eval(self, info: dict[str, Any]) -> dict[str, Any]:
        if not self._scored:
            return {}
        if mapping.ABC_INFO_TASK_EVAL not in info:
            raise RuntimeError(f'ABC reported no task metrics for a step of a scored task: {sorted(info)}')
        return info[mapping.ABC_INFO_TASK_EVAL]

    def _ever_success(self, task_eval: dict[str, Any]) -> bool:
        """ABC's verdict: its own eval reads ``ever_success`` at the trial's last step, for every scored task.

        A maintenance task sets it to ``not ever_failed``, and a count task to the count at that step."""
        # A task with no evaluator reports no metrics: ABC rolls it out to its horizon and counts no success.
        return bool(task_eval[mapping.ABC_EVAL_EVER_SUCCESS]) if self._scored else False

    def _trial_over(self, task_eval: dict[str, Any]) -> bool:
        """Whether the trial ends before its time limit, as ABC's own eval ends it.

        A task that reports ``ever_failed`` ends on it: a maintenance task when it fails, a count task never.
        Every other task ends at its first success."""
        if mapping.ABC_EVAL_EVER_FAILED in task_eval:
            return bool(task_eval[mapping.ABC_EVAL_EVER_FAILED])
        return self._ever_success(task_eval)

    def step(self, actions: dict[int, dict[str, Any]]) -> dict[str, Any]:
        (action,) = actions.values()  # ABC runs one scene per server, so this server serves one slot
        obs, _reward, _terminated, truncated, info = self._env.step(self._joint_action(action))
        task_eval = self._step_task_eval(info)
        return {
            protocol.SLOTS: [
                {
                    protocol.FRAME_OBS: self._observe(obs, task_eval),
                    protocol.FRAME_DONE: bool(truncated or self._trial_over(task_eval)),
                    protocol.FRAME_SUCCESS: self._ever_success(task_eval),
                    # A task such as multi_drawer_search moves its prompt to the next target as it goes.
                    protocol.FRAME_META: self._meta(obs),
                }
            ],
            protocol.FRAME_CONTROL_DT: self._control_dt(),
        }

    @staticmethod
    def _meta(obs: dict[str, Any]) -> dict[str, Any]:
        return {mapping.META_TASK: obs[mapping.ABC_OBS_PROMPT]}

    def _control_dt(self) -> float:
        # The scene sets the physics step and how many of them one action spans; ABC exposes neither.
        return float(self._env.model.opt.timestep * self._env._control_decimation)

    def _joint_action(self, action: dict[str, Any]) -> np.ndarray:
        """ABC's ``[joints(6), grip]`` per arm, in the order its robots are named."""
        self._sync_sites()
        per_arm = []
        for index, arm in enumerate(self._arms):
            command = action[protocol.arm_channel(protocol.ROBOT_COMMAND, arm.name)]
            measured = self._measured_q(arm)
            joints = (
                None
                if _joint_delta_width(command) not in (None, measured.shape[0])
                else arm_action.wire_command_to_arm_action(
                    command, measured, ik=functools.partial(self._ik, arm), current_eef=self._measured_eef(arm)
                )
            )
            # The rig's driver refuses such a command and holds, so a bad command fails the task, not the run.
            if joints is None or joints.shape != measured.shape or not np.all(np.isfinite(joints)):
                joints = measured
            grip = action[protocol.arm_channel(protocol.TARGET_GRIP, arm.name)]
            aperture = mapping.invert_grip(grip) if np.isfinite(grip) else self._last_reported_apertures[index]
            per_arm.append(np.append(joints, aperture))
        return np.concatenate(per_arm).astype(np.float32)

    def _mounts(self) -> dict[str, list[float]]:
        """Where each arm's base stands in the scene, keyed by the joint signal the recording names it by."""
        self._sync_sites()
        mounts = {}
        for arm in self._arms:
            signal = protocol.arm_channel(protocol.ROBOT_STATE, arm.name) + protocol.JOINTS_SUFFIX
            mounts[signal] = self._env.data.xpos[arm.base_body_id].tolist()
        return mounts

    def _sync_sites(self) -> None:
        # A site pose lags ``qpos`` by one forward pass after a reset or a step.
        mujoco.mj_kinematics(self._env.model, self._env.data)

    def _measured_q(self, arm: _Arm) -> np.ndarray:
        return np.asarray(self._env.data.qpos[arm.qpos_ids], dtype=np.float32)

    def _measured_eef(self, arm: _Arm) -> tuple[np.ndarray, np.ndarray]:
        data = self._env.data
        return data.site_xpos[arm.site_id].copy(), data.site_xmat[arm.site_id].reshape(3, 3).copy()

    def _ik(self, arm: _Arm, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray | None:
        """The arm's joints at the target, solved without moving the live scene; None where it does not
        converge."""
        rotation_error = np.empty(3)
        quat = np.empty(4)

        def pose_error(site_pos: np.ndarray, site_rot: np.ndarray) -> np.ndarray:
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(target_rot @ site_rot.T).reshape(9))
            mujoco.mju_quat2Vel(rotation_error, quat, 1.0)
            return np.concatenate([target_pos - site_pos, rotation_error])

        def converged(error: np.ndarray) -> bool:
            return bool(
                np.linalg.norm(error[:3]) < _IK_POSITION_TOLERANCE
                and np.linalg.norm(error[3:]) < _IK_ROTATION_TOLERANCE
            )

        solution = arm_action.damped_least_squares_ik(
            self._env.model,
            live_qpos=self._env.data.qpos,
            site_id=arm.site_id,
            qpos_ids=arm.qpos_ids,
            dof_ids=arm.dof_ids,
            start_q=self._measured_q(arm),
            lower=arm.lower,
            upper=arm.upper,
            pose_error=pose_error,
            converged=converged,
        )
        if solution.converged:
            return solution.q
        logger.warning(
            f'IK for the {arm.name} arm did not converge in {arm_action.IK_ITERATIONS} iterations: '
            f'{np.linalg.norm(solution.residual[:3]):.2e} m and {np.linalg.norm(solution.residual[3:]):.2e} rad '
            f'short of {target_pos.tolist()}; the arm holds'
        )
        return None

    def _observe(self, obs: dict[str, Any], task_eval: dict[str, Any]) -> dict[str, Any]:
        self._sync_sites()
        per_arm = np.asarray(obs[mapping.ABC_OBS_STATE]).reshape(len(self._arms), mapping.ARM_JOINTS + 1)
        self._last_reported_apertures = per_arm[:, -1]
        payload: dict[str, Any] = {
            mapping.OBS_SIM_STATE: mujoco_state.integration_state(self._env.model, self._env.data),
            mapping.OBS_TASK_EVAL: mapping.task_eval_signals(task_eval),
        }
        for index, arm in enumerate(self._arms):
            pos, rot = self._measured_eef(arm)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).reshape(9))
            payload.update({
                protocol.arm_channel(protocol.OBS_JOINT_POS, arm.name): self._measured_q(arm),
                protocol.arm_channel(protocol.OBS_JOINT_VEL, arm.name): np.asarray(
                    self._env.data.qvel[arm.dof_ids], dtype=np.float32
                ),
                protocol.arm_channel(protocol.OBS_EEF_POS, arm.name): pos.astype(np.float32),
                protocol.arm_channel(protocol.OBS_EEF_QUAT, arm.name): quat.astype(np.float32),
                protocol.arm_channel(protocol.OBS_GRIP, arm.name): np.float32(mapping.invert_grip(per_arm[index, -1])),
            })
        payload.update({name: np.ascontiguousarray(frame) for name, frame in obs[mapping.ABC_OBS_IMAGES].items()})
        return payload

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
