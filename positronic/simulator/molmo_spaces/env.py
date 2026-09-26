"""Serve MolmoSpaces benchmarks through the env-server protocol.

Runs in MolmoSpaces' isolated interpreter, with ``mapping``, ``server`` and ``protocol`` on ``PYTHONPATH``.
End-effector poses describe ``gripper/grasp_site`` in world coordinates.
"""

import argparse
import os
import sys
import types

# MuJoCo's CGL package and its ctypes bindings to Apple's OpenGL framework.
_CGL_PACKAGE = 'mujoco.cgl'
_CGL_MODULE = 'mujoco.cgl.cgl'


def _install_cgl_noop_stub() -> None:
    # HACK: MolmoSpaces treats every device_id=None context as CGL and imports ``from mujoco.cgl import cgl``
    # to unlock it. On Linux that import tries to load Apple's OpenGL framework, even with EGL or OSMesa.
    # Stub the CGL lock calls there; macOS needs the real bindings.
    if sys.platform == 'darwin' or _CGL_PACKAGE in sys.modules:
        return
    # ModuleType supports runtime attributes absent from its type definition.
    cgl = types.ModuleType(_CGL_MODULE)
    cgl.CGLLockContext = cgl.CGLUnlockContext = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    package = types.ModuleType(_CGL_PACKAGE)
    package.cgl = cgl  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules[_CGL_PACKAGE] = package
    sys.modules[_CGL_MODULE] = cgl


_install_cgl_noop_stub()

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import protocol  # noqa: E402
from server import EnvProtocol, EnvServer  # noqa: E402

# Loading the runner first avoids a circular import in json_eval_task_sampler.
import molmo_spaces.evaluation.json_eval_runner  # noqa: E402, F401  # pyright: ignore[reportMissingImports]
from molmo_spaces.configs.policy_configs import DummyPolicyConfig  # noqa: E402  # pyright: ignore[reportMissingImports]
from molmo_spaces.configs.robot_configs import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    ActionNoiseConfig,
    FrankaRobotConfig,
)
from molmo_spaces.evaluation.benchmark_schema import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    load_all_episodes,
)
from molmo_spaces.evaluation.configs.evaluation_configs import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    JsonBenchmarkEvalConfig,
)
from molmo_spaces.evaluation.eval_main import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    determine_task_horizon,
)
from molmo_spaces.tasks.json_eval_task_sampler import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    JsonEvalTaskSampler,
)


class _DroidEvalConfig(JsonBenchmarkEvalConfig):
    """Franka DROID configuration for benchmark episodes, with policy inference supplied by the client."""

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


class MolmoSpacesEnv(EnvProtocol):
    """A MolmoSpaces benchmark environment with one active episode."""

    def __init__(self, assets_dir: Path) -> None:
        self._assets_dir = assets_dir
        self._found = mapping.discover_benchmarks(assets_dir)
        self._episodes: dict[mapping.BenchmarkPath, list[Any]] = {}
        self._sampler: Any = None
        self._task: Any = None
        self._robot_view: Any = None
        self._control_dt: float | None = None
        self._meta: dict[str, Any] | None = None
        self._camera_names: list[str] = []

    def _episodes_of(self, bench: mapping.BenchmarkPath) -> list[Any]:
        if bench not in self._episodes:
            # Report available benchmarks when the requested directory is missing.
            mapping.select_benchmarks(self._found, bench._asdict())
            episodes = load_all_episodes(bench.under(self._assets_dir))
            if not episodes:
                raise ValueError(f'{bench.relative} holds no episodes')
            self._episodes[bench] = episodes
        return self._episodes[bench]

    def _assert_measures_at_grasp_site(self) -> None:
        """Require arm poses to track the benchmark's gripper grasp site."""
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
        if arm.leaf_frame_type != 'site':
            raise ValueError(f'arm move group measures at a {arm.leaf_frame_type}, not the expected site')
        name = mujoco.mj_id2name(arm.mj_model, mujoco.mjtObj.mjOBJ_SITE, arm.leaf_frame_id)
        if name != mapping.MOLMO_GRASP_SITE and not name.endswith(f'/{mapping.MOLMO_GRASP_SITE}'):
            raise ValueError(f'arm move group measures at site {name!r}, expected {mapping.MOLMO_GRASP_SITE!r}')

    def _build(self, bench: mapping.BenchmarkPath, episode_index: int, seed: int | None) -> None:
        if self._sampler is not None:
            self._sampler.close()
        episodes = self._episodes_of(bench)
        episode = episodes[episode_index]
        cfg = _DroidEvalConfig()
        cfg.seed = mapping.resolve_episode_seed(episode, episode_index, seed)
        # MolmoSpaces resolves one horizon from the full benchmark.
        cfg.task_horizon = determine_task_horizon(episodes, None, cfg.policy_dt_ms)
        self._sampler = JsonEvalTaskSampler(cfg, episode)
        # Task sampling places the objects; task.reset() alone does not restore the scene.
        self._task = self._sampler.sample_task(house_index=episode.house_index)
        self._robot_view = self._task.env.current_robot.robot_view
        self._assert_measures_at_grasp_site()
        self._control_dt = cfg.policy_dt_ms / 1000.0
        self._meta = {
            # Generated task descriptions can mislabel close episodes as "Open ...".
            mapping.META_TASK: episode.language.task_description,
            mapping.META_HOUSE_INDEX: episode.house_index,
        }

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Selected episode records, with names, reset parameters and horizons in seconds."""
        cfg = _DroidEvalConfig()
        selection = spec.get(mapping.SELECT_EPISODES)
        pinned = None if selection is None else [selection] if isinstance(selection, int) else list(selection)
        records = []
        for bench in mapping.select_benchmarks(self._found, spec):
            episodes = self._episodes_of(bench)
            count = len(episodes)
            indices = list(range(count)) if pinned is None else pinned
            out_of_range = [i for i in indices if not 0 <= i < count]
            if out_of_range:
                raise ValueError(f'episodes {out_of_range} out of range for the {count} episodes of {bench.relative}')
            horizon_sec = determine_task_horizon(episodes, None, cfg.policy_dt_ms) * cfg.policy_dt_ms / 1000.0
            records += [
                {
                    **bench._asdict(),
                    mapping.TOKEN_EPISODE_INDEX: i,
                    mapping.TASK_NAME: episodes[i].language.task_description,
                    mapping.TASK_HORIZON_SEC: horizon_sec,
                }
                for i in indices
            ]
        return records

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        bench = mapping.BenchmarkPath(**{d: token[d] for d in mapping.BenchmarkPath._fields})
        self._build(bench, token[mapping.TOKEN_EPISODE_INDEX], token.get(mapping.TOKEN_SEED))
        obs, _info = self._task.reset()  # One observation dict per environment.
        env_obs = obs[0]
        self._camera_names = [k for k, v in env_obs.items() if mapping.is_rgb_frame(v)]
        return {
            protocol.FRAME_OBS: self._observe(env_obs),
            protocol.FRAME_META: self._meta,
            protocol.FRAME_ROBOT_META: {},  # The client supplies the robot model through static_meta.
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        wire = protocol.single_arm(action)
        arm = mapping.wire_command_to_arm_action(
            wire[protocol.ROBOT_COMMAND], self._measured_arm_q(), ik=self._ik, current_eef=self._measured_eef_pose()
        )
        gripper = np.array([mapping.grip_command_to_actuator(wire[protocol.TARGET_GRIP])], dtype=np.float32)
        obs, _reward, _term, _trunc, _infos = self._task.step({
            mapping.MOLMO_ARM_GROUP: arm,
            mapping.MOLMO_GRIPPER_GROUP: gripper,
        })
        success = bool(self._task.judge_success())
        done = success or bool(self._task.is_done())
        return {
            protocol.FRAME_OBS: self._observe(obs[0]),
            protocol.FRAME_DONE: done,
            protocol.FRAME_SUCCESS: success,
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def _measured_arm_q(self) -> np.ndarray:
        return np.asarray(self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).joint_pos, dtype=np.float32)

    def _measured_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The grasp-site world pose as (translation, 3x3 rotation)."""
        eef_world = np.asarray(
            self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).leaf_frame_to_world, dtype=np.float64
        )
        return eef_world[:3, 3].copy(), eef_world[:3, :3].copy()

    def _ik(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        """Solve a world-frame grasp-site target with MolmoSpaces' IK."""
        pose = np.eye(4)
        pose[:3, 3] = target_pos
        pose[:3, :3] = target_rot
        solution = self._task.env.current_robot.kinematics.ik(
            move_group_id=mapping.MOLMO_ARM_GROUP,
            pose=pose,
            unlocked_move_group_ids=[mapping.MOLMO_ARM_GROUP],
            q0=self._robot_view.get_qpos_dict(),
            base_pose=self._robot_view.base.pose,
            rel_to_base=False,
        )
        if solution is None:
            raise RuntimeError(f'MolmoSpaces IK failed for world-frame target pose:\n{pose}')
        return solution[mapping.MOLMO_ARM_GROUP]

    def _observe(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Raw observations, including the grasp-site pose in world coordinates."""
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
        # MolmoSpaces' TCP observation is robot-relative.
        eef_world = np.asarray(arm.leaf_frame_to_world, dtype=np.float64)
        eef_quat = np.zeros(4)
        rot9 = np.ascontiguousarray(eef_world[:3, :3].reshape(9))
        mujoco.mju_mat2Quat(eef_quat, rot9)
        payload = {
            mapping.OBS_JOINT_POS: np.asarray(arm.joint_pos, dtype=np.float32),
            mapping.OBS_JOINT_VEL: np.asarray(arm.joint_vel, dtype=np.float32),
            mapping.OBS_EEF_POS: eef_world[:3, 3].astype(np.float32),
            mapping.OBS_EEF_QUAT: eef_quat.astype(np.float32),
            mapping.OBS_GRIP: np.float32(
                mapping.normalize_grip_qpos(env_obs[mapping.MOLMO_OBS_QPOS][mapping.MOLMO_GRIPPER_GROUP])
            ),
            mapping.OBS_SIM_STATE: self._full_physics_state(),
        }
        for name in self._camera_names:
            payload[name] = np.ascontiguousarray(env_obs[name])
        return payload

    def _full_physics_state(self) -> np.ndarray:
        """MuJoCo's ``mjSTATE_INTEGRATION`` vector for this scene."""
        data = self._robot_view.mj_data
        model = data.model
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
        mujoco.mj_getState(model, data, state, spec)
        return state

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
            self._task = None


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve MolmoSpaces over the env-server protocol.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    assets = os.environ.get(mapping.ASSETS_DIR_ENV)
    if not assets:
        parser.error(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    env = MolmoSpacesEnv(Path(assets))
    EnvServer(env, args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
