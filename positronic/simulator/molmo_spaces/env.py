"""MolmoSpaces — AllenAI's MuJoCo manipulation benchmark — behind the env-server protocol.

MolmoSpaces pins ``mujoco ~=3.5`` + its asset stack on Python 3.11. The launcher runs it with the molmospaces venv,
with the positronic-free ``server``/``protocol`` and this package's ``mapping`` module on ``PYTHONPATH``.
It imports ``molmo_spaces`` (+ mujoco/numpy) and those, never ``positronic``.

positronic owns the control loop: this server drives a single MolmoSpaces ``BaseMujocoTask`` per episode directly,
replacing MolmoSpaces' own ``JsonEvalRunner`` loop. The reset token selects the benchmark episode and an optional seed.
The client-side ``MolmoAdapter`` maps the raw payload this server reports into the canonical embodiment contract.

Command side: this server converts commands into joint targets, using MolmoSpaces' IK for Cartesian targets,
and steps the per-move-group ``{arm, gripper}`` action.
Observation side: MolmoSpaces' obs carries the joint positions/velocities and camera frames and the end-effector
cartesian pose is read from the robot view's grasp-site frame here, alongside the gripper closure.
"""

import argparse
import os
import sys
import types

import mapping  # positronic-free wire mappings, on PYTHONPATH; numpy only, so it pulls in no GL

# GL backend must be selected before any mujoco/molmo_spaces import.
# !!!! This module should not care about convinience of tests, tests should take care of this
# !!!! This leaks this knowledge in the env, which is not its problem. Ideally we should remove it
os.environ.setdefault(mapping.GL_BACKEND_ENV, mapping.GL_BACKEND_DEFAULT)


# MolmoSpaces' renderer module, which the stub below stands in for on Linux.
_CGL_PACKAGE = 'mujoco.cgl'
_CGL_MODULE = f'{_CGL_PACKAGE}.cgl'
# !!! so is it 'mujoco.cgl.cgl'???


def _install_cgl_noop_stub() -> None:
    # HACK: MolmoSpaces' renderer hardcodes a macOS CGL context on the CPU (device_id=None) render path
    # (opengl_rendering.py does ``from mujoco.cgl import cgl``), which dlopens Apple's OpenGL.framework and
    # crashes at renderer init on Linux — so a CPU-rendered server (MUJOCO_GL=osmesa or mesa software EGL)
    # dies before the first observation. CGL locking is a no-op off macOS, so stub the module: the import
    # resolves and the (un)lock does nothing. Untouched on a GPU box, where the EGL path never imports it.
    # macOS keeps the real module, where those locks guard an actual context.
    if sys.platform == 'darwin' or _CGL_PACKAGE in sys.modules:
        return
    cgl = types.ModuleType(_CGL_MODULE)
    cgl.CGLLockContext = cgl.CGLUnlockContext = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    package = types.ModuleType(_CGL_PACKAGE)
    package.cgl = cgl  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules[_CGL_PACKAGE] = package
    sys.modules[_CGL_MODULE] = cgl


_install_cgl_noop_stub()

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import protocol  # noqa: E402 -- the positronic-free wire contract, on PYTHONPATH  # pyright: ignore[reportMissingImports]

# server resolves to a module without these symbols under positronic's deps (the real one is on the molmo
# venv's PYTHONPATH), so the symbols read as unknown here.
from server import EnvProtocol, EnvServer  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]

import molmo_spaces.evaluation.json_eval_runner  # noqa: E402, F401 -- load first: breaks a circular import that importing json_eval_task_sampler directly hits  # pyright: ignore[reportMissingImports]
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


class _DroidPickEvalConfig(JsonBenchmarkEvalConfig):
    """The minimal eval config to build a Franka DROID pick task standalone.

    ``JsonBenchmarkEvalConfig`` defaults every ``MlSpacesExpConfig`` field except the robot and policy configs;
    the sampler overrides ``task_type``/``scene_dataset``/``data_split``/``camera_config``/``house_inds`` from the
    episode spec, so only these two are supplied. The policy config is a ``DummyPolicyConfig`` — positronic owns
    the policy, and ``sample_task`` never calls the framework's ``policy_factory`` (only reads
    ``force_enable_depth``).
    """

    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: DummyPolicyConfig = DummyPolicyConfig()

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.robot_config.action_noise_config = ActionNoiseConfig(enabled=False)


# MolmoSpaces reports a move group's leaf frame as one of MuJoCo's frame kinds; the arm's is a site.
_SITE_FRAME = 'site'


def _assert_measures_at_grasp_site(robot_view) -> None:
    """Fail unless the arm move group's leaf frame is ``mapping.MOLMO_GRASP_SITE``.

    That frame is what every pose this server reports is measured in, and the eval declares its recorded
    model's control frame at the same point. Nothing else ties the two together, so a scene whose arm resolves
    somewhere else would misframe every recorded pose silently — for the viewer, for offline IK and for any
    frame a policy asks for.
    """
    arm = robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
    if arm.leaf_frame_type != _SITE_FRAME:
        raise ValueError(f'arm move group measures at a {arm.leaf_frame_type}, not the expected site')
    name = mujoco.mj_id2name(arm.mj_model, mujoco.mjtObj.mjOBJ_SITE, arm.leaf_frame_id)  # pyright: ignore[reportAttributeAccessIssue]
    if name != mapping.MOLMO_GRASP_SITE and not name.endswith(f'/{mapping.MOLMO_GRASP_SITE}'):
        raise ValueError(f'arm move group measures at site {name!r}, expected {mapping.MOLMO_GRASP_SITE!r}')


class MolmoSpacesEnv(EnvProtocol):
    """The MolmoSpaces benchmarks under the asset packs, behind the ``tasks``/``reset``/``step``/``close`` the
    env server serves.

    ``tasks`` answers the episode records of the benchmarks a spec selects. Each reset builds from the token's
    benchmark, its episode index (into that benchmark's ``benchmark.json``) and its seed: MolmoSpaces'
    ``task.reset()`` does not re-place the scene — ``sample_task`` does — so each reset rebuilds the task for a
    clean, deterministic scene (benchmark episodes are exact-pose deterministic, so a rebuild reproduces them).
    ``step`` integrates the forwarded joint command onto the measured joints, drives the per-move-group action,
    and reports MolmoSpaces' ``is_done``/``judge_success``.
    """

    def __init__(self, assets_dir: Path) -> None:
        self._assets_dir = assets_dir
        self._found = mapping.discover_benchmarks(assets_dir)
        self._episodes: dict[mapping.BenchmarkPath, list[Any]] = {}  # a benchmark's specs, loaded on first use
        self._sampler: Any = None
        self._task: Any = None
        self._robot_view: Any = None
        self._control_dt: float | None = None
        self._meta: dict[str, Any] | None = None
        self._camera_names: list[str] = []

    def _episodes_of(self, bench: mapping.BenchmarkPath) -> list[Any]:
        if bench not in self._episodes:
            # Every dimension pinned: the selection is an existence check that lists what is there on a miss.
            mapping.select_benchmarks(self._found, bench._asdict())
            episodes = load_all_episodes(bench.under(self._assets_dir))
            if not episodes:
                raise ValueError(f'{bench.relative} holds no episodes')
            self._episodes[bench] = episodes
        return self._episodes[bench]

    def _build(self, bench: mapping.BenchmarkPath, episode_index: int, seed: int | None) -> None:
        if self._sampler is not None:
            self._sampler.close()  # release the prior episode's sim/renderer before building the next
        episodes = self._episodes_of(bench)
        episode = episodes[episode_index]
        cfg = _DroidPickEvalConfig()
        # Determinism enters at sampler construction (seed_task_sampling).
        cfg.seed = mapping.resolve_episode_seed(episode, episode_index, seed)
        # With ``task_horizon`` set, the task enforces it and ``is_done`` reports expiry, so a horizon-expired
        # trial ends with a terminal ``done`` exactly as the native benchmark scores it. The horizon is the
        # benchmark's, not an episode's.
        cfg.task_horizon = determine_task_horizon(episodes, None, cfg.policy_dt_ms)
        self._sampler = JsonEvalTaskSampler(cfg, episode)
        self._task = self._sampler.sample_task(house_index=episode.house_index)
        self._robot_view = self._task.env.current_robot.robot_view
        _assert_measures_at_grasp_site(self._robot_view)
        self._control_dt = cfg.policy_dt_ms / 1000.0
        # The authoritative benchmark prompt, straight from the episode spec — not
        # ``task.get_task_description()``, which upstream reconstructs per task type (e.g. OpeningTask emits
        # "Open the ..." even for a close episode), so a reconstruction could diverge from the benchmark goal.
        self._meta = {
            mapping.META_TASK: episode.language.task_description,
            mapping.META_HOUSE_INDEX: episode.house_index,
        }

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """The episode records ``spec`` selects: the benchmark dimensions pin benchmarks among those under the
        asset packs; ``episodes`` is one index, or a list of them, within each; absent, the whole benchmark.
        Every record carries the one horizon its benchmark enforces, converted against the config that
        enforces it."""
        cfg = _DroidPickEvalConfig()
        selection = spec.get('episodes')
        pinned = None if selection is None else [selection] if isinstance(selection, int) else list(selection)
        records = []
        for bench in mapping.select_benchmarks(self._found, spec):
            episodes = self._episodes_of(bench)
            count = len(episodes)
            indices = list(range(count)) if pinned is None else pinned
            # A negative index would silently run a from-the-end episode mislabeled by its own index.
            out_of_range = [i for i in indices if not 0 <= i < count]
            if out_of_range:
                raise ValueError(f'episodes {out_of_range} out of range for the {count} episodes of {bench.relative}')
            horizon_sec = determine_task_horizon(episodes, None, cfg.policy_dt_ms) * cfg.policy_dt_ms / 1000.0
            records += [
                {
                    **bench._asdict(),
                    'episode_index': i,
                    'name': episodes[i].language.task_description,
                    'task_horizon_sec': horizon_sec,
                }
                for i in indices
            ]
        return records

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        bench = mapping.BenchmarkPath(**{d: token[d] for d in mapping.BenchmarkPath._fields})
        self._build(bench, token[mapping.TOKEN_EPISODE_INDEX], token.get(mapping.TOKEN_SEED))
        obs, _info = self._task.reset()  # obs is a list, one dict per env; n_batch == 1
        env_obs = obs[0]
        self._camera_names = [k for k, v in env_obs.items() if mapping.is_rgb_frame(v)]
        # robot_meta is empty: this venv cannot import positronic to emit the Franka model, so the eval supplies
        # it via ``static_meta`` (``bundled_franka_model``).
        return {
            protocol.FRAME_OBS: self._observe(env_obs),
            protocol.FRAME_META: self._meta,
            protocol.FRAME_ROBOT_META: {},
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        arm = mapping.wire_command_to_arm_action(
            action[protocol.ACTION_COMMAND], self._measured_arm_q(), ik=self._ik, current_eef=self._measured_eef_pose()
        )
        gripper = np.array([mapping.grip_command_to_actuator(action[protocol.ACTION_GRIP])], dtype=np.float32)
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
        """The measured grasp-site world pose as ``(translation, 3x3 rotation)`` — the frame a Cartesian
        command targets and the one ``_observe`` reports, so command and observation share a frame."""
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
        """The raw observation payload for one env frame: measured joints, the eef world pose, grip, camera frames.

        The eef *world* pose is read from the arm move group's grasp-site frame, since MolmoSpaces' obs exposes
        only a robot-relative tcp pose.
        """
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
        eef_world = np.asarray(arm.leaf_frame_to_world, dtype=np.float64)  # 4x4 grasp-site world transform
        eef_quat = np.zeros(4)  # filled wxyz below
        rot9 = np.ascontiguousarray(eef_world[:3, :3].reshape(9))
        # mju_mat2Quat is a C binding absent from mujoco's type stubs, so pyright can't see the attribute.
        mujoco.mju_mat2Quat(eef_quat, rot9)  # pyright: ignore[reportAttributeAccessIssue]
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
        """The scene's complete integrable state: ``mjSTATE_INTEGRATION``, the minimal subset a deterministic
        MuJoCo sim restores from to reproduce its forward trajectory. Object poses in it let analysis recompute
        success. Positions start at index 1, after the scalar time."""
        data = self._robot_view.mj_data
        model = data.model
        spec = mujoco.mjtState.mjSTATE_INTEGRATION  # pyright: ignore[reportAttributeAccessIssue]
        state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_getState(model, data, state, spec)  # pyright: ignore[reportAttributeAccessIssue]
        return state

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
            self._task = None


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve MolmoSpaces over the env-server protocol.')
    parser.add_argument(protocol.OPT_HOST, default='localhost')
    parser.add_argument(protocol.OPT_PORT, type=int, required=True)
    args = parser.parse_args()
    assets = os.environ.get(mapping.ASSETS_DIR_ENV)
    if not assets:
        parser.error(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    env = MolmoSpacesEnv(Path(assets))
    EnvServer(env, args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
