"""MolmoSpaces — AllenAI's MuJoCo manipulation benchmark — behind the env-server protocol.

MolmoSpaces pins ``mujoco ~=3.5`` + its asset stack on Python 3.11, so this never shares positronic's venv: the
launcher runs it with the molmospaces ``.venv``'s python (``env.py --host ... --port ... --benchmark_dir ...``),
with the positronic-free ``server``/``protocol`` and this package's ``mapping`` module on ``PYTHONPATH``. It
imports only ``molmo_spaces`` (+ mujoco/numpy) and those, never ``positronic``.

positronic owns the control loop: this server drives a single MolmoSpaces ``BaseMujocoTask`` per episode directly
(``JsonEvalTaskSampler.sample_task`` builds the full sim/scene/renderer; ``reset``/``step``/``is_done``/
``judge_success`` drive it), replacing MolmoSpaces' own ``JsonEvalRunner`` loop. The reset token selects the
benchmark episode (index into ``benchmark.json``) and an optional seed; the client-side ``MolmoAdapter`` maps the
raw payload this server reports into the canonical embodiment contract.

Command side: the ``MolmoAdapter`` forwards a joint command (the DROID rig runs the joint-position controller);
this server integrates it onto the measured joints and steps the per-move-group ``{arm, gripper}`` action.
Observation side: MolmoSpaces' obs carries the joint positions/velocities and camera frames, but the
end-effector *world* pose is read from the robot view's grasp-site frame here, alongside the gripper closure, into
the raw payload the adapter assembles into a ``MujocoFrankaState``.
"""

import argparse
import os
import sys
import types

# MolmoSpaces renders MuJoCo scenes, so the GL backend must be selected before any mujoco/molmo_spaces import.
# The launcher sets MUJOCO_GL in the subprocess env (egl by default); default it here too so a direct invocation
# (e.g. a validate/e2e run) still boots. Set before the imports below.
os.environ.setdefault('MUJOCO_GL', 'egl')


def _install_cgl_noop_stub() -> None:
    # HACK: MolmoSpaces' renderer hardcodes a macOS CGL context on the CPU (device_id=None) render path
    # (opengl_rendering.py does ``from mujoco.cgl import cgl``), which dlopens Apple's OpenGL.framework and
    # crashes at renderer init on Linux — so a CPU-rendered server (MUJOCO_GL=osmesa or mesa software EGL)
    # dies before the first observation. CGL locking is a no-op off macOS, so stub the module: the import
    # resolves and the (un)lock does nothing. Untouched on a GPU box, where the EGL path never imports it.
    if 'mujoco.cgl' in sys.modules:
        return
    cgl = types.ModuleType('mujoco.cgl.cgl')
    cgl.CGLLockContext = cgl.CGLUnlockContext = lambda *args, **kwargs: None
    package = types.ModuleType('mujoco.cgl')
    package.cgl = cgl
    sys.modules['mujoco.cgl'] = package
    sys.modules['mujoco.cgl.cgl'] = cgl


_install_cgl_noop_stub()

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from server import EnvProtocol, EnvServer  # noqa: E402

import molmo_spaces.evaluation.json_eval_runner  # noqa: E402, F401 -- load first: breaks a circular import that importing json_eval_task_sampler directly hits
from molmo_spaces.configs.policy_configs import DummyPolicyConfig  # noqa: E402
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig  # noqa: E402
from molmo_spaces.evaluation.benchmark_schema import load_all_episodes  # noqa: E402
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig  # noqa: E402
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler  # noqa: E402


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


class MolmoSpacesEnv(EnvProtocol):
    """A MolmoSpaces benchmark episode behind the gym-style ``reset``/``step``/``close`` the env server serves.

    Built per reset from the token's episode index (into the loaded ``benchmark.json``) and its seed. MolmoSpaces'
    ``task.reset()`` does not re-place the scene — ``sample_task`` does — so each reset rebuilds the task for a
    clean, deterministic scene (benchmark episodes are exact-pose deterministic, so a rebuild reproduces them).
    ``step`` integrates the forwarded joint command onto the measured joints, drives the per-move-group action,
    and reports MolmoSpaces' ``is_done``/``judge_success``.
    """

    def __init__(self, benchmark_dir: str) -> None:
        self._episodes = load_all_episodes(Path(benchmark_dir))
        self._sampler = None
        self._task = None
        self._robot_view = None
        self._control_dt = None
        self._meta = None
        # The RGB camera keys the current episode renders — emitted every frame; the client's ``camera_dict``
        # selects which the policy sees.
        self._camera_names: list[str] = []

    def _build(self, episode_index: int, seed: int | None) -> None:
        if self._sampler is not None:
            self._sampler.close()  # release the prior episode's sim/renderer before building the next
        episode = self._episodes[episode_index]
        cfg = _DroidPickEvalConfig()
        # Determinism enters at sampler construction (seed_task_sampling); the token's seed overrides the spec's.
        cfg.seed = int(seed) if seed is not None else (episode.seed if episode.seed is not None else 42)
        # positronic's harness owns the episode deadline (the eval's timeout), so disable MolmoSpaces' internal
        # step horizon: JsonBenchmarkEvalConfig defaults it to 500 steps (~33s at the 66ms policy period), which
        # would self-terminate the task before the harness timeout and truncate the score. ``None`` -> the task
        # runs to an infinite horizon, so ``is_done`` reports only the task's own success/termination and the
        # harness stops the trial at its timeout.
        cfg.task_horizon = None
        self._sampler = JsonEvalTaskSampler(cfg, episode)
        self._task = self._sampler.sample_task(house_index=episode.house_index)
        self._robot_view = self._task.env.current_robot.robot_view
        self._control_dt = cfg.policy_dt_ms / 1000.0
        # The authoritative benchmark prompt, straight from the episode spec — not
        # ``task.get_task_description()``, which upstream reconstructs per task type (e.g. OpeningTask emits
        # "Open the ..." even for a close episode), so a reconstruction could diverge from the benchmark goal.
        self._meta = {'task': episode.language.task_description, 'house_index': episode.house_index}

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        self._build(token['episode_index'], token.get('seed'))
        obs, _info = self._task.reset()  # obs is a list, one dict per env; n_batch == 1
        env_obs = obs[0]
        self._camera_names = [k for k, v in env_obs.items() if _is_rgb_frame(v)]
        # robot_meta is empty: this venv cannot import positronic to emit the Franka model, so the eval supplies
        # it via ``static_meta`` (``bundled_franka_model``). ``meta`` carries the scene/task identity.
        return {'obs': self._observe(env_obs), 'meta': self._meta, 'robot_meta': {}, 'control_dt': self._control_dt}

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        arm = mapping.wire_command_to_arm_action(action['command'], self._measured_arm_q())
        gripper = np.array([mapping.grip_command_to_actuator(action['grip'])], dtype=np.float32)
        obs, _reward, _term, _trunc, _infos = self._task.step({'arm': arm, 'gripper': gripper})
        # positronic owns the deadline (the harness timeout), so the episode ends here on the task's own scored
        # success — end-on-success, the benchmark's scoring semantics — or any MolmoSpaces terminal (a done
        # action). The internal step horizon is disabled (see ``_build``), so ``is_done`` reports only
        # ``is_terminal``, never a timeout; without terminating on success here a successful rollout that keeps
        # sending joint commands would run to the harness timeout and be scored as a non-success.
        success = bool(self._task.judge_success())
        done = success or bool(self._task.is_done())
        return {'obs': self._observe(obs[0]), 'done': done, 'success': success, 'control_dt': self._control_dt}

    def _measured_arm_q(self) -> np.ndarray:
        return np.asarray(self._robot_view.get_move_group('arm').joint_pos, dtype=np.float32)

    def _observe(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        # MolmoSpaces' obs carries the joint positions/velocities and camera frames; the eef *world* pose is read
        # from the arm move group's grasp-site frame (obs only exposes a robot-relative tcp pose).
        arm = self._robot_view.get_move_group('arm')
        eef_world = np.asarray(arm.leaf_frame_to_world, dtype=np.float64)  # 4x4 grasp-site world transform
        eef_quat = np.zeros(4)
        mujoco.mju_mat2Quat(eef_quat, np.ascontiguousarray(eef_world[:3, :3].reshape(9)))  # -> wxyz
        payload = {
            'joint_pos': np.asarray(arm.joint_pos, dtype=np.float32),
            'joint_vel': np.asarray(arm.joint_vel, dtype=np.float32),
            'eef_pos': eef_world[:3, 3].astype(np.float32),
            'eef_quat': eef_quat.astype(np.float32),
            'grip': np.float32(mapping.normalize_grip_qpos(env_obs['qpos']['gripper'])),
        }
        for name in self._camera_names:
            payload[name] = np.ascontiguousarray(env_obs[name])
        return payload

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
            self._task = None


def _is_rgb_frame(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve MolmoSpaces over the env-server protocol.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--benchmark_dir', required=True, help='dir containing benchmark.json')
    args = parser.parse_args()
    if not os.environ.get('MLSPACES_ASSETS_DIR'):
        parser.error('MLSPACES_ASSETS_DIR must point at the MolmoSpaces asset packs')
    EnvServer(MolmoSpacesEnv(args.benchmark_dir), args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
