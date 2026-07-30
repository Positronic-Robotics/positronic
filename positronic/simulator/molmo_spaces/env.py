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

# molmo_spaces (+ its transitive configs/tasks) resolves only inside MolmoSpaces' own venv, where the launcher
# runs this module; pyright checks it against positronic's deps, which cannot see it. This module imports no
# positronic packages, so missing-import errors here are exclusively those foreign imports — suppress just that
# category file-wide; every other type check (wrong types, optional access, ...) stays active.
# pyright: reportMissingImports=false

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
    cgl.CGLLockContext = cgl.CGLUnlockContext = lambda *args, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    package = types.ModuleType('mujoco.cgl')
    package.cgl = cgl  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules['mujoco.cgl'] = package
    sys.modules['mujoco.cgl.cgl'] = cgl


_install_cgl_noop_stub()

from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mapping  # noqa: E402 -- positronic-free wire mappings, on PYTHONPATH
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import protocol  # noqa: E402 -- the positronic-free wire contract, on PYTHONPATH beside ``server``

# server resolves to a module without these symbols under positronic's deps (the real one is on the molmo
# venv's PYTHONPATH), so the symbols read as unknown here.
from server import EnvProtocol, EnvServer  # noqa: E402  # pyright: ignore[reportAttributeAccessIssue]

import molmo_spaces.evaluation.json_eval_runner  # noqa: E402, F401 -- load first: breaks a circular import that importing json_eval_task_sampler directly hits
from molmo_spaces.configs.policy_configs import DummyPolicyConfig  # noqa: E402
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig  # noqa: E402
from molmo_spaces.evaluation.benchmark_schema import load_all_episodes  # noqa: E402
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig  # noqa: E402
from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler  # noqa: E402

# Damped-least-squares differential IK, matching the LIBERO rig's solver (positronic/simulator/libero/env.py):
# the same iteration budget, damping and convergence tolerance, on MuJoCo's own site/body Jacobian.
_IK_ITERS = 100
_IK_DAMPING = 0.05
_IK_TOL = 1e-4


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

    def __init__(self, benchmark_dir: Path, task_horizon_steps: int | None = None) -> None:
        self._episodes = load_all_episodes(benchmark_dir)
        # An explicit per-run horizon override (steps), mirroring MolmoSpaces' ``--task_horizon_steps``; ``None``
        # reads the benchmark's own ``task_horizon_sec``.
        self._task_horizon_override = task_horizon_steps
        self._sampler: Any = None
        self._task: Any = None
        self._robot_view: Any = None
        self._control_dt: float | None = None
        # The episode's enforced horizon in sim-seconds (``task_horizon`` steps x the control period), reported at
        # reset so positronic can check its safety-net timeout is strictly weaker.
        self._horizon_sec: float | None = None
        self._meta: dict[str, Any] | None = None
        # The RGB camera keys the current episode renders — emitted every frame; the client's ``camera_dict``
        # selects which the policy sees.
        self._camera_names: list[str] = []
        # Scratch ``MjData`` the kinematics probes (``_fk``/``_ik``) run on, allocated once per episode and
        # refreshed from the live buffer per call — a Cartesian policy solves IK every control step, so the
        # allocation stays out of the loop. Rebuilt in ``_build``, since it is sized by the episode's model.
        self._scratch: Any = None

    def _build(self, episode_index: int, seed: int | None) -> None:
        if self._sampler is not None:
            self._sampler.close()  # release the prior episode's sim/renderer before building the next
        episode = self._episodes[episode_index]
        cfg = _DroidPickEvalConfig()
        # Determinism enters at sampler construction (seed_task_sampling); the token's seed overrides the spec's.
        cfg.seed = int(seed) if seed is not None else (episode.seed if episode.seed is not None else 42)
        # The sim owns the episode horizon: it is part of the task definition, so resolve the benchmark's own
        # ``task_horizon_sec`` into steps (``mapping.resolve_task_horizon_steps``; DROID Pick = 20 s -> 303 steps).
        # With ``task_horizon`` set, the task enforces it and ``is_done`` reports expiry, so a horizon-expired
        # trial ends with a terminal ``done`` exactly as the native benchmark scores it. The harness
        # ``Task.timeout`` is only a weaker safety net above this horizon.
        cfg.task_horizon = mapping.resolve_task_horizon_steps(episode, cfg.policy_dt_ms, self._task_horizon_override)
        self._sampler = JsonEvalTaskSampler(cfg, episode)
        self._task = self._sampler.sample_task(house_index=episode.house_index)
        self._robot_view = self._task.env.current_robot.robot_view
        self._scratch = None  # sized by this episode's model; allocated on the first probe
        self._control_dt = cfg.policy_dt_ms / 1000.0
        self._horizon_sec = cfg.task_horizon * self._control_dt
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
        # it via ``static_meta`` (``bundled_franka_model``). ``meta`` carries the scene/task identity; ``horizon``
        # is the sim-enforced episode deadline the harness checks its timeout against.
        return {
            'obs': self._observe(env_obs),
            'meta': self._meta,
            'robot_meta': {},
            'control_dt': self._control_dt,
            'horizon': self._horizon_sec,
            # This adoption covers the canonical command contract in full: the rig natively takes joint-position
            # targets alone, and ``wire_command_to_arm_action`` converts every other canonical type into one.
            'command_types': list(protocol.CANONICAL_COMMAND_TYPES),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        arm = mapping.wire_command_to_arm_action(
            action['command'], self._measured_arm_q(), ik=self._ik, current_eef=self._measured_eef_pose()
        )
        gripper = np.array([mapping.grip_command_to_actuator(action['grip'])], dtype=np.float32)
        obs, _reward, _term, _trunc, _infos = self._task.step({'arm': arm, 'gripper': gripper})
        # The trial ends on the task's judged success, on any MolmoSpaces terminal (a done action), or on native
        # horizon expiry — ``is_done`` covers the latter two now that the horizon is enabled (see ``_build``).
        # ``success`` is added explicitly for end-on-success, the benchmark's scoring semantics: without it a
        # successful rollout that kept sending joint commands would idle to the horizon and still score success.
        # ``success`` stays ``judge_success()`` alone, so a horizon expiry ends the trial with ``success=False``,
        # matching native scoring. The harness reads this ``done`` as the trial's true end; its timeout is a
        # safety net for a sim that never terminates.
        success = bool(self._task.judge_success())
        done = success or bool(self._task.is_done())
        return {'obs': self._observe(obs[0]), 'done': done, 'success': success, 'control_dt': self._control_dt}

    def _measured_arm_q(self) -> np.ndarray:
        return np.asarray(self._robot_view.get_move_group('arm').joint_pos, dtype=np.float32)

    def _measured_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """The measured grasp-site world pose as ``(translation, 3x3 rotation)`` — the frame a Cartesian
        command targets and the one ``observe_payload`` reports, so command and observation share a frame."""
        eef_world = np.asarray(self._robot_view.get_move_group('arm').leaf_frame_to_world, dtype=np.float64)
        return eef_world[:3, 3].copy(), eef_world[:3, :3].copy()

    def _scratch_data(self, move_group: Any) -> Any:
        """The scratch ``MjData``, refreshed from the live one, for off-sim kinematics probing.

        A fresh ``MjData`` seeded with ``qpos`` alone is NOT equivalent: MolmoSpaces places the robot in a scene
        whose pose also rides on state outside ``qpos`` (mocap bodies among it), which a fresh buffer resets to
        the model defaults — the grasp site then resolves metres away from the live one. Copying the whole
        struct keeps every such field, so the probe differs from the live scene only in the joints the caller
        sets, and copying into a retained buffer keeps the per-step allocation out of the control loop.
        """
        if self._scratch is None:
            self._scratch = mujoco.MjData(move_group.mj_model)  # pyright: ignore[reportAttributeAccessIssue]
        mujoco.mj_copyData(self._scratch, move_group.mj_model, move_group.mj_data)  # pyright: ignore[reportAttributeAccessIssue]
        return self._scratch

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The grasp-site world pose a candidate arm configuration reaches.

        Evaluated on a scratch ``MjData`` seeded from the live scene (objects intact), so the live sim is never
        perturbed: set the arm joints, propagate, read the leaf frame. The inverse of ``_ik``.
        """
        arm = self._robot_view.get_move_group('arm')
        data = self._scratch_data(arm)
        data.qpos[np.asarray(arm.joint_posadr)] = np.asarray(q, dtype=np.float64).reshape(-1)
        mujoco.mj_forward(arm.mj_model, data)  # pyright: ignore[reportAttributeAccessIssue]
        return _leaf_pose(arm, data)

    def _ik(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        """Absolute world grasp-site target -> the arm joint targets that reach it.

        Damped-least-squares differential IK on MuJoCo's own leaf-frame Jacobian, mirroring the LIBERO rig's
        solver. It iterates on a scratch ``MjData`` seeded from the live scene (objects intact), so probing
        candidate joint configurations never perturbs the sim being stepped. Joint targets stay inside the
        move group's limits, and a target the arm cannot reach yields the closest configuration the iteration
        reached rather than raising — an unreachable waypoint holds near the limit instead of aborting a trial.
        """
        arm = self._robot_view.get_move_group('arm')
        model = arm.mj_model
        posadr = np.asarray(arm.joint_posadr)
        veladr = np.asarray(arm.joint_veladr)
        limits = np.asarray(arm.joint_pos_limits, dtype=np.float64)
        data = self._scratch_data(arm)
        q = np.asarray(arm.joint_pos, dtype=np.float64).copy()
        for _ in range(_IK_ITERS):
            data.qpos[posadr] = q
            mujoco.mj_forward(model, data)  # pyright: ignore[reportAttributeAccessIssue]
            cur_pos, cur_rot = _leaf_pose(arm, data)
            err = _pose_error(target_pos, target_rot, cur_pos, cur_rot)
            if np.linalg.norm(err) < _IK_TOL:
                break
            jac = np.zeros((6, model.nv))
            _leaf_jacobian(arm, model, data, jac)
            jac = jac[:, veladr]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + _IK_DAMPING**2 * np.eye(6), err)
            q = np.clip(q + dq, limits[:, 0], limits[:, 1])
        return q

    def _observe(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        return observe_payload(self._robot_view, env_obs, self._camera_names)

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
            self._task = None


def observe_payload(robot_view: Any, env_obs: dict[str, Any], camera_names: list[str]) -> dict[str, Any]:
    """The raw observation payload for one env frame: measured joints, the eef world pose, grip, camera frames.

    MolmoSpaces' obs carries the joint positions/velocities and camera frames; the eef *world* pose is read from
    the arm move group's grasp-site frame (obs only exposes a robot-relative tcp pose). The parity reference
    (``parity_native.py``) shares this extraction so its comparison against the env-server path isolates the sim
    rollout, not the observation mapping.
    """
    arm = robot_view.get_move_group('arm')
    eef_world = np.asarray(arm.leaf_frame_to_world, dtype=np.float64)  # 4x4 grasp-site world transform
    eef_quat = np.zeros(4)  # filled wxyz below
    rot9 = np.ascontiguousarray(eef_world[:3, :3].reshape(9))
    # mju_mat2Quat is a C binding absent from mujoco's type stubs, so pyright can't see the attribute.
    mujoco.mju_mat2Quat(eef_quat, rot9)  # pyright: ignore[reportAttributeAccessIssue]
    payload = {
        'joint_pos': np.asarray(arm.joint_pos, dtype=np.float32),
        'joint_vel': np.asarray(arm.joint_vel, dtype=np.float32),
        'eef_pos': eef_world[:3, 3].astype(np.float32),
        'eef_quat': eef_quat.astype(np.float32),
        'grip': np.float32(mapping.normalize_grip_qpos(env_obs['qpos']['gripper'])),
        # The full MuJoCo generalized state (every body's pose + velocity, objects included) — privileged ground
        # truth the adapter routes to the recorder, so success can be recomputed/audited offline (like libero's
        # ``sim_state``), never fed to the policy.
        'sim_state': _full_physics_state(robot_view),
    }
    for name in camera_names:
        payload[name] = np.ascontiguousarray(env_obs[name])
    return payload


def _leaf_pose(move_group: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    """A move group's leaf-frame world pose read off *data* — which may be a scratch ``MjData``, unlike the
    group's own ``leaf_frame_to_world``, so IK can probe candidate joints without touching the live sim."""
    if move_group.leaf_frame_type == 'site':
        pos, mat = data.site_xpos[move_group.leaf_frame_id], data.site_xmat[move_group.leaf_frame_id]
    else:
        pos, mat = data.xpos[move_group.leaf_frame_id], data.xmat[move_group.leaf_frame_id]
    return np.array(pos, dtype=np.float64), np.array(mat, dtype=np.float64).reshape(3, 3)


def _leaf_jacobian(move_group: Any, model: Any, data: Any, out: np.ndarray) -> None:
    """The ``(6, nv)`` leaf-frame Jacobian into *out*, evaluated on *data*.

    Mirrors the move group's own ``get_jacobian`` but against a caller-supplied ``MjData``, which the IK
    iteration needs (the group's method is bound to the live one).
    """
    if move_group.leaf_frame_type == 'site':
        mujoco.mj_jacSite(model, data, out[:3], out[3:], move_group.leaf_frame_id)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        mujoco.mj_jacBody(model, data, out[:3], out[3:], move_group.leaf_frame_id)  # pyright: ignore[reportAttributeAccessIssue]


def _pose_error(target_pos: np.ndarray, target_rot: np.ndarray, cur_pos: np.ndarray, cur_rot: np.ndarray) -> np.ndarray:
    """The world-frame 6-vector error ``[translation, axis-angle rotation]`` from a measured to a target pose.

    Both halves are expressed in the world frame, matching the world-frame leaf Jacobian the IK step solves
    against. The rotation error is the axis-angle of ``R_target @ R_cur^T``, via MuJoCo's quaternion helpers.
    """
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray((target_rot @ cur_rot.T).reshape(9)))  # pyright: ignore[reportAttributeAccessIssue]
    rot_err = np.zeros(3)
    mujoco.mju_quat2Vel(rot_err, quat, 1.0)  # pyright: ignore[reportAttributeAccessIssue]
    return np.concatenate([np.asarray(target_pos, dtype=np.float64).reshape(3) - cur_pos, rot_err])


def _full_physics_state(robot_view: Any) -> np.ndarray:
    """The scene's full generalized state — ``qpos`` (all positions) then ``qvel`` (all velocities) — a
    deterministic MuJoCo sim replays from it, and object poses in ``qpos`` let analysis recompute success."""
    data = robot_view.mj_data
    return np.concatenate([np.asarray(data.qpos, dtype=np.float64), np.asarray(data.qvel, dtype=np.float64)])


def _is_rgb_frame(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[2] == 3 and value.dtype == np.uint8


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve MolmoSpaces over the env-server protocol.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--benchmark_dir', required=True, help='dir containing benchmark.json')
    parser.add_argument(
        '--task_horizon_steps', type=int, default=None, help='override the benchmark horizon (steps per episode)'
    )
    args = parser.parse_args()
    if not os.environ.get('MLSPACES_ASSETS_DIR'):
        parser.error('MLSPACES_ASSETS_DIR must point at the MolmoSpaces asset packs')
    env = MolmoSpacesEnv(Path(args.benchmark_dir), args.task_horizon_steps)
    EnvServer(env, args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
