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

# ``molmo_spaces`` (+ its transitive configs/tasks) and the flat ``protocol`` resolve only inside MolmoSpaces'
# own venv, where the launcher runs this module; pyright checks it against positronic's deps, which cannot see
# them. Each of those imports carries its own ``reportMissingImports`` suppression, so an import that should
# resolve here — anything from positronic, which this module must never take — still fails the check.

import argparse
import os
import sys
import types

import mapping  # positronic-free wire mappings, on PYTHONPATH; numpy only, so it pulls in no GL

# MolmoSpaces renders MuJoCo scenes, so the GL backend must be selected before any mujoco/molmo_spaces import.
# The launcher sets it in the subprocess env; default it here too so a direct invocation (e.g. a validate/e2e
# run) still boots. Set before the imports below.
os.environ.setdefault(mapping.GL_BACKEND_ENV, mapping.GL_BACKEND_DEFAULT)


# MolmoSpaces' renderer module, which the stub below stands in for on Linux.
_CGL_PACKAGE = 'mujoco.cgl'
_CGL_MODULE = f'{_CGL_PACKAGE}.cgl'


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

# Imported for its import-time ``_assert_data_versions_match()``: MolmoSpaces pins the asset versions a
# benchmark may be evaluated against, and this is the only place upstream enforces it. Driving the sampler
# directly skips the native entrypoint, so without this a run on mismatched asset packs would score where
# MolmoSpaces itself refuses to.
import molmo_spaces.evaluation.eval_main  # noqa: E402, F401  # pyright: ignore[reportMissingImports]
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
from molmo_spaces.tasks.json_eval_task_sampler import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    JsonEvalTaskSampler,
)

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


# MolmoSpaces reports a move group's leaf frame as one of MuJoCo's frame kinds; the arm's is a site.
_SITE_FRAME = 'site'


def _discovery_hint() -> str:
    """The benchmark dirs found under ``MLSPACES_ASSETS_DIR``, appended to a path that holds none."""
    assets = os.environ.get(mapping.ASSETS_DIR_ENV)
    if not assets:
        return f' Point {mapping.ASSETS_DIR_ENV} at the MolmoSpaces asset packs to have the available ones listed.'
    root = Path(assets) / mapping.ASSETS_BENCHMARKS_DIR
    found = sorted(str(p.parent) for p in root.rglob(mapping.MOLMO_BENCHMARK_MANIFEST))
    if not found:
        return f' No {mapping.MOLMO_BENCHMARK_MANIFEST} found under {root}.'
    return f' Available under {root}: {", ".join(found)}'


def _assert_measures_at_grasp_site(robot_view: Any) -> None:
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
    """A MolmoSpaces benchmark behind the ``tasks``/``reset``/``step``/``close`` the env server serves.

    ``tasks`` answers the benchmark's episode records. Each reset builds from the token's episode index (into
    the loaded ``benchmark.json``) and its seed: MolmoSpaces' ``task.reset()`` does not re-place the scene —
    ``sample_task`` does — so each reset rebuilds the task for a clean, deterministic scene (benchmark episodes
    are exact-pose deterministic, so a rebuild reproduces them). ``step`` integrates the forwarded joint command
    onto the measured joints, drives the per-move-group action, and reports MolmoSpaces'
    ``is_done``/``judge_success``.
    """

    def __init__(self, benchmark_dir: Path, task_horizon_steps: int | None = None) -> None:
        self._benchmark_dir = benchmark_dir
        self._episodes = load_all_episodes(benchmark_dir)
        # An explicit per-run horizon override (steps), mirroring MolmoSpaces' ``--task_horizon_steps``; ``None``
        # reads the benchmark's own ``task_horizon_sec``.
        self._task_horizon_override = task_horizon_steps
        self._sampler: Any = None
        self._task: Any = None
        self._robot_view: Any = None
        self._control_dt: float | None = None
        # The episode's enforced horizon in sim-seconds (``task_horizon`` steps x the control period), reported
        # at reset.
        self._horizon_sec: float | None = None
        self._meta: dict[str, Any] | None = None
        # The RGB camera keys the current episode renders, emitted every frame.
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
        # Determinism enters at sampler construction (seed_task_sampling).
        cfg.seed = mapping.resolve_episode_seed(episode, episode_index, seed)
        # With ``task_horizon`` set, the task enforces it and ``is_done`` reports expiry, so a horizon-expired
        # trial ends with a terminal ``done`` exactly as the native benchmark scores it. The horizon is the
        # benchmark's, not an episode's.
        cfg.task_horizon = mapping.resolve_task_horizon_steps(
            self._episodes, cfg.policy_dt_ms, self._task_horizon_override
        )
        self._sampler = JsonEvalTaskSampler(cfg, episode)
        self._task = self._sampler.sample_task(house_index=episode.house_index)
        self._robot_view = self._task.env.current_robot.robot_view
        _assert_measures_at_grasp_site(self._robot_view)
        self._scratch = None  # sized by this episode's model; allocated on the first probe
        self._control_dt = cfg.policy_dt_ms / 1000.0
        self._horizon_sec = cfg.task_horizon * self._control_dt
        # The authoritative benchmark prompt, straight from the episode spec — not
        # ``task.get_task_description()``, which upstream reconstructs per task type (e.g. OpeningTask emits
        # "Open the ..." even for a close episode), so a reconstruction could diverge from the benchmark goal.
        self._meta = {
            mapping.META_TASK: episode.language.task_description,
            mapping.META_HOUSE_INDEX: episode.house_index,
        }

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """The episode records ``spec`` selects: ``episodes`` is one index, or a list of them; absent, the whole
        benchmark. Every record carries the one horizon the benchmark enforces, converted against the config
        that enforces it."""
        if not self._episodes:
            raise ValueError(
                f'no benchmark episodes under {self._benchmark_dir}; expected a '
                f'{mapping.MOLMO_BENCHMARK_MANIFEST} or a legacy house_*/episode_*.json layout.{_discovery_hint()}'
            )
        count = len(self._episodes)
        selection = spec.get('episodes')
        indices = list(range(count)) if selection is None else [selection] if isinstance(selection, int) else selection
        # A negative index would silently run a from-the-end episode mislabeled by its own index.
        out_of_range = [i for i in indices if not 0 <= i < count]
        if out_of_range:
            raise ValueError(
                f'episodes {out_of_range} out of range for the {count} episodes under {self._benchmark_dir}'
            )
        cfg = _DroidPickEvalConfig()
        steps = mapping.resolve_task_horizon_steps(self._episodes, cfg.policy_dt_ms, self._task_horizon_override)
        horizon_sec = steps * cfg.policy_dt_ms / 1000.0
        return [
            {'name': self._episodes[i].language.task_description, 'episode_index': i, 'task_horizon_sec': horizon_sec}
            for i in indices
        ]

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        self._build(token[mapping.TOKEN_EPISODE_INDEX], token.get(mapping.TOKEN_SEED))
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
            protocol.FRAME_HORIZON: self._horizon_sec,
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        arm = mapping.wire_command_to_arm_action(
            action[protocol.ACTION_COMMAND],
            self._measured_arm_q(),
            ik=self._ik_robot_frame,
            current_eef=self._measured_eef_pose(),
        )
        gripper = np.array([mapping.grip_command_to_actuator(action[protocol.ACTION_GRIP])], dtype=np.float32)
        obs, _reward, _term, _trunc, _infos = self._task.step({
            mapping.MOLMO_ARM_GROUP: arm,
            mapping.MOLMO_GRIPPER_GROUP: gripper,
        })
        # The trial ends on the task's judged success, on a MolmoSpaces terminal, or on horizon expiry — the
        # latter two through ``is_done``. ``success`` is ORed in for end-on-success, the benchmark's scoring
        # semantics: without it a successful rollout that kept sending joint commands would idle to the horizon.
        # It stays ``judge_success()`` alone, so a horizon expiry ends the trial with ``success=False``, as
        # native scoring has it.
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
        """The measured grasp-site pose in the ROBOT frame as ``(translation, 3x3 rotation)`` — the frame a
        Cartesian command targets and the one ``_observe`` reports, so command and observation share a frame.

        Robot-frame rather than world: a scene-world pose puts the arm metres from the origin (the ProcTHOR
        house frame), which no DROID-trained checkpoint has ever seen as proprioception. MolmoSpaces' own
        ``tcp_pose`` sensor reports ``leaf_frame_to_robot`` for the same reason.
        """
        eef_robot = np.asarray(
            self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP).leaf_frame_to_robot, dtype=np.float64
        )
        return eef_robot[:3, 3].copy(), eef_robot[:3, :3].copy()

    def _robot_to_world(self) -> tuple[np.ndarray, np.ndarray]:
        """The robot base pose as ``(translation, 3x3 rotation)``, composing a robot-frame pose into world."""
        base = np.asarray(self._robot_view.base.pose, dtype=np.float64)
        return base[:3, 3].copy(), base[:3, :3].copy()

    def _ik_robot_frame(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        """A robot-frame Cartesian target -> arm joint targets, through the world-frame solver.

        Commands arrive in the frame ``_observe`` reports; ``_ik`` solves in world, so the base transform is
        applied here rather than duplicated in either.
        """
        t, r = self._robot_to_world()
        return self._ik(r @ np.asarray(target_pos, dtype=np.float64) + t, r @ np.asarray(target_rot, dtype=np.float64))

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
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
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
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
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
            self._leaf_jacobian(arm, model, data, jac)
            jac = jac[:, veladr]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + _IK_DAMPING**2 * np.eye(6), err)
            q = np.clip(q + dq, limits[:, 0], limits[:, 1])
        return q

    @staticmethod
    def _leaf_jacobian(move_group: Any, model: Any, data: Any, out: np.ndarray) -> None:
        """The ``(6, nv)`` leaf-frame Jacobian into *out*, evaluated on *data*.

        Mirrors the move group's own ``get_jacobian`` but against a caller-supplied ``MjData``, which the IK
        iteration needs (the group's method is bound to the live one).
        """
        if move_group.leaf_frame_type == _SITE_FRAME:
            mujoco.mj_jacSite(model, data, out[:3], out[3:], move_group.leaf_frame_id)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            mujoco.mj_jacBody(model, data, out[:3], out[3:], move_group.leaf_frame_id)  # pyright: ignore[reportAttributeAccessIssue]

    def _observe(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """The raw observation payload for one env frame: measured joints, the eef world pose, grip, camera frames.

        MolmoSpaces' obs carries the joint positions/velocities and camera frames; the eef pose is read from the
        arm move group's grasp-site frame, in the robot frame the policy is trained against.
        """
        arm = self._robot_view.get_move_group(mapping.MOLMO_ARM_GROUP)
        eef_robot = np.asarray(arm.leaf_frame_to_robot, dtype=np.float64)  # 4x4 grasp-site robot-frame transform
        eef_quat = np.zeros(4)  # filled wxyz below
        rot9 = np.ascontiguousarray(eef_robot[:3, :3].reshape(9))
        # mju_mat2Quat is a C binding absent from mujoco's type stubs, so pyright can't see the attribute.
        mujoco.mju_mat2Quat(eef_quat, rot9)  # pyright: ignore[reportAttributeAccessIssue]
        payload = {
            mapping.OBS_JOINT_POS: np.asarray(arm.joint_pos, dtype=np.float32),
            mapping.OBS_JOINT_VEL: np.asarray(arm.joint_vel, dtype=np.float32),
            mapping.OBS_EEF_POS: eef_robot[:3, 3].astype(np.float32),
            mapping.OBS_EEF_QUAT: eef_quat.astype(np.float32),
            mapping.OBS_GRIP: np.float32(
                mapping.normalize_grip_qpos(env_obs[mapping.MOLMO_OBS_QPOS][mapping.MOLMO_GRIPPER_GROUP])
            ),
            # The full MuJoCo generalized state: every body's pose + velocity, objects included.
            mapping.OBS_SIM_STATE: self._full_physics_state(),
        }
        for name in self._camera_names:
            payload[name] = np.ascontiguousarray(env_obs[name])
        return payload

    def _full_physics_state(self) -> np.ndarray:
        """The scene's complete integrable state: ``mjSTATE_INTEGRATION``, the minimal subset a deterministic
        MuJoCo sim restores from to reproduce its forward trajectory — positions and velocities, and with them
        mocap bodies, actuator activation, controls and the solver warm-start. Object poses in it let analysis
        recompute success. Positions start at index 1, after the scalar time."""
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


# rules-allow: stranded-definition — this file keeps its pure MuJoCo helpers at module scope as a set:
# `_assert_measures_at_grasp_site`, `_leaf_pose` and `_pose_error` all take plain model/data arguments, hold no
# `self`, and are called only from `MolmoSpacesEnv`. Moving one into the class splits the set for no gain;
# moving all three is a layout decision for the file, not a fix to this definition.
def _leaf_pose(move_group: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    """A move group's leaf-frame world pose read off *data* — which may be a scratch ``MjData``, unlike the
    group's own ``leaf_frame_to_world``, so IK can probe candidate joints without touching the live sim."""
    if move_group.leaf_frame_type == _SITE_FRAME:
        pos, mat = data.site_xpos[move_group.leaf_frame_id], data.site_xmat[move_group.leaf_frame_id]
    else:
        pos, mat = data.xpos[move_group.leaf_frame_id], data.xmat[move_group.leaf_frame_id]
    return np.array(pos, dtype=np.float64), np.array(mat, dtype=np.float64).reshape(3, 3)


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


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve MolmoSpaces over the env-server protocol.')
    parser.add_argument(protocol.OPT_HOST, default='localhost')
    parser.add_argument(protocol.OPT_PORT, type=int, required=True)
    parser.add_argument(
        mapping.OPT_BENCHMARK_DIR, required=True, help=f'dir containing {mapping.MOLMO_BENCHMARK_MANIFEST}'
    )
    parser.add_argument(
        mapping.OPT_TASK_HORIZON_STEPS,
        type=int,
        default=None,
        help='override the benchmark horizon (steps per episode)',
    )
    args = parser.parse_args()
    if not os.environ.get(mapping.ASSETS_DIR_ENV):
        parser.error(f'{mapping.ASSETS_DIR_ENV} must point at the MolmoSpaces asset packs')
    env = MolmoSpacesEnv(Path(args.benchmark_dir), args.task_horizon_steps)
    EnvServer(env, args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
