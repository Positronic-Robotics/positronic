"""RoboLab — NVIDIA's Isaac Lab manipulation benchmark — behind the env-server protocol.

RoboLab pins ``isaacsim 5.0`` + ``isaaclab 2.2`` into its own uv project, so this never shares positronic's
venv: the launcher runs ``uv run --project <robolab clone> env.py --host ... --port ... --headless`` with the
positronic-free ``server``/``protocol`` modules on ``PYTHONPATH``. It imports only robolab + isaaclab, never
``positronic``.

One jointpos-substrate env serves every wire command: RoboLab's leaderboard stack drives
``DroidJointPositionActionCfg`` (8-dim ``[q1..q7 absolute rad, gripper]`` at 15 Hz), so ``joint_pos`` passes
through bit-identically and every other command converts server-side into the 7 joint targets — ``joint_vel``
anchors on the measured joints, ``hold`` re-commands them, and Cartesian poses solve through IsaacLab's
differential IK with the controller parameters of RoboLab's own AbsIK registration (``DroidIKActionCfg``).
Cartesian wire poses are in the eef control frame (``eef_frame`` = base_link ∘ ``EEF_OFFSET_ROT``; position
env-local, rotation world) — the same frame the observations report, so observed and commanded pose share one
frame.

The reset token carries the RoboLab env name and the instruction variant; the env builds that task on the
first reset and caches it, rebuilding only when the key changes. There is no seed anywhere: RoboLab's eval
path has no seed hook, so a recorded seed would only mislead.
"""

import argparse
import os
import tempfile
import time
from typing import Any

import cv2  # noqa: F401 -- robolab requires cv2 imported before isaaclab
import numpy as np
import torch
from isaaclab.app import AppLauncher
from protocol import decode
from server import EnvProtocol, EnvServer

# Isaac's rigid bring-up order: parse CLI args, launch the app, and only then import anything that touches the
# omni/isaaclab runtime — which forces the second import block below and the module-level parse. ``validate.py``
# imports this module without ``--port``, so ``main`` checks the port instead of argparse.
parser = argparse.ArgumentParser(description='Serve RoboLab over the env-server protocol.')
parser.add_argument('--host', default='localhost')
parser.add_argument('--port', type=int)
parser.add_argument('--camera-res', type=int, nargs=2, default=(1280, 720), metavar=('WIDTH', 'HEIGHT'))
parser.add_argument('--disable-viewport', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True  # not a CLI flag: every robolab runner forces it (the image obs need rendering)
simulation_app = AppLauncher(args).app

import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    matrix_from_quat,
    quat_from_matrix,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.logging.results import get_all_env_subtask_infos  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from robolab.robots import droid  # noqa: E402
from robolab.robots.droid import EEF_OFFSET_ROT  # noqa: E402
from robolab.variations.camera import OverShoulderLeftCameraCfg  # noqa: E402

# Both flags gate recorder construction in the env cfg's ``__post_init__``, so they are set before any
# ``create_env``: subtask progress feeds the wire ``subtask`` observation; per-step image recording only bloats
# the HDF5 the recorder writes. ``create_env`` also writes ``env_cfg.json`` and recordings into the global
# output dir — point it at a temp dir instead of robolab's repo checkout.
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.set_output_dir(tempfile.mkdtemp(prefix='robolab-env-'))

# The policy cameras (RoboLab's WRIST_LEFT preset) render at a stock 1280x720 while pi05-class policies
# consume ~224x224, so most of the tile render is discarded. The resolution override mutates the ORIGINAL
# ``TiledCameraCfg`` objects: isaaclab's ``configclass`` strips class-level mutables into dataclass fields
# whose ``default_factory`` deepcopies the captured original on every cfg instantiation, so writing the
# originals (before any registration) reaches every task's scene. The wrist original is the ``droid`` module
# global (``DroidCfg`` and ``WristCameraCfg`` both capture it); the over-shoulder original exists only inside
# its factory's closure.
_over_shoulder = (
    OverShoulderLeftCameraCfg
    .__dataclass_fields__['over_shoulder_left_camera']
    .default_factory.__closure__[0]
    .cell_contents
)
assert isinstance(_over_shoulder, TiledCameraCfg), _over_shoulder
for _camera in (droid._WRIST_CAM, _over_shoulder):
    _camera.width, _camera.height = args.camera_res

if args.disable_viewport:
    # The default viewport ('/OmniverseKit_Persp', 1280x720) renders every render step even headless, and
    # nothing in this server consumes it.
    get_active_viewport().updates_enabled = False


def _load_robot_meta() -> dict[str, Any]:
    """The DROID rig's model (the viewer's URDF + Robotiq meshes) decoded from the wire codec and emitted as
    ``robot_meta``. This env runs in RoboLab's Isaac interpreter and cannot import positronic to build the model
    itself, so the launcher builds ``bundled_franka_model()`` on the positronic side, serializes it, and passes
    the path in ``ROBOLAB_ROBOT_META``. Empty when a caller (e.g. ``validate.py``) imports the env without it."""
    path = os.environ.get('ROBOLAB_ROBOT_META')
    if path is None:
        return {}
    with open(path, 'rb') as f:
        return decode(f.read())


class RobolabEnv(EnvProtocol):
    """A RoboLab task behind the gym-style ``reset``/``step``/``close`` the env server serves.

    Built from the reset token's key (RoboLab env name, instruction variant) and cached; ``reset`` rebuilds
    when the key changes and re-randomizes the scene through RoboLab's own reset events, or restores an exact
    recorded state when the token carries one (demo replay). ``step`` maps the forwarded command into the
    8-dim jointpos action and reports ``done`` (episode over) and ``success`` (the task's success termination
    fired).
    """

    def __init__(self):
        self._key = None
        self._env = None
        self._env_cfg = None
        self._meta = None
        self._robot_meta = _load_robot_meta()
        self._control_dt = None
        # Tasks registered with gymnasium in this process; re-registering rebuilds the cfg class and churns
        # the registry, so each registers once.
        self._registered: set[str] = set()
        # Rebuilt with the env: robot handles, model-structure indices, and the standalone IK controller.
        self._robot = None
        self._joint_ids = None
        self._body_idx = None
        self._eef_frame_idx = None
        self._eef_offset_rot = None
        self._ik = None
        self._timeline = omni.timeline.get_timeline_interface()
        self._kit_app = omni.kit.app.get_app()

    def _timed_phase(self, phase: str, method):
        """``method`` with its wall time accumulated into ``self._phase_s[phase]``."""

        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                self._phase_s[phase] += time.perf_counter() - start

        return timed

    def _build(self, task: str, instruction_type: str) -> None:
        if self._env is not None:
            self._env.close()  # release the prior task's env before create_env opens a fresh USD stage
        if task not in self._registered:
            auto_register_droid_envs(task=[task])
            self._registered.add(task)
        env_name = get_envs(task=task)[0]
        self._env, self._env_cfg = create_env(
            env_name, device=args.device, num_envs=1, instruction_type=instruction_type
        )
        self._control_dt = self._env_cfg.sim.dt * self._env_cfg.decimation
        # The sim context's ``step`` (physics substeps) and ``render`` are the two native phases inside
        # ``env.step``; both are re-wrapped per build (a new env brings a fresh SimulationContext). The sums
        # feed the step response's ``timing`` and are zeroed at each ``step`` call, so reset-time rendering
        # never leaks into a step's decomposition.
        self._phase_s = {'physics': 0.0, 'render': 0.0}
        self._env.sim.step = self._timed_phase('physics', self._env.sim.step)
        self._env.sim.render = self._timed_phase('render', self._env.sim.render)
        # ``instruction`` is the resolved language goal (``create_env`` picks the variant); the rest is the
        # task identity the episode records.
        self._meta = {
            'task': self._env_cfg.instruction,
            'env_name': env_name,
            'task_name': self._env_cfg._task_name,
            'attributes': list(self._env_cfg._task_attributes),
        }
        robot = self._env.scene['robot']
        self._robot = robot
        self._joint_ids = [i for i, name in enumerate(robot.data.joint_names) if name.startswith('panda_joint')]
        self._body_idx = robot.data.body_names.index('base_link')  # the Robotiq gripper mount flange
        self._eef_frame_idx = self._env.scene['frames'].data.target_frame_names.index('eef_frame')
        self._eef_offset_rot = torch.tensor([EEF_OFFSET_ROT], dtype=torch.float32, device=self._env.device)
        # The controller parameters of RoboLab's AbsIK registration (``DroidIKActionCfg``).
        cfg = DifferentialIKControllerCfg(command_type='pose', use_relative_mode=False, ik_method='dls')
        self._ik = DifferentialIKController(cfg, num_envs=1, device=self._env.device)

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        key = (token['task'], token['instruction_type'])
        if key != self._key:
            self._build(*key)
            self._key = key
        else:
            self._env.reset_eval_state()  # unfreeze terminated envs so the next trial runs on the cached env
        # RoboLab's own episode runner resets twice — the first randomizes, the second settles the freshly
        # placed scene into the sensors (robolab/eval/episode.py).
        self._env.reset()
        obs, _extras = self._env.reset()
        state = token.get('state')
        if state is not None:
            # Exact-state replay (a demo's own scene). RoboLab records states env-origin relative
            # (``scene.get_state(is_relative=True)``), so restore them the same way. The recorded entries
            # overlay the scene's own post-reset state: ``reset_to`` wants every scene entity, and a recording
            # may predate entities that never move (e.g. the table joined the scene's rigid objects for
            # contact sensing after the shipped demos were recorded).
            merged = self._merged_state(self._env.scene.get_state(is_relative=True), state)
            obs, _extras = self._env.reset_to(merged, None, is_relative=True)
        # ``robot_meta`` carries the DROID rig's model (the launcher serialized it, since this server can't
        # build it); ``meta`` carries the task identity and the resolved instruction. The reset frame's subtask
        # progress is zeros: the recorder's infos refresh only after a step, so reading them here would replay
        # the prior trial's final values.
        return {
            'obs': self._observe(obs, np.zeros(4, dtype=np.float32)),
            'meta': self._meta,
            'robot_meta': self._robot_meta,
            'control_dt': self._control_dt,
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        start = time.perf_counter()
        self._phase_s['physics'] = self._phase_s['render'] = 0.0
        # Isaac pauses its timeline while assets stream in; stepping a paused sim stalls, so pump the kit
        # update loop until it plays again (robolab's episode loop does the same before every step).
        while not self._timeline.is_playing():
            self._kit_app.update()
        act = torch.zeros(1, 8, device=self._env.device)
        act[0, :7] = self._joint_targets(action['command'])
        # The binary gripper term closes above 0.5, so the wire grip ([0, 1], 1 = closed) feeds it as-is.
        act[0, 7] = float(action['grip'])
        obs, _reward, _term, _trunc, _info = self._env.step(act)
        # ``done``/``success`` key off RoboLab's frozen-env accounting, not the raw term/trunc flags: a
        # termination within the first two steps is a physics artifact its env resets in place and keeps
        # running (never frozen, no verdict), while a real success/time-out freezes the env and records one.
        done = self._env.all_terminated
        return {
            'obs': self._observe(obs, self._subtask_progress()),
            'done': done,
            'success': done and bool(self._env.get_env_results()[0]['success']),
            'control_dt': self._control_dt,
            # The server's own step decomposition: physics substeps, sensor/viewport rendering, and this
            # call's whole wall (observation materialisation included) — the client records it against its
            # socket-level step time.
            'timing': {
                'physics_s': self._phase_s['physics'],
                'render_s': self._phase_s['render'],
                'wall_s': time.perf_counter() - start,
            },
        }

    def _joint_targets(self, command: dict[str, Any]) -> torch.Tensor:
        """The wire command as the 7 absolute joint targets the jointpos substrate steps."""
        match command['type']:
            case 'joint_pos':  # pass through untouched — bit-identical to RoboLab's own leaderboard stack
                return torch.as_tensor(command['q'], dtype=torch.float32, device=self._env.device)
            case 'joint_vel':  # ``dq`` is a per-step joint delta (positronic applies ``JointDelta`` as q + dq)
                dq = torch.as_tensor(command['dq'], dtype=torch.float32, device=self._env.device)
                return self._measured_q() + dq
            case 'hold':  # re-command the measured joints; a zero action would drive the arm to the zero pose
                return self._measured_q()
            case 'cartesian':
                return self._solve_ik(*self._wire_pose(command['pose']))
            case 'cartesian_delta':
                cur_pos, cur_quat = self._eef_pose()
                delta_pos, delta_quat = self._wire_pose(command['delta'])
                # The world-frame compose of positronic's ``apply_cartesian_delta``: translation adds and
                # rotation left-multiplies onto the measured eef pose.
                return self._solve_ik(cur_pos + delta_pos, quat_mul(delta_quat, cur_quat))
            case other:
                raise ValueError(f'unknown command type {other!r}')

    def _solve_ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """The 7 joint targets reaching an eef-control-frame pose: one differential-IK linearization.

        Lifts IsaacLab's ``DifferentialInverseKinematicsAction`` (task_space_actions.py): the controller
        tracks base_link in the robot root frame, so the commanded eef_frame orientation is un-offset first
        (``q_base = q_eef ⊗ R_offset⁻¹``, the ``run_abs_ik_demo.py`` conversion; positions are shared — the
        offset translation is zero), then target and measured pose move into the root frame alongside the
        root-frame-rotated jacobian.
        """
        robot = self._robot
        target_quat_w = quat_mul(target_quat, quat_inv(self._eef_offset_rot))
        target_pos_w = target_pos + self._env.scene.env_origins
        root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
        target_pos_b, target_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, robot.data.body_pos_w[:, self._body_idx], robot.data.body_quat_w[:, self._body_idx]
        )
        # Fixed-base articulation: jacobian rows drop the root body (body row - 1), columns are the joint ids.
        jacobian = robot.root_physx_view.get_jacobians()[:, self._body_idx - 1, :, self._joint_ids]
        root_rot = matrix_from_quat(quat_inv(root_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot, jacobian[:, 3:, :])
        joint_pos = robot.data.joint_pos[:, self._joint_ids]
        self._ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1), ee_pos_b, ee_quat_b)
        return self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)[0]

    def _measured_q(self) -> torch.Tensor:
        return self._robot.data.joint_pos[0, self._joint_ids]

    def _eef_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The measured eef control-frame pose (env-local position, world-frame quat wxyz), batched (1, ...)."""
        frames = self._env.scene['frames'].data
        pos = frames.target_pos_w[:, self._eef_frame_idx] - self._env.scene.env_origins
        return pos, frames.target_quat_w[:, self._eef_frame_idx]

    def _wire_pose(self, vec: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """The adapter's flat ``[translation(3), rotation_matrix(9)]`` wire pose, batched (pos (1,3), quat (1,4))."""
        vec = torch.as_tensor(vec, dtype=torch.float32, device=self._env.device)
        return vec[:3].unsqueeze(0), quat_from_matrix(vec[3:].reshape(1, 3, 3))

    def _merged_state(self, base: Any, overlay: Any) -> Any:
        """The scene's live state tree with the recording's entries written over it, as device tensors."""
        if not isinstance(base, dict):
            return torch.as_tensor(overlay, dtype=base.dtype, device=self._env.device)
        return {
            key: self._merged_state(value, overlay[key]) if key in overlay else value for key, value in base.items()
        }

    def _subtask_progress(self) -> np.ndarray:
        """The env's live subtask progress as the wire ``[status, completed, total, score]`` vector."""
        subtask = np.zeros(4, dtype=np.float32)
        infos = get_all_env_subtask_infos(self._env)
        if infos is not None:
            info = infos[0]
            subtask[:] = (info['status'], info['completed'], info['total'], info['score'])
        return subtask

    def _observe(self, obs: dict[str, Any], subtask: np.ndarray) -> dict[str, Any]:
        # The eef pose is reported in the control frame Cartesian commands arrive in (``eef_frame``: env-local
        # position, world quat wxyz), so the observed and commanded pose share a frame. The proprio group has
        # no joint velocities, so they read straight from the articulation.
        proprio = obs['proprio_obs']
        images = obs['image_obs']
        return {
            'joint_pos': proprio['arm_joint_pos'][0].cpu().numpy(),
            'joint_vel': self._robot.data.joint_vel[0, self._joint_ids].cpu().numpy(),
            'eef_pos': proprio['eef_pos'][0].cpu().numpy(),
            'eef_quat': proprio['eef_quat'][0].cpu().numpy(),
            'grip': float(proprio['gripper_pos'][0, 0]),
            'over_shoulder_left_camera': images['over_shoulder_left_camera'][0].cpu().numpy(),
            'wrist_cam': images['wrist_cam'][0].cpu().numpy(),
            'subtask': subtask,
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def main() -> None:
    if args.port is None:
        parser.error('--port is required')
    EnvServer(RobolabEnv(), args.host, args.port).serve_forever()
    simulation_app.close()


if __name__ == '__main__':
    main()
