# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "msgpack",
#     "websockets",
#     "robosuite==1.4.1",
#     "mujoco==3.2.3",  # pin to openpi's LIBERO eval engine version, so our physics matches the policy's training/eval
#     "numpy<2",
#     # LIBERO is not listed here: it declares ``install_requires=[]`` and ships ``libero`` as a PEP 420 namespace
#     # package, so no wheel carries it — the launcher puts a source checkout on ``PYTHONPATH``. These are the
#     # packages LIBERO's benchmark/env path imports but never declares.
#     "torch<2.6",  # torch 2.6 flipped ``torch.load`` to weights_only=True; LIBERO's init-states are plain pickles
#     "bddl==1.0.1",
#     "gym==0.25.2",
#     "hydra-core",
#     "easydict",
#     "einops",
#     "cloudpickle",
#     "future",
#     "matplotlib",
#     "pyyaml",
# ]
# ///
"""LIBERO behind the env-server protocol: a standalone server for LIBERO tasks in its own interpreter.

positronic is pinned ``>=3.11`` and LIBERO needs ``<=3.10``, so this never shares positronic's venv:
``uv run env.py`` reads the inline metadata above to build a 3.10 environment with LIBERO, and positronic
launches it as a subprocess that ``EnvConnection`` connects to over the socket. It imports only ``libero`` +
robosuite + the positronic-free ``server``/``protocol`` modules (placed on ``PYTHONPATH`` by the launcher),
never ``positronic`` itself.

The client-side ``LiberoAdapter`` forwards the held command (an absolute Cartesian pose, joint positions, or
joint velocities); this server maps it into the active controller's action and speaks only LIBERO's raw gym
payloads. The control mode in the reset token selects the controller LIBERO uses — ``OSC_POSE`` for
Cartesian, ``JOINT_POSITION``/``JOINT_VELOCITY`` for joints — and any command is bridged into it: a Cartesian
goal is solved to joints with damped-least-squares IK and a joint goal is turned into an OSC pose delta with
forward kinematics, both on the MuJoCo site Jacobian OSC itself computes. The reset token carries the task spec
(suite, task_id, camera resolution, control mode) plus the per-trial seed; the env builds that task on the first
reset and caches it, rebuilding only when the spec changes. The seed selects a saved init-state and re-randomizes
object positions; a ``None`` seed draws one at random.
"""

import argparse
import builtins
import pathlib
import random
from typing import Any

import mujoco
import numpy as np
from robosuite.utils.transform_utils import get_pose_error, make_pose, mat2quat, quat2axisangle
from server import EnvProtocol, EnvServer

# LIBERO's package __init__ prompts on stdin for a dataset path when ``~/.libero/config.yaml`` is absent, which
# would hang this unattended server on a clean machine. Answer its first-run prompt with the default ('n') so it
# writes the config from LIBERO's own package-relative paths, non-interactively.
builtins.input = lambda *_args, **_kwargs: 'n'

from libero.libero import benchmark, get_libero_path  # noqa: E402 -- must follow the stdin shim above
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

_IK_ITERS = 100
_IK_DAMPING = 0.05
_IK_TOL = 1e-4


def _unpack_pose(vec: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split the adapter's flat ``[translation(3), rotation_matrix(9)]`` wire pose into ``(pos, 3x3 rot)``."""
    vec = np.asarray(vec)
    return vec[:3], vec[3:].reshape(3, 3)


class LiberoEnv(EnvProtocol):
    """A LIBERO task behind the gym-style ``reset``/``step``/``close`` the env server serves.

    Built from the reset token's task spec (suite, task_id, resolution, control mode) and cached; ``reset``
    rebuilds when the spec changes, then re-seeds and selects a saved init-state (drawing the seed when the token
    carries ``None``), or restores an exact recorded full state when the token carries one (demo replay).
    ``step`` maps the forwarded command into the active controller's action (FK/IK bridging pose<->joint) and
    returns LIBERO's raw obs plus the full physics state — the privileged ground truth, so success is
    recomputable downstream — and ``done``.
    """

    def __init__(self):
        # The task spec (suite, task_id, resolution, control mode) the current env was built for; ``reset``
        # rebuilds when the token's spec differs, so one server can serve any task without a restart.
        self._key = None
        self._control_mode = None
        self._env = None
        self._init_states = None
        self._meta = None
        self._control_dt = None
        # Cached at build: model-structure indices, invariant across resets of the same task XML.
        self._qpos_idx = None
        self._qvel_idx = None
        self._eef_site_id = None
        self._jnt_low = None
        self._jnt_high = None
        self._grip_open_aperture = None

    # LIBERO runs with ``hard_reset=True``, so each reset destroys and recreates the sim, model, and controller
    # objects; these read the live ones from the env rather than caching a copy that would go stale after a reset.
    @property
    def _sim(self):
        return self._env.env.sim

    @property
    def _model(self):
        return self._env.env.sim.model._model

    @property
    def _controller(self):
        return self._env.env.robots[0].controller

    def _build(self, suite_name: str, task_id: int, camera_resolution: int, control_mode: str) -> None:
        if self._env is not None:
            self._env.close()  # release the prior task's sim/renderer before building a different one
        self._control_mode = control_mode
        task_suite = benchmark.get_benchmark_dict()[suite_name]()
        task = task_suite.get_task(task_id)
        self._init_states = task_suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        # OSC_POSE is the controller LIBERO configures for its own policies (its ``env_wrapper`` hands the
        # controller *name* to ``load_controller_config``); operational-space control maps a per-step
        # end-effector pose delta to joint motion — LIBERO ships no IK of its own. The joint modes select
        # robosuite's joint controllers. Pass the name, not a config dict, so the wrapper loads it as LIBERO does.
        controller_names = {'ee': 'OSC_POSE', 'joint': 'JOINT_POSITION', 'joint_delta': 'JOINT_VELOCITY'}
        self._env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            controller=controller_names[control_mode],
            camera_heights=camera_resolution,
            camera_widths=camera_resolution,
        )
        self._meta = {'suite': suite_name, 'task_id': task_id, 'task': task.language}
        self._control_dt = 1.0 / self._env.env.control_freq
        robot = self._env.env.robots[0]
        self._qpos_idx = np.asarray(robot._ref_joint_pos_indexes)
        self._qvel_idx = np.asarray(robot._ref_joint_vel_indexes)
        self._eef_site_id = self._sim.model.site_name2id(robot.controller.eef_name)
        self._jnt_low, self._jnt_high = self._sim.model.jnt_range[robot._ref_joint_indexes].T
        # The Panda's two finger joints open in opposite directions from a shared closed pose at 0, so the
        # aperture between them is their full travel when open; normalize ``grip`` against it (see ``_observe``).
        gripper_jids = [self._sim.model.joint_name2id(j) for j in robot.gripper.joints]
        self._grip_open_aperture = float(np.sum(np.diff(self._sim.model.jnt_range[gripper_jids], axis=1)))

    def reset(self, token: dict[str, Any]) -> dict[str, Any]:
        key = (token['suite'], token['task_id'], token['camera_resolution'], token['control_mode'])
        if key != self._key:
            self._build(*key)
            self._key = key
        state = token.get('state')
        if state is not None:
            # Exact-state replay: restore a recorded full physics state (a demo's own scene). The seed/init-state
            # selection is moot — ``set_init_state`` overwrites the whole state.
            self._env.reset()
            raw = self._env.set_init_state(np.asarray(state))
        else:
            # An absent seed draws a fresh scene, matching ``MujocoSim.reset(seed=None)``.
            seed = int(token['seed']) if token['seed'] is not None else random.randrange(2**31)
            self._env.seed(seed)
            self._env.reset()
            raw = self._env.set_init_state(self._init_states[seed % len(self._init_states)])
            # ``set_init_state`` only places objects kinematically (``sim.forward``, no dynamics), so they start
            # mid-fall. Step a held arm with the gripper open so they settle before the first observation, the
            # zero-pose-delta/open-gripper action openpi waits out as ``LIBERO_DUMMY_ACTION``.
            for _ in range(token.get('settle_steps', 10)):
                raw, _reward, _done, _info = self._env.step(np.zeros(len(self._controller.input_max)).tolist() + [-1.0])
        # ``robot_meta`` is empty: the 3.10 server can't import positronic to emit the Panda model, so the eval
        # supplies it via ``static_meta`` (``bundled_panda_model``). ``meta`` carries the scene/task identity.
        return {'obs': self._observe(raw), 'meta': self._meta, 'robot_meta': {}, 'control_dt': self._control_dt}

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        arm = self._arm_action(action['command'])
        # positronic grip in [0, 1] maps to robosuite's [-1, 1]; robosuite's PandaGripper opens at -1 and closes
        # at +1, so grip=1 closes.
        grip = action['grip'] * 2.0 - 1.0
        # LIBERO's ``BDDLBaseDomain.step`` overrides robosuite's horizon-based ``done`` with ``_check_success()``,
        # so ``done`` is the task-success flag (the adapter's ``eval.success``), not a step-limit timeout.
        raw, _reward, done, _info = self._env.step(np.concatenate([arm, [grip]]).tolist())
        return {'obs': self._observe(raw), 'done': bool(done), 'control_dt': self._control_dt}

    def _arm_action(self, command: dict[str, Any]) -> np.ndarray:
        # All-to-all: each command becomes the physical pre-scale quantity the active controller's set_goal adds to
        # the current setpoint, then ``_normalize`` inverts the controller's scaling. The pose<->joint cells use
        # FK/IK on the site Jacobian. ``dq`` is a per-step joint delta (positronic applies ``JointDelta`` as
        # ``q + dq``, never as a rate), so it bridges as a delta everywhere except the JOINT_VELOCITY controller,
        # which wants rad/s — there it is divided by the control period.
        match (self._control_mode, command['type']):
            case ('ee', 'cartesian'):  # OSC_POSE: world-frame pose error
                physical = self._pose_error(*_unpack_pose(command['pose']))
            case ('ee', 'cartesian_delta'):  # OSC_POSE: the world-frame delta composed onto the live eef pose
                physical = self._pose_error(*self._delta_target(command))
            case ('ee', 'joint_pos'):
                physical = self._pose_error(*self._fk(command['q']))
            case ('ee', 'joint_vel'):
                physical = self._pose_error(*self._fk(self._cur_q() + command['dq']))
            case ('ee', 'hold'):
                physical = np.zeros(6)
            case ('joint', 'joint_pos'):  # JOINT_POSITION: joint delta from current
                physical = command['q'] - self._cur_q()
            case ('joint', 'joint_vel'):
                physical = command['dq']
            case ('joint', 'cartesian'):
                physical = self._ik(*_unpack_pose(command['pose'])) - self._cur_q()
            case ('joint', 'cartesian_delta'):
                physical = self._ik(*self._delta_target(command)) - self._cur_q()
            case ('joint', 'hold'):
                physical = np.zeros(len(self._qpos_idx))
            case ('joint_delta', 'joint_vel'):  # JOINT_VELOCITY: the per-step delta as a rate over the control period
                physical = command['dq'] / self._control_dt
            case ('joint_delta', 'joint_pos'):
                physical = (command['q'] - self._cur_q()) / self._control_dt
            case ('joint_delta', 'cartesian'):
                physical = (self._ik(*_unpack_pose(command['pose'])) - self._cur_q()) / self._control_dt
            case ('joint_delta', 'cartesian_delta'):
                physical = (self._ik(*self._delta_target(command)) - self._cur_q()) / self._control_dt
            case ('joint_delta', 'hold'):
                physical = np.zeros(len(self._qpos_idx))
            case (mode, ctype):
                raise ValueError(f'control mode {mode!r} cannot map command {ctype!r}')
        return self._normalize(physical)

    def _normalize(self, physical: np.ndarray) -> np.ndarray:
        # The exact inverse of robosuite ``Controller.scale_action``: every controller — OSC_POSE, JOINT_POSITION,
        # JOINT_VELOCITY — maps a normalized [input_min, input_max] action to its output range, so each cell builds
        # the physical pre-scale quantity and this inverts the scaling. A target beyond one step's output range
        # saturates at the clip, exactly as for LIBERO's own policies.
        c = self._controller
        scale = np.abs(c.output_max - c.output_min) / np.abs(c.input_max - c.input_min)
        out_mid = (c.output_max + c.output_min) / 2
        in_mid = (c.input_max + c.input_min) / 2
        return np.clip((physical - out_mid) / scale + in_mid, c.input_min, c.input_max)

    def _pose_error(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        # The physical pre-scale OSC delta the controller's set_goal expects: world-frame translation plus the
        # world-frame axis-angle rotation error (``goal_pos = ee_pos + Δpos``; ``goal_ori = R(Δrot) @ ee_ori``).
        cur_pos, cur_rot = self._cur_pose()
        return np.concatenate([target_pos - cur_pos, quat2axisangle(mat2quat(target_rot @ cur_rot.T))])

    def _delta_target(self, command: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        # The absolute pose a world-frame ``cartesian_delta`` targets: translation adds and rotation left-multiplies
        # onto the live eef pose (``goal_pos = ee_pos + Δpos``; ``goal_ori = R(Δrot) @ ee_ori``) — the same compose
        # each control mode then bridges, as an OSC pose error or via IK to joints.
        cur_pos, cur_rot = self._cur_pose()
        delta_pos, delta_rot = _unpack_pose(command['delta'])
        return cur_pos + delta_pos, delta_rot @ cur_rot

    def _cur_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(self._sim.data.site_xpos[self._eef_site_id])
        rot = np.array(self._sim.data.site_xmat[self._eef_site_id]).reshape(3, 3)
        return pos, rot

    def _cur_q(self) -> np.ndarray:
        return np.array(self._sim.data.qpos[self._qpos_idx])

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Forward kinematics on a scratch ``MjData`` seeded from the live scene (objects intact), so the live sim
        # is never perturbed: set the arm joints, propagate, read the eef site.
        data = mujoco.MjData(self._model)
        data.qpos[:] = self._sim.data.qpos
        data.qpos[self._qpos_idx] = q
        mujoco.mj_forward(self._model, data)
        pos = np.array(data.site_xpos[self._eef_site_id])
        rot = np.array(data.site_xmat[self._eef_site_id]).reshape(3, 3)
        return pos, rot

    def _ik(self, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        # Damped-least-squares differential IK on the same MuJoCo site Jacobian OSC uses (no pybullet), iterated
        # on a scratch ``MjData`` from the current joints. ``get_pose_error`` returns the world-frame error that
        # matches the world-frame site Jacobian; ``validate.py`` checks the ``_fk(_ik(pose))`` round-trip.
        data = mujoco.MjData(self._model)
        data.qpos[:] = self._sim.data.qpos
        target_pose = make_pose(target_pos, target_rot)
        q = self._cur_q()
        for _ in range(_IK_ITERS):
            data.qpos[self._qpos_idx] = q
            mujoco.mj_forward(self._model, data)
            cur_pose = make_pose(
                np.array(data.site_xpos[self._eef_site_id]), np.array(data.site_xmat[self._eef_site_id]).reshape(3, 3)
            )
            err = get_pose_error(target_pose, cur_pose)
            if np.linalg.norm(err) < _IK_TOL:
                break
            jacp = np.zeros((3, self._model.nv))
            jacr = np.zeros((3, self._model.nv))
            mujoco.mj_jacSite(self._model, data, jacp, jacr, self._eef_site_id)
            jac = np.vstack([jacp, jacr])[:, self._qvel_idx]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + _IK_DAMPING**2 * np.eye(6), err)
            q = np.clip(q + dq, self._jnt_low, self._jnt_high)
        return q

    def _observe(self, raw: dict[str, Any]) -> dict[str, Any]:
        # Report the eef pose in the grip-site frame OSC controls — the frame ``_arm_action`` recovers
        # commands against — so the observed pose and the commanded pose share a frame. robosuite's raw
        # ``robot0_eef_quat`` is the hand *body* orientation, a fixed offset from the grip site, which would
        # desync observation and command and break absolute-pose control.
        eef_pos, eef_rot = self._cur_pose()
        # Closure in [0, 1] (0 open, 1 closed) — the same convention the grip *command* uses (``grip=1`` closes)
        # and the native sim reports. Summing the two finger qpos cancels (they mirror); the aperture between
        # them is the openness signal.
        fingers = np.asarray(raw['robot0_gripper_qpos'])
        grip = 1.0 - abs(fingers[0] - fingers[1]) / self._grip_open_aperture
        return {
            'agentview_image': raw['agentview_image'],
            'eye_in_hand_image': raw['robot0_eye_in_hand_image'],
            'joint_pos': raw['robot0_joint_pos'],
            'joint_vel': raw['robot0_joint_vel'],
            'eef_pos': eef_pos,
            'eef_quat': mat2quat(eef_rot),
            'grip': float(np.clip(grip, 0.0, 1.0)),
            'sim_state': self._env.env.sim.get_state().flatten(),
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def main() -> None:
    parser = argparse.ArgumentParser(description='Serve LIBERO over the env-server protocol.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    EnvServer(LiberoEnv(), args.host, args.port).serve_forever()


if __name__ == '__main__':
    main()
