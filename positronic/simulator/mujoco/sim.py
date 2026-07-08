import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.ik import qpos_from_site_pose
from positronic.drivers.roboarm.models import bundled_panda_model
from positronic.simulator.mujoco.transforms import MujocoSceneTransform, load_spec, load_spec_from_file, np_seed

logger = logging.getLogger(__name__)


# mjSTATE_INTEGRATION is MuJoCo's complete integrable state (qpos, qvel, act, ctrl, warm-start, ...):
# the minimal subset that restores the sim and reproduces its forward trajectory exactly. The other
# specs are subsets of it, so recording them too only duplicates these values.
STATE_SPECS = [mj.mjtState.mjSTATE_INTEGRATION]


def save_state(model, data) -> dict[str, np.ndarray]:
    """
    Saves full state of the simulator.
    This state could be used to restore the exact state of the simulator.
    Returns:
        data: A dictionary containing the full state of the simulator.
    """
    state_data = {}

    for spec in STATE_SPECS:
        size = mj.mj_stateSize(model, spec)
        state_data[spec.name] = np.empty(size, np.float64)
        mj.mj_getState(model, data, state_data[spec.name], spec)

    return state_data


class MujocoFrankaState(State, pimm.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(7 + 7 + 7 + 1,), dtype=np.float32)
        self.array.fill(0.0)
        self.array[14 + 7] = RobotStatus.AVAILABLE.value

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        # Return a copy so downstream consumers don't hold a view into the shared state buffer
        return self.array[:7].copy()

    @property
    def dq(self) -> np.ndarray:
        # Return a copy for the same reason as q
        return self.array[7:14].copy()

    @property
    def ee_pose(self) -> geom.Transform3D:
        return geom.Transform3D(self.array[14 : 14 + 3], geom.Rotation.from_quat(self.array[14 + 3 : 14 + 7]))

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[14 + 7]))

    def set_error(self):
        self.array[14 + 7] = RobotStatus.ERROR.value

    def clear_error(self):
        self.array[14 + 7] = RobotStatus.AVAILABLE.value

    def encode(self, q, dq, ee_pose):
        self.array[:7] = q
        self.array[7:14] = dq
        self.array[14 : 14 + 3] = ee_pose.translation
        self.array[14 + 3 : 14 + 7] = ee_pose.rotation.as_quat
        self.array[14 + 7] = self.status.value


class _Cadence:
    """Per-stream emission gate: ``fps=None`` fires on every physics tick."""

    def __init__(self, fps: float | None):
        self._period = None if fps is None else 1.0 / fps
        self._next_due = 0.0

    def __call__(self, now: float) -> bool:
        if self._period is None:
            return True
        if now < self._next_due:
            return False
        self._next_due = now + self._period
        return True


class MujocoSim(pimm.ControlSystem):
    """The MuJoCo embodiment in one control system: scene, Franka arm, gripper, and cameras.

    ``reset`` rebuilds the scene and flags frame-0 publication; the run loop publishes that post-reset
    scene on its next turn — in sequence, before any step — so the first inference reads it and the
    recorder logs it. Every other turn applies the due command waypoints, steps once, and emits the due
    streams (post-step, Gym-style). The sim sleeps one control period each turn, so it is the eval's sole
    time-master. Each stream has an independent rate (``*_fps``, ``None`` = every physics tick).
    """

    def __init__(
        self,
        mujoco_model_path: str,
        loaders: Sequence[MujocoSceneTransform] = (),
        *,
        suffix: str = '_ph',
        gripper_actuator: str = 'actuator8_ph',
        gripper_joint: str = 'finger_joint1_ph',
        camera_resolution: tuple[int, int] = (320, 240),
        camera_fps: float | None = 30,
        state_fps: float | None = None,
        grip_fps: float | None = None,
        sim_state_fps: float | None = None,
    ):
        self.mujoco_model_path = mujoco_model_path
        self.loaders = loaders
        self.warmup_steps = 1000
        self.fps_counter = pimm.utils.RateCounter('MujocoSim')

        self._ee_name = f'end_effector{suffix}'
        self._joint_names = [f'joint{i}{suffix}' for i in range(1, 8)]
        self._actuator_names = [f'actuator{i}{suffix}' for i in range(1, 8)]
        self._gripper_actuator = gripper_actuator
        self._gripper_joint = gripper_joint
        self._camera_resolution = camera_resolution
        self._camera_fps = camera_fps
        self._state_fps = state_fps
        self._grip_fps = grip_fps
        self._sim_state_fps = sim_state_fps
        self._renderer: mj.Renderer | None = None
        self._ik_data: mj.MjData | None = None

        self._load_scene()
        self._home()
        self._error = False
        self._adapters: dict[str, pimm.shared_memory.NumpySMAdapter] | None = None
        self._arm_player = roboarm_command.TrajectoryPlayer(reduce=roboarm_command.reduce)
        self._grip_player = roboarm_command.TrajectoryPlayer()
        self._last_grip = 0.0
        # Set by ``reset``; the run loop publishes frame-0 (instead of stepping) on its next turn and clears it.
        self._reset_pending = False

        self.commands: pimm.SignalReceiver[roboarm_command.CommandType] = pimm.ControlSystemReceiver(self, default=None)
        self.state: pimm.SignalEmitter[MujocoFrankaState] = pimm.ControlSystemEmitter(self)
        self.robot_meta = pimm.ControlSystemEmitter(self)
        self.target_grip: pimm.SignalReceiver[float] = pimm.ControlSystemReceiver(self, default=0.0)
        self.grip: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self.cameras: pimm.EmitterDict = pimm.EmitterDict(self)
        # Privileged ground truth: the full ``save_state`` dict, spec keys prefixed with '.' so the
        # writer expands them into ``<signal>.<spec>`` signals. Scoring is computed downstream, not
        # live: it rebuilds the episode's model from the ``scene_xml`` in its static meta and
        # replays these states through it (``mj_setState`` + ``mj_forward``).
        self.sim_state: pimm.SignalEmitter[dict[str, np.ndarray]] = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self._emit_robot_meta()
        state_due = _Cadence(self._state_fps)
        grip_due = _Cadence(self._grip_fps)
        sim_state_due = _Cadence(self._sim_state_fps)
        cameras_due = _Cadence(self._camera_fps)

        while not should_stop.value:
            yield pimm.Sleep(self.model.opt.timestep)
            if self._reset_pending:
                # The reset is this turn's step: publish the prepared scene as frame-0 (no ``mj_step``), so
                # the recorder samples it before any step advances the sim. ``reset`` already loaded the scene.
                self._reset_pending = False
                self._emit_robot_meta()
                self._publish_frame()
                continue
            now = clock.now()
            cmd_msg = self.commands.read()
            if cmd_msg.updated:
                self._arm_player.set(cmd_msg.data)
            cmd = self._arm_player.advance(clock.now_ns())
            if cmd is not None:
                self._apply_command(cmd)
            grip_msg = self.target_grip.read()
            if grip_msg.updated:
                self._grip_player.set(grip_msg.data)
            grip = self._grip_player.advance(clock.now_ns())
            if grip is not None:
                self._last_grip = grip
            self._apply_grip(self._last_grip)

            self.step()
            self.fps_counter.tick()
            if state_due(now):
                self._emit_state()
            if grip_due(now):
                self._emit_grip()
            if sim_state_due(now):
                self._emit_sim_state()
            if cameras_due(now):
                self._emit_cameras()

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def reset(self, seed: int | None = None):
        """Re-randomize the scene from ``seed`` and arm frame-0 publication for the next turn.

        The model and data are rebuilt wholesale, so model-level loader effects (fixed-body poses,
        colors, cameras) re-randomize too; the renderer and IK physics rebind lazily. The run loop
        publishes the prepared scene as frame-0 on its next turn — in sequence, before any step — so the
        first inference reads the reset state and the recorder logs it. Stale commands queued while idle
        (e.g. the inter-episode home) are dropped and the run-loop's trajectory players and held grip are
        cleared, so the first step neither applies a queued command nor replays the previous episode's
        trajectory on the freshly reset scene.
        """
        self._load_scene(seed)
        self._home()
        self._error = False
        self.commands.read()
        self.target_grip.read()
        self._arm_player.set([])
        self._grip_player.set([])
        self._last_grip = 0.0
        self._reset_pending = True

    def _load_scene(self, seed: int | None = None):
        """Apply the loaders to the model file and bind the result; ``scene_xml`` captures the draw."""
        with np_seed(seed):
            spec, self.metadata = load_spec_from_file(self.mujoco_model_path, self.loaders)
        self.model = spec.compile()
        self.scene_xml = spec.to_xml()
        self._bind_model()

    def _emit_robot_meta(self):
        # Emit the full robot model (URDF + meshes + frames + gripper) at record time, like franka.py,
        # plus the per-episode scene_xml that restores the MuJoCo scene.
        self.robot_meta.emit({**bundled_panda_model(), 'scene_xml': self.scene_xml})

    def _publish_frame(self):
        """Emit every observation stream once for the current scene — the post-reset frame-0."""
        self._emit_state()
        self._emit_grip()
        self._emit_sim_state()
        self._emit_cameras()

    def _emit_state(self):
        state = MujocoFrankaState()
        state.encode(self._q, self._dq, self._ee_pose)
        if self._error:
            state.set_error()
        self.state.emit(state)

    def _emit_grip(self):
        self.grip.emit(self._current_grip())

    def _emit_sim_state(self):
        if self.sim_state.num_bound:
            self.sim_state.emit({f'.{name}': arr for name, arr in self.save_state().items() if arr.size})

    def _emit_cameras(self):
        if self._adapters is None:
            self._adapters = self._camera_adapters()
        if self._adapters:
            self._render(self._adapters)

    def _bind_model(self):
        """Derive everything that hangs off ``self.model``; runs at construction and on every rebuild."""
        self.data = mj.MjData(self.model)
        self.initial_ctrl = [float(x) for x in self.metadata.get('initial_ctrl').split(',')]
        self._joint_qpos_ids = [self.model.joint(name).qposadr.item() for name in self._joint_names]
        min_grip, max_grip = self.model.actuator(self._gripper_actuator).ctrlrange
        self._grip_range = (float(min_grip), float(max_grip))
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._ik_data = None

    def _home(self):
        """Drive the actuators to their default controls (``initial_ctrl``) and step through the settling
        transient — the arm's default pose, whether after a scene build or on a ``Reset`` command."""
        self.data.ctrl = self.initial_ctrl
        mj.mj_step(self.model, self.data, self.warmup_steps)

    def load_state(self, state: dict, reset_time: bool = True):
        """Restore a recorded scene: the model from ``scene_xml`` (when present), then the MuJoCo state.

        Episodes recorded before scene capture carry no ``scene_xml``; their state restores onto
        this sim's own model draw, so only ``qpos``-borne (freejointed) randomization replays
        faithfully for them.
        """
        if 'scene_xml' in state:
            self.scene_xml = state['scene_xml']
            scene, self.metadata = load_spec(self.scene_xml, Path(self.mujoco_model_path).parent)
            self.model = scene.compile()
            self._bind_model()
        mj.mj_resetData(self.model, self.data)
        for spec in STATE_SPECS:
            mj.mj_setState(self.model, self.data, np.array(state[spec.name]), spec)

        if reset_time:
            self.data.time = 0

    def save_state(self) -> dict[str, np.ndarray]:
        """
        Saves full state of the simulator.

        This state could be used to restore the exact state of the simulator.

        Returns:
            data: A dictionary containing the full state of the simulator.
        """
        return save_state(self.model, self.data)

    def step(self, duration: float | None = None) -> None:
        duration = duration or self.model.opt.timestep
        target_time = self.data.time + duration
        while self.data.time < target_time:
            mj.mj_step(self.model, self.data)

    def _apply_command(self, cmd):
        match cmd:
            case roboarm_command.CartesianPosition(pose=pose):
                q = self._recalculate_ik(pose)
                if q is not None:
                    self._set_actuator_values(q)
                else:
                    logger.warning(f'IK failed for ee_pose: {pose}')
                    self._error = True
            case roboarm_command.CartesianDelta(delta=delta):
                target = roboarm_command.apply_cartesian_delta(self._ee_pose, delta)
                q = self._recalculate_ik(target)
                if q is not None:
                    self._set_actuator_values(q)
                else:
                    logger.warning(f'IK failed for delta target: {target}')
                    self._error = True
            case roboarm_command.JointPosition(positions=positions):
                self._set_actuator_values(positions)
            case roboarm_command.JointDelta(velocities=delta):
                self._set_actuator_values(self._q + delta)
            case roboarm_command.Reset():
                self._home()  # the Reset command homes the arm; re-randomizing the scene is ``reset``'s job
                self._error = False
            case roboarm_command.Recover():
                self._error = False
            case _:
                raise ValueError(f'Unknown command type: {type(cmd)}')

    def _recalculate_ik(self, target: geom.Transform3D) -> np.ndarray | None:
        if self._ik_data is None:
            self._ik_data = mj.MjData(self.model)
            self._ik_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self._ee_name)
            self._ik_dof_ids = np.array([self.model.joint(n).dofadr.item() for n in self._joint_names])
        # Search on a scratch copy warm-started from the robot's current pose, leaving the live sim intact.
        self._ik_data.qpos[:] = self.data.qpos
        qpos, _, success = qpos_from_site_pose(
            self.model,
            self._ik_data,
            self._ik_site_id,
            self._ik_dof_ids,
            target.translation,
            target.rotation.as_quat,
            rot_weight=0.5,
        )
        return qpos[self._joint_qpos_ids] if success else None

    def _set_actuator_values(self, values: np.ndarray):
        for name, value in zip(self._actuator_names, values, strict=True):
            self.data.actuator(name).ctrl = value

    def _apply_grip(self, target: float):
        """Convert [0, 1] target grip (0 = open, 1 = closed) to the actuator control range."""
        min_grip, max_grip = self._grip_range
        self.data.actuator(self._gripper_actuator).ctrl = max_grip - target * (max_grip - min_grip)

    def _current_grip(self) -> float:
        """Convert the current grip joint position to [0, 1] (0 = open, 1 = closed)."""
        min_grip, max_grip = self._grip_range
        return 1.0 - (self.data.joint(self._gripper_joint).qpos.item() - min_grip) / (max_grip - min_grip)

    @property
    def _q(self) -> np.ndarray:
        return np.array([self.data.qpos[i] for i in self._joint_qpos_ids])

    @property
    def _dq(self) -> np.ndarray:
        return np.array([self.data.qvel[i] for i in self._joint_qpos_ids])

    @property
    def _ee_pose(self) -> geom.Transform3D:
        site = self.data.site(self._ee_name)
        quat = np.empty(4)
        mj.mju_mat2Quat(quat, site.xmat.copy())
        return geom.Transform3D(translation=site.xpos.copy(), rotation=geom.Rotation.from_quat(quat))

    def _camera_adapters(self) -> dict[str, pimm.shared_memory.NumpySMAdapter]:
        existing = {self.model.camera(i).name for i in range(self.model.ncam)}
        width, height = self._camera_resolution
        adapters = {}
        for name in self.cameras.keys():
            if name not in existing:
                raise RuntimeError(
                    f"Camera '{name}' is bound but does not exist in the mujoco model. Available cameras: {existing}"
                )
            adapters[name] = pimm.shared_memory.NumpySMAdapter(shape=(height, width, 3), dtype=np.uint8)
        return adapters

    def _render(self, adapters: dict[str, pimm.shared_memory.NumpySMAdapter]):
        if self._renderer is None:
            width, height = self._camera_resolution
            self._renderer = mj.Renderer(self.model, height=height, width=width)
        for name, adapter in adapters.items():
            self._renderer.update_scene(self.data, camera=name)
            self._renderer.render(out=adapter.array)
            self.cameras[name].emit(adapter)
