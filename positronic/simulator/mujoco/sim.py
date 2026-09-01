import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np

import pimm
from positronic import geom, telemetry, telemetry_keys
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm.ik import qpos_from_site_pose
from positronic.drivers.roboarm.models import bundled_panda_model
from positronic.drivers.utils import Moves, MoveStatus
from positronic.eval import EVAL_SEED
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

    def set_status(self, status: RobotStatus):
        self.array[14 + 7] = status.value

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

    ``reset`` rebuilds the scene and publishes it, and ``sync_move`` puts the arm where the trial starts
    it. Every other turn applies whatever command has just arrived, steps once, and emits the due streams
    (post-step, Gym-style). The sim sleeps one control period each turn, so it is the eval's sole
    time-master. Each stream has an independent rate (``*_fps``, ``None`` = every physics tick).
    """

    _MOVE_TOL = 0.05  # radians; the position actuators hold the arm a few hundredths short of their ctrl
    _MOVE_TIMEOUT_S = 5.0  # sim seconds a move gets before it gives up on the arm reaching its target

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
        self.mujoco_model_path = str(Path(mujoco_model_path).expanduser())
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
        self._apply_initial_ctrl()
        self._error = False
        self._adapters: dict[str, pimm.shared_memory.NumpySMAdapter] | None = None
        self._last_grip = 0.0

        self.commands = pimm.ControlSystemReceiver[roboarm_command.CommandType](self)
        self.sync_move = pimm.calls.ControlSystemHandler[roboarm_command.CommandType, None](self)
        self.env_reset = pimm.calls.ControlSystemHandler[Any, None](self)
        self.state = pimm.ControlSystemEmitter[MujocoFrankaState](self)
        self.robot_meta = pimm.ControlSystemEmitter(self)
        self.target_grip = pimm.ControlSystemReceiver[float](self)
        self.grip = pimm.ControlSystemEmitter[float](self)
        self.cameras: pimm.EmitterDict = pimm.EmitterDict(self)
        # Privileged ground truth: the full ``save_state`` dict, spec keys prefixed with '.' so the
        # writer expands them into ``<signal>.<spec>`` signals. Scoring is computed downstream, not
        # live: it rebuilds the episode's model from the ``scene_xml`` in its static meta and
        # replays these states through it (``mj_setState`` + ``mj_forward``).
        self.sim_state = pimm.ControlSystemEmitter[dict[str, np.ndarray]](self)

        self._moves = Moves[roboarm_command.CommandType](self.sync_move, self.commands)
        # The state has a second reason to go out — a move that ended — so it is not one of ``_streams``.
        self._state_due = _Cadence(self._state_fps)
        self._streams = [
            (_Cadence(self._grip_fps), self._emit_grip),
            (_Cadence(self._sim_state_fps), self._emit_sim_state),
            (_Cadence(self._camera_fps), self._emit_cameras),
        ]

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        self._emit_robot_meta()
        with self._moves:
            while not should_stop.value:
                yield pimm.Sleep(self.model.opt.timestep)
                now = clock.now()
                # The scene is drawn before the arm is asked for anything, so a move's target is computed
                # against the model the redraw just built.
                redraw = next(self.env_reset.incoming(), None)
                if redraw is not None:
                    with pimm.calls.raise_to(redraw):
                        self.reset(dict(redraw.request or {}).get(EVAL_SEED))
                        redraw.set_result(None)

                command = self._moves.next_request()
                if isinstance(command, pimm.calls.Call):
                    self._accept_move(command, now)
                elif self._error:
                    self._error = False
                elif command is not None:
                    self._apply_command(command)
                if redraw is None:  # a redraw is the turn's whole work
                    self._step_and_emit(now)

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _accept_move(self, call: pimm.calls.Call[roboarm_command.CommandType, None], now: float) -> None:
        """Aim the actuators at what ``call`` asks for; the arm is that move's until it reads back there."""
        with pimm.calls.raise_to(call):
            target = self._to_joints(call.request)
            self._set_actuator_values(target)
            self._moves.accept(call, target, self._MOVE_TOL, now, self._MOVE_TIMEOUT_S)
            # Already there: answered on the spot, after the state that says so and without a step.
            if self._settle_move(now) is MoveStatus.ARRIVED:
                self._emit_state()
                self._moves.answer()

    def _settle_move(self, now: float) -> MoveStatus:
        """Where the move in flight stands, once the arm it leaves behind is accounted for."""
        status = self._moves.settle(self._q, now)
        if status is not MoveStatus.MOVING:
            self._error = self._moves.errored  # a move that lands clears what one that gave up left
            if self._error:
                # The actuators still drive at the target the move never reached; left there, the arm
                # would resume it the moment whatever held it back goes away.
                self._set_actuator_values(self._q)
        return status

    def _step_and_emit(self, now: float) -> None:
        if (grip := pimm.value_updated(self.target_grip)) is not None:
            self._last_grip = grip
        self._apply_grip(self._last_grip)

        # An env step is the sim advance plus the observations it produces, rendering included
        # (``_emit_cameras``): on an image-heavy scene the rendering is most of the step, and outside
        # this span the wall split reads it as overhead.
        with telemetry.span(telemetry_keys.SPAN_ENV_STEP):
            self.step()
            self.fps_counter.tick()
            ended = self._moves.active and self._settle_move(now) is not MoveStatus.MOVING
            # A move that ended says where the arm got to, whatever rate the state stream runs at
            if self._state_due(now) or ended:
                self._emit_state()
            for due, emit in self._streams:
                if due(now):
                    emit()
        self._moves.answer()  # after the state that says where the arm got to, never mid-travel

    def reset(self, seed: int | None = None):
        """Re-randomize the scene from ``seed`` and publish what it draws.

        The model and data are rebuilt wholesale, so model-level loader effects (fixed-body poses,
        colors, cameras) re-randomize too; the renderer and IK physics rebind lazily. Stale commands
        queued while idle are dropped and the held grip is cleared, so the first step does not apply a
        queued command on the freshly reset scene.
        """
        self._load_scene(seed)
        self._apply_initial_ctrl()
        self._error = False
        self.commands.read()
        self.target_grip.read()
        self._last_grip = 0.0
        self._emit_robot_meta()
        self._publish_frame()

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
        """Emit every observation stream once for the current scene."""
        self._emit_state()
        self._emit_grip()
        self._emit_sim_state()
        self._emit_cameras()

    def _emit_state(self):
        state = MujocoFrankaState()
        state.encode(self._q, self._dq, self._ee_pose)
        if self._error:
            state.set_status(RobotStatus.ERROR)
        elif self._moves.active:
            state.set_status(RobotStatus.BUSY)  # a move owns the arm, and the command stream goes unread
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
        self.initial_joints = np.array([self.initial_ctrl[self.model.actuator(n).id] for n in self._actuator_names])
        self._joint_qpos_ids = [self.model.joint(name).qposadr.item() for name in self._joint_names]
        min_grip, max_grip = self.model.actuator(self._gripper_actuator).ctrlrange
        self._grip_range = (float(min_grip), float(max_grip))
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._ik_data = None

    def _apply_initial_ctrl(self):
        """Drive the actuators to their default controls (``initial_ctrl``) and step through the settling
        transient, leaving the arm in the pose the scene draws it in."""
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
        target_time = self.data.time + (duration or self.model.opt.timestep)
        while self.data.time < target_time:
            mj.mj_step(self.model, self.data)

    def _to_joints(self, cmd: roboarm_command.CommandType) -> np.ndarray:
        """The joints ``cmd`` asks the arm to hold. A control mode it pins is not honored: the sim runs its
        own law."""
        match cmd:
            case roboarm_command.CartesianPosition(pose=pose):
                return self._ik(pose)
            case roboarm_command.CartesianDelta() as delta_cmd:
                return self._ik(delta_cmd.apply(self._ee_pose))
            case roboarm_command.JointPosition(positions=positions):
                return self._joints(positions)
            case roboarm_command.JointDelta(velocities=delta):
                return self._q + self._joints(delta)
            case other:
                raise NotImplementedError(f'Unsupported command {other}')

    def _joints(self, values: np.ndarray) -> np.ndarray:
        """``values`` as a joint vector; NumPy would broadcast one that names fewer joints across them all."""
        joints = np.asarray(values, dtype=np.float64)
        if joints.shape != (len(self._joint_names),):
            raise ValueError(f'{joints} does not name every joint')
        return joints

    def _apply_command(self, cmd: roboarm_command.CommandType) -> None:
        """Aim the actuators at what ``cmd`` asks for, leaving the arm where it is if it cannot be met."""
        try:
            self._set_actuator_values(self._to_joints(cmd))
        # rules-allow: swallowed-error — a command stream cannot end the run; the arm reads ERROR instead
        except ValueError as exc:
            logger.warning(f'{cmd} not applied: {exc}')
            self._error = True

    def _ik(self, target: geom.Transform3D) -> np.ndarray:
        """The joints that put the end effector at ``target``; raises what the arm cannot reach."""
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
        if not success:
            raise ValueError(f'{target} is out of reach')
        return qpos[self._joint_qpos_ids]

    def _set_actuator_values(self, values: np.ndarray) -> None:
        """Aim every arm actuator at ``values``; a target the arm cannot be put at leaves it alone."""
        # Checked whole before the first write, so a refused target retargets nothing. A non-finite one
        # steps the sim to NaN, and no comparison against NaN ever reports the arm arrived.
        if not np.all(np.isfinite(values)):
            raise ValueError(f'{np.asarray(values)} is not a joint target')
        for name, value in list(zip(self._actuator_names, values, strict=True)):
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
