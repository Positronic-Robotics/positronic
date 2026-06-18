import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik

import pimm
from positronic import geom
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as roboarm_command
from positronic.simulator.mujoco.transforms import MujocoSceneTransform, load_spec, load_spec_from_file, np_seed
from positronic.utils import package_assets_path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def bundled_panda_model() -> dict:
    """The simulated Franka panda (arm + hand) for the 3D viewer and offline IK reconstruction: the
    panda URDF, its collision meshes, the joint names, and the ``end_effector`` control frame — the
    grasp site where the sim measures ``robot_state.ee_pose``. The viewer renders this and
    ``ik_joints_from_episode`` inverts against it, so both share one frame-consistent model. The real
    arm uses ``drivers/roboarm/fr3.urdf``, whose end_effector sits at the physical flange instead.
    """
    urdf_path = Path(package_assets_path('assets/mujoco/panda.urdf'))
    urdf = urdf_path.read_text()
    mesh_dir = urdf_path.parent / 'assets'
    mesh_files = {mesh.get('filename') for mesh in ET.fromstring(urdf).iter('mesh')}
    return {
        'urdf': urdf,
        'meshes': {name: (mesh_dir / name).read_bytes() for name in sorted(mesh_files)},
        'joint_names': [f'joint{i}' for i in range(1, 8)],
        'control_frame': 'end_effector',
    }


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

    Every tick applies the due command waypoints, steps physics once, then emits the due
    streams — so all signals carry the post-step state of the same physics instant. Each
    stream has an independent rate (``*_fps``, ``None`` = every physics tick).
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
        self._ik_physics: dm_mujoco.Physics | None = None

        self._load_scene()
        self._warmup()

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
        adapters = self._camera_adapters()
        state = MujocoFrankaState()
        arm_player = roboarm_command.TrajectoryPlayer()
        grip_player = roboarm_command.TrajectoryPlayer()
        last_grip = 0.0
        state_due = _Cadence(self._state_fps)
        grip_due = _Cadence(self._grip_fps)
        sim_state_due = _Cadence(self._sim_state_fps)
        cameras_due = _Cadence(self._camera_fps)

        while not should_stop.value:
            cmd_msg = self.commands.read()
            if cmd_msg.updated:
                arm_player.set(cmd_msg.data)
            for cmd in arm_player.advance(clock.now_ns()):
                self._apply_command(cmd, state)
            grip_msg = self.target_grip.read()
            if grip_msg.updated:
                grip_player.set(grip_msg.data)
            for grip in grip_player.advance(clock.now_ns()):
                last_grip = grip
            self._apply_grip(last_grip)

            self.step()
            self.fps_counter.tick()

            now = clock.now()
            if state_due(now):
                state.encode(self._q, self._dq, self._ee_pose)
                self.state.emit(state)
            if grip_due(now):
                self.grip.emit(self._current_grip())
            if self.sim_state.num_bound and sim_state_due(now):
                self.sim_state.emit({f'.{name}': arr for name, arr in self.save_state().items() if arr.size})
            if adapters and cameras_due(now):
                self._render(adapters, clock.now_ns())

            # One physics step is one tick of simulated time; sleeping a timestep paces the world clock.
            yield pimm.Sleep(self.model.opt.timestep)

        if self._renderer is not None:
            self._renderer.close()

    def reset(self, seed: int | None = None):
        """Re-randomize the scene by re-applying the loaders; ``seed`` makes the draw deterministic.

        The model and data are rebuilt wholesale, so model-level loader effects (fixed-body
        poses, colors, cameras) re-randomize too; the renderer and the IK physics rebind to
        the new model lazily. Re-emits ``robot_meta``, so episodes record the fresh scene's
        ``scene_xml``.
        """
        self._load_scene(seed)
        self._warmup()
        self._emit_robot_meta()

    def _load_scene(self, seed: int | None = None):
        """Apply the loaders to the model file and bind the result; ``scene_xml`` captures the draw."""
        with np_seed(seed):
            spec, self.metadata = load_spec_from_file(self.mujoco_model_path, self.loaders)
        self.model = spec.compile()
        self.scene_xml = spec.to_xml()
        self._bind_model()

    def _emit_robot_meta(self):
        self.robot_meta.emit({**bundled_panda_model(), 'scene_xml': self.scene_xml})

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
        self._ik_physics = None

    def _warmup(self):
        """Settle the freshly built scene: apply the initial controls and step through the transient."""
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

    def _apply_command(self, cmd, state: MujocoFrankaState):
        match cmd:
            case roboarm_command.CartesianPosition(pose=pose):
                q = self._recalculate_ik(pose)
                if q is not None:
                    self._set_actuator_values(q)
                else:
                    logger.warning(f'IK failed for ee_pose: {pose}')
                    state.set_error()
            case roboarm_command.JointPosition(positions=positions):
                self._set_actuator_values(positions)
            case roboarm_command.JointDelta(velocities=delta):
                self._set_actuator_values(self._q + delta)
            case roboarm_command.Reset():
                self.reset()
                state.clear_error()
            case roboarm_command.Recover():
                state.clear_error()
            case _:
                raise ValueError(f'Unknown command type: {type(cmd)}')

    def _recalculate_ik(self, target: geom.Transform3D) -> np.ndarray | None:
        if self._ik_physics is None:
            # Wraps the live ``data``, so IK warm-starts from the robot's current pose.
            self._ik_physics = dm_mujoco.Physics.from_model(self.data)
        result = ik.qpos_from_site_pose(
            physics=self._ik_physics,
            site_name=self._ee_name,
            target_pos=target.translation,
            target_quat=target.rotation.as_quat,
            joint_names=self._joint_names,
            rot_weight=0.5,
        )
        return result.qpos[self._joint_qpos_ids] if result.success else None

    def _set_actuator_values(self, values: np.ndarray):
        for name, value in zip(self._actuator_names, values, strict=True):
            self.data.actuator(name).ctrl = value

    def _apply_grip(self, target: float):
        """Convert [0, 1] target grip to the actuator control range."""
        min_grip, max_grip = self._grip_range
        self.data.actuator(self._gripper_actuator).ctrl = min_grip + target * (max_grip - min_grip)

    def _current_grip(self) -> float:
        """Convert the current grip joint position to the range of [0, 1]."""
        min_grip, max_grip = self._grip_range
        return (self.data.joint(self._gripper_joint).qpos.item() - min_grip) / (max_grip - min_grip)

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

    def _render(self, adapters: dict[str, pimm.shared_memory.NumpySMAdapter], ts_ns: int):
        if self._renderer is None:
            width, height = self._camera_resolution
            self._renderer = mj.Renderer(self.model, height=height, width=width)
        for name, adapter in adapters.items():
            self._renderer.update_scene(self.data, camera=name)
            self._renderer.render(out=adapter.array)
            self.cameras[name].emit(adapter, ts=ts_ns)
