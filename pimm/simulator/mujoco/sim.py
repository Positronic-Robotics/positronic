from typing import Any, Dict, Sequence, Tuple
import logging

import geom
import ironic as ir
import mujoco as mj
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik
import numpy as np

from pimm.drivers.roboarm import RobotStatus, State
from pimm.drivers.roboarm import command as roboarm_command
from pimm.simulator.mujoco.transforms import MujocoSceneTransform, load_model_from_spec_file

logger = logging.getLogger(__name__)


def load_from_xml_path(model_path: str, loaders: Sequence[MujocoSceneTransform] = ()) -> mj.MjModel:
    model, metadata = load_model_from_spec_file(model_path, loaders)

    return model, metadata


STATE_SPECS = [
    mj.mjtState.mjSTATE_FULLPHYSICS,
    mj.mjtState.mjSTATE_USER,
    mj.mjtState.mjSTATE_INTEGRATION,
    mj.mjtState.mjSTATE_WARMSTART,
]


class MujocoSim(ir.Clock):
    def __init__(self, mujoco_model_path: str, loaders: Sequence[MujocoSceneTransform] = ()):
        self.model, self.metadata = load_from_xml_path(mujoco_model_path, loaders)
        self.data = mj.MjData(self.model)
        self.fps_counter = ir.utils.RateCounter("MujocoSim")
        self.initial_ctrl = [float(x) for x in self.metadata.get('initial_ctrl').split(',')]
        self.warmup_steps = 1000
        self.reset()

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock):
        while not should_stop.value:
            self.step()
            self.fps_counter.tick()
            yield ir.Pass()

    def now(self) -> float:
        return self.data.time

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        if self.initial_ctrl is not None:
            self.data.ctrl = self.initial_ctrl
        mj.mj_step(self.model, self.data, self.warmup_steps)
        self.data.time = 0

    def load_state(self, state: dict, reset_time: bool = True):
        mj.mj_resetData(self.model, self.data)
        for spec in STATE_SPECS:
            mj.mj_setState(self.model, self.data, np.array(state[spec.name]), spec)

        if reset_time:
            self.data.time = 0

    def save_state(self) -> Dict[str, np.ndarray]:
        """
        Saves full state of the simulator.

        This state could be used to restore the exact state of the simulator.

        Returns:
            data: A dictionary containing the full state of the simulator.
        """
        data = {}

        for spec in STATE_SPECS:
            size = mj.mj_stateSize(self.model, spec)
            data[spec.name] = np.empty(size, np.float64)
            mj.mj_getState(self.model, self.data, data[spec.name], spec)

        return data

    def step(self, duration: float | None = None) -> None:
        duration = duration or self.model.opt.timestep
        target_time = self.data.time + duration
        while self.data.time < target_time:
            mj.mj_step(self.model, self.data)


class MujocoCamera:
    frame: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, model, data, camera_name: str, resolution: Tuple[int, int], fps: int = 30):
        super().__init__()
        self.model = model
        self.data = data
        self.render_resolution = resolution
        self.camera_name = camera_name
        self.fps = fps
        self.fps_counter = ir.utils.RateCounter("MujocoCamera")

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock):
        renderer = mj.Renderer(self.model, height=self.render_resolution[1], width=self.render_resolution[0])

        while not should_stop.value:
            renderer.update_scene(self.data, camera=self.camera_name)
            frame = renderer.render()
            self.frame.emit({'image': frame}, ts=clock.now_ns())
            self.fps_counter.tick()
            yield ir.Sleep(1 / self.fps)

        renderer.close()


class MujocoFrankaState(State, ir.shared_memory.NumpySMAdapter):
    def __init__(self):
        super().__init__(shape=(7 + 7 + 7 + 1,), dtype=np.float32)

    def instantiation_params(self) -> tuple[Any, ...]:
        return ()

    @property
    def q(self) -> np.ndarray:
        return self.array[:7]

    @property
    def dq(self) -> np.ndarray:
        return self.array[7:14]

    @property
    def ee_pose(self) -> geom.Transform3D:
        return geom.Transform3D(self.array[14:14 + 3], self.array[14 + 3:14 + 7])

    @property
    def status(self) -> RobotStatus:
        return RobotStatus(int(self.array[14 + 7]))

    def encode(self, q, dq, ee_pose):
        self.array[:7] = q
        self.array[7:14] = dq
        self.array[14:14 + 3] = ee_pose.translation
        self.array[14 + 3:14 + 7] = ee_pose.rotation.as_quat
        self.array[14 + 7] = self.status.value


class MujocoFranka:
    commands: ir.SignalReader[roboarm_command.CommandType] = ir.NoOpReader()

    state: ir.SignalEmitter[MujocoFrankaState] = ir.NoOpEmitter()

    def __init__(self, sim: MujocoSim, suffix: str = ''):
        self.sim = sim
        self.physics = dm_mujoco.Physics.from_model(sim.data)
        self.ee_name = f'end_effector{suffix}'
        self.joint_names = [f'joint{i}{suffix}' for i in range(1, 8)]
        self.actuator_names = [f'actuator{i}{suffix}' for i in range(1, 8)]
        self.joint_qpos_ids = [self.sim.model.joint(joint).qposadr.item() for joint in self.joint_names]

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock):
        commands = ir.DefaultReader(ir.ValueUpdated(self.commands), (None, False))
        state = MujocoFrankaState()

        while not should_stop.value:
            command, is_updated = commands.value
            if is_updated:
                match command:
                    case roboarm_command.CartesianMove(pose=pose):
                        self.set_ee_pose(pose)
                    case roboarm_command.JointMove(positions=positions):
                        self.set_actuator_values(positions)
                    case roboarm_command.Reset():
                        # TODO: it's not clear how to make reset, because interleave breakes if time goes backwards
                        pass
                    case _:
                        raise ValueError(f"Unknown command type: {type(command)}")

            # TODO: still a copy here
            state.encode(self.q, self.dq, self.ee_pose)

            self.state.emit(state)
            yield ir.Pass()

    def _recalculate_ik(self, target_robot_position: geom.Transform3D) -> np.ndarray | None:
        result = ik.qpos_from_site_pose(
            physics=self.physics,
            site_name=self.ee_name,
            target_pos=target_robot_position.translation,
            target_quat=target_robot_position.rotation.as_quat,
            joint_names=self.joint_names,
            rot_weight=0.5,
        )

        if result.success:
            return result.qpos[self.joint_qpos_ids]

        return None

    @property
    def q(self) -> np.ndarray:
        return np.array([self.sim.data.qpos[i] for i in self.joint_qpos_ids])

    @property
    def dq(self) -> np.ndarray:
        return np.array([self.sim.data.qvel[i] for i in self.joint_qpos_ids])

    @property
    def ee_pose(self) -> geom.Transform3D:
        translation = self.sim.data.site(self.ee_name).xpos.copy()
        rotmat: np.ndarray = self.sim.data.site(self.ee_name).xmat.copy()
        quat = self._xmat_to_quat(rotmat)
        return geom.Transform3D(translation=translation, rotation=geom.Rotation.from_quat(quat))

    def set_actuator_values(self, actuator_values: np.ndarray):
        for i in range(7):
            self.sim.data.actuator(self.actuator_names[i]).ctrl = actuator_values[i]

    def set_ee_pose(self, ee_pose: geom.Transform3D):
        q = self._recalculate_ik(ee_pose)
        if q is not None:
            self.set_actuator_values(q)
        else:
            logger.warning(f"Failed to calculate IK for ee_pose: {ee_pose}")

    def _xmat_to_quat(self, xmat: np.ndarray) -> np.ndarray:
        site_quat = np.empty(4)
        mj.mju_mat2Quat(site_quat, xmat)
        return site_quat


class MujocoGripper:
    target_grip: ir.SignalReader[float] = ir.NoOpReader()
    grip: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, sim: MujocoSim, actuator_name: str, joint_name: str):
        self.sim = sim
        self.actuator_name = actuator_name
        self.joint_name = joint_name

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock):
        target_grip_reader = ir.DefaultReader(self.target_grip, 0.0)

        while not should_stop.value:
            target_grip = target_grip_reader.value
            self.set_target_grip(target_grip)

            self.grip.emit(self.sim.data.joint(self.joint_name).qpos.item())
            yield ir.Pass()

    def set_target_grip(self, target_grip: float):
        self.sim.data.actuator(self.actuator_name).ctrl = target_grip
