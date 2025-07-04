from typing import Sequence, Tuple

import geom
import ironic2 as ir
import mujoco as mj
from dm_control import mujoco as dm_mujoco
from dm_control.utils import inverse_kinematics as ik
import numpy as np

from positronic.simulator.mujoco.scene.transforms import MujocoSceneTransform, load_model_from_spec_file


def load_from_xml_path(model_path: str, loaders: Sequence[MujocoSceneTransform] = ()) -> mj.MjModel:
    model, _ = load_model_from_spec_file(model_path, loaders)

    return model


# TODO: maybe MujocoSim is clock?
class MujocoClock(ir.Clock):
    def __init__(self, sim: "MujocoSim"):
        self.sim = sim

    def now(self) -> float:
        return self.sim.data.time

    def sleep(self, duration: float) -> None:
        self.sim.step(duration)


class MujocoSim:
    def __init__(self, mujoco_model_path: str, loaders: Sequence[MujocoSceneTransform] = ()):
        self.model = load_from_xml_path(mujoco_model_path, loaders)
        self.data = mj.MjData(self.model)

    def run(self, should_stop: ir.SignalReader):
        while not ir.is_true(should_stop):
            self.step()
            yield

    def get_clock(self) -> ir.Clock:
        return MujocoClock(self)

    def step(self, duration: float | None = None) -> None:
        duration = duration or self.model.opt.timestep
        while self.data.time < self.data.time + duration:
            mj.mj_step(self.model, self.data)


class MujocoCamera:
    frame: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, model, data, camera_name: str, resolution: Tuple[int, int]):
        super().__init__()
        self.model = model
        self.data = data
        self.render_resolution = resolution
        self.camera_name = camera_name

    def run(self, should_stop: ir.SignalReader):
        renderer = mj.Renderer(self.model, height=self.render_resolution[1], width=self.render_resolution[0])

        while not ir.is_true(should_stop):
            renderer.update_scene(self.data, camera=self.camera_name)
            frame = renderer.render()
            self.frame.emit({'frame': frame})
            yield

        renderer.close()


class MujocoFranka:
    commands: ir.SignalReader = ir.NoOpReader()

    state: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, sim: MujocoSim, suffix: str = ''):
        self.sim = sim
        self.physics = dm_mujoco.Physics.from_model(sim.data)
        self.ee_name = f'end_effector{suffix}'
        self.joint_names = [f'joint{i}{suffix}' for i in range(1, 8)]
        self.actuator_names = [f'actuator{i}{suffix}' for i in range(1, 8)]
        self.joint_qpos_ids = [self.sim.model.joint(joint).qposadr.item() for joint in self.joint_names]

    def run(self, should_stop: ir.SignalReader):
        self.commands = ir.DefaultReader(self.commands, None)

        while not ir.is_true(should_stop):
            command = self.commands.value
            if command is not None:
                q = self._recalculate_ik(command)
                if q is not None:
                    self.set_actuator_values(q)

            self.state.emit(self.sim.data.qpos[:])
            yield

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
    def joints(self) -> np.ndarray:
        return np.array([self.sim.data.qpos[i] for i in self.joint_qpos_ids])

    def set_actuator_values(self, actuator_values: np.ndarray):
        for i in range(7):
            self.sim.data.actuator(self.actuator_names[i]).ctrl = actuator_values[i]


class MujocoGripper:
    target_grip: ir.SignalReader = ir.NoOpReader()
    grip: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, sim: MujocoSim, actuator_name: str):
        self.sim = sim
        self.actuator_name = actuator_name

    def run(self, should_stop: ir.SignalReader):
        self.target_grip = ir.DefaultReader(self.target_grip, 0)

        while not ir.is_true(should_stop):
            target_grip = self.target_grip.value
            self.sim.data.actuator(self.actuator_name).ctrl = target_grip

            # TODO: this is wrong but follows previous implementation. We need to get proper qpos instead of ctrl
            self.grip.emit(self.sim.data.actuator(self.actuator_name).ctrl)
            yield
