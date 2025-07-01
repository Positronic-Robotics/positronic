import time
from typing import Dict, Sequence

import ironic2 as ir
from positronic.simulator.mujoco.sim import create_from_config, MujocoSimulatorEnv, MujocoSceneTransform


class MujocoSim:
    target_position: ir.SignalReader = ir.NoOpReader()

    state: ir.SignalEmitter = ir.NoOpEmitter()
    frames: Dict[str, ir.SignalEmitter] = {}

    sim: MujocoSimulatorEnv

    def __init__(
            self,
            mujoco_model_path: str,
            simulation_hz: float,
            camera_names: Sequence[str],
            camera_width: int,
            camera_height: int,
            loaders: Sequence[MujocoSceneTransform] = ()
    ):
        self.sim = create_from_config(
            mujoco_model_path, simulation_hz, camera_names, camera_width, camera_height, loaders
        )

    @property
    def camera_names(self):
        return self.sim.renderer.camera_names

    def run(self, should_stop: ir.SignalReader):
        self.sim.renderer.initialize()
        while not ir.is_true(should_stop):
            self.sim.simulator.step()
            self.state.emit(self.sim.simulator.joints)

            frames = self.sim.renderer.render()
            for cam_name, frame in frames.items():
                self.frames[cam_name].emit({'frame': frame})

            time.sleep(1 / 500)