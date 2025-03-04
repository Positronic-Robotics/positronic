from typing import Callable, Optional, Sequence, Tuple

import ironic as ir
from positronic.drivers.roboarm import RobotStatus
from geom import Transform3D
from ironic.utils import Throttler
from positronic.simulator.mujoco.sim import (
    CompositeMujocoMetricCalculator,
    InverseKinematics,
    MujocoMetricCalculator,
    MujocoRenderer,
    MujocoSimulator,
)


@ir.ironic_system(
    input_ports=["actuator_values", "gripper_target_grasp", "reset", "robot_target_position"],
    output_ports=["images", "robot_status"],
    output_props=[
        "robot_position",
        "grip",
        "joints",
        "ext_force_ee",
        "ext_force_base",
        "actuator_values",
        "metrics",
        "episode_metadata",
    ])
class MujocoSimulatorCS(ir.ControlSystem):
    def __init__(
            self,
            simulator_factory: Callable[[], Tuple[MujocoSimulator, MujocoRenderer, InverseKinematics]],
            simulation_rate: float = 1 / 500,
            render_rate: float = 1 / 60,
            metric_calculators: Optional[Sequence[MujocoMetricCalculator]] = None,
    ):
        super().__init__()
        self.simulator_factory = simulator_factory
        self.simulator = None
        self._renderer = None
        self.inverse_kinematics = None
        self.do_simulation = Throttler(every_sec=simulation_rate)
        self.simulation_rate = simulation_rate
        self.pending_actions = []
        self.do_render = Throttler(every_sec=render_rate)
        self.metric_calculator = CompositeMujocoMetricCalculator(metric_calculators or [])

        self.resetting = False

    @property
    def renderer(self) -> MujocoRenderer:
        return self._renderer

    @renderer.setter
    def renderer(self, value):
        if self._renderer is not None:
            self._renderer.close()
        print('setting renderer')
        self._renderer = value

    @ir.out_property
    async def robot_position(self):
        return ir.Message(
            Transform3D(
                translation=self.simulator.robot_position.translation,
                rotation=self.simulator.robot_position.rotation
            ),
            self.ts
        )

    @ir.out_property
    async def grip(self):
        return ir.Message(self.simulator.grip, self.ts)

    @ir.out_property
    async def joints(self):
        return ir.Message(self.simulator.joints, self.ts)

    @ir.out_property
    async def ext_force_ee(self):
        return ir.Message(self.simulator.ext_force_ee, self.ts)

    @ir.out_property
    async def ext_force_base(self):
        return ir.Message(self.simulator.ext_force_base, self.ts)

    @ir.out_property
    async def actuator_values(self):
        return ir.Message(self.simulator.actuator_values, self.ts)

    @ir.out_property
    async def metrics(self):
        return ir.Message(self.metric_calculator.get_metrics(), self.ts)

    @ir.out_property
    async def episode_metadata(self):
        return ir.Message(self.simulator.save_state(), self.ts)

    @property
    def ts(self) -> int:
        return self.simulator.ts_ns

    def simulate(self):
        self.simulator.step()
        self.metric_calculator.update()

    async def render(self):
        if self.renderer is not None:
            images = self.renderer.render_frames()
            await self.outs.images.write(ir.Message(images, self.ts))

    async def _init_position(self):
        await self.outs.robot_status.write(ir.Message(RobotStatus.RESETTING, 0))
        self.resetting = True
        self.simulator, self.renderer, self.inverse_kinematics = self.simulator_factory()
        self.simulator.reset()
        self.metric_calculator.reset()
        self.renderer.initialize()

        await self.outs.robot_status.write(ir.Message(RobotStatus.AVAILABLE, self.ts))
        self.resetting = False

    def _init_renderer(self):
        if self.renderer is not None:
            self.renderer.initialize()

    @ir.on_message('reset')
    async def on_reset(self, _message: ir.Message):
        await self._init_position()

    @ir.on_message('gripper_target_grasp')
    async def on_gripper_target_grasp(self, message: ir.Message):
        self.simulator.set_grip(message.data)

    @ir.on_message('robot_target_position')
    async def on_robot_target_position(self, message: ir.Message):
        actuator_values = self.inverse_kinematics.recalculate_ik(message.data)

        if actuator_values is not None:
            self.simulator.set_actuator_values(actuator_values)

    @ir.on_message('actuator_values')
    async def on_actuator_values(self, message: ir.Message):
        self.simulator.set_actuator_values(message.data)

    async def setup(self):
        await self._init_position()

    async def step(self):
        for _ in range(self.do_simulation()):
            self.simulate()

        if self.do_render():
            await self.render()
        return ir.State.ALIVE

    async def cleanup(self):
        if self.renderer is not None:
            self.renderer.close()
