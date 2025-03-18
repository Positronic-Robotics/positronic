import asyncio
import logging
import numpy as np
import fire

import ironic as ir
from positronic.drivers.tidybot2 import Tidybot
from positronic.drivers.ui.joystick import JoystickCS

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


@ir.ironic_system(
    input_props=["joystick_axis", "joystick_buttons"],
    output_ports=["target_velocity_local", "target_velocity_global", "start_control", "stop_control"],
)
class TidybotTeleop(ir.ControlSystem):
    """Control system for teleoperating the Tidybot using a joystick."""

    def __init__(self, max_vel=(1.0, 1.0, 3.14)):
        """
        Initialize the Tidybot teleop control system.

        Args:
            max_vel (tuple): Maximum velocity in (x, y, theta) directions
        """
        super().__init__()
        self.max_vel = np.array(max_vel)
        self.vehicle_started = False

    async def step(self):
        button_handler = (await self.ins.joystick_buttons()).data

        if button_handler.is_pressed("Back") and self.vehicle_started:
            logging.info("Stopping Tidybot control")
            await self.outs.stop_control.write(ir.Message(None))
            self.vehicle_started = False
            return ir.State.FINISHED

        update_port = None
        if button_handler.is_pressed("Left bumper"):
            update_port = self.outs.target_velocity_local
        elif button_handler.is_pressed("Right bumper"):
            update_port = self.outs.target_velocity_global

        if update_port is not None:
            axis = await self.ins.joystick_axis()
            axis_values = axis.data
            if len(axis_values) < 4:
                logging.error("Expected 4 axis values, got %d, terminating teleop", len(axis_values))
                return ir.State.FINISHED

            x = -axis_values[1]  # Left analog stick forward/backward
            y = -axis_values[0]  # Left analog stick left/right
            th = -axis_values[3]  # Right analog stick left/right

            target_velocity = self.max_vel * np.array([x, y, th])

            await update_port.write(ir.Message(target_velocity, axis.timestamp))

        return ir.State.ALIVE


@ir.config(joystick_id=0, fps=200, deadzone_size=0.1)
def joystick(joystick_id, fps, deadzone_size):
    return JoystickCS(joystick_id=joystick_id, fps=fps, deadzone_size=deadzone_size)


@ir.config(max_vel=(1.0, 1.0, 3.14), max_accel=(0.5, 0.5, 2.36))
def tidybot(max_vel, max_accel):
    return Tidybot(max_vel=max_vel, max_accel=max_accel)


@ir.config(max_vel=(1.0, 1.0, 3.14))
def teleop(max_vel):
    return TidybotTeleop(max_vel=max_vel)


async def _main(joystick: ir.ControlSystem, tidybot: ir.ControlSystem, teleop: ir.ControlSystem):
    teleop.bind(joystick_axis=joystick.outs.axis, joystick_buttons=joystick.outs.buttons)
    tidybot.bind(target_velocity_local=teleop.outs.target_velocity_local,
                 target_velocity_global=teleop.outs.target_velocity_global,
                #  start_control=teleop.outs.start_control,
                #  stop_control=teleop.outs.stop_control
                 )

    system = ir.compose(joystick, teleop, tidybot)
    await ir.utils.run_gracefully(system)


main = ir.Config(_main, joystick=joystick, tidybot=tidybot, teleop=teleop)


async def async_main(**kwargs):
    await main.override_and_instantiate(**kwargs)


def custom_main(**kwargs):
    if 'help' in kwargs:
        del kwargs['help']
        config = main.override(**kwargs)
        print(config)
        return
    asyncio.run(async_main(**kwargs))


if __name__ == "__main__":
    fire.Fire(custom_main)
