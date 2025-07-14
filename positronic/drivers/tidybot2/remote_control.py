# Script for teleoperating the Tidybot using a gamepad.

import asyncio
import logging

import fire
import numpy as np

import configuronic as cfgc
import ironic as ir
import positronic.cfg.hardware.tidybot
import positronic.cfg.ui

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


@ir.ironic_system(
    input_props=["gamepad_axis", "gamepad_buttons"],
    output_ports=["target_velocity_local", "target_velocity_global", "start_control", "stop_control"],
)
class TidybotTeleop(ir.ControlSystem):
    """Control system for teleoperating the Tidybot using a gamepad."""

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
        button_handler = (await self.ins.gamepad_buttons()).data

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
            axis = await self.ins.gamepad_axis()
            axis_values = axis.data
            if len(axis_values) < 4:
                logging.error(f"Expected 4 axis values, got {len(axis_values)}, terminating teleop")
                return ir.State.FINISHED

            x = -axis_values[1]  # Left analog stick forward/backward
            y = -axis_values[0]  # Left analog stick left/right
            th = -axis_values[3]  # Right analog stick left/right

            target_velocity = self.max_vel * np.array([x, y, th])

            await update_port.write(ir.Message(target_velocity, axis.timestamp))

        return ir.State.ALIVE


teleop = cfgc.Config(TidybotTeleop, max_vel=(1.0, 1.0, 3.14))


async def _main(gamepad: ir.ControlSystem, tidybot: ir.ControlSystem, teleop: ir.ControlSystem):
    teleop.bind(gamepad_axis=gamepad.outs.axis, gamepad_buttons=gamepad.outs.buttons)
    tidybot.bind(target_velocity_local=teleop.outs.target_velocity_local,
                 target_velocity_global=teleop.outs.target_velocity_global)

    system = ir.compose(gamepad, teleop, tidybot)
    await ir.utils.run_gracefully(system)


main = cfgc.Config(_main,
                   gamepad=positronic.cfg.ui.gamepad,
                   tidybot=positronic.cfg.hardware.tidybot.vehicle0,
                   teleop=teleop)


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
