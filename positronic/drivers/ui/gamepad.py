import logging
import numpy as np
import pygame
from pygame.joystick import Joystick

import ironic as ir
from positronic.tools.buttons import ButtonHandler


LOGITECH_F710_BUTTON_MAPPING = {
    0: "A",
    1: "B",
    2: "X",
    3: "Y",
    4: "Left bumper",
    5: "Right bumper",
    6: "Back",
    7: "Start",
    8: "Left analog stick",
    9: "Right analog stick",
    10: "Left trigger",
    11: "Right trigger",
}


def apply_deadzone(arr, deadzone_size=0.05):
    """Apply deadzone to an array of joystick values."""
    return np.where(np.abs(arr) <= deadzone_size, 0, np.sign(arr) * (np.abs(arr) - deadzone_size) / (1 - deadzone_size))


@ir.ironic_system(output_props=["buttons", "axis"])
class GamepadCS(ir.ControlSystem):
    """A joystick control system."""

    def __init__(self, joystick_id=0, fps=200, deadzone_size=0.05, button_mapping=LOGITECH_F710_BUTTON_MAPPING):
        """
        Initialize the gamepad control system.

        Args:
            joystick_id (int): ID of the joystick to use (default: 0)
            fps (int): How many times per second to read the joystick (default: 200)
            deadzone_size (float): Size of the deadzone to apply to axis values (default: 0.05)
            button_mapping (dict): Mapping of button indices to names for ButtonHandler
                                   If None, will use default numeric indices as names
        """
        super().__init__()
        self.joystick_id = joystick_id
        self.deadzone_size = deadzone_size
        self.button_mapping = button_mapping
        self.joystick = None
        self.button_handler = ButtonHandler()
        self.pygame_pump = ir.utils.ThrottledCallback(pygame.event.pump, fps)

    async def setup(self):
        """Initialize pygame and the joystick."""
        pygame.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No joysticks connected")
            return

        if self.joystick_id >= pygame.joystick.get_count():
            logging.error(
                f"Joystick ID {self.joystick_id} out of range. Only {pygame.joystick.get_count()} joysticks available.")
            self.joystick_id = 0

        self.joystick = Joystick(self.joystick_id)
        self.joystick.init()

    async def step(self):
        """Read joystick state and update button handler."""
        if self.joystick is None:
            logging.error("Joystick not initialized")
            return ir.State.FINISHED

        self.pygame_pump()
        return ir.State.ALIVE

    async def cleanup(self):
        """Clean up pygame resources."""
        if self.joystick:
            self.joystick.quit()
        pygame.quit()

    @ir.out_property
    async def buttons(self):
        """Get the current button states."""
        button_states = {}

        for i in range(self.joystick.get_numbuttons()):
            button_name = i
            if self.button_mapping and i in self.button_mapping:
                button_name = self.button_mapping[i]
            button_states[button_name] = float(self.joystick.get_button(i))

        self.button_handler.update_buttons(button_states)
        # TODO: Make ButtonHandler "frozen" for the readers.
        # This requires changing ButtonHandler behavior in different places.
        return ir.Message(self.button_handler)

    @ir.out_property
    async def axis(self):
        """Get the current axis values with deadzone applied."""
        axis_values = np.array([self.joystick.get_axis(i) for i in range(self.joystick.get_numaxes())])
        axis_values = apply_deadzone(axis_values, self.deadzone_size)
        return ir.Message(axis_values)
