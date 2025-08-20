import enum

import pimm

from positronic.drivers.webxr import WebXR
from positronic.utils.buttons import ButtonHandler


class DataCollectionCommand(enum.Enum):
    START_TRACKING = "start_tracking"
    STOP_TRACKING = "stop_tracking"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    RESET_ROBOT = "reset_robot"
    NO_OP = "no_op"

    def __str__(self) -> str:
        return self.value


def _parse_webxr_buttons(buttons: dict, button_handler: ButtonHandler):
    for side in ['left', 'right']:
        if buttons[side] is None:
            continue

        mapping = {
            f'{side}_A': buttons[side][4],
            f'{side}_B': buttons[side][5],
            f'{side}_trigger': buttons[side][0],
            f'{side}_thumb': buttons[side][1],
            f'{side}_stick': buttons[side][3]
        }
        button_handler.update_buttons(mapping)


def webxr_controls(webxr_buttons: pimm.SignalReader[dict]):
    button_handler = ButtonHandler()

    def _convert_buttons_to_commands(buttons: dict):
        _parse_webxr_buttons(buttons, button_handler)

        if button_handler.just_pressed('right_A'):
            return DataCollectionCommand.START_TRACKING
        elif button_handler.just_released('right_A'):
            return DataCollectionCommand.STOP_TRACKING
        elif button_handler.just_pressed('right_B'):
            return DataCollectionCommand.START_RECORDING
        elif button_handler.just_released('right_B'):
            return DataCollectionCommand.STOP_RECORDING
        elif button_handler.just_pressed('right_trigger'):
            return DataCollectionCommand.RESET_ROBOT
        else:
            return DataCollectionCommand.NO_OP

    return pimm.map(webxr_buttons, _convert_buttons_to_commands)