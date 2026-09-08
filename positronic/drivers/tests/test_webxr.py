"""What the WebXR driver makes of the payload from the headset."""

import numpy as np
import pytest

from positronic.drivers.webxr import BUTTONS, CONTROLLERS, ORIENTATION, POSITION, SIDES, _parse_controller_data

TOUCH_BUTTONS = [0.4, 0.0, 0.0, 0.0, 1.0, 0.0]  # trigger, squeeze, unused, stick, A, B


def _payload(**controllers):
    return {CONTROLLERS: {**dict.fromkeys(SIDES), **controllers}}


def _controller(buttons):
    return {POSITION: [0.1, 0.2, 0.3], ORIENTATION: [1.0, 0.0, 0.0, 0.0], BUTTONS: buttons}


def test_gamepad_buttons_reach_the_caller():
    _, buttons = _parse_controller_data(_payload(right=_controller(TOUCH_BUTTONS)))

    right = buttons['right']
    assert right is not None
    np.testing.assert_allclose(right, TOUCH_BUTTONS)


def test_an_input_source_that_cannot_drive_the_arm_is_refused():
    """Every teleoperation control is a button, so a source with none of them controls nothing."""
    with pytest.raises(ValueError, match=r'shaped \(0,\)'):
        _parse_controller_data(_payload(right=_controller([])))


def test_a_controller_short_of_a_button_is_refused():
    with pytest.raises(ValueError, match=r'shaped \(5,\)'):
        _parse_controller_data(_payload(right=_controller(TOUCH_BUTTONS[:5])))


def test_buttons_that_are_not_one_row_are_refused():
    """Six buttons in a nested list count six, and the reads that follow take one of the rows for a value."""
    with pytest.raises(ValueError, match=r'shaped \(1, 6\)'):
        _parse_controller_data(_payload(right=_controller([TOUCH_BUTTONS])))
