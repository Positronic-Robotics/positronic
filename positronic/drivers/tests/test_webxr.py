"""What the WebXR driver makes of the payload from the headset."""

import numpy as np

from positronic.drivers.webxr import _parse_controller_data

TOUCH_BUTTONS = [0.4, 0.0, 0.0, 0.0, 1.0, 0.0]  # trigger, squeeze, unused, stick, A, B


def _payload(**controllers):
    return {'controllers': {'left': None, 'right': None, **controllers}}


def _controller(buttons):
    return {'position': [0.1, 0.2, 0.3], 'orientation': [1.0, 0.0, 0.0, 0.0], 'buttons': buttons}


def test_gamepad_buttons_reach_the_caller():
    _, buttons = _parse_controller_data(_payload(right=_controller(TOUCH_BUTTONS)))

    right = buttons['right']
    assert right is not None
    np.testing.assert_allclose(right, TOUCH_BUTTONS)


def test_input_source_without_gamepad_has_no_buttons():
    positions, buttons = _parse_controller_data(_payload(right=_controller([])))

    right = positions['right']
    assert buttons['right'] is None
    assert right is not None
    np.testing.assert_allclose(right.translation, [0.1, 0.2, 0.3])
