import pickle

import pytest

import pimm

try:
    from positronic.gui.dpg import DearpyguiUi
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DearpyguiUi = None


@pytest.mark.skipif(DearpyguiUi is None, reason='dearpygui is not available')
def test_dearpygui_ui_is_picklable():
    gui = DearpyguiUi()
    original_receiver = gui.cameras['cam']

    data = pickle.dumps(gui)
    restored = pickle.loads(data)

    assert isinstance(original_receiver, pimm.ControlSystemReceiver)
    assert isinstance(restored.cameras['cam'], pimm.ControlSystemReceiver)
    assert isinstance(restored.cameras['new_cam'], pimm.ControlSystemReceiver)
