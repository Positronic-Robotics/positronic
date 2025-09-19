import pickle

import pimm
from positronic.gui.dpg import DearpyguiUi


def test_dearpygui_ui_is_picklable():
    gui = DearpyguiUi()
    original_receiver = gui.cameras['cam']

    data = pickle.dumps(gui)
    restored = pickle.loads(data)

    assert isinstance(original_receiver, pimm.ControlSystemReceiver)
    assert isinstance(restored.cameras['cam'], pimm.ControlSystemReceiver)
    assert isinstance(restored.cameras['new_cam'], pimm.ControlSystemReceiver)
