import pickle
import sys

import pytest

import pimm
from positronic.vendors.gr00t import server as groot_server
from positronic.vendors.gr00t import train as groot_train

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


def test_groot_resolve_python_bin_defaults_to_sys_executable():
    assert groot_server._resolve_python_bin(None) == sys.executable
    assert groot_train._resolve_python_bin(None) == sys.executable


def test_groot_resolve_python_bin_respects_override_path():
    assert groot_server._resolve_python_bin('/tmp/groot-venv') == '/tmp/groot-venv/bin/python'
    assert groot_train._resolve_python_bin('/tmp/groot-venv') == '/tmp/groot-venv/bin/python'
