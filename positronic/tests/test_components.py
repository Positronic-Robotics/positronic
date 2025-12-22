import pickle
from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

import pytest

import pimm
from pimm.core import ControlSystem


def _optional_import(module: str, symbol: str) -> Any | None:
    try:
        return getattr(import_module(module), symbol)
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return None


DearpyguiUi = cast(Any | None, _optional_import('positronic.gui.dpg', 'DearpyguiUi'))
SLCamera = cast(Any | None, _optional_import('positronic.drivers.camera.zed', 'SLCamera'))
LinuxVideo = cast(Any | None, _optional_import('positronic.drivers.camera.linux_video', 'LinuxVideo'))
OpenCVCamera = cast(Any | None, _optional_import('positronic.drivers.camera.opencv', 'OpenCVCamera'))
LuxonisCamera = cast(Any | None, _optional_import('positronic.drivers.camera.luxonis', 'LuxonisCamera'))
SoundSystem = cast(Any | None, _optional_import('positronic.drivers.sound', 'SoundSystem'))


def _make_linux_video() -> ControlSystem:
    if LinuxVideo is None:
        pytest.skip('linux_video dependencies are not available')
    return LinuxVideo('/dev/video0', 640, 480, 30, 'MJPG')


def _make_opencv_camera() -> ControlSystem:
    if OpenCVCamera is None:
        pytest.skip('opencv dependencies are not available')
    return OpenCVCamera(0, (640, 480), 30)


def _make_luxonis_camera() -> ControlSystem:
    if LuxonisCamera is None:
        pytest.skip('depthai is not available')
    return LuxonisCamera(30)


@pytest.mark.skipif(DearpyguiUi is None, reason='dearpygui is not available')
def test_dearpygui_ui_is_picklable():
    assert DearpyguiUi is not None
    gui = DearpyguiUi()
    original_receiver = gui.cameras['cam']

    data = pickle.dumps(gui)
    restored = pickle.loads(data)

    assert isinstance(original_receiver, pimm.ControlSystemReceiver)
    assert isinstance(restored.cameras['cam'], pimm.ControlSystemReceiver)
    assert isinstance(restored.cameras['new_cam'], pimm.ControlSystemReceiver)


@pytest.mark.skipif(SLCamera is None, reason='pyzed is not available')
def test_zed_slcamera_is_picklable():
    assert SLCamera is not None
    cam = SLCamera(depth_mode='none')
    pickle.dumps(cam)


@pytest.mark.parametrize(
    'factory',
    [
        pytest.param(_make_linux_video, id='LinuxVideo'),
        pytest.param(_make_opencv_camera, id='OpenCVCamera'),
        pytest.param(_make_luxonis_camera, id='LuxonisCamera'),
    ],
)
def test_camera_drivers_are_picklable(factory: Callable[[], ControlSystem]):
    cam = factory()
    assert isinstance(cam, ControlSystem)
    pickle.dumps(cam)


def test_sound_system_level_to_frequency_clamps_extreme_values():
    if SoundSystem is None:
        pytest.skip('sound dependencies are not available')

    sound = SoundSystem()
    # Extremely large level should not overflow (regression for pow overflow).
    master_volume, frequency = sound._level_to_frequency(1e300)
    assert master_volume == sound.enable_master_volume
    assert 0.0 < frequency < sound.sample_rate / 2.0
