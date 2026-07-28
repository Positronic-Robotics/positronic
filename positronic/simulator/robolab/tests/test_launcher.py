import functools
from pathlib import Path

from positronic.simulator.robolab import launcher
from positronic.simulator.robolab.launcher import _env_command


def test_env_command_omits_camera_flags_by_default():
    command = _env_command(Path('/robolab'), 'localhost', 5000, None)
    assert '--camera-res' not in command
    assert '--disable-viewport' not in command
    assert command[-1] == '--headless'


def test_env_command_sets_resolution_and_disables_viewport():
    command = _env_command(Path('/robolab'), 'localhost', 5000, (320, 180))
    res_at = command.index('--camera-res')
    assert command[res_at + 1 : res_at + 3] == ['320', '180']
    assert '--disable-viewport' in command


def test_serve_robolab_threads_resolution_into_spawn(monkeypatch):
    captured = {}
    monkeypatch.setattr(launcher, 'serve_subprocess', lambda spawn, host: captured.update(spawn=spawn, host=host))
    launcher.serve_robolab(camera_resolution=(320, 180))
    spawn = captured['spawn']
    assert isinstance(spawn, functools.partial)
    assert spawn.func is launcher._spawn
    assert spawn.keywords == {'camera_resolution': (320, 180)}


def test_serve_robolab_defaults_to_no_resolution(monkeypatch):
    captured = {}
    monkeypatch.setattr(launcher, 'serve_subprocess', lambda spawn, host: captured.update(spawn=spawn))
    launcher.serve_robolab()
    assert captured['spawn'].keywords == {'camera_resolution': None}
