"""The setup steps a benchmark launcher composes, checked without a benchmark."""

import fcntl
import subprocess
import sys
from pathlib import Path

import pytest

from positronic.simulator.env_server import launcher


# rules-allow: hardcoded-keys — these assertions spell the names the subprocess reads, so a test built from
# the launcher's own constants would check the derivation against itself.
@pytest.mark.parametrize('platform, backend', [('linux', 'egl'), ('darwin', 'cgl')])
def test_a_server_gets_the_platform_renderer_when_the_operator_sets_none(monkeypatch, platform, backend):
    monkeypatch.delenv('MUJOCO_GL', raising=False)
    monkeypatch.setattr(sys, 'platform', platform)
    assert launcher.mujoco_server_env([])['MUJOCO_GL'] == backend


def test_an_operators_renderer_is_never_overridden(monkeypatch):
    monkeypatch.setenv('MUJOCO_GL', 'osmesa')
    assert launcher.mujoco_server_env([])['MUJOCO_GL'] == 'osmesa'


def test_the_server_path_keeps_the_given_order():
    env = launcher.mujoco_server_env([Path('/a'), Path('/b')])
    assert env['PYTHONPATH'].split(':') == ['/a', '/b']


def test_a_second_setup_waits_while_the_first_holds_the_lock(tmp_path):
    cache = tmp_path / 'cache'
    with launcher.setup_lock(cache):
        with open(cache / 'setup.lock', 'w') as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(cache / 'setup.lock', 'w') as other:
        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_a_venv_is_made_only_where_it_is_absent(tmp_path, monkeypatch):
    runs: list[list[str]] = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: runs.append(command))
    venv = tmp_path / '.venv'

    assert launcher.ensure_venv(venv, '3.12') == venv / 'bin' / 'python'
    venv.mkdir()
    launcher.ensure_venv(venv, '3.12')

    assert runs == [['uv', 'venv', '--python', '3.12', str(venv)]]


def test_an_install_goes_into_the_named_venv(tmp_path, monkeypatch):
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))

    launcher.uv_pip_install(tmp_path, ['-c', 'pins.txt', 'numpy'], cwd=tmp_path / 'src')

    ((command, kwargs),) = calls
    assert command == ['uv', 'pip', 'install', '-c', 'pins.txt', 'numpy']
    assert kwargs['env']['VIRTUAL_ENV'] == str(tmp_path)
    assert kwargs['cwd'] == str(tmp_path / 'src')
