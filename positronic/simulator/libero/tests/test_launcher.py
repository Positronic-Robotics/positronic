"""The environment the LIBERO env-server subprocess is launched with."""

from pathlib import Path

import pytest

from positronic.simulator.libero import launcher


@pytest.fixture
def spawn_env(monkeypatch):
    """Runs ``_spawn`` with the checkout and the subprocess stubbed out, and returns the environment it built."""
    captured: dict[str, str] = {}

    def fake_popen(command, env):
        captured.update(env)

    monkeypatch.setattr(launcher, '_ensure_libero_src', lambda: Path('/libero'))
    monkeypatch.setattr(launcher.subprocess, 'Popen', fake_popen)

    def spawn() -> dict[str, str]:
        launcher._spawn('localhost', 5555)
        return captured

    return spawn


def test_linux_renders_offscreen(monkeypatch, spawn_env):
    monkeypatch.setattr(launcher.sys, 'platform', 'linux')
    monkeypatch.delenv('MUJOCO_GL', raising=False)

    assert spawn_env()['MUJOCO_GL'] == 'egl'


def test_a_backend_chosen_by_the_operator_is_kept(monkeypatch, spawn_env):
    monkeypatch.setattr(launcher.sys, 'platform', 'linux')
    monkeypatch.setenv('MUJOCO_GL', 'osmesa')

    assert spawn_env()['MUJOCO_GL'] == 'osmesa'


def test_macos_leaves_the_backend_to_mujoco(monkeypatch, spawn_env):
    monkeypatch.setattr(launcher.sys, 'platform', 'darwin')
    monkeypatch.delenv('MUJOCO_GL', raising=False)

    assert 'MUJOCO_GL' not in spawn_env()
