"""The environment the launcher hands its env-server subprocess.

Unlike ``test_e2e``, nothing here clones LIBERO or spawns anything: the checkout and ``Popen`` are stubbed so
the assertions are about what ``_spawn`` decides, which is where the renderer choice belongs.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from positronic.simulator.libero import launcher


@pytest.fixture
def spawn_env(monkeypatch):
    """``_spawn``'s subprocess environment, for a platform and whatever the operator already exported."""

    def build(platform: str) -> dict[str, str]:
        captured: dict[str, str] = {}
        monkeypatch.setattr(sys, 'platform', platform)
        monkeypatch.setattr(launcher, '_ensure_libero_src', lambda: Path('/libero/src'))
        monkeypatch.setattr(subprocess, 'Popen', lambda command, env: captured.update(env))
        launcher._spawn('localhost', 4242)
        return captured

    return build


def test_headless_linux_gets_a_renderer_without_being_told(spawn_env, monkeypatch):
    """The point of the whole thing: a GPU host has no display, and nobody outside this file should have to
    know that MuJoCo defaults to GLFW."""
    monkeypatch.delenv('MUJOCO_GL', raising=False)
    assert spawn_env('linux')['MUJOCO_GL'] == 'egl'


def test_an_operators_renderer_is_never_overridden(spawn_env, monkeypatch):
    """A software-rendering box (CI) exports its own backend; the default must not win over it."""
    monkeypatch.setenv('MUJOCO_GL', 'osmesa')
    assert spawn_env('linux')['MUJOCO_GL'] == 'osmesa'


def test_macos_keeps_glfw(spawn_env, monkeypatch):
    """robosuite forces GLFW on macOS, and GLFW must init on the main thread — naming any backend here breaks
    rendering rather than fixing it."""
    monkeypatch.delenv('MUJOCO_GL', raising=False)
    assert 'MUJOCO_GL' not in spawn_env('darwin')
