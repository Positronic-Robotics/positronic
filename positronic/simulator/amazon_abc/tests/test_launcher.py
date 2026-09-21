"""What the launcher hands its env-server subprocess, and which assets it asks for.

Nothing here clones ABC or spawns it: the checkout, the installs and ``Popen`` are stubbed, so what is under
test is the choice ``_spawn`` makes rather than the simulation it leads to.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from positronic.simulator.amazon_abc import launcher


@pytest.fixture
def spawned(monkeypatch):
    """The subprocess environment and the asset commands ``_spawn`` runs, for a platform and a task selection."""

    def build(platform: str, tasks=('put_plastic_bottles_in_bin',)):
        runs: list[list[str]] = []
        captured: dict[str, str] = {}
        monkeypatch.setattr(sys, 'platform', platform)
        monkeypatch.setattr(launcher, 'ensure_pinned_checkout', lambda *args, **kwargs: Path('/abc/src'))
        monkeypatch.setattr(subprocess, 'run', lambda command, **kwargs: runs.append([str(c) for c in command]))
        monkeypatch.setattr(subprocess, 'Popen', lambda command, env: captured.update(env))
        launcher._spawn('localhost', 4242, tasks)
        return captured, runs

    return build


def test_headless_linux_gets_a_renderer_without_being_told(spawned, monkeypatch):
    """A GPU host has no display, and nobody running an eval should have to know that."""
    monkeypatch.delenv('MUJOCO_GL', raising=False)
    env, _ = spawned('linux')
    assert env['MUJOCO_GL'] == 'egl'


def test_an_operators_renderer_is_never_overridden(spawned, monkeypatch):
    """A software-rendering box exports its own backend; the default must not win over it."""
    monkeypatch.setenv('MUJOCO_GL', 'osmesa')
    env, _ = spawned('linux')
    assert env['MUJOCO_GL'] == 'osmesa'


def test_the_server_reaches_the_wire_modules_and_abc_itself(spawned):
    env, _ = spawned('linux')
    assert env['PYTHONPATH'].split(':') == [
        str(Path(launcher.__file__).parents[1] / 'env_server'),
        str(Path(launcher.__file__).parent),
        str(launcher._ABC_SRC),
    ]


def test_only_the_named_tasks_assets_are_downloaded(spawned):
    _, runs = spawned('linux', tasks=['put_plastic_bottles_in_bin', 'turn_mug_right_side_up'])
    assert runs[-1][-3:] == ['--sim-task', 'put_plastic_bottles_in_bin', 'turn_mug_right_side_up']


def test_a_sweep_over_the_catalogue_downloads_every_package(spawned):
    """An unbound task list runs whatever ABC offers, so every scene's meshes have to be there."""
    _, runs = spawned('linux', tasks=None)
    assert runs[-1][-1] == '--sim'
