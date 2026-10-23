"""What the launcher hands its env-server subprocess, and which assets it asks for."""

import re
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


# rules-allow: hardcoded-keys — these assertions spell the names the subprocess reads, so a test built from
# the launcher's own constants would check the derivation against itself.
def test_headless_linux_gets_a_renderer_without_being_told(spawned, monkeypatch):
    monkeypatch.delenv('MUJOCO_GL', raising=False)
    env, _ = spawned('linux')
    assert env['MUJOCO_GL'] == 'egl'


def test_an_operators_renderer_is_never_overridden(spawned, monkeypatch):
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


def test_a_sweep_over_the_catalogue_downloads_every_package_with_the_robocasa_packs(spawned):
    _, runs = spawned('linux', tasks=None)
    assert runs[-1][-2:] == ['--sim', '--sim-robocasa']


def test_named_tasks_leave_the_robocasa_choice_to_their_own_asset_map(spawned):
    _, runs = spawned('linux', tasks=['put_plastic_bottles_in_bin'])
    assert '--sim-robocasa' not in runs[-1]


def _pins(constraints: Path) -> dict[str, str]:
    lines = [line.strip() for line in constraints.read_text().splitlines()]
    return dict(line.split('==') for line in lines if line and not line.startswith('#'))


def test_every_package_of_the_abc_environment_installs_at_one_pinned_version(spawned):
    _, runs = spawned('linux')
    (install,) = [run for run in runs if run[:3] == ['uv', 'pip', 'install']]

    assert '-c' in install, 'the ABC environment installs with no constraints'
    pins = _pins(Path(install[install.index('-c') + 1]))
    # ABC ships no lockfile, and its contact-rich tasks score differently on another MuJoCo.
    named = [arg for arg in install[3:] if arg != '-c' and not arg.endswith('.txt')]
    for requirement in named:
        name = re.split('[<>=~!]', requirement)[0]
        assert name in pins, f'{name} has no pinned version'
    assert all(re.fullmatch(r'[0-9][0-9A-Za-z.+]*', version) for version in pins.values())
