"""How a run reaches its RoboLab env server: the clone count on the server's own command line, and an eval
that shares a server some other process runs.

Nothing here spawns RoboLab: the spawn needs its pinned checkout and the whole Isaac Lab stack.
"""

import os
import subprocess
import sys
from typing import Any

import pytest

from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import launcher as env_launcher
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.server import EnvProtocol
from positronic.simulator.env_server.tests.conftest import serve_env
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab import launcher


class _StubProcess:
    """Stands in for the spawned server: with the bind wait stubbed out, it is only ever terminated."""

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.fixture
def spawned(monkeypatch) -> list[list[str]]:
    """The command lines ``serve_robolab`` would have spawned, with the checkout and the sync stubbed out."""
    commands: list[list[str]] = []
    monkeypatch.setattr(launcher, 'ensure_pinned_checkout', lambda *args, **kwargs: launcher._ROBOLAB_SRC)
    monkeypatch.setattr(
        subprocess, 'run', lambda command, **kwargs: subprocess.CompletedProcess(command, 0, 'uv 0.13.7\n')
    )
    # Nothing binds the port here, so the wait for it is stubbed out.
    monkeypatch.setattr(env_launcher, '_await_bind', lambda *args, **kwargs: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda command, **kwargs: commands.append(command) or _StubProcess())
    return commands


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_a_run_that_names_no_clone_count_serves_one_scene(spawned):
    with launcher.serve_robolab(robolab_keys.WRIST_LEFT):
        pass

    assert _flag(spawned[0], '--max-envs') == '1'


def test_the_clone_count_and_the_host_reach_the_env_server(spawned):
    with launcher.serve_robolab(robolab_keys.WRIST_LEFT, '0.0.0.0', max_envs=16):
        pass

    assert _flag(spawned[0], '--max-envs') == '16'
    assert _flag(spawned[0], '--host') == '0.0.0.0'
    assert _flag(spawned[0], '--cameras') == robolab_keys.WRIST_LEFT


def test_the_join_deadline_of_a_shared_server_reaches_the_env_server(spawned, monkeypatch):
    argv = ['launcher', '--max-envs', '8', '--port', '9000', '--join-deadline', '45']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit):
        launcher.main()

    assert _flag(spawned[0], '--join-deadline') == '45.0'


_LOCKED_PROJECT = """[project]
name = "robolab"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
isaac50 = []

[tool.uv]
package = false
constraint-dependencies = []
"""


def test_the_setup_installs_the_shipped_lock_and_leaves_the_checkout_as_pinned(monkeypatch, tmp_path):
    src = tmp_path / 'robolab'
    src.mkdir()
    (src / 'pyproject.toml').write_text(_LOCKED_PROJECT)
    monkeypatch.setenv('UV_OFFLINE', '1')
    monkeypatch.setenv('UV_CACHE_DIR', str(tmp_path / 'uv-cache'))
    monkeypatch.delenv('UV_PROJECT_ENVIRONMENT', raising=False)
    subprocess.run(['uv', 'lock', '--quiet', '--project', str(src)], check=True)
    # The shipped lock is stale against the pyproject: a sync that resolves rewrites it.
    pyproject = _LOCKED_PROJECT.replace('version = "0.1.0"', 'version = "0.1.1"')
    (src / 'pyproject.toml').write_text(pyproject)
    shipped_lock = (src / 'uv.lock').read_text()
    monkeypatch.setattr(launcher, 'ensure_pinned_checkout', lambda *args, **kwargs: src)
    monkeypatch.setattr(launcher, '_ROBOLAB_SRC', src)
    monkeypatch.setattr(launcher, '_ROBOLAB_PYTHON', sys.executable)
    monkeypatch.setattr(env_launcher, '_await_bind', lambda *args, **kwargs: None)
    popen = subprocess.Popen
    # The install runs for real; only the server spawn is stubbed out.
    monkeypatch.setattr(
        subprocess,
        'Popen',
        lambda command, **kwargs: _StubProcess() if command[:2] == ['uv', 'run'] else popen(command, **kwargs),
    )

    with launcher.serve_robolab(robolab_keys.WRIST_LEFT):
        pass

    assert (src / '.venv' / 'pyvenv.cfg').is_file()
    assert (src / 'uv.lock').read_text() == shipped_lock
    assert (src / 'pyproject.toml').read_text() == pyproject


def test_the_setup_refuses_a_uv_too_old_for_the_robolab_build_config(monkeypatch, tmp_path):
    calls = tmp_path / 'uv-calls'
    uv = tmp_path / 'bin' / 'uv'
    uv.parent.mkdir()
    uv.write_text(f'#!/bin/sh\necho "$@" >> {calls}\necho "uv 0.9.30"\n')
    uv.chmod(0o755)
    monkeypatch.setenv('PATH', f'{uv.parent}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setattr(launcher, 'ensure_pinned_checkout', lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(launcher, '_ROBOLAB_SRC', tmp_path / 'robolab')
    monkeypatch.setattr(env_launcher, '_await_bind', lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match=r'uv 0\.10\.0 or newer'), launcher.serve_robolab(robolab_keys.WRIST_LEFT):
        pass

    assert 'sync' not in calls.read_text()


class _OneTaskEnv(EnvProtocol):
    """Answers the task listing a RoboLab server of two clones would give for one task."""

    @property
    def num_slots(self) -> int:
        return 2

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                robolab_keys.RECORD_NAME: 'BananaInBowlTask',
                robolab_keys.RECORD_EPISODE_LENGTH_S: 30.0,
                protocol.TASK_INSTRUCTION: 'Put the banana in the bowl',
                robolab_keys.RECORD_INSTRUCTION_TYPE: spec[robolab_keys.SELECT_INSTRUCTION_TYPE],
            }
        ]

    def reset(self, token: Any, slots: int | None = None) -> dict[str, Any]:
        raise AssertionError('the listing needs no reset')

    def step(self, actions: dict[int, dict[str, Any]]) -> dict[str, Any]:
        raise AssertionError('the listing needs no step')

    def close(self) -> None:
        pass


def test_an_eval_with_an_env_server_drives_that_server_and_launches_none(monkeypatch):
    def launch(*args, **kwargs):
        raise AssertionError('an eval that names an env server launched one of its own')

    monkeypatch.setattr(robolab_cfg, 'serve_robolab', launch)
    with serve_env(_OneTaskEnv()) as (host, port):
        ev = robolab_cfg.banana_in_bowl.override(env_server=f'{host}:{port}').instantiate()
        trials = ev.tasks()

    assert [trial.prepare_args[eval_keys.SCENE][eval_keys.TASK] for trial in trials] == ['BananaInBowlTask']
