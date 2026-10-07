"""How a run reaches its RoboLab env server: the clone count on the server's own command line, and an eval
that shares a server some other process runs.

Nothing here spawns RoboLab: the spawn needs its pinned checkout and the whole Isaac Lab stack.
"""

import subprocess
from typing import Any

import pytest

from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import launcher as env_launcher
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
    monkeypatch.setattr(launcher, '_ensure_robolab_src', lambda: launcher._ROBOLAB_SRC)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)
    # Nothing binds the port here, so the wait for it is stubbed out.
    monkeypatch.setattr(env_launcher, '_await_bind', lambda *args, **kwargs: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda command, **kwargs: commands.append(command) or _StubProcess())
    return commands


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_a_run_that_names_no_clone_count_serves_one_scene(spawned):
    with launcher.serve_robolab(robolab_keys.WRIST_LEFT):
        pass

    assert _flag(spawned[0], '--num-envs') == '1'


def test_the_clone_count_and_the_host_reach_the_env_server(spawned):
    with launcher.serve_robolab(robolab_keys.WRIST_LEFT, '0.0.0.0', num_envs=16):
        pass

    assert _flag(spawned[0], '--num-envs') == '16'
    assert _flag(spawned[0], '--host') == '0.0.0.0'
    assert _flag(spawned[0], '--cameras') == robolab_keys.WRIST_LEFT


class _OneTaskEnv(EnvProtocol):
    """Answers the task listing a RoboLab server of two clones would give for one task."""

    @property
    def num_slots(self) -> int:
        return 2

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        return [{'name': 'BananaInBowlTask', 'episode_length_s': 30.0}]

    def reset(self, token: Any) -> dict[str, Any]:
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
