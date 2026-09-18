"""How many scene clones a run asks for, from the eval config down to the env server's own command line.

Nothing here spawns RoboLab: the spawn needs its pinned checkout and the whole Isaac Lab stack, which
``e2e.py`` exercises on a RoboLab box. What is checked is that the number a run names is the number the
server is told to clone.
"""

import subprocess
from typing import Any

import pytest

from positronic.cfg.eval.sim import robolab as robolab_cfg
from positronic.simulator.env_server import launcher as env_launcher
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab import launcher


@pytest.fixture
def spawned(monkeypatch) -> list[list[str]]:
    """The command lines ``serve_robolab`` would have spawned, with the checkout and the sync stubbed out."""
    commands: list[list[str]] = []
    monkeypatch.setattr(launcher, '_ensure_robolab_src', lambda: launcher._ROBOLAB_SRC)
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: None)
    # Nothing binds the port here, so the wait for it is stubbed out; ``test_remote_env`` covers that wait.
    monkeypatch.setattr(env_launcher, '_await_bind', lambda *args, **kwargs: None)
    monkeypatch.setattr(subprocess, 'Popen', lambda command, **kwargs: commands.append(command) or _StubProcess())
    return commands


class _StubProcess:
    """Stands in for the spawned server: with the bind wait stubbed out, it is only ever terminated."""

    def terminate(self) -> None:
        pass

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_a_run_that_names_no_clone_count_serves_one_scene(spawned):
    """With no clone count named, the server is told to serve one scene behind the socket."""
    with launcher.serve_robolab(robolab_keys.WRIST_LEFT):
        pass

    assert _flag(spawned[0], '--num-envs') == '1'


def test_the_clone_count_reaches_the_env_server(spawned):
    """A run that asks for 16 clones tells the server to clone 16 times."""
    with launcher.serve_robolab(robolab_keys.WRIST_LEFT, num_envs=16):
        pass

    assert _flag(spawned[0], '--num-envs') == '16'
    assert _flag(spawned[0], '--cameras') == robolab_keys.WRIST_LEFT


def test_the_eval_config_surfaces_the_clone_count(monkeypatch):
    """``--eval.num_envs`` is what an operator sets; it reaches the launcher unchanged."""
    asked: list[Any] = []

    def serve_robolab(cameras: str, num_envs: int = 1):
        asked.append((cameras, num_envs))
        return None

    monkeypatch.setattr(robolab_cfg, 'serve_robolab', serve_robolab)

    robolab_cfg.banana_in_bowl.override(num_envs=8).instantiate()

    assert asked == [(robolab_keys.WRIST_LEFT_RIGHT, 8)]
