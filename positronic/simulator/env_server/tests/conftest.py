import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from positronic.simulator.env_server.launcher import free_port
from positronic.simulator.env_server.server import EnvProtocol, EnvServer
from positronic.simulator.env_server.tests.mujoco_env import CAMERAS, make_mujoco_env


@contextmanager
def serve_env(env: EnvProtocol) -> Iterator[tuple[str, int]]:
    """Serve one ``EnvProtocol`` on a localhost port in a background thread.

    A thread (not a subprocess) shares this process, so a test's ``mj.Renderer`` monkeypatch reaches
    the server's render path.
    """
    host, port = 'localhost', free_port()
    server = EnvServer(env, host, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
    else:
        raise RuntimeError('Env server failed to start')

    try:
        yield host, port
    finally:
        server.shutdown()
        thread.join(timeout=2.0)


@pytest.fixture
def env_server() -> Iterator[tuple[str, int]]:
    """A ``MujocoSim``-backed ``stack_cubes`` env server."""
    with serve_env(make_mujoco_env(list(CAMERAS.values()))) as addr:
        yield addr
