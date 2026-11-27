"""The env server answers from an isolated interpreter whose packages a simulator's own venv pins."""

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from websockets.sync.client import connect

from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.env_server.launcher import PYTHONPATH_ENV, _await_bind, free_port, terminate

_ENV_SERVER_DIR = Path(__file__).parents[1]

# The websockets that RoboLab's Isaac Lab venv resolves to.
_OLDEST_WEBSOCKETS = '12.0'

# Lists one task; each observation is the step count, and each episode ends at its second step.
_SERVE_ONE_TASK = """
import sys
import numpy as np
from protocol import FRAME_CONTROL_DT, FRAME_DONE, FRAME_META, FRAME_OBS, FRAME_ROBOT_META, SLOTS, TASK_INSTRUCTION
from server import EnvProtocol, EnvServer

class OneTask(EnvProtocol):
    def tasks(self, spec):
        return [{'name': 'idle', TASK_INSTRUCTION: 'wait'}]
    def reset(self, token, slots=None):
        self.steps = 0
        return {SLOTS: [{FRAME_OBS: 0}], FRAME_META: {}, FRAME_ROBOT_META: {}, FRAME_CONTROL_DT: 0.1}
    def step(self, actions):
        self.steps += 1
        slot = {FRAME_OBS: self.steps, FRAME_DONE: self.steps >= 2}
        return {SLOTS: [slot], FRAME_CONTROL_DT: 0.1}
    def close(self):
        pass

EnvServer(OneTask(), 'localhost', int(sys.argv[1])).serve_forever()
"""


@pytest.fixture
def uv() -> str:
    found = shutil.which('uv')
    assert found is not None, 'uv is not on PATH, and this test builds its interpreter with it'
    return found


# On a cold host the isolated `uv` environment installs its packages before the server binds.
BIND_DEADLINE_S = 300.0

# Longer than the bind deadline, so a slow start fails on that deadline and the test body still runs.
pytestmark = pytest.mark.timeout(BIND_DEADLINE_S + 60)


@pytest.fixture
def oldest_websockets_server_port(uv: str) -> Iterator[int]:
    """The port of an env server that runs on the oldest websockets that a simulator's venv carries."""
    port = free_port()
    command = [
        uv,
        'run',
        '--no-project',
        '--isolated',
        '--python',
        '3.11',
        '--with',
        f'websockets=={_OLDEST_WEBSOCKETS}',
        '--with',
        'msgpack',
        '--with',
        'numpy',
        'python',
        '-c',
        _SERVE_ONE_TASK,
        str(port),
    ]
    env = {**os.environ, PYTHONPATH_ENV: str(_ENV_SERVER_DIR)}
    server = subprocess.Popen(command, env=env)
    try:
        _await_bind(server, 'localhost', port, deadline=BIND_DEADLINE_S)
        yield port
    finally:
        terminate(server)


def test_a_server_on_the_oldest_websockets_answers_a_request(oldest_websockets_server_port):
    conn = EnvConnection('localhost', oldest_websockets_server_port)
    try:
        assert conn.tasks({}) == [{'name': 'idle', protocol.TASK_INSTRUCTION: 'wait'}]
    finally:
        conn.close()


def test_a_server_on_the_oldest_websockets_runs_an_episode_to_its_end(oldest_websockets_server_port):
    conn = EnvConnection('localhost', oldest_websockets_server_port)
    try:
        frames = [conn.reset(None), conn.step({}), conn.step({})]
    finally:
        conn.close()

    assert [frame[protocol.FRAME_OBS] for frame in frames] == [0, 1, 2]
    assert [frame.get(protocol.FRAME_DONE) for frame in frames] == [None, False, True]


def test_a_server_on_the_oldest_websockets_frees_the_slot_of_a_client_that_drops(oldest_websockets_server_port):
    """The client closes its socket with no close request, so only the server's own handling frees the slot."""
    with connect(f'ws://localhost:{oldest_websockets_server_port}/', compression=None) as dropped:
        dropped.send(protocol.encode({protocol.CMD: protocol.Command.RESET.value, protocol.TOKEN: None}))
        dropped.recv(timeout=10.0)

    deadline = time.monotonic() + 10.0
    while True:
        remaining = max(deadline - time.monotonic(), 0.0)
        conn = EnvConnection('localhost', oldest_websockets_server_port, connect_deadline=remaining)
        try:
            frame = conn.reset(None)
            break
        except RuntimeError:
            # The server frees the dropped slot after its handler returns, so a reset can still find it taken.
            if time.monotonic() >= deadline:
                raise
        finally:
            conn.close()
        time.sleep(0.1)

    assert frame[protocol.FRAME_OBS] == 0
