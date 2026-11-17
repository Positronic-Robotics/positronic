"""The env server answers from an isolated interpreter whose packages a simulator's own venv pins."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.env_server.launcher import PYTHONPATH_ENV, _await_bind, free_port, terminate

_ENV_SERVER_DIR = Path(__file__).parents[1]

_OLDEST_WEBSOCKETS = '12.0'

_SERVE_ONE_TASK = """
import sys
from protocol import TASK_INSTRUCTION
from server import EnvProtocol, EnvServer

class OneTask(EnvProtocol):
    def tasks(self, spec):
        return [{'name': 'idle', TASK_INSTRUCTION: 'wait'}]
    def reset(self, token):
        raise NotImplementedError
    def step(self, actions):
        raise NotImplementedError
    def close(self):
        pass

EnvServer(OneTask(), 'localhost', int(sys.argv[1])).serve_forever()
"""


@pytest.fixture
def uv() -> str:
    found = shutil.which('uv')
    assert found is not None, 'uv is not on PATH, and this test builds its interpreter with it'
    return found


def test_a_server_on_the_oldest_websockets_answers_a_request(uv):
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
        _await_bind(server, 'localhost', port, deadline=300.0)
        conn = EnvConnection('localhost', port)
        try:
            assert conn.tasks({}) == [{'name': 'idle', protocol.TASK_INSTRUCTION: 'wait'}]
        finally:
            conn.close()
    finally:
        terminate(server)
