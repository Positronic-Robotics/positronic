import asyncio
import os
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Generator, Mapping
from unittest.mock import MagicMock

import pytest
import uvicorn

from positronic.offboard.server import WS_IMPL, PolicyServer
from positronic.policy import Policy, Session
from positronic.policy.executor import Executor
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.spec import ModelSource, PolicySource, remote


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


RunningServers = list[tuple[uvicorn.Server, threading.Thread]]

StartServer = Callable[..., tuple[str, int, PolicyServer]]

StartUnixServer = Callable[..., PolicyServer]


@pytest.fixture
def running_servers() -> Generator[RunningServers, None, None]:
    """Every server a test started, stopped and joined at teardown."""
    running: RunningServers = []
    yield running
    for uv_server, thread in running:
        uv_server.should_exit = True
        thread.join(timeout=5.0)


def _serve_in_background(server: PolicyServer, config: uvicorn.Config, running: RunningServers) -> None:
    uv_server = uvicorn.Server(config)

    async def _run():
        await server._startup()
        await uv_server.serve()

    thread = threading.Thread(target=asyncio.run, args=(_run(),), daemon=True)
    thread.start()
    running.append((uv_server, thread))


def _wait_until_it_accepts(dial: Callable[[], None]) -> None:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            dial()
            return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError('Server failed to start')


@pytest.fixture
def socket_path() -> Generator[str, None, None]:
    """A path for a Unix socket, short enough for the 104-byte limit that ``tmp_path`` can pass."""
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
        yield os.path.join(directory, 's.sock')


@pytest.fixture
def start_server(running_servers: RunningServers) -> StartServer:
    """Factory serving pipelines on daemon threads."""

    def start(pipeline, **server_kwargs) -> tuple[str, int, PolicyServer]:
        server = PolicyServer(pipeline, host='localhost', port=_find_free_port(), **server_kwargs)
        config = uvicorn.Config(server.app, host=server.host, port=server.port, log_level='warning', ws=WS_IMPL)
        _serve_in_background(server, config, running_servers)
        _wait_until_it_accepts(lambda: socket.create_connection((server.host, server.port), timeout=0.1).close())
        return server.host, server.port, server

    return start


def _dial_unix(path: str) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.1)
        sock.connect(path)


@pytest.fixture
def start_unix_server(running_servers: RunningServers) -> Generator[StartUnixServer, None, None]:
    """Factory serving pipelines on a Unix socket, as ``PolicyServer.serve`` claims one."""
    claimed: list[socket.socket] = []

    def start(pipeline, uds: str, **server_kwargs) -> PolicyServer:
        server = PolicyServer(pipeline, uds=uds, **server_kwargs)
        sock = PolicyServer.claim_socket_path(uds)
        claimed.append(sock)
        config = uvicorn.Config(server.app, fd=sock.fileno(), log_level='warning', ws=WS_IMPL)
        _serve_in_background(server, config, running_servers)
        _wait_until_it_accepts(lambda: _dial_unix(uds))
        return server

    yield start
    for sock in claimed:
        sock.close()


@pytest.fixture
def open_session() -> Generator[Callable[..., tuple[Session, Executor]], None, None]:
    """Opens a policy's session against a runtime that serves its functions, as the harness does."""
    runtimes: list[Executor] = []

    def make(policy: Policy) -> tuple[Session, Executor]:
        runtimes.append(Executor(policy.functions))
        return policy.new_session(None, runtimes[-1]), runtimes[-1]

    yield make
    for runtime in runtimes:
        runtime.close()


# How long a round trip against a local server may take before a test calls it lost.
ANSWER_SEC = 5.0


def round_trip(session: Session, rt: Executor, obs, time_ns: int = 0) -> list[dict] | None:
    """What ``session`` answers for ``obs``, over the two calls one round trip takes.

    Both calls get the same ``time_ns``, so a chunk comes back anchored at the value the test passed.
    """
    assert session(obs, time_ns) is None, 'a round-trip was already in flight'
    rt.wait(ANSWER_SEC)
    assert not rt.in_flight, 'the round-trip never came back'
    return session(obs, time_ns)


def _make_mock_policy(action, meta):
    """Create a mock policy with session-based API."""
    session = MagicMock()
    session.return_value = action
    session.meta = meta
    session.close = MagicMock()

    policy = MagicMock()
    policy.new_session.return_value = session
    policy.functions = {}  # `Policy.functions` is a mapping, and MagicMock's stand-in is not
    policy._mock_session = session  # expose for assertions
    return policy


@pytest.fixture
def make_mock_policy() -> Callable[..., MagicMock]:
    return _make_mock_policy


class _DictSource(ModelSource):
    """Multi-model source over ready policies; the dict's first key is the default."""

    def __init__(self, policies: Mapping[str, Policy]):
        self._policies = policies

    def get_models(self) -> list[str]:
        return list(self._policies)

    def resolve(self, model_id: str | None) -> str:
        if model_id is None:
            return next(iter(self._policies))
        if model_id not in self._policies:
            raise ValueError(f'Unknown model {model_id!r}. Available: {list(self._policies)}')
        return model_id

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        return self._policies[model_id]


@pytest.fixture
def mock_policy() -> MagicMock:
    """Mock policy for testing."""
    return _make_mock_policy({'action_data': [1, 2, 3]}, {'model_name': 'test_model'})


@pytest.fixture
def mock_policy_registry() -> dict[str, MagicMock]:
    return {
        'alpha': _make_mock_policy({'action_data': ['alpha']}, {'model_name': 'alpha'}),
        'beta': _make_mock_policy({'action_data': ['beta']}, {'model_name': 'beta'}),
    }


@pytest.fixture
def inference_server(start_server: StartServer, mock_policy: MagicMock) -> tuple[str, int]:
    """A served single-policy pipeline.

    Returns:
        tuple[str, int]: (host, port)
    """
    host, port, _server = start_server(ChunkedSchedule() | remote | PolicySource(mock_policy))
    return host, port


@pytest.fixture
def multi_policy_server(
    start_server: StartServer, mock_policy_registry: dict[str, MagicMock]
) -> tuple[str, int, dict[str, MagicMock]]:
    host, port, _server = start_server(ChunkedSchedule() | remote | _DictSource(mock_policy_registry))
    return host, port, mock_policy_registry
