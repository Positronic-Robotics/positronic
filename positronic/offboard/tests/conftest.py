import asyncio
import socket
import threading
import time
from collections.abc import Callable, Generator, Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest
import uvicorn

from positronic.offboard.server import PolicyServer
from positronic.policy import AnySession, ChunkSession, Policy, Session
from positronic.policy.executor import Executor
from positronic.policy.layers import ChunkPlayer
from positronic.policy.spec import ModelSource, PolicySource, remote


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


StartServer = Callable[..., tuple[str, int, PolicyServer]]


@pytest.fixture
def start_server() -> Generator[StartServer, None, None]:
    """Factory serving pipelines on daemon threads; every started server is stopped and joined at teardown."""
    running: list[tuple[uvicorn.Server, threading.Thread]] = []

    def start(pipeline, **server_kwargs) -> tuple[str, int, PolicyServer]:
        server = PolicyServer(pipeline, host='localhost', port=_find_free_port(), **server_kwargs)
        uv_server = uvicorn.Server(uvicorn.Config(server.app, host=server.host, port=server.port, log_level='warning'))

        async def _run():
            await server._startup()
            await uv_server.serve()

        thread = threading.Thread(target=asyncio.run, args=(_run(),), daemon=True)
        thread.start()
        running.append((uv_server, thread))

        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                with socket.create_connection((server.host, server.port), timeout=0.1):
                    return server.host, server.port, server
            except (ConnectionRefusedError, OSError):
                time.sleep(0.05)
        raise RuntimeError('Server failed to start')

    yield start
    for uv_server, thread in running:
        uv_server.should_exit = True
        thread.join(timeout=5.0)


@pytest.fixture
def open_session() -> Generator[Callable[..., tuple[AnySession, Executor]], None, None]:
    """Opens a policy's session against a runtime that serves its functions, as the harness does."""
    runtimes: list[Executor] = []

    def make(policy: Policy) -> tuple[AnySession, Executor]:
        runtimes.append(Executor(policy.functions))
        return policy.new_session(None, runtimes[-1]), runtimes[-1]

    yield make
    for runtime in runtimes:
        runtime.close()


# How long a round trip against a local server may take before a test calls it lost.
ANSWER_SEC = 5.0


def round_trip(session: ChunkSession, rt: Executor, obs, time_ns: int = 0) -> list[dict] | dict | None:
    """What ``session`` answers for ``obs``, over the two calls one round trip takes.

    Both calls get the same ``time_ns``, so a chunk comes back anchored at the value the test passed.
    """
    assert session(obs, time_ns) is None, 'a round-trip was already in flight'
    rt.wait(ANSWER_SEC)
    assert not rt.in_flight, 'the round-trip never came back'
    return session(obs, time_ns)


def played_round_trip(session: Session, rt: Executor, obs, time_ns: int = 0) -> Mapping[str, Any]:
    """What ``session`` commands for ``obs``, over the two calls one round trip takes.

    For a session topped by a ``ChunkPlayer``: both calls get the same ``time_ns``, so the chunk comes back
    anchored at the value the test passed and the answer is the waypoints due at it.
    """
    assert session(obs, time_ns)[0] == {}, 'a round-trip was already in flight'
    rt.wait(ANSWER_SEC)
    assert not rt.in_flight, 'the round-trip never came back'
    return session(obs, time_ns)[0]


def _make_mock_policy(action, meta):
    """Create a mock policy with session-based API."""
    session = MagicMock()
    session.return_value = action
    session.meta = meta
    session.close = MagicMock()

    policy = MagicMock()
    policy.new_session.return_value = session
    policy.meta = meta
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
    host, port, _server = start_server(ChunkPlayer() | remote | PolicySource(mock_policy))
    return host, port


@pytest.fixture
def multi_policy_server(
    start_server: StartServer, mock_policy_registry: dict[str, MagicMock]
) -> tuple[str, int, dict[str, MagicMock]]:
    host, port, _server = start_server(ChunkPlayer() | remote | _DictSource(mock_policy_registry))
    return host, port, mock_policy_registry
