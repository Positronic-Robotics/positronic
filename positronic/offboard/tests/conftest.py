import asyncio
import socket
import threading
import time
from collections.abc import Callable, Generator, Mapping
from unittest.mock import MagicMock

import pytest
import uvicorn

from positronic.offboard.server import PolicyServer
from positronic.policy import Policy
from positronic.policy.layers import ChunkedSchedule
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


def _make_mock_policy(action, meta):
    """Create a mock policy with session-based API."""
    session = MagicMock()
    session.return_value = action
    session.meta = meta
    session.close = MagicMock()

    policy = MagicMock()
    policy.new_session.return_value = session
    policy.meta = meta
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
