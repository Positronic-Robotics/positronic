import os
import tempfile
import threading
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
from positronic_wire import wire
from positronic_wire.grpc import GrpcClientWire
from positronic_wire.websocket import WebsocketClientWire, WebsocketUnixClientWire

from positronic.offboard import grpc_wire, server_wire, websocket_wire
from positronic.offboard.server import PolicyServer
from positronic.policy import Policy, Session
from positronic.policy.executor import Executor
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.spec import ModelSource, PolicySource, remote


class Served(NamedTuple):
    """A running server, and the addresses its wires took."""

    host: str
    port: int
    server: PolicyServer
    grpc_port: int | None
    uds: Path | None = None

    def ws(self, model: str = '', query: str = '') -> tuple[wire.ClientWire, wire.SessionAddress]:
        """The websocket wire, and this server's session on it: ``InferenceClient(*served.ws())``."""
        return WebsocketClientWire(), wire.SessionAddress(self.host, self.port, wire.session_path(model), query)

    def grpc(self, model: str = '', query: str = '') -> tuple[wire.ClientWire, wire.SessionAddress]:
        """The gRPC wire, and this server's session on it."""
        assert self.grpc_port is not None, 'the server serves no gRPC wire'
        return GrpcClientWire(), wire.SessionAddress(self.host, self.grpc_port, wire.session_path(model), query)

    def unix(self, model: str = '', query: str = '') -> tuple[wire.ClientWire, wire.SessionAddress]:
        """The socket wire, and this server's session on the socket it bound."""
        assert self.uds is not None, 'the server bound no socket'
        return WebsocketUnixClientWire(), wire.SessionAddress(
            self.host, self.port, wire.session_path(model), query, self.uds
        )


StartServer = Callable[..., Served]


@pytest.fixture
def start_server() -> Generator[StartServer, None, None]:
    """Factory serving pipelines on daemon threads; teardown stops and joins every started server.

    Each wire asks for port 0, and servers started in parallel never draw the same port. ``grpc=True``
    serves the gRPC wire beside the websocket one. ``uds`` binds the websocket wire to that socket path
    instead, as ``serve --uds`` does; the port it reports is then 0.
    """
    running: list[tuple[PolicyServer, threading.Thread]] = []

    def start(pipeline, *, grpc: bool = False, uds: str | None = None, **server_kwargs) -> Served:
        host = server_kwargs.pop('host', 'localhost')
        server = PolicyServer(pipeline, **server_kwargs)
        wires: list[server_wire.Wire] = [websocket_wire.WebsocketWire(host, 0, server.api, uds=uds)]
        if grpc:
            wires.append(grpc_wire.GrpcWire(host, 0))
        ready = threading.Event()
        thread = threading.Thread(target=server.serve, args=(wires, ready.set), daemon=True)
        thread.start()
        running.append((server, thread))
        if not ready.wait(timeout=10.0):
            raise RuntimeError('Server failed to start')
        return Served(
            host, wires[0].endpoint.port, server, wires[1].endpoint.port if grpc else None, wires[0].endpoint.uds
        )

    yield start
    for server, thread in running:
        server.shutdown()
        thread.join(timeout=10.0)
        assert not thread.is_alive(), 'the server did not stop when asked'


@pytest.fixture
def socket_path() -> Generator[str, None, None]:
    """A path for a Unix socket, short enough for the 104-byte limit that ``tmp_path`` can pass."""
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
        yield os.path.join(directory, 's.sock')


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


class DictSource(ModelSource):
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
def inference_server(start_server: StartServer, mock_policy: MagicMock) -> Served:
    """A served single-policy pipeline."""
    return start_server(ChunkedSchedule() | remote | PolicySource(mock_policy))


@pytest.fixture
def multi_policy_server(
    start_server: StartServer, mock_policy_registry: dict[str, MagicMock]
) -> tuple[Served, dict[str, MagicMock]]:
    return start_server(ChunkedSchedule() | remote | DictSource(mock_policy_registry)), mock_policy_registry
