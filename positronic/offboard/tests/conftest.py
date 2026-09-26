import os
import tempfile
import threading
from collections.abc import Callable, Generator, Mapping
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock

import pytest
from positronic_wire import wire
from positronic_wire.grpc import GrpcClientWire
from positronic_wire.websocket import WebsocketClientWire, WebsocketUnixClientWire

from positronic.offboard import grpc_wire, server_wire, websocket_wire
from positronic.offboard.server import PolicyServer
from positronic.offboard.spec import Model, ModelSource, PolicyDeployment
from positronic.policy.processors import ChunkedSchedule


class Served(NamedTuple):
    """A running server, and the addresses its wires took."""

    host: str
    port: int
    server: PolicyServer
    grpc_port: int | None
    uds: Path | None = None

    def ws(self, model: str = '', query: str = '') -> tuple[wire.ClientWire[Any], wire.HostPortAddress]:
        """The websocket wire, and this server's session on it: ``InferenceClient(*served.ws())``."""
        return WebsocketClientWire(), wire.HostPortAddress(self.host, self.port, wire.session_path(model), query)

    def grpc(self, model: str = '', query: str = '') -> tuple[wire.ClientWire[Any], wire.HostPortAddress]:
        """The gRPC wire, and this server's session on it."""
        assert self.grpc_port is not None, 'the server serves no gRPC wire'
        return GrpcClientWire(), wire.HostPortAddress(self.host, self.grpc_port, wire.session_path(model), query)

    def unix(self, model: str = '', query: str = '') -> tuple[wire.ClientWire[Any], wire.UnixSocketAddress]:
        """The socket wire, and this server's session on the socket it bound."""
        assert self.uds is not None, 'the server bound no socket'
        return WebsocketUnixClientWire(), wire.UnixSocketAddress(self.uds, wire.session_path(model), query)


StartServer = Callable[..., Served]


@pytest.fixture
def start_server() -> Generator[StartServer, None, None]:
    """Factory serving pipelines on daemon threads; teardown stops and joins every started server.

    Each wire asks for port 0, and servers started in parallel never draw the same port. ``grpc=True``
    serves the gRPC wire beside the websocket one. ``uds`` binds the websocket wire to that socket path
    instead, as a socket-bound websocket wire does; the port it reports is then 0.
    """
    running: list[tuple[PolicyServer, threading.Thread]] = []

    def start(pipeline, *, grpc: bool = False, uds: str | None = None, **server_kwargs) -> Served:
        host = server_kwargs.pop('host', 'localhost')
        server = PolicyServer(pipeline, **server_kwargs)
        binds: server_wire.ServedAddress = (
            server_wire.ServedHostPort(host, 0) if uds is None else websocket_wire.ServedUnixSocket(Path(uds))
        )
        wires: list[server_wire.Wire] = [websocket_wire.WebsocketWire(binds)]
        if grpc:
            wires.append(grpc_wire.GrpcWire(server_wire.ServedHostPort(host, 0)))
        ready = threading.Event()
        thread = threading.Thread(target=server.serve, args=(wires, ready.set), daemon=True)
        thread.start()
        running.append((server, thread))
        if not ready.wait(timeout=10.0):
            raise RuntimeError('Server failed to start')
        served = wires[0].served_address
        bound = served.port if isinstance(served, server_wire.ServedHostPort) else 0
        grpc_served = wires[1].served_address if grpc else None
        return Served(
            host,
            bound,
            server,
            grpc_served.port if isinstance(grpc_served, server_wire.ServedHostPort) else None,
            served.uds if isinstance(served, websocket_wire.ServedUnixSocket) else None,
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
def make_mock_model():
    def make(action, meta):
        model = MagicMock(spec=Model)
        model.return_value = action
        model.meta.return_value = meta
        return model

    return make


class DictSource(ModelSource):
    """Multi-model source over ready models; the dict's first key is the default."""

    def __init__(self, models: Mapping[str, Model]):
        self._models = models

    def get_models(self) -> list[str]:
        return list(self._models)

    def resolve(self, model_id: str | None) -> str:
        if model_id is None:
            return next(iter(self._models))
        if model_id not in self._models:
            raise ValueError(f'Unknown model {model_id!r}. Available: {list(self._models)}')
        return model_id

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model:
        return self._models[model_id]


@pytest.fixture
def mock_model(make_mock_model) -> MagicMock:
    """Callable model with independently configurable results and metadata."""
    return make_mock_model({'action_data': [1, 2, 3]}, {'model_name': 'test_model'})


@pytest.fixture
def mock_model_registry(make_mock_model) -> dict[str, MagicMock]:
    return {
        'alpha': make_mock_model({'action_data': ['alpha']}, {'model_name': 'alpha'}),
        'beta': make_mock_model({'action_data': ['beta']}, {'model_name': 'beta'}),
    }


@pytest.fixture
def inference_server(start_server: StartServer, mock_model: MagicMock) -> tuple[str, int]:
    """A served single-policy pipeline.

    Returns:
        tuple[str, int]: (host, port)
    """
    host, port, *_ = start_server(PolicyDeployment(DictSource({'default': mock_model}), ChunkedSchedule(fps=10)))
    return host, port


@pytest.fixture
def multi_model_server(
    start_server: StartServer, mock_model_registry: dict[str, MagicMock]
) -> tuple[str, int, dict[str, MagicMock]]:
    host, port, *_ = start_server(PolicyDeployment(DictSource(mock_model_registry), ChunkedSchedule(fps=10)))
    return host, port, mock_model_registry
