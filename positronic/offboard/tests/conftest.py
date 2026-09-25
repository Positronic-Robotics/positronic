import os
import tempfile
import threading
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock

import pytest
from positronic_wire import wire
from positronic_wire.grpc import GrpcClientWire
from positronic_wire.websocket import WebsocketClientWire, WebsocketUnixClientWire

from positronic.offboard import grpc_wire, server_wire, websocket_wire
from positronic.offboard.server import PolicyServer
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.observation import ObservationCodec


class Served(NamedTuple):
    """A running server, and the addresses its wires took."""

    host: str
    port: int
    server: PolicyServer
    grpc_port: int | None
    uds: Path | None = None

    def ws(self, query: str = '') -> tuple[wire.ClientWire[Any], wire.HostPortAddress]:
        """The websocket wire, and this server's session on it: ``InferenceClient(*served.ws())``."""
        return WebsocketClientWire(), wire.HostPortAddress(self.host, self.port, wire.SESSION_PATH, query)

    def grpc(self, query: str = '') -> tuple[wire.ClientWire[Any], wire.HostPortAddress]:
        """The gRPC wire, and this server's session on it."""
        assert self.grpc_port is not None, 'the server serves no gRPC wire'
        return GrpcClientWire(), wire.HostPortAddress(self.host, self.grpc_port, wire.SESSION_PATH, query)

    def unix(self, query: str = '') -> tuple[wire.ClientWire[Any], wire.UnixSocketAddress]:
        """The socket wire, and this server's session on the socket it bound."""
        assert self.uds is not None, 'the server bound no socket'
        return WebsocketUnixClientWire(), wire.UnixSocketAddress(self.uds, wire.SESSION_PATH, query)


StartServer = Callable[..., Served]


@pytest.fixture
def start_server() -> Generator[StartServer, None, None]:
    """Factory serving a model through a pipeline on daemon threads; teardown stops and joins every server.

    Each wire asks for port 0, and servers started in parallel never draw the same port. ``grpc=True``
    serves the gRPC wire beside the websocket one. ``uds`` binds the websocket wire to that socket path
    instead, as a socket-bound websocket wire does; the port it reports is then 0.
    """
    running: list[tuple[PolicyServer, threading.Thread]] = []

    def start(model: Model, pipeline, *, grpc: bool = False, uds: str | None = None, **server_kwargs) -> Served:
        host = server_kwargs.pop('host', 'localhost')
        server = PolicyServer(lambda: model, pipeline, **server_kwargs)
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


# Not ``keys.TASK``, so a test that reads the prompt back here proves the codec encoded the warm.
WARM_PROMPT_FIELD = 'prompt'


def warm_pipeline() -> PolicyDeployment:
    """A served deployment whose server codec builds the observation a warm runs on."""
    codec = ObservationCodec(state={}, images={}, task_field=WARM_PROMPT_FIELD)
    return PolicyDeployment(ChunkedSchedule(fps=10), codec)


@pytest.fixture
def mock_model(make_mock_model) -> MagicMock:
    """Callable model with independently configurable results and metadata."""
    return make_mock_model({'action_data': [1, 2, 3]}, {'model_name': 'test_model'})


@pytest.fixture
def inference_server(start_server: StartServer, mock_model: MagicMock) -> tuple[str, int]:
    """A served single-policy pipeline.

    Returns:
        tuple[str, int]: (host, port)
    """
    host, port, *_ = start_server(mock_model, PolicyDeployment(ChunkedSchedule(fps=10)))
    return host, port
