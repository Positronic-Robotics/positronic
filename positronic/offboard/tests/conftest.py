import threading
from collections.abc import Callable, Generator, Mapping
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest

from positronic.offboard import grpc_wire, websocket_wire, wire
from positronic.offboard.server import PolicyServer
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.spec import Model, ModelSource, Pipeline


class Served(NamedTuple):
    """A running server, and the ports its wires took."""

    host: str
    port: int
    server: PolicyServer
    grpc_port: int | None


StartServer = Callable[..., Served]


@pytest.fixture
def start_server() -> Generator[StartServer, None, None]:
    """Factory serving pipelines on daemon threads; teardown stops and joins every started server.

    Each wire asks for port 0, and servers started in parallel never draw the same port. ``grpc=True``
    serves the gRPC wire beside the websocket one.
    """
    running: list[tuple[PolicyServer, threading.Thread]] = []

    def start(pipeline, *, grpc: bool = False, **server_kwargs) -> Served:
        host = server_kwargs.pop('host', 'localhost')
        server = PolicyServer(pipeline, **server_kwargs)
        wires: list[wire.Wire] = [websocket_wire.WebsocketWire(host, 0, server.api)]
        if grpc:
            wires.append(grpc_wire.GrpcWire(host, 0))
        ready = threading.Event()
        thread = threading.Thread(target=server.serve, args=(wires, ready.set), daemon=True)
        thread.start()
        running.append((server, thread))
        if not ready.wait(timeout=10.0):
            raise RuntimeError('Server failed to start')
        return Served(host, wires[0].endpoint.port, server, wires[1].endpoint.port if grpc else None)

    yield start
    for server, thread in running:
        server.shutdown()
        thread.join(timeout=10.0)
        assert not thread.is_alive(), 'the server did not stop when asked'


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
    host, port, *_ = start_server(Pipeline(DictSource({'default': mock_model}), ChunkedSchedule(fps=10)))
    return host, port


@pytest.fixture
def multi_model_server(
    start_server: StartServer, mock_model_registry: dict[str, MagicMock]
) -> tuple[str, int, dict[str, MagicMock]]:
    host, port, *_ = start_server(Pipeline(DictSource(mock_model_registry), ChunkedSchedule(fps=10)))
    return host, port, mock_model_registry
