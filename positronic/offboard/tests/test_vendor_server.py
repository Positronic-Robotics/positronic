import asyncio
import socket
import threading
import time
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
import uvicorn

from positronic.offboard.client import InferenceClient
from positronic.offboard.vendor_server import VendorServer
from positronic.policy import Codec, RemotePolicy
from positronic.policy.codec import ActionTimestamp
from positronic.policy.spec import inline, remote
from positronic.policy.wrappers import ChunkedSchedule


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class _StubVendorServer(VendorServer):
    def __init__(self, definition=remote, **kwargs):
        super().__init__(definition=definition, **kwargs)
        self.mock_session = MagicMock()
        self.mock_session.return_value = [{'action': [1, 2, 3]}]
        self.mock_session.meta = {'model_name': 'stub'}
        self.mock_session.close = MagicMock()

        self.mock_policy = MagicMock()
        self.mock_policy.new_session.return_value = self.mock_session
        self.mock_policy.meta = {}
        self.metadata = {'type': 'stub'}
        self.warmup_called = False

    async def resolve_model(self, model_id, websocket):
        return 'dummy_handle', {'checkpoint_id': model_id or 'default'}

    def create_policy(self, model_handle):
        return self.mock_policy

    async def get_models(self):
        return {'models': ['stub']}

    async def warmup(self, policy):
        self.warmup_called = True


def _start_server(server: VendorServer) -> tuple[str, int, _StubVendorServer]:
    async def _run():
        await server._startup()
        config = uvicorn.Config(server.app, host=server.host, port=server.port, log_level='warning')
        await uvicorn.Server(config).serve()

    thread = threading.Thread(target=asyncio.run, args=(_run(),), daemon=True)
    thread.start()

    start = time.time()
    while time.time() - start < 5.0:
        try:
            with socket.create_connection((server.host, server.port), timeout=0.1):
                return server.host, server.port, server
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    raise RuntimeError('Server failed to start')


@pytest.fixture
def stub_server() -> Generator[tuple[str, int, _StubVendorServer], None, None]:
    yield _start_server(_StubVendorServer(host='localhost', port=find_free_port()))


def test_full_inference_cycle(stub_server):
    host, port, server = stub_server
    client = InferenceClient(host, port)
    session = client.new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        assert session.metadata['type'] == 'stub'
        assert session.metadata['local_stack'] == {'seq': []}
        assert 'positronic_version' in session.metadata

        obs = {'image': 'test'}
        result = session.infer(obs)
        assert result == [{'action': [1, 2, 3]}]
        server.mock_session.assert_called_with(obs)
    finally:
        session.close()


def test_warmup_called_on_startup(stub_server):
    _host, _port, server = stub_server
    assert server.warmup_called


def test_no_codec(stub_server):
    host, port, server = stub_server
    assert server._remote is None

    client = InferenceClient(host, port)
    session = client.new_session()
    try:
        result = session.infer({'obs': 'data'})
        assert result == [{'action': [1, 2, 3]}]
    finally:
        session.close()


def test_checkpoint_id_in_route(stub_server):
    host, port, server = stub_server
    client = InferenceClient(host, port)
    session = client.new_session('my_checkpoint')
    try:
        assert session.metadata['checkpoint_id'] == 'my_checkpoint'
    finally:
        session.close()


class _LatestTrackingServer(VendorServer):
    """Stub whose 'latest' checkpoint can change after startup; resolve_model(None)
    returns the current latest, mirroring real vendor servers."""

    def __init__(self, **kwargs):
        super().__init__(definition=remote, **kwargs)
        self.latest = '100'
        self.mock_session = MagicMock()
        self.mock_session.return_value = [{'action': [1, 2, 3]}]
        self.mock_session.meta = {}
        self.mock_session.close = MagicMock()
        self.mock_policy = MagicMock()
        self.mock_policy.new_session.return_value = self.mock_session
        self.mock_policy.meta = {}

    async def resolve_model(self, model_id, websocket):
        resolved = model_id if model_id is not None else self.latest
        return 'handle', {'checkpoint_id': resolved}

    def create_policy(self, model_handle):
        return self.mock_policy

    async def get_models(self):
        return {'models': [self.latest]}


def test_latest_checkpoint_pinned_once_at_startup():
    server = _LatestTrackingServer(host='localhost', port=find_free_port())
    host, port, _ = _start_server(server)
    # A newer checkpoint lands after startup (e.g. a training job writes it)...
    server.latest = '200'
    client = InferenceClient(host, port)
    # ...but a default session still serves the checkpoint pinned at startup.
    session = client.new_session()
    try:
        assert session.metadata['checkpoint_id'] == '100'
    finally:
        session.close()
    # Explicit requests still load the named checkpoint.
    session = client.new_session('200')
    try:
        assert session.metadata['checkpoint_id'] == '200'
    finally:
        session.close()


class _IdentityCodec(Codec):
    def encode(self, data):
        return data

    def _decode_single(self, data, context):
        return data

    @property
    def meta(self):
        return {'codec': 'identity'}

    def dummy_encoded(self, data=None):
        return data or {}


@pytest.fixture
def codec_server() -> Generator[tuple[str, int, _StubVendorServer], None, None]:
    definition = remote | _IdentityCodec()
    yield _start_server(_StubVendorServer(definition=definition, host='localhost', port=find_free_port()))


def test_codec_wrapping(codec_server):
    host, port, server = codec_server
    client = InferenceClient(host, port)
    session = client.new_session()
    try:
        assert session.metadata['codec'] == 'identity'
        result = session.infer({'obs': 'data'})
        assert result == [{'action': [1, 2, 3]}]
    finally:
        session.close()


@pytest.fixture
def declaring_server() -> Generator[tuple[str, int, _StubVendorServer], None, None]:
    definition = ChunkedSchedule() | remote | _IdentityCodec()
    yield _start_server(_StubVendorServer(definition=definition, host='localhost', port=find_free_port()))


def test_local_stack_declared_in_handshake(declaring_server):
    host, port, _server = declaring_server
    client = InferenceClient(host, port)
    session = client.new_session()
    try:
        assert session.metadata['local_stack'] == {'name': 'chunked_schedule'}
    finally:
        session.close()


class _ScriptedPolicy:
    """Deterministic base policy: every session returns the same untimestamped chunk."""

    def __init__(self):
        self.meta = {}

    def new_session(self, context=None, now=None):
        session = MagicMock()
        session.return_value = [{'a': 1.0}, {'a': 2.0}, {'a': 3.0}]
        session.meta = {}
        return session

    def close(self):
        pass


def test_in_process_equals_remote_for_same_definition():
    """The same definition must behave identically served in-process and over the wire."""

    def definition():
        return ChunkedSchedule() | remote | ActionTimestamp(fps=10.0)

    clock = [100.0]

    server = _StubVendorServer(definition=definition(), host='localhost', port=find_free_port())
    server.create_policy = lambda handle: _ScriptedPolicy()
    host, port, _ = _start_server(server)
    remote_session = RemotePolicy(host, port).new_session(now=lambda: clock[0])

    local_session = inline(definition()).wrap(_ScriptedPolicy()).new_session(now=lambda: clock[0])

    remote_actions = remote_session({'obs_time_ns': 0})
    local_actions = local_session({'obs_time_ns': 0})
    assert remote_actions == local_actions
    assert [a['timestamp'] for a in local_actions] == [100.0, 100.1, 100.2]

    # Both gate identically while the chunk plays out.
    clock[0] = 100.15
    assert remote_session({'obs_time_ns': 0}) is None
    assert local_session({'obs_time_ns': 0}) is None

    remote_session.close()
