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
from positronic.policy import Codec, Policy


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class _StubVendorServer(VendorServer):
    def __init__(self, codec=None, **kwargs):
        super().__init__(codec=codec, **kwargs)
        self.mock_policy = MagicMock(spec=Policy)
        self.mock_policy.select_action.return_value = [{'action': [1, 2, 3]}]
        self.mock_policy.meta = {'model_name': 'stub'}
        self.mock_policy.reset.return_value = None
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

        obs = {'image': 'test'}
        result = session.infer(obs)
        assert result == [{'action': [1, 2, 3]}]
        server.mock_policy.select_action.assert_called_with(obs)
    finally:
        session.close()


def test_warmup_called_on_startup(stub_server):
    _host, _port, server = stub_server
    assert server.warmup_called


def test_no_codec(stub_server):
    host, port, server = stub_server
    assert server.codec is None

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
    yield _start_server(_StubVendorServer(codec=_IdentityCodec(), host='localhost', port=find_free_port()))


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
