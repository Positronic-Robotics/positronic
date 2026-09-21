import asyncio
import logging
import os
import socket
import threading
import time
import urllib.parse
from collections.abc import Callable, Generator
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock, patch

import configuronic as cfn
import httpx
import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response
from websockets.sync.client import connect

from positronic.offboard import keys as offboard_keys
from positronic.offboard import protocol, websocket_wire, wire
from positronic.offboard.client import InferenceClient, InferenceSession, _ConnectRetries
from positronic.offboard.protocol import deserialise, serialise
from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV, PolicyServer, bearer
from positronic.offboard.server_utils import warmup
from positronic.offboard.spec import Model, ModelSource, PolicyDeployment
from positronic.offboard.tests.conftest import DictSource
from positronic.offboard.websocket_wire import WebsocketClientConnection
from positronic.policy import Codec
from positronic.policy.codec import ActionTimestamp
from positronic.policy.layers import ChunkedSchedule, TemporalStack
from positronic.policy.sequential import Sequential


class _StubSource(ModelSource):
    """Serves one loaded model under any requested id, so route-supplied checkpoints resolve as-is."""

    def __init__(self, policy: Model, name: str = 'stub'):
        self._policy = policy
        self._name = name

    def get_models(self) -> list[str]:
        return [self._name]

    def resolve(self, model_id: str | None) -> str:
        return model_id if model_id is not None else self._name

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model:
        return self._policy

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'type': 'stub'}


# Short enough for a quick test, long enough that a loaded box reaches the first poll.
_A_MOMENT_IDLE = 0.5


class _FailingWire(wire.Wire):
    """Serves for ``after`` seconds, then raises."""

    def __init__(self, after: float):
        self._after = after
        self.stopped = False

    @property
    def endpoint(self) -> wire.Endpoint:
        return wire.Endpoint('localhost', 0)

    async def start(self, session: wire.SessionHandler, authorized: wire.Authorized) -> None:
        pass

    async def serve(self) -> None:
        await asyncio.sleep(self._after)
        raise RuntimeError(f'the {self._after}s wire fell over')

    async def stop(self) -> None:
        self.stopped = True


class _UnbindableWire(wire.Wire):
    """A wire whose port is taken."""

    @property
    def endpoint(self) -> wire.Endpoint:
        raise AssertionError('it never bound')

    async def start(self, session: wire.SessionHandler, authorized: wire.Authorized) -> None:
        raise OSError('that port is taken')

    async def serve(self) -> None:
        raise AssertionError('it never served')

    async def stop(self) -> None:
        pass


def test_a_server_with_no_wire_refuses_to_serve(make_mock_model):
    """A server that binds nothing answers nobody, so it raises instead of reporting itself ready."""
    server = PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)))
    with pytest.raises(ValueError, match='at least one wire'):
        server.serve([], on_ready=lambda: pytest.fail('it reported ready with no wire bound'))


def test_a_wire_that_cannot_bind_stops_the_ones_that_did(make_mock_model):
    """A wire binds when it starts, and a startup that gives up frees the port an earlier wire took."""
    server = PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)))
    bound = _FailingWire(_A_MOMENT_IDLE)
    with pytest.raises(OSError, match='that port is taken'):
        server.serve([bound, _UnbindableWire()])
    assert bound.stopped, 'the wire that had bound was left holding its port'


def _rebind_and_release(host: str, port: int) -> None:
    """Bind ``host`` on ``port`` and let it go again. It raises while anything else holds the port."""
    for sock in websocket_wire._listening_sockets(host, port):
        sock.close()


def test_a_websocket_wire_releases_its_port_when_startup_rolls_back(make_mock_model):
    """A ``WebsocketWire`` binds a real socket when it starts, and a startup that rolls back frees it."""
    server = PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)))
    bound = websocket_wire.WebsocketWire('localhost', 0, server.api)
    with pytest.raises(OSError, match='that port is taken'):
        server.serve([bound, _UnbindableWire()])
    # A leaked listener holds the port, and a fresh bind to it raises.
    _rebind_and_release('localhost', bound.endpoint.port)


def test_a_websocket_wire_served_once_still_releases_its_port_on_a_later_rollback(make_mock_model):
    """A wire that served and stopped starts again with a fresh socket, and a rollback before it serves frees it."""
    server = PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)))
    bound = websocket_wire.WebsocketWire('localhost', 0, server.api)
    serving = threading.Thread(target=server.serve, args=([bound],))
    serving.start()
    time.sleep(_A_MOMENT_IDLE)
    server.shutdown()
    serving.join(timeout=10.0)
    assert not serving.is_alive(), 'the first serve did not end'
    with pytest.raises(OSError, match='that port is taken'):
        server.serve([bound, _UnbindableWire()])
    _rebind_and_release('localhost', bound.endpoint.port)


def test_a_host_with_two_addresses_binds_each_of_them_on_one_port(monkeypatch):
    """A dual-stack host answers on both families, so a client reaches it at either address."""

    def both_loopbacks(host, port, *_args, **_kwargs):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('::1', port, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('127.0.0.1', port)),
        ]

    monkeypatch.setattr(socket, 'getaddrinfo', both_loopbacks)
    sockets = websocket_wire._listening_sockets('dual-stack.test', 0)
    try:
        assert [sock.getsockname()[0] for sock in sockets] == ['::1', '127.0.0.1']
        assert len({sock.getsockname()[1] for sock in sockets}) == 1, 'the two addresses took different ports'
    finally:
        for sock in sockets:
            sock.close()


def test_an_address_resolved_twice_binds_once(monkeypatch):
    """A second bind on the same address fails the whole set, so a repeated answer counts once."""

    def loopback_twice(host, port, *_args, **_kwargs):
        entry = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', ('127.0.0.1', port))
        return [entry, entry]

    monkeypatch.setattr(socket, 'getaddrinfo', loopback_twice)
    sockets = websocket_wire._listening_sockets('twice.test', 0)
    try:
        assert [sock.getsockname()[0] for sock in sockets] == ['127.0.0.1']
    finally:
        for sock in sockets:
            sock.close()


def test_a_host_with_one_address_binds_one_socket_and_names_the_port_it_took(make_mock_model):
    server = PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)))
    bound = websocket_wire.WebsocketWire('127.0.0.1', 0, server.api)
    asyncio.run(bound.start(MagicMock(), lambda _headers: True))
    try:
        assert len(bound._sockets) == 1
        assert bound.endpoint.port == bound._sockets[0].getsockname()[1] != 0
    finally:
        asyncio.run(bound.stop())
    _rebind_and_release('127.0.0.1', bound.endpoint.port)


def test_a_failing_wire_reaches_the_caller_and_the_rest_are_logged(make_mock_model, caplog):
    """No wire ends in silence: one failure raises out of ``serve``, and ``serve`` logs every other one."""
    server = PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)))
    with caplog.at_level(logging.ERROR, logger='positronic.offboard.server'):
        with pytest.raises(RuntimeError, match='the 0.05s wire fell over'):
            server.serve([_FailingWire(0.05), _FailingWire(0.1)])
    assert any('the 0.1s wire fell over' in record.getMessage() for record in caplog.records)


def test_an_idle_server_stops_itself(make_mock_model):
    """The idle watchdog ends every wire, and ``serve`` returns with no ``shutdown`` call."""
    server = PolicyServer(
        PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)),
        idle_timeout_min=_A_MOMENT_IDLE / 60,
    )
    serving = threading.Thread(target=server.serve, args=([websocket_wire.WebsocketWire('localhost', 0, server.api)],))
    serving.start()
    serving.join(timeout=_A_MOMENT_IDLE * 20)
    assert not serving.is_alive(), 'the idle watchdog left the server running'


@pytest.fixture
def stub_server(start_server, make_mock_model) -> tuple[str, int, PolicyServer, MagicMock]:
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, server, _ = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)))
    return host, port, server, policy


def test_full_inference_cycle(stub_server):
    host, port, _server, policy = stub_server
    client = InferenceClient.from_url(f'{host}:{port}')
    session = client.new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        assert session.metadata['type'] == 'stub'
        assert session.metadata['local_stack'] == {'name': 'chunked_schedule', 'args': {'fps': 10}}
        assert offboard_keys.POSITRONIC_VERSION in session.metadata

        obs = {'image': 'test'}
        result = session.infer(obs)
        assert result == [{'action': [1, 2, 3]}]
        policy.assert_called_with(obs)
    finally:
        session.close()


def test_no_codec(stub_server):
    host, port, _server, _policy = stub_server
    client = InferenceClient.from_url(f'{host}:{port}')
    session = client.new_session()
    try:
        result = session.infer({'obs': 'data'})
        assert result == [{'action': [1, 2, 3]}]
    finally:
        session.close()


@pytest.mark.parametrize(
    'checkpoint_id',
    [
        'my_checkpoint',
        'GEAR-Dreams/DreamZero-DROID',
        's3://bucket/ckpt-1',
        's3://bucket/checkpoint-500/',
        'weird?x#y',
        '100%done',
    ],
)
def test_checkpoint_id_in_route(stub_server, checkpoint_id):
    host, port, _server, _policy = stub_server
    # ``safe='/'`` keeps a path-shaped id's separators as path segments, and encodes the characters that
    # would otherwise end the path (``?``, ``#``) or be decoded away (``%``).
    quoted = urllib.parse.quote(checkpoint_id, safe='/')
    client = InferenceClient.from_url(f'{host}:{port}/api/v1/session/{quoted}')
    session = client.new_session()
    try:
        assert session.metadata['checkpoint_id'] == checkpoint_id
    finally:
        session.close()


class _LatestSource(ModelSource):
    """Source whose 'latest' checkpoint can change after startup; ``resolve(None)``
    returns the current latest, mirroring real vendor sources."""

    def __init__(self, policy: Model):
        self._policy = policy
        self.latest = '100'

    def get_models(self) -> list[str]:
        return [self.latest]

    def resolve(self, model_id: str | None) -> str:
        return model_id if model_id is not None else self.latest

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model:
        return self._policy


def test_latest_checkpoint_pinned_once_at_startup(start_server, make_mock_model):
    source = _LatestSource(make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'}))
    host, port, *_ = start_server(PolicyDeployment(source, ChunkedSchedule(fps=10)))
    # A newer checkpoint lands after startup (e.g. a training job writes it)...
    source.latest = '200'
    client = InferenceClient.from_url(f'{host}:{port}')
    # ...but a default session still serves the checkpoint pinned at startup.
    session = client.new_session()
    try:
        assert session.metadata['checkpoint_id'] == '100'
    finally:
        session.close()
    # Explicit requests still load the named checkpoint.
    session = InferenceClient.from_url(f'{host}:{port}/api/v1/session/200').new_session()
    try:
        assert session.metadata['checkpoint_id'] == '200'
    finally:
        session.close()


class _ProgressSource(_StubSource):
    """Reports load progress, so switching models exercises the ``loading`` frame stream."""

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model:
        if on_progress is not None:
            on_progress('halfway there')
        return self._policy


def test_load_progress_frames_reach_the_client(start_server, make_mock_model):
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, *_ = start_server(PolicyDeployment(_ProgressSource(policy), ChunkedSchedule(fps=10)))
    # Requesting a non-pinned id forces a load inside the handshake; the source's progress
    # callbacks must arrive as ``loading`` frames before ``ready``.
    ws = connect(f'ws://{host}:{port}/api/v1/session/other')
    try:
        frames = []
        while not any(f.get('status') == 'ready' for f in frames):
            frames.append(deserialise(ws.recv(timeout=10)))
        messages = [f.get('message', '') for f in frames if f.get('status') == 'loading']
        assert any('halfway there' in m for m in messages)
    finally:
        ws.close()


def test_a_client_that_leaves_mid_inference_is_a_lost_peer_not_an_error(stub_server, caplog):
    """The answer meets a closed socket; the wire reports a lost peer, and the server logs no error."""
    host, port, _server, policy = stub_server
    policy.side_effect = lambda *_: time.sleep(0.3) or [{'action': [1, 2, 3]}]
    ws = connect(f'ws://{host}:{port}/api/v1/session')
    while deserialise(ws.recv(timeout=10)).get('status') != 'ready':
        pass
    with caplog.at_level(logging.INFO, logger='positronic.offboard.server'):
        ws.send(serialise({'image': 'test'}))
        ws.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any('Client disconnected' in r.getMessage() for r in caplog.records):
            time.sleep(0.05)
    assert any('Client disconnected' in r.getMessage() for r in caplog.records)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


class _IdentityCodec(Codec):
    def encode(self, data):
        return data

    def _decode_single(self, data):
        return data

    @property
    def meta(self):
        return {'codec': 'identity'}


@pytest.fixture
def codec_server(start_server, make_mock_model) -> tuple[str, int, MagicMock]:
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, *_ = start_server(
        PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10), codec=_IdentityCodec())
    )
    return host, port, policy


def test_codec_wrapping(codec_server):
    host, port, _policy = codec_server
    client = InferenceClient.from_url(f'{host}:{port}')
    session = client.new_session()
    try:
        assert session.metadata['codec'] == 'identity'
        result = session.infer({'obs': 'data'})
        assert result == [{'action': [1, 2, 3]}]
    finally:
        session.close()


def test_a_failed_inference_leaves_no_served_timing_behind(stub_server):
    """The timing block belongs to the answer it arrived with; a failed round trip has none."""
    host, port, _server, policy = stub_server
    policy.side_effect = [[{'action': [1, 2, 3]}], RuntimeError('shape mismatch'), [{'action': [1, 2, 3]}]]
    session = InferenceClient.from_url(f'{host}:{port}').new_session()
    try:
        session.infer({'image': 'test'})
        assert protocol.TIMING_SERVED in session.served_timing
        with pytest.raises(RuntimeError, match='shape mismatch'):
            session.infer({'image': 'test'})
        assert session.served_timing == {}
        session.infer({'image': 'test'})
        with pytest.raises(TypeError):
            session.infer({'image': object()})
        assert session.served_timing == {}
    finally:
        session.close()


def test_warmup_calls_the_model_without_closing_it(make_mock_model):
    policy = make_mock_model([{'action': [1, 2, 3]}], {})
    obs = {'obs': 'zeros'}

    warmup(policy, obs)

    policy.assert_called_once_with(obs)
    policy.close.assert_not_called()


def test_warmup_failure_propagates_without_closing_the_model(make_mock_model):
    policy = make_mock_model([], {})
    policy.side_effect = RuntimeError('shape mismatch')

    with pytest.raises(RuntimeError, match='shape mismatch'):
        warmup(policy, {})

    policy.close.assert_not_called()


def test_local_stack_declared_in_handshake(start_server, make_mock_model):
    stub = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    pipeline = PolicyDeployment(_StubSource(stub), ChunkedSchedule(fps=10), codec=_IdentityCodec())
    host, port, *_ = start_server(pipeline)
    client = InferenceClient.from_url(f'{host}:{port}')
    session = client.new_session()
    try:
        assert session.metadata['local_stack'] == {'name': 'chunked_schedule', 'args': {'fps': 10}}
    finally:
        session.close()


class _ScriptedModel(Model):
    """A model returning the same untimestamped chunk on every call."""

    def __call__(self, obs):
        return [{'a': 1.0}, {'a': 2.0}, {'a': 3.0}]


def _tunable_pipe(source: ModelSource, offsets: tuple[float, ...] = (-0.1, 0.0), pad_start: bool = True):
    return PolicyDeployment(
        source,
        Sequential(TemporalStack(keys=('x',), offsets_sec=offsets, pad_start=pad_start), ChunkedSchedule(fps=10)),
    )


def _param_session(host: str, port: int, query: list[tuple[str, str]]) -> InferenceSession:
    uri = f'ws://{host}:{port}/api/v1/session?' + urllib.parse.urlencode(query)
    return InferenceSession(WebsocketClientConnection(connect(uri)))


@pytest.fixture
def param_server(start_server, make_mock_model) -> Generator[tuple[str, int], None, None]:
    stub = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    pipe_cfg = cfn.Config(_tunable_pipe, source=cfn.Config(_StubSource, policy=stub))
    host, port, *_ = start_server(pipe_cfg)
    yield host, port


def test_session_params_override_declared_local_stack(param_server):
    host, port = param_server
    session = _param_session(host, port, [('offsets', '[-0.5, 0.0]')])
    try:
        stack = session.metadata['local_stack']['seq'][0]
        assert stack['name'] == 'temporal_stack'
        assert stack['args']['offsets_sec'] == [-0.5, 0.0]
    finally:
        session.close()


def test_session_params_coerce_json_values(param_server):
    host, port = param_server
    session = _param_session(host, port, [('pad_start', 'false')])
    try:
        assert session.metadata['local_stack']['seq'][0]['args']['pad_start'] is False
    finally:
        session.close()


def _fps_pipe(source: ModelSource, fps: float = 10.0):
    return PolicyDeployment(source, ChunkedSchedule(fps=10), codec=ActionTimestamp(fps=fps))


def test_session_param_retunes_the_served_remote_half(start_server):
    pipe_cfg = cfn.Config(_fps_pipe, source=cfn.Config(DictSource, models={'default': _ScriptedModel()}))
    host, port, *_ = start_server(pipe_cfg)

    # The wire carries the server-side half's output: relative timestamps spaced 1/fps.
    default_session = _param_session(host, port, [])
    tuned_session = _param_session(host, port, [('fps', '5')])
    try:
        assert [a['timestamp'] for a in default_session.infer({})] == pytest.approx([0.0, 0.1, 0.2, 0.3])
        assert [a['timestamp'] for a in tuned_session.infer({})] == pytest.approx([0.0, 0.2, 0.4, 0.6])
    finally:
        default_session.close()
        tuned_session.close()


def test_model_id_is_named_by_path_not_query(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='model_id'):
        _param_session(host, port, [('model_id', 'other')])

    uri = f'ws://{host}:{port}/api/v1/session/other?pad_start=false'
    session = InferenceSession(WebsocketClientConnection(connect(uri)))
    try:
        assert session.metadata['checkpoint_id'] == 'other'
        assert session.metadata['local_stack']['seq'][0]['args']['pad_start'] is False
    finally:
        session.close()


def test_unknown_session_param_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='nonexistent'):
        _param_session(host, port, [('nonexistent', '1')])


def test_import_string_session_params_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='import syntax'):
        _param_session(host, port, [('pad_start', '"@os.system"')])
    # The relative form is no safer: leading dots walk up the module tree from the key's current
    # value, and `source` holds a config, so they resolve.
    with pytest.raises(RuntimeError, match='import syntax'):
        _param_session(host, port, [('source', '".....os.system"')])
    # Nested values are refused too, and the error names the position inside the value.
    with pytest.raises(RuntimeError, match=r'offsets\[0\]'):
        _param_session(host, port, [('offsets', '["@os.getcwd"]')])


def test_dotted_param_is_data_where_it_could_not_import(param_server):
    """A leading-dot value stays a plain string on a key that gives imports no base to resolve
    against — so ordinary path-ish values are usable as data."""
    host, port = param_server
    session = _param_session(host, port, [('pad_start', '"./data"')])
    try:
        assert session.metadata['local_stack']['seq'][0]['args']['pad_start'] == './data'
    finally:
        session.close()


def test_source_touching_session_param_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='fixed at launch'):
        _param_session(host, port, [('source.name', '"other"')])


def test_plain_pipe_server_rejects_session_params(start_server, make_mock_model):
    stub = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, *_ = start_server(_tunable_pipe(_StubSource(stub)))
    with pytest.raises(RuntimeError, match='config-launched'):
        _param_session(host, port, [('pad_start', 'false')])


def test_duplicate_session_param_keys_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='[Dd]uplicate'):
        _param_session(host, port, [('pad_start', 'false'), ('pad_start', 'true')])


_TOKEN = 'test-secret-token'

# The deployed endpoint the ``endpoint`` marker's tests address. Unset, those tests serve their own server
# and prove its behaviour; set, the same assertions run through whatever ingress fronts that deployment,
# which is the only place the two can disagree.
ENDPOINT_URL_ENV = 'POSITRONIC_ENDPOINT_URL'
_LIVE_ENDPOINT = os.environ.get(ENDPOINT_URL_ENV)


@pytest.fixture
def authed_endpoint(start_server, make_mock_model) -> tuple[str, str]:
    """An authenticated server's URL, and the token gating it."""
    if _LIVE_ENDPOINT:
        return _LIVE_ENDPOINT, os.environ[AUTH_TOKEN_ENV]
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, *_ = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)), auth_token=_TOKEN)
    return f'{host}:{port}', _TOKEN


@pytest.mark.endpoint
@pytest.mark.parametrize(
    'make_header',
    [
        pytest.param(lambda token: None, id='absent'),
        pytest.param(lambda token: bearer(f'not-{token}'), id='wrong-token'),
        pytest.param(lambda token: token, id='no-bearer-prefix'),
    ],
)
def test_auth_rejects_requests_without_the_token(authed_endpoint, make_header, monkeypatch):
    # A 403 buys retries for a backend that may be merely cold. This one is refusing, so those attempts and
    # the waits between them are dead time; `TestNewSessionRetriesRefusedConnects` tests the budget.
    monkeypatch.setattr(_ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    url, token = authed_endpoint
    header = make_header(token)
    client = InferenceClient.from_url(url, headers=None if header is None else {AUTH_HEADER: header})
    with pytest.raises(wire.ConnectRefused) as refused:
        client.new_session()
    assert refused.value.refusal is wire.Refusal.FORBIDDEN
    with pytest.raises(httpx.HTTPStatusError):
        client.list_models()


@pytest.mark.parametrize(
    ('status', 'refusal'),
    [
        (HTTPStatus.FORBIDDEN, wire.Refusal.FORBIDDEN),
        (HTTPStatus.TOO_MANY_REQUESTS, wire.Refusal.COLD),
        (HTTPStatus.SERVICE_UNAVAILABLE, wire.Refusal.COLD),
        (HTTPStatus.BAD_GATEWAY, wire.Refusal.COLD),
        (HTTPStatus.UNAUTHORIZED, wire.Refusal.FINAL),
        (HTTPStatus.NOT_FOUND, wire.Refusal.FINAL),
    ],
)
def test_a_non_101_answer_to_the_upgrade_says_what_the_server_is(status, refusal):
    refused_upgrade = InvalidStatus(Response(status, 'refused', Headers()))
    with (
        patch('positronic.offboard.websocket_wire.connect', side_effect=refused_upgrade),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        address = wire.SessionAddress('localhost', 8000, wire.SESSION_PATH, '', secure=False)
        websocket_wire.WebsocketClientWire().dial(address, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is refused_upgrade


@pytest.mark.endpoint
def test_auth_accepts_the_token(authed_endpoint):
    url, token = authed_endpoint
    client = InferenceClient.from_url(url, headers={AUTH_HEADER: bearer(token)})
    assert client.list_models()
    session = client.new_session()
    try:
        # Reaching the handshake metadata means the upgrade completed and the server's first frame arrived.
        # An ingress that drops ``Upgrade`` never gets that far: it answers the handshake with a plain 200.
        assert offboard_keys.POSITRONIC_VERSION in session.metadata
    finally:
        session.close()


# A managed ingress closes a connection it has read nothing from — Nebius' does after ~90s, shorter than one
# cold inference. Nothing crosses the wire while a session waits for actions except the client's keepalive
# pings, so idling past that window and still getting a pong is those pings doing their job.
_IDLE_WINDOW_SEC = 120.0


@pytest.mark.endpoint
@pytest.mark.skipif(not _LIVE_ENDPOINT, reason=f'no ingress to idle against; set {ENDPOINT_URL_ENV}')
def test_session_outlives_an_idle_ingress_window(authed_endpoint):
    url, token = authed_endpoint
    session = InferenceClient.from_url(url, headers={AUTH_HEADER: bearer(token)}).new_session()
    try:
        time.sleep(_IDLE_WINDOW_SEC)
        conn = session._conn
        assert isinstance(conn, WebsocketClientConnection), "the idle window is the websocket wire's"
        assert conn._websocket.ping().wait(timeout=30.0)
    finally:
        session.close()


def test_server_without_a_token_serves_open(stub_server):
    host, port, _server, _policy = stub_server
    assert InferenceClient.from_url(f'{host}:{port}').list_models() == ['stub']


@pytest.mark.parametrize(
    'token',
    [
        pytest.param('', id='empty'),
        pytest.param('tökén', id='non-ascii'),
        # What a secret read from a file that ends in one looks like.
        pytest.param('a-token\n', id='trailing-newline'),
        pytest.param('a token', id='space'),
    ],
)
def test_a_token_that_could_never_gate_fails_closed_at_startup(make_mock_model, token):
    with pytest.raises(ValueError, match='ASCII'):
        PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)), auth_token=token)


def test_a_non_ascii_authorization_header_is_refused_rather_than_crashing(start_server, make_mock_model):
    """A header carries bytes, and Starlette hands them over latin-1 decoded, so a peer can put a
    non-ASCII ``str`` in front of the token comparison."""
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, *_ = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)), auth_token=_TOKEN)
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall(
            b'GET /api/v1/models HTTP/1.1\r\nHost: localhost\r\n'
            b'Authorization: Bearer t\xf6ken\r\nConnection: close\r\n\r\n'
        )
        status = sock.recv(64).split(b' ')[1]
    assert status == b'401'


def test_shutdown_closes_the_loaded_model(start_server, make_mock_model):
    model = make_mock_model([], {})
    closed = threading.Event()
    model.close.side_effect = closed.set
    served = start_server(PolicyDeployment(_StubSource(model), ChunkedSchedule(fps=10)))
    session = InferenceClient.from_url(f'{served.host}:{served.port}').new_session()
    session.close()
    model.close.assert_not_called()
    served.server.shutdown()
    assert closed.wait(timeout=5)
    model.close.assert_called_once()
