import asyncio
import errno
import logging
import os
import pathlib
import socket
import stat
import threading
import time
import urllib.parse
from collections.abc import Callable, Generator
from unittest.mock import MagicMock, patch

import configuronic as cfn
import pytest
from fastapi import APIRouter
from positronic_wire import registry, wire
from positronic_wire import websocket as client_websocket
from positronic_wire.websocket import WebsocketClientConnection
from websockets.exceptions import ConnectionClosedOK
from websockets.sync.client import connect, unix_connect

from positronic.offboard import keys as offboard_keys
from positronic.offboard import protocol, server_wire, websocket_wire
from positronic.offboard.client import ConnectRetries, InferenceClient, InferenceSession
from positronic.offboard.protocol import deserialise, serialise
from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV, PolicyServer, bearer
from positronic.offboard.server_utils import warmup
from positronic.offboard.spec import Model, ModelSource, PolicyDeployment
from positronic.offboard.tests.conftest import DictSource, Served
from positronic.policy import Codec
from positronic.policy.base import ARGS
from positronic.policy.processors import ChunkedSchedule, TemporalStack
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


# Short enough for a quick test, long enough that a loaded box reaches the first poll.
_A_MOMENT_IDLE = 0.5


def _bound_port(bound: websocket_wire.WebsocketWire) -> int:
    """The port a wire bound. A wire serving a socket bound none, and no test here asks one for a port."""
    served = bound.served_address
    assert isinstance(served, server_wire.ServedHostPort), 'the wire bound a socket, not a port'
    return served.port


class _FailingWire(server_wire.Wire):
    """Serves for ``after`` seconds, then raises."""

    def __init__(self, after: float):
        self._after = after
        self.stopped = False

    @property
    def served_address(self) -> server_wire.ServedHostPort:
        return server_wire.ServedHostPort('localhost', 0)

    async def start(
        self, session: server_wire.SessionHandler, authorized: server_wire.Authorized, api: APIRouter
    ) -> None:
        pass

    async def serve(self) -> None:
        await asyncio.sleep(self._after)
        raise RuntimeError(f'the {self._after}s wire fell over')

    async def stop(self) -> None:
        self.stopped = True


class _UnbindableWire(server_wire.Wire):
    """A wire whose port is taken."""

    @property
    def served_address(self) -> server_wire.ServedHostPort:
        raise AssertionError('it never bound')

    async def start(
        self, session: server_wire.SessionHandler, authorized: server_wire.Authorized, api: APIRouter
    ) -> None:
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
    bound = websocket_wire.WebsocketWire(server_wire.ServedHostPort('localhost', 0))
    with pytest.raises(OSError, match='that port is taken'):
        server.serve([bound, _UnbindableWire()])
    # A leaked listener holds the port, and a fresh bind to it raises.
    _rebind_and_release('localhost', _bound_port(bound))


def test_a_websocket_wire_served_once_still_releases_its_port_on_a_later_rollback(make_mock_model):
    """A wire that served and stopped starts again with a fresh socket, and a rollback before it serves frees it."""
    server = PolicyServer(PolicyDeployment(_StubSource(make_mock_model([], {})), ChunkedSchedule(fps=10)))
    bound = websocket_wire.WebsocketWire(server_wire.ServedHostPort('localhost', 0))
    serving = threading.Thread(target=server.serve, args=([bound],))
    serving.start()
    time.sleep(_A_MOMENT_IDLE)
    server.shutdown()
    serving.join(timeout=10.0)
    assert not serving.is_alive(), 'the first serve did not end'
    with pytest.raises(OSError, match='that port is taken'):
        server.serve([bound, _UnbindableWire()])
    _rebind_and_release('localhost', _bound_port(bound))


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
    bound = websocket_wire.WebsocketWire(server_wire.ServedHostPort('127.0.0.1', 0))
    asyncio.run(bound.start(MagicMock(), lambda _headers: True, server.api))
    try:
        assert len(bound._sockets) == 1
        assert _bound_port(bound) == bound._sockets[0].getsockname()[1] != 0
    finally:
        asyncio.run(bound.stop())
    _rebind_and_release('127.0.0.1', _bound_port(bound))


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
    serving = threading.Thread(
        target=server.serve, args=([websocket_wire.WebsocketWire(server_wire.ServedHostPort('localhost', 0))],)
    )
    serving.start()
    serving.join(timeout=_A_MOMENT_IDLE * 20)
    assert not serving.is_alive(), 'the idle watchdog left the server running'


@pytest.fixture
def stub_server(start_server, make_mock_model) -> tuple[str, int, PolicyServer, MagicMock]:
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'})
    host, port, server, *_ = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)))
    return host, port, server, policy


def test_full_inference_cycle(stub_server):
    host, port, _server, policy = stub_server
    client = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    )
    session = client.new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        assert session.metadata['type'] == 'stub'
        assert session.metadata['local_stack'] == {'name': 'chunked_schedule', 'version': 2, 'args': {'fps': 10}}
        assert offboard_keys.POSITRONIC_VERSION in session.metadata

        obs = {'image': 'test'}
        result = session.infer(obs)
        assert result == [{'action': [1, 2, 3]}]
        policy.assert_called_with(obs, session_id=session.session_id)
    finally:
        session.close()


def test_the_server_negotiates_no_deflate_with_a_client_that_offers_it(stub_server):
    """A stock websockets client offers permessage-deflate, and the session still opens uncompressed."""
    host, port, *_ = stub_server
    with connect(f'ws://{host}:{port}{wire.SESSION_PATH}') as ws:
        assert ws.protocol.extensions == []
        assert deserialise(ws.recv(timeout=10))[protocol.STATUS] == protocol.ServerStatus.READY


def test_no_codec(stub_server):
    host, port, _server, _policy = stub_server
    client = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    )
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
    client = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.session_path(checkpoint_id), '')
    )
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
    source = _LatestSource(make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'}))
    host, port, *_ = start_server(PolicyDeployment(source, ChunkedSchedule(fps=10)))
    # A newer checkpoint lands after startup (e.g. a training job writes it)...
    source.latest = '200'
    client = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    )
    # ...but a default session still serves the checkpoint pinned at startup.
    session = client.new_session()
    try:
        assert session.metadata['checkpoint_id'] == '100'
    finally:
        session.close()
    # Explicit requests still load the named checkpoint.
    session = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.session_path('200'), '')
    ).new_session()
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
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'})
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
    finished = threading.Event()
    ended = threading.Event()

    def infer(obs, *, session_id):
        time.sleep(0.3)
        finished.set()
        return [{'action': [1, 2, 3]}]

    def end_session(session_id):
        assert finished.is_set()
        ended.set()

    policy.side_effect = infer
    policy.end_session.side_effect = end_session
    ws = connect(f'ws://{host}:{port}/api/v1/session')
    while (ready := deserialise(ws.recv(timeout=10))).get(protocol.STATUS) != protocol.ServerStatus.READY:
        pass
    with caplog.at_level(logging.INFO, logger='positronic.offboard.server'):
        ws.send(serialise({protocol.SESSION_ID: ready[protocol.SESSION_ID], protocol.OBSERVATION: {'image': 'test'}}))
        ws.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any('Client disconnected' in r.getMessage() for r in caplog.records):
            time.sleep(0.05)
    assert any('Client disconnected' in r.getMessage() for r in caplog.records)
    assert ended.wait(timeout=5)
    policy.end_session.assert_called_once_with(ready[protocol.SESSION_ID])
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
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'})
    host, port, *_ = start_server(
        PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10), codec=_IdentityCodec())
    )
    return host, port, policy


def test_codec_wrapping(codec_server):
    host, port, _policy = codec_server
    client = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    )
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
    session = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    ).new_session()
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

    session_id = policy.call_args.kwargs['session_id']
    policy.assert_called_once_with(obs, session_id=session_id)
    policy.end_session.assert_called_once_with(session_id)
    policy.close.assert_not_called()


def test_warmup_failure_propagates_without_closing_the_model(make_mock_model):
    policy = make_mock_model([], {})
    policy.side_effect = RuntimeError('shape mismatch')

    with pytest.raises(RuntimeError, match='shape mismatch'):
        warmup(policy, {})

    policy.end_session.assert_called_once_with(policy.call_args.kwargs['session_id'])
    policy.close.assert_not_called()


def test_local_stack_declared_in_handshake(start_server, make_mock_model):
    stub = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'})
    pipeline = PolicyDeployment(_StubSource(stub), ChunkedSchedule(fps=10), codec=_IdentityCodec())
    host, port, *_ = start_server(pipeline)
    client = InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    )
    session = client.new_session()
    try:
        assert session.metadata['local_stack'] == {'name': 'chunked_schedule', 'version': 2, 'args': {'fps': 10}}
    finally:
        session.close()


@pytest.fixture
def unix_stub_server(start_server, socket_path, make_mock_model) -> tuple[Served, MagicMock]:
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)), uds=socket_path)
    return served, policy


def test_a_pipeline_served_over_a_unix_socket(unix_stub_server, socket_path):
    served, policy = unix_stub_server
    client = InferenceClient(*served.unix())

    assert client.list_models() == ['stub']
    session = client.new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        assert session.metadata[offboard_keys.LOCAL_STACK] == {
            'name': 'chunked_schedule',
            'version': 2,
            'args': {'fps': 10},
        }
        assert session.metadata[offboard_keys.UDS] == socket_path
        assert offboard_keys.HOST not in session.metadata
        assert offboard_keys.PORT not in session.metadata

        obs = {'image': 'test'}
        assert session.infer(obs) == [{'action': [1, 2, 3]}]
        policy.assert_called_with(obs, session_id=session.session_id)
    finally:
        session.close()


def test_a_session_over_a_socket_carries_the_model_id(unix_stub_server):
    """The route is the address's, not a URL's, so a socket carries a model id like any other wire."""
    served, _policy = unix_stub_server

    session = InferenceClient(*served.unix(model='10000')).new_session()
    try:
        assert session.metadata[offboard_keys.CHECKPOINT_ID] == '10000'
    finally:
        session.close()


@pytest.mark.parametrize('model_id', ['', 'stub'])
@pytest.mark.parametrize('client_closes', [True, False])
def test_session_end_leaves_time_to_read_the_ack(unix_stub_server, socket_path, monkeypatch, model_id, client_closes):
    if not client_closes:
        monkeypatch.setattr(websocket_wire.WebsocketWire, 'CLOSE_TIMEOUT_SEC', 0.05)
    _served, model = unix_stub_server
    with unix_connect(socket_path, uri=f'ws://localhost{wire.session_path(model_id)}') as conn:
        ready = deserialise(conn.recv(timeout=5))
        message = {protocol.SESSION_ID: ready[protocol.SESSION_ID], protocol.END_SESSION: True}
        conn.send(serialise(message))
        assert deserialise(conn.recv(timeout=5)) == message
        model.end_session.assert_called_once_with(ready[protocol.SESSION_ID])
        if client_closes:
            assert conn.ping().wait(5), 'The server closed before the client could read the acknowledgement'
        else:
            with pytest.raises(ConnectionClosedOK):
                conn.recv(timeout=5)


def test_a_probe_over_a_socket_answers_for_the_server_that_bound_it(unix_stub_server):
    """A coordinator preflights an endpoint with ``probe`` alone, and a socket answers it like a port."""
    served, _policy = unix_stub_server
    client_wire, address = served.unix()

    assert client_wire.probe(address, None, 5.0) is None


def test_a_probe_of_a_socket_nothing_has_bound_is_cold(socket_path):
    """A path no server has bound yet can still become one, so the probe says to wait rather than refuse."""
    client_wire = registry.client_wire('websocket_unix')
    address = wire.UnixSocketAddress(pathlib.Path(socket_path), wire.session_path(), '')

    assert client_wire.probe(address, None, 1.0) is wire.Refusal.COLD


def test_a_socket_path_that_reads_as_a_url_is_dialled_as_the_filename_it_is(start_server, socket_path, make_mock_model):
    """The socket is a path the wire hands to the kernel, so the spellings a URL would have to escape
    — a space, a percent, the route marker itself — reach it unchanged."""
    odd = pathlib.Path(socket_path).parent / 'a b%c' / 'api' / 'v1'
    odd.mkdir(parents=True)
    uds = str(odd / 's.sock')
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)), uds=uds)

    client = InferenceClient(*served.unix())

    assert str(served.uds) == uds
    assert client.list_models() == ['stub']
    session = client.new_session()
    try:
        assert session.infer({'obs': 'data'}) == [{'action': [1, 2, 3]}]
    finally:
        session.close()


def _dial_socket(uds: str, **settings) -> InferenceClient:
    """A client for a socket no server need have bound yet, as a co-located one is built."""
    client_wire = registry.client_wire('websocket_unix')
    address = wire.UnixSocketAddress(pathlib.Path(uds), wire.session_path(), '')
    return InferenceClient(client_wire, address, **settings)


@pytest.mark.timeout(60.0)
def test_a_client_waits_for_a_socket_the_server_has_not_bound_yet(start_server, socket_path, make_mock_model):
    """``serve`` binds only once the model has loaded, so a co-located client starting beside its
    server finds no socket at all for that interval."""
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    pipeline = PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10))
    late = threading.Timer(1.5, lambda: start_server(pipeline, uds=socket_path))
    late.start()

    try:
        session = _dial_socket(socket_path, connect_deadline=30.0).new_session()
    finally:
        late.join()
    try:
        assert session.metadata['model_name'] == 'stub'
        assert session.infer({'obs': 'data'}) == [{'action': [1, 2, 3]}]
    finally:
        session.close()


@pytest.mark.timeout(60.0)
def test_a_client_waits_for_a_server_restarting_over_the_socket_it_left(start_server, socket_path, make_mock_model):
    """A bound path whose server has gone refuses the dial, and the successor binds over it. The wait
    covers that restart as it covers a first start."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as gone:
        gone.bind(socket_path)
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    pipeline = PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10))
    late = threading.Timer(1.5, lambda: start_server(pipeline, uds=socket_path))
    late.start()

    try:
        session = _dial_socket(socket_path, connect_deadline=30.0).new_session()
    finally:
        late.join()
    try:
        assert session.infer({'obs': 'data'}) == [{'action': [1, 2, 3]}]
    finally:
        session.close()


def test_a_dial_this_process_broke_fails_at_once_over_a_live_socket(unix_stub_server, socket_path):
    """A descriptor limit is this process's own, so no server appearing clears it. The socket is live
    and the path says so, which is exactly when reading the path alone would wait out the deadline."""
    started = time.monotonic()

    with patch('positronic_wire.websocket.unix_connect') as dial:
        dial.side_effect = OSError(errno.EMFILE, 'Too many open files')
        with pytest.raises(wire.ConnectRefused) as refusal:
            _dial_socket(socket_path, connect_deadline=30.0).new_session()

    assert refusal.value.refusal is wire.Refusal.FINAL
    assert 'Too many open files' in str(refusal.value)
    assert time.monotonic() - started < 5.0


def test_a_dial_at_a_path_holding_something_that_is_not_a_socket_fails_at_once(socket_path):
    """No waiting clears a wrong path. Which errno says so differs by platform, so this asserts the
    connect deadline goes unspent."""
    pathlib.Path(socket_path).write_text('not a socket')
    started = time.monotonic()

    with pytest.raises(wire.ConnectRefused) as refusal:
        _dial_socket(socket_path, connect_deadline=30.0).new_session()

    assert refusal.value.refusal is wire.Refusal.FINAL
    assert time.monotonic() - started < 5.0


def test_a_server_binds_over_the_socket_an_earlier_run_left(start_server, socket_path, make_mock_model):
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
        stale.bind(socket_path)

    served = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)), uds=socket_path)

    assert InferenceClient(*served.unix()).list_models() == ['stub']


def test_a_path_that_is_not_a_socket_is_refused_and_left_alone(socket_path):
    """A wrong ``uds`` is refused, and the file it names stays as it was."""
    path = pathlib.Path(socket_path)
    path.write_text('not a socket')

    with pytest.raises(OSError) as refusal:
        websocket_wire.claim_socket_path(path)

    assert refusal.value.errno == errno.EADDRINUSE
    assert path.read_text() == 'not a socket'


@pytest.mark.timeout(30.0)
def test_a_second_claim_on_one_path_is_refused_and_the_first_goes_on_serving(socket_path):
    """The bind is the claim, so two servers starting on one absent path cannot both pass it."""
    held = websocket_wire.claim_socket_path(pathlib.Path(socket_path))
    try:
        with pytest.raises(OSError) as refusal:
            websocket_wire.claim_socket_path(pathlib.Path(socket_path))
        assert refusal.value.errno == errno.EADDRINUSE
        assert socket_path in str(refusal.value)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
        assert held.accept()[0].close() is None
    finally:
        held.close()


def test_a_claimed_socket_keeps_the_mode_the_umask_gives(socket_path):
    """A restrictive umask is the deployment's choice, and widening it would open the socket to every
    local account that can reach the directory."""
    previous = os.umask(0o077)
    try:
        sock = websocket_wire.claim_socket_path(pathlib.Path(socket_path))
    finally:
        os.umask(previous)
    try:
        assert stat.S_IMODE(os.stat(socket_path).st_mode) & 0o077 == 0
    finally:
        sock.close()


@pytest.mark.timeout(30.0)
def test_a_server_refuses_a_socket_a_live_server_listens_on(socket_path, make_mock_model):
    """``asyncio.create_unix_server`` unlinks the file it finds, so only a refusal here keeps the
    address with the server that owns it."""
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    server = PolicyServer(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)))

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as live:
        live.bind(socket_path)
        live.listen()

        with pytest.raises(OSError) as refusal:
            server.serve([websocket_wire.WebsocketWire(websocket_wire.ServedUnixSocket(pathlib.Path(socket_path)))])
        assert refusal.value.errno == errno.EADDRINUSE
        assert socket_path in str(refusal.value)

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(socket_path)
        assert live.accept()[0].close() is None


def test_a_server_refuses_a_relative_socket_path():
    """A relative path is resolved against the directory the server was started from, so the path an
    operator wrote and the path a client dials would part company on the next start."""
    with pytest.raises(ValueError, match='relative socket path'):
        websocket_wire.WebsocketWire(websocket_wire.ServedUnixSocket(pathlib.Path('policy.sock')))


_INFER = 'infer'


class _ScriptedModel(Model):
    """A model returning the same untimestamped chunk on every call."""

    def __call__(self, obs, *, session_id: str):
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
    stub = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'})
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
    return PolicyDeployment(source, ChunkedSchedule(fps=fps))


def test_session_param_retunes_the_client_schedule(start_server):
    pipe_cfg = cfn.Config(_fps_pipe, source=cfn.Config(DictSource, models={'default': _ScriptedModel()}))
    host, port, *_ = start_server(pipe_cfg)

    default_session = _param_session(host, port, [])
    tuned_session = _param_session(host, port, [('fps', '5')])
    try:
        assert default_session.metadata[offboard_keys.LOCAL_STACK][ARGS]['fps'] == 10
        assert tuned_session.metadata[offboard_keys.LOCAL_STACK][ARGS]['fps'] == 5
        assert default_session.infer({}) == tuned_session.infer({})
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
    stub = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'})
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
ENDPOINT_HOST_ENV = 'POSITRONIC_ENDPOINT_HOST'
ENDPOINT_PORT_ENV = 'POSITRONIC_ENDPOINT_PORT'
ENDPOINT_WIRE_ENV = 'POSITRONIC_ENDPOINT_WIRE'
_LIVE_HOST = os.environ.get(ENDPOINT_HOST_ENV)


@pytest.fixture
def authed_endpoint(start_server, make_mock_model) -> tuple[tuple[wire.ClientWire, wire.SessionAddress], str]:
    """An authenticated server's wire and session address, and the token gating it."""
    if _LIVE_HOST:
        address = wire.HostPortAddress(_LIVE_HOST, int(os.environ[ENDPOINT_PORT_ENV]), wire.SESSION_PATH, '')
        return (registry.client_wire(os.environ[ENDPOINT_WIRE_ENV]), address), os.environ[AUTH_TOKEN_ENV]
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(PolicyDeployment(_StubSource(policy), ChunkedSchedule(fps=10)), auth_token=_TOKEN)
    return served.ws(), _TOKEN


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
    monkeypatch.setattr(ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    endpoint, token = authed_endpoint
    header = make_header(token)
    client = InferenceClient(*endpoint, headers=None if header is None else {AUTH_HEADER: header})
    with pytest.raises(wire.ConnectRefused) as refused:
        client.new_session()
    assert refused.value.refusal is wire.Refusal.FORBIDDEN
    # The catalogue refuses in the same vocabulary: the wire reads it, so it raises what a dial raises.
    with pytest.raises(wire.ConnectRefused) as catalogue:
        client.list_models()
    assert catalogue.value.refusal is wire.Refusal.FINAL


@pytest.mark.endpoint
def test_auth_accepts_the_token(authed_endpoint):
    endpoint, token = authed_endpoint
    client = InferenceClient(*endpoint, headers={AUTH_HEADER: bearer(token)})
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
@pytest.mark.skipif(not _LIVE_HOST, reason=f'no ingress to idle against; set {ENDPOINT_HOST_ENV}')
def test_session_outlives_an_idle_ingress_window(authed_endpoint):
    endpoint, token = authed_endpoint
    session = InferenceClient(*endpoint, headers={AUTH_HEADER: bearer(token)}).new_session()
    try:
        time.sleep(_IDLE_WINDOW_SEC)
        conn = session._conn
        assert isinstance(conn, WebsocketClientConnection), "the idle window is the websocket wire's"
        assert conn._websocket.ping().wait(timeout=30.0)
    finally:
        session.close()


def test_server_without_a_token_serves_open(stub_server):
    host, port, _server, _policy = stub_server
    assert InferenceClient(
        client_websocket.WebsocketClientWire(), wire.HostPortAddress(host, port, wire.SESSION_PATH, '')
    ).list_models() == ['stub']


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
    policy = make_mock_model([{'action': [1, 2, 3]}], {'model_name': 'stub', 'type': 'stub'})
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
    session = InferenceClient(*served.ws()).new_session()
    session.close()
    model.close.assert_not_called()
    served.server.shutdown()
    assert closed.wait(timeout=5)
    model.close.assert_called_once()
