"""Native model lifecycle over independently installed transports, without Positronic."""

import asyncio
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from dataclasses import dataclass, field
from http import HTTPStatus
from http.client import HTTPConnection
from importlib.util import find_spec
from typing import Any

import numpy as np
import pytest
from positronic_model_server import keys, protocol, serialization, server_wire, spec
from positronic_model_server.server import Model, ModelServer, Session
from positronic_wire import wire

TRANSPORTS = []
if find_spec('uvicorn') is not None and find_spec('websockets') is not None:
    from positronic_model_server.websocket_wire import WebsocketWire
    from positronic_wire.websocket import WebsocketClientWire

    TRANSPORTS.append(pytest.param((WebsocketWire, WebsocketClientWire), id='websocket'))
if find_spec('grpc') is not None:
    from positronic_model_server.grpc_wire import GrpcWire
    from positronic_wire.grpc import GrpcClientWire

    TRANSPORTS.append(pytest.param((GrpcWire, GrpcClientWire), id='grpc'))


class Probe:
    def __init__(self):
        self.calls = []
        self.loading = threading.Event()
        self.load_release = threading.Event()
        self.preparing = threading.Event()
        self.prepare_release = threading.Event()
        self.inferring = threading.Event()
        self.infer_release = threading.Event()
        self.cleaned = threading.Event()
        for gate in (self.load_release, self.prepare_release, self.infer_release):
            gate.set()

    def record(self, operation: str) -> None:
        self.calls.append((operation, os.getpid(), threading.get_ident()))

    def load(self) -> Model:
        self.record('load')
        self.loading.set()
        assert self.load_release.wait(10)
        return Model(self.prepare, parameters={'scale': 1}, metadata={'checkpoint': 'fake'}, close=self.close)

    def close(self) -> None:
        self.record('close_model')

    def prepare(self, params: dict[str, Any]) -> Session:
        self.record('prepare')
        self.preparing.set()
        assert self.prepare_release.wait(10)
        if not isinstance(params['scale'], int):
            raise ValueError('scale must be an integer')
        count = 0

        def infer(obs):
            nonlocal count
            self.record('infer')
            self.inferring.set()
            assert self.infer_release.wait(10)
            if obs.get('fail'):
                raise ValueError('model rejected observation')
            count += 1
            return {'count': count, 'scaled': params['scale'] * obs.get('value', 1), 'native': obs}

        def close():
            self.record('close_session')
            self.cleaned.set()

        return Session(infer, spec.component('chunked_schedule', version=2, fps=20), close=close)


@dataclass
class Peer:
    connection: wire.ClientConnection
    session_id: str
    metadata: dict

    def infer(self, observation: Any) -> dict:
        self.connection.send(
            serialization.serialise({protocol.SESSION_ID: self.session_id, protocol.OBSERVATION: observation})
        )
        return serialization.deserialise(self.connection.recv(timeout=5))

    def close(self) -> None:
        message = {protocol.SESSION_ID: self.session_id, protocol.END_SESSION: True}
        with suppress(wire.PeerDisconnected):
            self.connection.send(serialization.serialise(message))
        assert serialization.deserialise(self.connection.recv(timeout=5)) == message
        self.connection.close()


@dataclass
class Running:
    probe: Probe
    server: ModelServer
    transport: server_wire.Wire
    client_wire: wire.ClientWire
    ready: threading.Event
    thread: threading.Thread
    errors: list[BaseException]
    connections: list[wire.ClientConnection] = field(default_factory=list)

    def connect(self, query: str = '', headers: dict[str, str] | None = None) -> wire.ClientConnection:
        address = self.transport.served_address
        assert isinstance(address, server_wire.ServedHostPort)
        connection = self.client_wire.dial(
            wire.HostPortAddress(address.host, address.port, wire.SESSION_PATH, query), headers=headers, open_timeout=5
        )
        self.connections.append(connection)
        return connection

    def session(self, query: str = '', headers: dict[str, str] | None = None) -> Peer:
        connection = self.connect(query, headers)
        while True:
            message = serialization.deserialise(connection.recv(timeout=5))
            if protocol.ERROR in message:
                raise RuntimeError(message[protocol.ERROR])
            if message[protocol.STATUS] == protocol.ServerStatus.READY:
                assert message[protocol.PROTOCOL_VERSION] == protocol.ProtocolVersion.V3
                return Peer(connection, message[protocol.SESSION_ID], message[protocol.META])
            assert message[protocol.STATUS] == protocol.ServerStatus.WAITING


@pytest.fixture(params=TRANSPORTS)
def transport_types(request):
    return request.param


@pytest.fixture
def start(transport_types):
    servers: list[Running] = []

    def launch(probe: Probe | None = None, *, wait: bool = True, **kwargs) -> Running:
        probe = Probe() if probe is None else probe
        transport_type, client_type = transport_types
        transport = transport_type(server_wire.ServedHostPort('127.0.0.1', 0))
        server = ModelServer(probe.load, **kwargs)
        server.WAITING_INTERVAL_SEC = 0.02
        ready = threading.Event()
        errors: list[BaseException] = []

        def run():
            try:
                server.serve([transport], ready.set)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        running = Running(probe, server, transport, client_type(), ready, thread, errors)
        servers.append(running)
        thread.start()
        if wait:
            assert ready.wait(5), errors
        return running

    yield launch
    for running in servers:
        for gate in (running.probe.load_release, running.probe.prepare_release, running.probe.infer_release):
            gate.set()
        for connection in running.connections:
            connection.close()
        running.server.shutdown()
        running.thread.join(10)
        assert not running.thread.is_alive(), 'Server failed to stop'
        assert not running.errors


def test_sessions_keep_parameters_history_and_native_results(start):
    running = start()
    first = running.session('scale=3')
    second = running.session()
    assert first.metadata[keys.SESSION_PARAMS] == {'scale': 3}
    assert first.metadata[keys.EFFECTIVE_PARAMS] == {'scale': 3}
    assert second.metadata[keys.SESSION_PARAMS] == {}
    assert second.metadata[keys.EFFECTIVE_PARAMS] == {'scale': 1}
    assert first.session_id != second.session_id
    native = {'value': 2, 'robot_command': {'vendor': 'opaque'}, b'__cmd__': 'opaque', 'array': np.arange(6)}
    answer = first.infer(native)
    assert answer[protocol.RESULT]['scaled'] == 6
    np.testing.assert_array_equal(answer[protocol.RESULT]['native']['array'], native['array'])
    assert answer[protocol.RESULT]['native'][b'__cmd__'] == 'opaque'
    assert first.infer({})[protocol.RESULT]['count'] == 2
    assert second.infer({})[protocol.RESULT]['count'] == 1
    first.close()
    second.close()
    running.server.shutdown()
    running.thread.join(5)
    operations, processes, threads = zip(*running.probe.calls, strict=True)
    assert operations.count('load') == operations.count('close_model') == 1
    assert operations.count('close_session') == 2
    assert set(processes) == {os.getpid()}
    assert len(set(threads)) == 1


@pytest.mark.parametrize('query', ['unknown=1', 'scale=1&scale=2', 'scale=NaN', 'scale="wrong"'])
def test_invalid_session_parameters_are_refused(start, query):
    running = start()
    with pytest.raises(RuntimeError):
        running.session(query)
    assert running.session().infer({})[protocol.RESULT]['scaled'] == 1


def test_inference_error_keeps_the_session_usable(start):
    peer = start().session()
    assert 'model rejected' in peer.infer({'fail': True})[protocol.ERROR]
    assert peer.infer({})[protocol.RESULT]['count'] == 1
    peer.close()


def test_a_wrong_session_id_closes_only_the_requesting_session(start):
    running = start()
    first, second = running.session(), running.session()
    with suppress(wire.PeerDisconnected):
        first.connection.send(
            serialization.serialise({protocol.SESSION_ID: second.session_id, protocol.OBSERVATION: {}})
        )
    message = serialization.deserialise(first.connection.recv(timeout=5))
    assert message[protocol.STATUS] == protocol.ServerStatus.ERROR
    assert 'session ID' in message[protocol.ERROR]
    assert second.infer({})[protocol.RESULT]['count'] == 1
    second.close()


def test_shared_warmup_precedes_binding_and_shutdown_is_not_lost(start):
    probe = Probe()
    probe.load_release.clear()
    running = start(probe, wait=False)
    assert probe.loading.wait(5)
    assert not running.ready.is_set()
    with pytest.raises(AssertionError, match='not started'):
        _ = running.transport.served_address
    running.server.shutdown()
    probe.load_release.set()
    running.thread.join(5)
    assert not running.thread.is_alive()
    assert [name for name, *_ in probe.calls] == ['load', 'close_model']


def test_shutdown_before_the_event_loop_starts_is_not_lost(start, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    run = asyncio.run

    def delayed_run(coro):
        entered.set()
        assert release.wait(5)
        return run(coro)

    monkeypatch.setattr(asyncio, 'run', delayed_run)
    running = start(wait=False)
    try:
        assert entered.wait(5)
        running.server.shutdown()
    finally:
        release.set()
    running.thread.join(5)

    assert not running.thread.is_alive()
    assert not running.ready.is_set()
    assert [name for name, *_ in running.probe.calls] == ['load', 'close_model']


def test_a_server_can_serve_again_after_shutdown(transport_types):
    probe = Probe()
    server = ModelServer(probe.load)
    transport_type, _ = transport_types

    def ready():
        probe.record('ready')
        server.shutdown()
        server.shutdown()

    for _ in range(2):
        server.serve([transport_type(server_wire.ServedHostPort('127.0.0.1', 0))], ready)

    assert [name for name, *_ in probe.calls] == ['load', 'ready', 'close_model'] * 2


def test_partial_listener_startup_releases_all_ports_and_model(transport_types, monkeypatch):
    transport_type, _ = transport_types
    listeners = [transport_type(server_wire.ServedHostPort('127.0.0.1', 0)) for _ in range(2)]
    start = listeners[1].start

    async def failing_start(*args):
        await start(*args)
        raise RuntimeError('listener startup failed')

    monkeypatch.setattr(listeners[1], 'start', failing_start)
    probe = Probe()
    with pytest.raises(RuntimeError, match='listener startup failed'):
        ModelServer(probe.load).serve(listeners)
    assert [name for name, *_ in probe.calls] == ['load', 'close_model']
    for listener in listeners:
        address = listener.served_address
        with socket.socket() as reclaimed:
            reclaimed.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            reclaimed.bind((address.host, address.port))


def test_waiting_keeps_preparation_alive_and_keepalive_answers(start):
    probe = Probe()
    probe.prepare_release.clear()
    running = start(probe, idle_timeout_min=1)
    connection = running.connect()
    for _ in range(3):
        assert serialization.deserialise(connection.recv(timeout=2))[protocol.STATUS] == protocol.ServerStatus.WAITING
    assert probe.preparing.wait(5)
    address = running.transport.served_address
    assert isinstance(address, server_wire.ServedHostPort)
    assert (
        running.client_wire.keepalive(
            wire.HostPortAddress(address.host, address.port, wire.SESSION_PATH, ''), headers=None, timeout=5
        )
        == 60
    )
    probe.prepare_release.set()
    while True:
        message = serialization.deserialise(connection.recv(timeout=5))
        if message[protocol.STATUS] == protocol.ServerStatus.READY:
            break
    peer = Peer(connection, message[protocol.SESSION_ID], message[protocol.META])
    assert peer.infer({})[protocol.RESULT]['count'] == 1
    peer.close()


def test_shutdown_waits_for_inference_before_session_and_model_cleanup(start):
    probe = Probe()
    running = start(probe)
    peer = running.session()
    probe.infer_release.clear()
    peer.connection.send(serialization.serialise({protocol.SESSION_ID: peer.session_id, protocol.OBSERVATION: {}}))
    assert probe.inferring.wait(5)
    running.server.shutdown()
    assert not probe.cleaned.wait(0.05)
    probe.infer_release.set()
    running.thread.join(10)
    assert not running.thread.is_alive()
    assert [name for name, *_ in probe.calls][-3:] == ['infer', 'close_session', 'close_model']


def test_disconnect_during_preparation_releases_the_prepared_session(start):
    probe = Probe()
    probe.prepare_release.clear()
    running = start(probe)
    connection = running.connect()
    assert probe.preparing.wait(5)
    connection.close()
    probe.prepare_release.set()
    assert probe.cleaned.wait(5)


def test_sessions_serialize_inference_and_keep_independent_state(start):
    running = start()
    peers = [running.session(f'scale={value}') for value in range(1, 5)]
    with ThreadPoolExecutor(max_workers=4) as clients:
        answers = list(clients.map(lambda peer: peer.infer({}), peers))
    assert [answer[protocol.RESULT]['scaled'] for answer in answers] == [1, 2, 3, 4]
    assert all(answer[protocol.RESULT]['count'] == 1 for answer in answers)
    assert len({thread for _, _, thread in running.probe.calls}) == 1


def test_result_encoding_keeps_model_buffers_exclusive_and_keepalive_available(start, monkeypatch):
    running = start()
    first, second = running.session(), running.session()
    encoding = threading.Event()
    release = running.probe.infer_release

    def encode(value, encodings):
        release.clear()
        encoding.set()
        assert release.wait(10)
        return value

    monkeypatch.setattr(serialization, 'encode_images', encode)
    first.connection.send(serialization.serialise({protocol.SESSION_ID: first.session_id, protocol.OBSERVATION: {}}))
    assert encoding.wait(5)
    second.connection.send(serialization.serialise({protocol.SESSION_ID: second.session_id, protocol.OBSERVATION: {}}))
    address = running.transport.served_address
    assert isinstance(address, server_wire.ServedHostPort)
    running.client_wire.keepalive(
        wire.HostPortAddress(address.host, address.port, wire.SESSION_PATH, ''), headers=None, timeout=5
    )
    assert [name for name, *_ in running.probe.calls].count('infer') == 1
    monkeypatch.undo()
    release.set()
    for peer in (first, second):
        assert protocol.RESULT in serialization.deserialise(peer.connection.recv(timeout=5))
        peer.close()


def test_authentication_gates_sessions(start):
    running = start(auth_token='secret')
    with pytest.raises(wire.ConnectRefused):
        running.session()
    address = running.transport.served_address
    assert isinstance(address, server_wire.ServedHostPort)
    endpoint = wire.HostPortAddress(address.host, address.port, wire.SESSION_PATH, '')
    with pytest.raises(wire.ConnectRefused):
        running.client_wire.keepalive(endpoint, headers=None, timeout=5)
    assert (
        running.client_wire.keepalive(endpoint, headers={protocol.AUTH_HEADER: protocol.bearer('secret')}, timeout=5)
        is None
    )
    peer = running.session(headers={protocol.AUTH_HEADER: protocol.bearer('secret')})
    assert peer.infer({})[protocol.RESULT]['count'] == 1
    peer.close()


@pytest.mark.parametrize(
    'path,method,token,status',
    [
        (wire.KEEPALIVE_PATH, 'POST', 'secret', HTTPStatus.OK),
        (wire.KEEPALIVE_PATH, 'POST', 'wrong', HTTPStatus.UNAUTHORIZED),
        (wire.KEEPALIVE_PATH, 'GET', 'secret', HTTPStatus.METHOD_NOT_ALLOWED),
        ('/missing', 'POST', 'secret', HTTPStatus.NOT_FOUND),
    ],
)
def test_http_routes_preserve_status_and_do_not_prepare_sessions(start, transport_types, path, method, token, status):
    _, client_type = transport_types
    if client_type.NAME != 'websocket':
        pytest.skip('HTTP routes belong to the WebSocket transport')
    running = start(auth_token='secret')
    address = running.transport.served_address
    assert isinstance(address, server_wire.ServedHostPort)
    with closing(HTTPConnection(address.host, address.port, timeout=5)) as connection:
        connection.request(method, path, headers={protocol.AUTH_HEADER: protocol.bearer(token)})
        response = connection.getresponse()
        assert response.status == status
        if status == HTTPStatus.METHOD_NOT_ALLOWED:
            assert response.getheader('Allow') == 'POST'
        if status == HTTPStatus.OK:
            assert response.getheader('Content-Type') == 'application/json'
    assert [name for name, *_ in running.probe.calls] == ['load']
    endpoint = wire.HostPortAddress(address.host, address.port, wire.SESSION_PATH, '')
    assert running.client_wire.probe(endpoint, headers=None, open_timeout=5) is None


def test_idle_expiry_closes_the_model(start):
    running = start(idle_timeout_min=0.05 / 60)
    running.thread.join(5)
    assert not running.thread.is_alive()
    assert [name for name, *_ in running.probe.calls] == ['load', 'close_model']


def test_shutdown_releases_the_listening_port(start):
    running = start()
    address = running.transport.served_address
    assert isinstance(address, server_wire.ServedHostPort)
    running.server.shutdown()
    running.thread.join(5)
    assert not running.thread.is_alive()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((address.host, address.port))
