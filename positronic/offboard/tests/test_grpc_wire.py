"""The gRPC wire: a session runs over it as it runs over the websocket."""

import asyncio
import datetime
import ipaddress
import logging
import pathlib
import queue
import ssl
import tempfile
import threading
import time
from collections.abc import Callable, Generator
from unittest.mock import ANY, MagicMock

import configuronic as cfn
import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
from cryptography.x509.oid import NameOID
from positronic_wire import grpc as client_grpc
from positronic_wire import wire

from positronic.offboard import grpc_wire
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import InferenceClient, _ConnectRetries
from positronic.offboard.server import AUTH_HEADER, bearer
from positronic.offboard.tests.conftest import DictSource, Served, StartServer
from positronic.policy.base import SEQ
from positronic.policy.layers import ChunkedSchedule, TemporalStack
from positronic.policy.spec import ModelSource, PolicySource, remote

_TOKEN = 'test-secret-token'


@pytest.fixture
def both_wires(start_server: StartServer, make_mock_policy) -> tuple[Served, MagicMock]:
    """A server that offers both wires over one policy."""
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True)
    return served, policy


def test_a_grpc_session_handshakes_and_infers(both_wires):
    served, policy = both_wires
    session = InferenceClient(*served.grpc()).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        obs = {'image': 'test'}
        assert session.infer(obs) == [{'action': [1, 2, 3]}]
        policy._mock_session.assert_called_with(obs, ANY)
    finally:
        session.close()


def _apart_from_the_endpoint(meta: dict) -> dict:
    return {key: value for key, value in meta.items() if key not in (offboard_keys.HOST, offboard_keys.PORT)}


def test_both_wires_answer_one_observation_alike(both_wires):
    served, _policy = both_wires
    obs = {'image': 'test'}
    over_ws = InferenceClient(*served.ws()).new_session()
    over_grpc = InferenceClient(*served.grpc()).new_session()
    try:
        assert _apart_from_the_endpoint(over_grpc.metadata) == _apart_from_the_endpoint(over_ws.metadata)
        assert over_grpc.infer(obs) == over_ws.infer(obs)
    finally:
        over_ws.close()
        over_grpc.close()


def test_each_wire_names_its_own_port_in_the_meta(both_wires):
    served, _policy = both_wires
    over_ws = InferenceClient(*served.ws()).new_session()
    over_grpc = InferenceClient(*served.grpc()).new_session()
    try:
        assert over_ws.metadata[offboard_keys.PORT] == served.port
        assert over_grpc.metadata[offboard_keys.PORT] == served.grpc_port
    finally:
        over_ws.close()
        over_grpc.close()


def test_both_wires_report_what_their_close_saw(both_wires, caplog):
    """The server holds the slot of a session whose close it never saw, and the next handshake waits on it."""
    served, _policy = both_wires
    over_ws = InferenceClient(*served.ws()).new_session()
    over_grpc = InferenceClient(*served.grpc()).new_session()

    with caplog.at_level(logging.INFO, logger='positronic.offboard.client'):
        over_ws.close()
        over_grpc.close()

    ws_report, grpc_report = (r.getMessage() for r in caplog.records if 'InferenceSession.close' in r.getMessage())
    assert 'close code 1000' in ws_report  # the server answered the close frame
    assert 'server ended it within 5.0s True' in grpc_report


def test_closing_a_session_ends_it_on_the_server(both_wires):
    """``close`` returns after the server has released the session."""
    served, _policy = both_wires
    session = InferenceClient(*served.grpc()).new_session()
    assert served.server._active_sessions == 1
    session.close()
    assert served.server._active_sessions == 0


def test_a_failed_inference_reaches_the_client_as_an_exception(both_wires):
    served, policy = both_wires
    session = InferenceClient(*served.grpc()).new_session()
    try:
        policy._mock_session.side_effect = RuntimeError('no such joint')
        with pytest.raises(RuntimeError, match='no such joint'):
            session.infer({'image': 'test'})
    finally:
        session.close()


def test_a_session_that_cannot_open_reaches_the_client_as_an_exception(start_server, make_mock_policy):
    """A model the source refuses fails in the handshake, before the session serves anything."""
    policies = {'alpha': make_mock_policy([{'action': [1]}], {'model_name': 'alpha'})}
    served = start_server(ChunkedSchedule() | remote | DictSource(policies), grpc=True)
    with pytest.raises(RuntimeError, match='Unknown model'):
        InferenceClient(*served.grpc(model='beta')).new_session()


def test_the_session_path_names_the_model(start_server, make_mock_policy):
    policies = {
        'alpha': make_mock_policy([{'action': ['alpha']}], {'model_name': 'alpha'}),
        'beta': make_mock_policy([{'action': ['beta']}], {'model_name': 'beta'}),
    }
    served = start_server(ChunkedSchedule() | remote | DictSource(policies), grpc=True)
    session = InferenceClient(*served.grpc(model='beta')).new_session()
    try:
        assert session.metadata['model_name'] == 'beta'
        assert session.infer({'obs': 'beta'}) == [{'action': ['beta']}]
    finally:
        session.close()


def _tunable_pipe(source: ModelSource, offsets: tuple[float, ...] = (-0.1, 0.0)):
    return TemporalStack(keys=('x',), offsets_sec=offsets) | ChunkedSchedule() | remote | source


def test_the_query_carries_the_session_params(start_server, make_mock_policy):
    policies = {'alpha': make_mock_policy([{'action': ['alpha']}], {'model_name': 'alpha'})}
    pipe = cfn.Config(_tunable_pipe, source=cfn.Config(DictSource, policies=policies))
    served = start_server(pipe, grpc=True)
    session = InferenceClient(*served.grpc(query='offsets=[-0.5, 0.0]')).new_session()
    try:
        stack = session.metadata[offboard_keys.LOCAL_STACK][SEQ]
        assert stack[0]['args']['offsets_sec'] == [-0.5, 0.0]
    finally:
        session.close()


@pytest.fixture
def authed_server(start_server: StartServer, make_mock_policy) -> Served:
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True, auth_token=_TOKEN)
    return served


def test_the_grpc_wire_gates_on_the_bearer_token(authed_server):
    session = InferenceClient(*authed_server.grpc(), headers={AUTH_HEADER: bearer(_TOKEN)}).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
    finally:
        session.close()


@pytest.mark.parametrize('header', [None, bearer('wrong'), _TOKEN])
def test_the_grpc_wire_refuses_a_session_without_the_token(authed_server, header, monkeypatch):
    # A refused credential answers like a cold backend, and the client retries it; one attempt shows the
    # refusal.
    monkeypatch.setattr(_ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    headers = None if header is None else {AUTH_HEADER: header}
    with pytest.raises(wire.ConnectRefused) as refused:
        InferenceClient(*authed_server.grpc(), headers=headers).new_session()
    assert refused.value.refusal is wire.Refusal.FORBIDDEN


# An address, and no name that resolves to two families: gRPC reports the last address it failed on,
# and a refused second family would hide what the first blamed.
EDGE_HOST = '127.0.0.1'


def _self_signed(host: str) -> tuple[bytes, bytes]:
    """A certificate and key for ``host``, PEM encoded, valid from yesterday."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, host)])
    day = datetime.timedelta(days=1)
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509
        .CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - day)
        .not_valid_after(now + day)
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address(host))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    private = key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    return certificate.public_bytes(Encoding.PEM), private


async def _copy(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    # A session ends with one end closing first, and the other half of the pair then writes into a dead
    # socket. Any other error is the edge's own, and fails the test.
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


@pytest.fixture
def tls_edge() -> Generator[Callable[[str, int], tuple[int, bytes]], None, None]:
    """Starts a TLS front over a plaintext gRPC port; an authenticated endpoint stands behind one.

    The front terminates TLS, selects HTTP/2 over ALPN and copies the bytes on. It answers its own port
    and the root to verify it against. ``alpn=False`` selects no protocol, as a front over a raw TCP
    port does, and ``certificate_host`` names the address its certificate covers.
    """
    stops: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    def start(
        backend_host: str, backend_port: int, alpn: bool = True, certificate_host: str = EDGE_HOST
    ) -> tuple[int, bytes]:
        certificate, private = _self_signed(certificate_host)
        started: queue.SimpleQueue = queue.SimpleQueue()

        async def _serve_edge() -> None:
            with tempfile.TemporaryDirectory() as keys:
                chain, key_file = pathlib.Path(keys, 'chain.pem'), pathlib.Path(keys, 'key.pem')
                chain.write_bytes(certificate)
                key_file.write_bytes(private)
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(chain, key_file)
                if alpn:
                    context.set_alpn_protocols(['h2'])

                async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                    backend_r, backend_w = await asyncio.open_connection(backend_host, backend_port)
                    await asyncio.gather(_copy(reader, backend_w), _copy(backend_r, writer))

                edge = await asyncio.start_server(_handle, EDGE_HOST, 0, ssl=context)
                stop = asyncio.Event()
                started.put((edge.sockets[0].getsockname()[1], asyncio.get_running_loop(), stop))
                async with edge:
                    await stop.wait()

        threading.Thread(target=asyncio.run, args=(_serve_edge(),), daemon=True).start()
        port, loop, stop = started.get(timeout=5.0)
        stops.append((loop, stop))
        return port, certificate

    yield start
    for loop, stop in stops:
        loop.call_soon_threadsafe(stop.set)


def _trust_only(monkeypatch, root: bytes) -> None:
    """Verify every channel this test opens against ``root``, in place of the system's own."""
    system_roots = grpc.ssl_channel_credentials
    monkeypatch.setattr(grpc, 'ssl_channel_credentials', lambda: system_roots(root))


def _through_edge(port: int) -> tuple[wire.ClientWire, wire.SessionAddress]:
    """The TLS member of the gRPC wire, and the session behind the edge on ``port``."""
    return client_grpc.GrpcTlsClientWire(), wire.HostPortAddress(EDGE_HOST, port, wire.SESSION_PATH, '')


@pytest.fixture
def edged(tls_edge, monkeypatch) -> Callable[[Served], tuple[wire.ClientWire, wire.SessionAddress]]:
    """A server reached through a TLS edge, on the gRPC wire's TLS member; the client trusts the edge's root."""

    def endpoint(served: Served) -> tuple[wire.ClientWire, wire.SessionAddress]:
        port, root = tls_edge(served.host, served.grpc_port)
        _trust_only(monkeypatch, root)
        return _through_edge(port)

    return endpoint


def test_a_session_through_a_tls_edge_handshakes_and_infers(both_wires, edged):
    served, policy = both_wires
    session = InferenceClient(*edged(served)).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        obs = {'image': 'test'}
        assert session.infer(obs) == [{'action': [1, 2, 3]}]
        policy._mock_session.assert_called_with(obs, ANY)
    finally:
        session.close()


def test_a_tls_edge_carries_the_bearer_token(authed_server, edged):
    session = InferenceClient(*edged(authed_server), headers={AUTH_HEADER: bearer(_TOKEN)}).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
    finally:
        session.close()


def test_a_tls_edge_session_without_the_token_is_refused(authed_server, edged, monkeypatch):
    monkeypatch.setattr(_ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    with pytest.raises(wire.ConnectRefused) as refused:
        InferenceClient(*edged(authed_server)).new_session()
    assert refused.value.refusal is wire.Refusal.FORBIDDEN


def test_a_grpc_session_names_the_session_port_alone(both_wires):
    served, _policy = both_wires
    client = InferenceClient(*served.grpc())
    assert client.session_url == f'{served.host}:{served.grpc_port}/api/v1/session'
    with pytest.raises(ValueError, match='carries sessions alone'):
        client.list_models()


@pytest.mark.parametrize(
    ('session_path', 'model_id'),
    [
        (wire.SESSION_PATH, None),
        (f'{wire.SESSION_PATH}/10000', '10000'),
        (f'{wire.SESSION_PATH}/GEAR-Dreams/DreamZero-DROID', 'GEAR-Dreams/DreamZero-DROID'),
        (f'{wire.SESSION_PATH}/s3%3A//bucket/ckpt-1', 's3://bucket/ckpt-1'),
    ],
)
def test_the_session_path_decodes_as_the_websocket_route_does(session_path, model_id):
    assert grpc_wire.model_id_of(session_path) == model_id


def test_a_path_outside_the_session_route_is_refused():
    with pytest.raises(ValueError, match='Unexpected session path'):
        grpc_wire.model_id_of('/api/v2/session/10000')


def test_a_port_that_never_answers_is_named_at_the_deadline():
    """Nothing listens on port 1; the channel never becomes ready."""
    address = wire.HostPortAddress('localhost', 1, wire.SESSION_PATH, '')
    client = InferenceClient(client_grpc.GrpcClientWire(), address, open_timeout=0.2, connect_deadline=0.0)
    with pytest.raises(TimeoutError, match='localhost:1'):
        client.new_session()


def test_an_open_timeout_under_the_probe_budget_still_opens(both_wires):
    served, _policy = both_wires
    budget = client_grpc._REFUSAL_PROBE_SEC / 2
    session = InferenceClient(*served.grpc(), open_timeout=budget, connect_deadline=0.0).new_session()
    try:
        assert session.infer({'image': 'test'}) == [{'action': [1, 2, 3]}]
    finally:
        session.close()


def test_an_ipv6_host_binds_in_brackets(start_server: StartServer, make_mock_policy):
    """The bind target carries the brackets gRPC's syntax needs, and a session opens on the bound server."""
    assert client_grpc.target('::', 9000) == '[::]:9000'
    assert client_grpc.target('0.0.0.0', 9000) == '0.0.0.0:9000'

    policy = make_mock_policy([{'action': [4]}], {'model_name': 'stub'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True, host='::1')
    session = InferenceClient(*served.grpc()).new_session()
    try:
        assert session.infer({'image': 'test'}) == [{'action': [4]}]
    finally:
        session.close()


def test_a_refused_handshake_closes_the_connection(both_wires):
    """A refusal in a protocol frame raises past the transport handlers, and the connection holds a reader
    thread until it is closed."""
    client = InferenceClient(*both_wires[0].grpc(model='unknown-model'))
    opened = []
    client_wire = client._wire

    class _Recording(wire.ClientWire[wire.HostPortAddress]):
        """The client's wire, recording every connection it dials."""

        NAME = client_wire.NAME
        ADDRESS = wire.HostPortAddress

        def session_url(self, address):
            return client_wire.session_url(address)

        def list_models(self, address, headers, open_timeout):
            return client_wire.list_models(address, headers, open_timeout)

        def dial(self, address, headers, open_timeout):
            opened.append(client_wire.dial(address, headers, open_timeout))
            return opened[-1]

        def probe(self, address, headers, open_timeout):
            return client_wire.probe(address, headers, open_timeout)

    client._wire = _Recording()
    with pytest.raises(RuntimeError):
        client.new_session()
    assert opened, 'the session never opened a connection'
    assert opened[0]._closed, 'the refused session left its connection open'


# Long enough for the client to send more pings than gRPC's own server default tolerates.
_SILENCE_SEC = 8.0


@pytest.fixture
def chatty_client(monkeypatch) -> None:
    """Pings every 500 ms, and a silence of seconds stands in for one of minutes."""
    monkeypatch.setattr(client_grpc, 'PING_EVERY_MS', 500)


def _silent_then_infer(served: Served) -> list[dict]:
    session = InferenceClient(*served.grpc()).new_session()
    try:
        time.sleep(_SILENCE_SEC)
        return session.infer({'image': 'test'})
    finally:
        session.close()


def test_a_session_answers_after_a_silence_no_frame_crossed(both_wires, chatty_client):
    """The wire's own pings hold the stream open through an inference that outlasts a front's idle close."""
    assert _silent_then_infer(both_wires[0]) == [{'action': [1, 2, 3]}]


def test_a_server_on_the_grpc_ping_defaults_kills_the_silent_session(
    start_server, make_mock_policy, chatty_client, monkeypatch
):
    """gRPC's own server defaults answer those pings with ``GOAWAY too_many_pings``, and the session is lost."""
    monkeypatch.setattr(grpc_wire, '_server_options', lambda: list(client_grpc.MESSAGE_SIZE_OPTIONS))
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True)
    with pytest.raises(wire.PeerDisconnected, match='Too many pings'):
        _silent_then_infer(served)


def _surfaces_at_once(port: int, blamed: str) -> None:
    """Assert that a connect through the edge on ``port`` fails, names ``blamed``, and spends no retry deadline."""
    client = InferenceClient(*_through_edge(port), open_timeout=2.0, connect_deadline=20.0)
    started = time.monotonic()
    with pytest.raises(wire.ConnectRefused, match=blamed) as refused:
        client.new_session()
    assert refused.value.refusal is wire.Refusal.FINAL
    assert time.monotonic() - started < 8.0, 'the connect retried a permanent failure'


def test_a_certificate_the_client_cannot_verify_is_not_retried(both_wires, tls_edge, monkeypatch):
    port, _root = tls_edge(both_wires[0].host, both_wires[0].grpc_port)
    unrelated, _key = _self_signed(EDGE_HOST)
    _trust_only(monkeypatch, unrelated)
    _surfaces_at_once(port, client_grpc._UNUSABLE_EDGE_DETAILS[0])


def test_a_certificate_that_covers_another_host_is_not_retried(both_wires, tls_edge, monkeypatch):
    """The client trusts this root, and the edge presents a certificate for an address nobody dialled."""
    port, root = tls_edge(both_wires[0].host, both_wires[0].grpc_port, certificate_host='127.0.0.2')
    _trust_only(monkeypatch, root)
    _surfaces_at_once(port, client_grpc._UNUSABLE_EDGE_DETAILS[2])


def test_an_edge_that_selects_no_alpn_is_not_retried(both_wires, tls_edge, monkeypatch):
    """A front over a raw TCP port terminates TLS and names no ALPN protocol, and gRPC refuses it."""
    port, root = tls_edge(both_wires[0].host, both_wires[0].grpc_port, alpn=False)
    _trust_only(monkeypatch, root)
    _surfaces_at_once(port, client_grpc._UNUSABLE_EDGE_DETAILS[1])


def test_a_timed_out_session_refuses_the_next_inference(both_wires):
    """The timeout closes the connection, and the server may answer inside the close's own wait."""
    served, policy = both_wires
    policy._mock_session.side_effect = lambda *_: time.sleep(1.0) or [{'action': [1, 2, 3]}]
    session = InferenceClient(*served.grpc(), infer_timeout=0.2).new_session()
    with pytest.raises(TimeoutError):
        session.infer({'image': 'test'})
    # The late answer is the first observation's actions.
    with pytest.raises(wire.PeerDisconnected):
        session.infer({'image': 'test'})


def test_a_status_after_the_first_frame_surfaces_as_a_lost_peer(both_wires):
    """A stream that ends after frames have crossed raises a lost peer, which the connect retry reads as cold."""
    served, _policy = both_wires
    address = served.grpc(model='unknown-model')[1]
    conn = client_grpc.GrpcClientWire().dial(address, None, 10.0)
    try:
        conn.recv(timeout=10.0)
        # The server refuses the model in a frame, then ends the stream with that status.
        with pytest.raises(wire.PeerDisconnected) as gone:
            conn.recv(timeout=10.0)
        assert isinstance(gone.value.__cause__, grpc.RpcError)
    finally:
        conn.close()


def test_a_connection_refuses_to_send_once_the_server_ends_the_stream(both_wires):
    """``send`` raises as soon as the terminal status is read, and the write never reaches the outbox."""
    served, _policy = both_wires
    address = served.grpc(model='unknown-model')[1]
    conn = client_grpc.GrpcClientWire().dial(address, None, 10.0)
    try:
        conn.recv(timeout=10.0)
        with pytest.raises(wire.PeerDisconnected):
            conn.recv(timeout=10.0)
        refused = time.monotonic()
        with pytest.raises(wire.PeerDisconnected):
            conn.send(b'an observation the stream can no longer carry')
        assert time.monotonic() - refused < 1.0, 'the refused send waited for a write instead of raising'
        # The close report says the peer ended the stream.
        assert 'peer had ended the stream True' in conn.close()
    finally:
        conn.close()
