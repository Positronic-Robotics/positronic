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

from positronic.offboard import grpc_wire, wire
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import InferenceClient, _ConnectRetries
from positronic.offboard.server import AUTH_HEADER, bearer
from positronic.offboard.tests.conftest import DictSource, Served, StartServer
from positronic.policy.base import SEQ
from positronic.policy.layers import ChunkedSchedule, TemporalStack
from positronic.policy.spec import ModelSource, PolicySource, remote

_TOKEN = 'test-secret-token'


def grpc_url(served: Served, path: str = '') -> str:
    return f'grpc://{served.host}:{served.grpc_port}{path}'


@pytest.fixture
def both_wires(start_server: StartServer, make_mock_policy) -> tuple[Served, MagicMock]:
    """A server that offers both wires over one policy."""
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True)
    return served, policy


def test_a_grpc_session_handshakes_and_infers(both_wires):
    served, policy = both_wires
    session = InferenceClient(grpc_url(served)).new_session()
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
    over_ws = InferenceClient(f'{served.host}:{served.port}').new_session()
    over_grpc = InferenceClient(grpc_url(served)).new_session()
    try:
        assert _apart_from_the_endpoint(over_grpc.metadata) == _apart_from_the_endpoint(over_ws.metadata)
        assert over_grpc.infer(obs) == over_ws.infer(obs)
    finally:
        over_ws.close()
        over_grpc.close()


def test_each_wire_names_its_own_port_in_the_meta(both_wires):
    served, _policy = both_wires
    over_ws = InferenceClient(f'{served.host}:{served.port}').new_session()
    over_grpc = InferenceClient(grpc_url(served)).new_session()
    try:
        assert over_ws.metadata[offboard_keys.PORT] == served.port
        assert over_grpc.metadata[offboard_keys.PORT] == served.grpc_port
    finally:
        over_ws.close()
        over_grpc.close()


def test_both_wires_report_what_their_close_saw(both_wires, caplog):
    """The server holds the slot of a session whose close it never saw, and the next handshake waits on it."""
    served, _policy = both_wires
    over_ws = InferenceClient(f'{served.host}:{served.port}').new_session()
    over_grpc = InferenceClient(grpc_url(served)).new_session()

    with caplog.at_level(logging.INFO, logger='positronic.offboard.client'):
        over_ws.close()
        over_grpc.close()

    ws_report, grpc_report = (r.getMessage() for r in caplog.records if 'InferenceSession.close' in r.getMessage())
    assert 'close code 1000' in ws_report  # the server answered the close frame
    assert 'server ended it within 5.0s True' in grpc_report


def test_closing_a_session_ends_it_on_the_server(both_wires):
    """``close`` returns after the server has released the session."""
    served, _policy = both_wires
    session = InferenceClient(grpc_url(served)).new_session()
    assert served.server._active_sessions == 1
    session.close()
    assert served.server._active_sessions == 0


def test_a_failed_inference_reaches_the_client_as_an_exception(both_wires):
    served, policy = both_wires
    session = InferenceClient(grpc_url(served)).new_session()
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
        InferenceClient(grpc_url(served, f'{wire.SESSION_PATH}/beta')).new_session()


def test_the_session_path_names_the_model(start_server, make_mock_policy):
    policies = {
        'alpha': make_mock_policy([{'action': ['alpha']}], {'model_name': 'alpha'}),
        'beta': make_mock_policy([{'action': ['beta']}], {'model_name': 'beta'}),
    }
    served = start_server(ChunkedSchedule() | remote | DictSource(policies), grpc=True)
    session = InferenceClient(grpc_url(served, f'{wire.SESSION_PATH}/beta')).new_session()
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
    session = InferenceClient(grpc_url(served, f'{wire.SESSION_PATH}?offsets=[-0.5, 0.0]')).new_session()
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
    session = InferenceClient(grpc_url(authed_server), headers={AUTH_HEADER: bearer(_TOKEN)}).new_session()
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
        InferenceClient(grpc_url(authed_server), headers=headers).new_session()
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
    # The end that closes first leaves the other half of the pair writing into a dead socket, which is how
    # a session ends. Any other error is the edge's own, and fails the test.
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


@pytest.fixture
def tls_edge() -> Generator[Callable[[str, int], tuple[int, bytes]], None, None]:
    """Starts a TLS front over a plaintext gRPC port, as an authenticated endpoint is served.

    The front terminates TLS, selects HTTP/2 over ALPN and copies the bytes on. It answers its own port
    and the root to verify it against. ``alpn=False`` selects no protocol, as a front over a raw TCP
    port does.
    """
    stops: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []

    def start(backend_host: str, backend_port: int, alpn: bool = True) -> tuple[int, bytes]:
        certificate, private = _self_signed(EDGE_HOST)
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


@pytest.fixture
def edged(tls_edge, monkeypatch) -> Callable[[Served], str]:
    """The ``grpcs://`` URL of a server reached through a TLS edge; the client trusts the edge's root."""

    def url(served: Served) -> str:
        port, root = tls_edge(served.host, served.grpc_port)
        _trust_only(monkeypatch, root)
        return f'grpcs://{EDGE_HOST}:{port}'

    return url


def test_a_session_through_a_tls_edge_handshakes_and_infers(both_wires, edged):
    served, policy = both_wires
    session = InferenceClient(edged(served)).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        obs = {'image': 'test'}
        assert session.infer(obs) == [{'action': [1, 2, 3]}]
        policy._mock_session.assert_called_with(obs, ANY)
    finally:
        session.close()


def test_a_tls_edge_carries_the_bearer_token(authed_server, edged):
    session = InferenceClient(edged(authed_server), headers={AUTH_HEADER: bearer(_TOKEN)}).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
    finally:
        session.close()


def test_a_tls_edge_session_without_the_token_is_refused(authed_server, edged, monkeypatch):
    monkeypatch.setattr(_ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    with pytest.raises(wire.ConnectRefused) as refused:
        InferenceClient(edged(authed_server)).new_session()
    assert refused.value.refusal is wire.Refusal.FORBIDDEN


@pytest.mark.parametrize(
    ('code', 'details', 'refusal'),
    [
        (grpc.StatusCode.PERMISSION_DENIED, 'Invalid or missing bearer token', wire.Refusal.FORBIDDEN),
        (grpc.StatusCode.UNAVAILABLE, 'connection refused', wire.Refusal.COLD),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, '', wire.Refusal.COLD),
        (grpc.StatusCode.DEADLINE_EXCEEDED, '', wire.Refusal.COLD),
        (grpc.StatusCode.UNAVAILABLE, 'Cannot check peer: missing selected ALPN property', wire.Refusal.FINAL),
        (grpc.StatusCode.UNAVAILABLE, 'CERTIFICATE_VERIFY_FAILED', wire.Refusal.FINAL),
        (
            grpc.StatusCode.UNAVAILABLE,
            'address lookup failed for gpu-host:443: Domain name not found',
            wire.Refusal.FINAL,
        ),
        (
            grpc.StatusCode.UNAVAILABLE,
            'address lookup failed for gpu-host:443: DNS server returned answer with no data',
            wire.Refusal.FINAL,
        ),
        (
            grpc.StatusCode.UNAVAILABLE,
            'address lookup failed for gpu-host:443: Timeout while contacting DNS servers',
            wire.Refusal.COLD,
        ),
        (grpc.StatusCode.UNIMPLEMENTED, '', wire.Refusal.FINAL),
        (grpc.StatusCode.INTERNAL, '', wire.Refusal.FINAL),
    ],
)
def test_a_status_that_refuses_the_call_reads_as_its_http_status_does(code, details, refusal):
    status = MagicMock()
    status.code.return_value = code
    status.details.return_value = details
    assert grpc_wire._refusal(status) is refusal


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match='Unsupported scheme'):
        InferenceClient('tcp://gpu-host:9000')


@pytest.mark.parametrize('url', ['grpc://gpu-host:9000', 'grpcs://gpu-host:9000'])
def test_a_grpc_url_names_the_session_port_alone(url):
    client = InferenceClient(url)
    assert client.session_url == f'{url}/api/v1/session'
    with pytest.raises(ValueError, match='gRPC session port'):
        client.list_models()


@pytest.mark.parametrize(
    ('url', 'target', 'secure'),
    [
        ('grpc://gpu-host', 'gpu-host:80', False),
        ('grpcs://gpu-host', 'gpu-host:443', True),
        ('grpcs://gpu-host:9000', 'gpu-host:9000', True),
    ],
)
def test_the_scheme_fixes_the_port_and_the_tls(url, target, secure):
    client = InferenceClient(url)
    assert (client._grpc_target, client._grpc_secure) == (target, secure)


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
    client = InferenceClient('grpc://localhost:1', open_timeout=0.2, connect_deadline=0.0)
    with pytest.raises(TimeoutError, match='grpc://localhost:1'):
        client.new_session()


def test_an_open_timeout_under_the_probe_budget_still_opens(both_wires):
    served, _policy = both_wires
    budget = grpc_wire._REFUSAL_PROBE_SEC / 2
    session = InferenceClient(grpc_url(served), open_timeout=budget, connect_deadline=0.0).new_session()
    try:
        assert session.infer({'image': 'test'}) == [{'action': [1, 2, 3]}]
    finally:
        session.close()


def test_an_ipv6_host_binds_in_brackets(start_server: StartServer, make_mock_policy):
    """A bare '::1' binds as ':::<port>', which gRPC refuses."""
    assert grpc_wire._bind_target('::', 9000) == '[::]:9000'
    assert grpc_wire._bind_target('0.0.0.0', 9000) == '0.0.0.0:9000'

    policy = make_mock_policy([{'action': [4]}], {'model_name': 'stub'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True, host='::1')
    session = InferenceClient(f'grpc://[{served.host}]:{served.grpc_port}').new_session()
    try:
        assert session.infer({'image': 'test'}) == [{'action': [4]}]
    finally:
        session.close()


def test_a_refused_handshake_closes_the_connection(both_wires):
    """A refusal in a protocol frame raises past the transport handlers, and the connection holds a reader
    thread until it is closed."""
    client = InferenceClient(grpc_url(both_wires[0], f'{wire.SESSION_PATH}/unknown-model'))
    opened = []
    connect = client._connect

    def record():
        opened.append(connect())
        return opened[-1]

    client._connect = record
    with pytest.raises(RuntimeError):
        client.new_session()
    assert opened, 'the session never opened a connection'
    assert opened[0]._closed, 'the refused session left its connection open'


# Long enough for the client to send more pings than gRPC's own server default tolerates.
_SILENCE_SEC = 8.0


@pytest.fixture
def chatty_client(monkeypatch) -> None:
    """Pings every 500 ms, and a silence of seconds stands in for one of minutes."""
    monkeypatch.setattr(grpc_wire, '_PING_EVERY_MS', 500)


def _silent_then_infer(served: Served) -> list[dict]:
    session = InferenceClient(grpc_url(served)).new_session()
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
    monkeypatch.setattr(grpc_wire, '_server_options', lambda: list(grpc_wire._MESSAGE_SIZE_OPTIONS))
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    served = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True)
    with pytest.raises(wire.PeerDisconnected, match='Too many pings'):
        _silent_then_infer(served)


def _surfaces_at_once(url: str, blamed: str) -> None:
    """Assert that a connect to ``url`` fails, names ``blamed``, and spends no retry deadline."""
    client = InferenceClient(url, open_timeout=2.0, connect_deadline=20.0)
    started = time.monotonic()
    with pytest.raises(wire.ConnectRefused, match=blamed) as refused:
        client.new_session()
    assert refused.value.refusal is wire.Refusal.FINAL
    assert time.monotonic() - started < 8.0, 'the connect retried a permanent failure'


def test_a_certificate_the_client_cannot_verify_is_not_retried(both_wires, tls_edge, monkeypatch):
    port, _root = tls_edge(both_wires[0].host, both_wires[0].grpc_port)
    unrelated, _key = _self_signed(EDGE_HOST)
    _trust_only(monkeypatch, unrelated)
    _surfaces_at_once(f'grpcs://{EDGE_HOST}:{port}', grpc_wire._UNUSABLE_EDGE_DETAILS[0])


def test_an_edge_that_selects_no_alpn_is_not_retried(both_wires, tls_edge, monkeypatch):
    """A front over a raw TCP port terminates TLS and names no ALPN protocol, and gRPC refuses it."""
    port, root = tls_edge(both_wires[0].host, both_wires[0].grpc_port, alpn=False)
    _trust_only(monkeypatch, root)
    _surfaces_at_once(f'grpcs://{EDGE_HOST}:{port}', grpc_wire._UNUSABLE_EDGE_DETAILS[1])


def test_a_timed_out_session_refuses_the_next_inference(both_wires):
    """The timeout closes the connection, and the server may answer inside the close's own wait."""
    served, policy = both_wires
    policy._mock_session.side_effect = lambda *_: time.sleep(1.0) or [{'action': [1, 2, 3]}]
    session = InferenceClient(grpc_url(served), infer_timeout=0.2).new_session()
    with pytest.raises(TimeoutError):
        session.infer({'image': 'test'})
    # The late answer is the first observation's actions.
    with pytest.raises(wire.PeerDisconnected):
        session.infer({'image': 'test'})


def test_a_status_after_the_first_frame_surfaces_as_a_lost_peer(both_wires):
    """A stream that ends after frames have crossed raises a lost peer, which the connect retry reads as cold."""
    served, _policy = both_wires
    target = f'{served.host}:{served.grpc_port}'
    conn = grpc_wire.dial(target, f'{wire.SESSION_PATH}/unknown-model', '', None, 10.0, secure=False)
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
    target = f'{served.host}:{served.grpc_port}'
    conn = grpc_wire.dial(target, f'{wire.SESSION_PATH}/unknown-model', '', None, 10.0, secure=False)
    try:
        conn.recv(timeout=10.0)
        with pytest.raises(wire.PeerDisconnected):
            conn.recv(timeout=10.0)
        with pytest.raises(wire.PeerDisconnected):
            conn.send(b'an observation the stream can no longer carry')
        # The close report says the peer ended the stream.
        assert 'peer had ended the stream True' in conn.close()
    finally:
        conn.close()
