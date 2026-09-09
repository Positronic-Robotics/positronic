"""The gRPC wire: one session runs over it exactly as it runs over the websocket."""

import asyncio
import datetime
import ipaddress
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
from positronic.offboard.server import AUTH_HEADER, PolicyServer, bearer
from positronic.offboard.tests.conftest import DictSource, StartServer
from positronic.policy.base import SEQ
from positronic.policy.layers import ChunkedSchedule, TemporalStack
from positronic.policy.spec import ModelSource, PolicySource, remote

_TOKEN = 'test-secret-token'


def grpc_url(server: PolicyServer, path: str = '') -> str:
    return f'grpc://{server.host}:{server.grpc_port}{path}'


@pytest.fixture
def both_wires(start_server: StartServer, make_mock_policy) -> tuple[PolicyServer, MagicMock]:
    """A server offering both wires over one policy."""
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    _host, _port, server = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True)
    return server, policy


def test_a_grpc_session_handshakes_and_infers(both_wires):
    server, policy = both_wires
    session = InferenceClient(grpc_url(server)).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        obs = {'image': 'test'}
        assert session.infer(obs) == [{'action': [1, 2, 3]}]
        policy._mock_session.assert_called_with(obs, ANY)
    finally:
        session.close()


def test_both_wires_answer_one_observation_alike(both_wires):
    server, _policy = both_wires
    obs = {'image': 'test'}
    over_ws = InferenceClient(f'{server.host}:{server.port}').new_session()
    over_grpc = InferenceClient(grpc_url(server)).new_session()
    try:
        assert over_grpc.metadata == over_ws.metadata
        assert over_grpc.infer(obs) == over_ws.infer(obs)
    finally:
        over_ws.close()
        over_grpc.close()


def test_closing_a_session_ends_it_on_the_server(both_wires):
    """``close`` half-closes the stream and waits, so the server releases the session before it returns."""
    server, _policy = both_wires
    session = InferenceClient(grpc_url(server)).new_session()
    assert server._active_sessions == 1
    session.close()
    assert server._active_sessions == 0


def test_a_failed_inference_reaches_the_client_as_an_exception(both_wires):
    server, policy = both_wires
    session = InferenceClient(grpc_url(server)).new_session()
    try:
        policy._mock_session.side_effect = RuntimeError('no such joint')
        with pytest.raises(RuntimeError, match='no such joint'):
            session.infer({'image': 'test'})
    finally:
        session.close()


def test_a_session_that_cannot_open_reaches_the_client_as_an_exception(start_server, make_mock_policy):
    """A model the source refuses fails in the handshake, before the session serves anything."""
    policies = {'alpha': make_mock_policy([{'action': [1]}], {'model_name': 'alpha'})}
    _host, _port, server = start_server(ChunkedSchedule() | remote | DictSource(policies), grpc=True)
    with pytest.raises(RuntimeError, match='Unknown model'):
        InferenceClient(grpc_url(server, f'{wire.SESSION_PATH}/beta')).new_session()


def test_the_session_path_names_the_model(start_server, make_mock_policy):
    policies = {
        'alpha': make_mock_policy([{'action': ['alpha']}], {'model_name': 'alpha'}),
        'beta': make_mock_policy([{'action': ['beta']}], {'model_name': 'beta'}),
    }
    _host, _port, server = start_server(ChunkedSchedule() | remote | DictSource(policies), grpc=True)
    session = InferenceClient(grpc_url(server, f'{wire.SESSION_PATH}/beta')).new_session()
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
    _host, _port, server = start_server(pipe, grpc=True)
    session = InferenceClient(grpc_url(server, f'{wire.SESSION_PATH}?offsets=[-0.5, 0.0]')).new_session()
    try:
        stack = session.metadata[offboard_keys.LOCAL_STACK][SEQ]
        # `args` and the layer's own constructor keyword are the spec grammar's, written wherever a
        # layer renders itself; this reader spells them as the wire carries them.
        assert stack[0]['args']['offsets_sec'] == [-0.5, 0.0]
    finally:
        session.close()


@pytest.fixture
def authed_server(start_server: StartServer, make_mock_policy) -> PolicyServer:
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    _host, _port, server = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True, auth_token=_TOKEN)
    return server


def test_the_grpc_wire_gates_on_the_bearer_token(authed_server):
    session = InferenceClient(grpc_url(authed_server), headers={AUTH_HEADER: bearer(_TOKEN)}).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
    finally:
        session.close()


@pytest.mark.parametrize('header', [None, bearer('wrong'), _TOKEN])
def test_the_grpc_wire_refuses_a_session_without_the_token(authed_server, header, monkeypatch):
    # A refused credential and a cold backend answer alike, so the client spends attempts on it; one
    # is enough to see the refusal.
    monkeypatch.setattr(_ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    headers = None if header is None else {AUTH_HEADER: header}
    with pytest.raises(grpc.RpcError) as refused:
        InferenceClient(grpc_url(authed_server), headers=headers).new_session()
    assert refused.value.code() is grpc.StatusCode.PERMISSION_DENIED


# The edge answers on one address, not on both families a name resolves to: gRPC reports the last
# address it failed on, so a second leg refusing the connection would hide what the first blamed.
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
    # Whichever end closes first leaves the other half of the pair writing into a dead socket, which
    # is how a session ends. Anything else is the edge itself failing and belongs in the test's face.
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()


@pytest.fixture
def tls_edge() -> Generator[Callable[[str, int], tuple[int, bytes]], None, None]:
    """Starts a TLS front over a plaintext gRPC port, the shape an authenticated endpoint takes.

    It terminates TLS, selects HTTP/2 over ALPN and copies the bytes on, so the client and the server
    speak one h2 connection end to end and the server holds no certificate. Answers the front's own
    port and the root to verify it against. ``alpn=False`` selects no protocol at all, which is what
    a front fronting a raw TCP port does.
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
def edged(tls_edge, monkeypatch) -> Callable[[PolicyServer], str]:
    """The ``grpcs://`` URL of a server reached through a TLS edge, with the client trusting its root."""

    def url(server: PolicyServer) -> str:
        port, root = tls_edge(server.host, server.grpc_port)
        _trust_only(monkeypatch, root)
        return f'grpcs://{EDGE_HOST}:{port}'

    return url


def test_a_session_through_a_tls_edge_handshakes_and_infers(both_wires, edged):
    server, policy = both_wires
    session = InferenceClient(edged(server)).new_session()
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
    with pytest.raises(grpc.RpcError) as refused:
        InferenceClient(edged(authed_server)).new_session()
    assert refused.value.code() is grpc.StatusCode.PERMISSION_DENIED


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
    """Nothing listens on port 1, so the channel never becomes ready and the connect deadline passes."""
    client = InferenceClient('grpc://localhost:1', open_timeout=0.2, connect_deadline=0.0)
    with pytest.raises(TimeoutError, match='grpc://localhost:1'):
        client.new_session()


def test_an_ipv6_host_binds_in_brackets(start_server: StartServer, make_mock_policy):
    """gRPC's target syntax brackets an IPv6 literal, so a bare '::1' would bind ':::<port>' and fail."""
    assert grpc_wire._bind_target('::', 9000) == '[::]:9000'
    assert grpc_wire._bind_target('0.0.0.0', 9000) == '0.0.0.0:9000'

    policy = make_mock_policy([{'action': [4]}], {'model_name': 'stub'})
    _host, _port, server = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True, host='::1')
    session = InferenceClient(f'grpc://[{server.host}]:{server.grpc_port}').new_session()
    try:
        assert session.infer({'image': 'test'}) == [{'action': [4]}]
    finally:
        session.close()


def test_a_refused_handshake_closes_the_connection(both_wires):
    """A model the source does not know is refused in a protocol frame, past the transport handlers,
    and the gRPC connection behind it holds a reader thread until something closes it."""
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
    """Pings often enough that a silence measured in seconds stands in for one measured in minutes."""
    monkeypatch.setattr(grpc_wire, '_PING_EVERY_MS', 500)


def _silent_then_infer(server: PolicyServer) -> list[dict]:
    session = InferenceClient(grpc_url(server)).new_session()
    try:
        time.sleep(_SILENCE_SEC)
        return session.infer({'image': 'test'})
    finally:
        session.close()


def test_a_session_answers_after_a_silence_no_frame_crossed(both_wires, chatty_client):
    """One inference can outlast a front's idle close, so the wire's own pings hold the stream open."""
    assert _silent_then_infer(both_wires[0]) == [{'action': [1, 2, 3]}]


def test_a_server_on_the_grpc_ping_defaults_kills_the_silent_session(
    start_server, make_mock_policy, chatty_client, monkeypatch
):
    """gRPC's own server defaults answer those pings with ``GOAWAY too_many_pings``."""
    monkeypatch.setattr(grpc_wire, '_server_options', lambda: list(grpc_wire._MESSAGE_SIZE_OPTIONS))
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    _host, _port, server = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True)
    with pytest.raises(grpc.RpcError, match='Too many pings'):
        _silent_then_infer(server)


def test_a_certificate_the_client_cannot_verify_is_not_retried(both_wires, tls_edge, monkeypatch):
    """A root that does not cover the edge is permanent, so it surfaces on the first attempt."""
    port, _root = tls_edge(both_wires[0].host, both_wires[0].grpc_port)
    unrelated, _key = _self_signed(EDGE_HOST)
    _trust_only(monkeypatch, unrelated)
    _surfaces_at_once(f'grpcs://{EDGE_HOST}:{port}', grpc_wire.UNUSABLE_EDGE[0])


def test_an_edge_that_selects_no_alpn_is_not_retried(both_wires, tls_edge, monkeypatch):
    """A front fronting a raw TCP port terminates TLS and names no protocol, which gRPC cannot use."""
    port, root = tls_edge(both_wires[0].host, both_wires[0].grpc_port, alpn=False)
    _trust_only(monkeypatch, root)
    _surfaces_at_once(f'grpcs://{EDGE_HOST}:{port}', grpc_wire.UNUSABLE_EDGE[1])


def _surfaces_at_once(url: str, blamed: str) -> None:
    """Assert a connect to ``url`` fails naming ``blamed``, without spending its retry deadline."""
    client = InferenceClient(url, open_timeout=2.0, connect_deadline=20.0)
    started = time.monotonic()
    with pytest.raises(grpc.RpcError, match=blamed):
        client.new_session()
    assert time.monotonic() - started < 8.0, 'the connect retried a permanent failure'


def test_a_timed_out_session_refuses_the_next_inference(both_wires):
    """The timeout closes the connection, and the server may answer inside the close's own wait."""
    server, policy = both_wires
    policy._mock_session.side_effect = lambda *_: time.sleep(1.0) or [{'action': [1, 2, 3]}]
    session = InferenceClient(grpc_url(server), infer_timeout=0.2).new_session()
    with pytest.raises(TimeoutError):
        session.infer({'image': 'test'})
    # Without the guard this answers the first observation's actions, against the second's state.
    with pytest.raises(wire.PeerDisconnected):
        session.infer({'image': 'test'})
