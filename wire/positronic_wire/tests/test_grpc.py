"""The client side of the gRPC wire."""

import dataclasses
import ipaddress
import queue
import socket
import threading
import time
from concurrent import futures
from typing import cast
from unittest.mock import MagicMock

import grpc
import pytest
from positronic_wire import grpc as client_grpc
from positronic_wire import wire


@pytest.mark.parametrize(
    ('code', 'details', 'refusal'),
    [
        (grpc.StatusCode.PERMISSION_DENIED, 'Invalid or missing bearer token', wire.Refusal.FORBIDDEN),
        (grpc.StatusCode.UNAVAILABLE, 'upstream connect error', wire.Refusal.COLD),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, '', wire.Refusal.COLD),
        (grpc.StatusCode.DEADLINE_EXCEEDED, 'Deadline Exceeded', wire.Refusal.SILENT),
        (
            grpc.StatusCode.UNAVAILABLE,
            'failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:1: Connection refused',
            wire.Refusal.SILENT,
        ),
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
            wire.Refusal.SILENT,
        ),
        (
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            'received metadata size exceeds hard limit (value length 200000 vs. 16384)',
            wire.Refusal.FINAL,
        ),
        (
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            'CLIENT: Received message larger than max (85 vs. 10)',
            wire.Refusal.FINAL,
        ),
        (
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            'Sent message larger than max (20000000 vs. 16777216)',
            wire.Refusal.FINAL,
        ),
        (grpc.StatusCode.UNIMPLEMENTED, '', wire.Refusal.FINAL),
        (grpc.StatusCode.INTERNAL, '', wire.Refusal.FINAL),
    ],
)
def test_a_status_that_refuses_the_call_reads_as_its_http_status_does(code, details, refusal):
    status = MagicMock()
    status.code.return_value = code
    status.details.return_value = details
    assert client_grpc._refusal(status) is refusal


_ADDRESS = wire.HostPortAddress('gpu-host', 9000, wire.SESSION_PATH, '')


def _dialled_target(host: str, monkeypatch) -> str:
    """The gRPC target ``dial`` builds for ``host``, without opening a channel."""
    targets = []

    def refuse(channel: grpc.Channel, target: str, open_timeout: float) -> grpc.Channel:
        targets.append(target)
        raise wire.ConnectRefused(wire.Refusal.FINAL, 'this test opens no channel')

    monkeypatch.setattr(client_grpc, '_ready_channel', refuse)
    with pytest.raises(wire.ConnectRefused):
        client_grpc.GrpcClientWire().dial(dataclasses.replace(_ADDRESS, host=host), None, 1.0)
    return targets[0]


def test_an_ipv6_host_dials_in_brackets(monkeypatch):
    """An address holds the host raw; the dial target carries the brackets gRPC's syntax needs."""
    assert _dialled_target('::1', monkeypatch) == '[::1]:9000'


@pytest.mark.parametrize('host', ['127.0.0.1', 'gpu-host'])
def test_a_host_that_is_no_ipv6_literal_dials_unchanged(host, monkeypatch):
    assert _dialled_target(host, monkeypatch) == f'{host}:9000'


def _status(code: grpc.StatusCode) -> grpc.RpcError:
    status = MagicMock()
    status.code.return_value = code
    status.details.return_value = ''
    return status


def _probing(monkeypatch, answer: grpc.RpcError | None) -> list[tuple]:
    """Answer every probe call with ``answer``, and record what each was sent."""
    sent = []

    def call(channel, metadata, timeout):
        sent.append((channel, metadata, timeout))
        return answer

    monkeypatch.setattr(client_grpc, '_connect_refusal', call)
    return sent


def test_a_probe_the_server_answers_unimplemented_reads_as_the_server(monkeypatch):
    _probing(monkeypatch, _status(grpc.StatusCode.UNIMPLEMENTED))
    assert client_grpc.GrpcClientWire().probe(_ADDRESS, None, 1.0) is None


@pytest.mark.parametrize(
    ('code', 'refusal'),
    [
        (grpc.StatusCode.UNAVAILABLE, wire.Refusal.COLD),
        (grpc.StatusCode.DEADLINE_EXCEEDED, wire.Refusal.SILENT),
        (grpc.StatusCode.PERMISSION_DENIED, wire.Refusal.FORBIDDEN),
        (grpc.StatusCode.INTERNAL, wire.Refusal.FINAL),
    ],
)
def test_a_probe_reads_a_refusing_status_as_dial_does(code, refusal, monkeypatch):
    _probing(monkeypatch, _status(code))
    assert client_grpc.GrpcClientWire().probe(_ADDRESS, None, 1.0) is refusal


def test_a_probe_carries_the_headers_as_metadata_and_closes_the_channel(monkeypatch):
    channel = MagicMock()
    monkeypatch.setattr(grpc, 'insecure_channel', lambda target, options: channel)
    sent = _probing(monkeypatch, _status(grpc.StatusCode.UNIMPLEMENTED))
    assert client_grpc.GrpcClientWire().probe(_ADDRESS, {'Modal-Key': 'k'}, 2.0) is None
    assert sent == [(channel, (('modal-key', 'k'),), 2.0)]
    channel.close.assert_called_once()


def test_a_port_that_never_answers_is_no_answer():
    """Nothing listens on port 1; the call never reaches a server."""
    assert (
        client_grpc.GrpcClientWire().probe(dataclasses.replace(_ADDRESS, host='localhost', port=1), None, 0.2)
        is wire.Refusal.SILENT
    )


def test_the_plain_member_opens_an_insecure_channel(monkeypatch):
    opened = []
    monkeypatch.setattr(grpc, 'insecure_channel', lambda target, options: opened.append(target) or MagicMock())
    _probing(monkeypatch, None)
    assert client_grpc.GrpcClientWire().probe(_ADDRESS, None, 1.0) is None
    assert opened == ['gpu-host:9000']


def test_the_tls_member_opens_a_secure_channel(monkeypatch):
    opened = []
    monkeypatch.setattr(grpc, 'ssl_channel_credentials', lambda: 'roots')
    monkeypatch.setattr(
        grpc, 'secure_channel', lambda target, credentials, options: opened.append((target, credentials)) or MagicMock()
    )
    _probing(monkeypatch, None)
    assert client_grpc.GrpcTlsClientWire().probe(_ADDRESS, None, 1.0) is None
    assert opened == [('gpu-host:9000', 'roots')]


@pytest.mark.parametrize(
    ('client_wire', 'address', 'spelled'),
    [
        (client_grpc.GrpcClientWire(), _ADDRESS, 'gpu-host:9000/api/v1/session'),
        (
            client_grpc.GrpcClientWire(),
            dataclasses.replace(_ADDRESS, host='::1', query='fps=10'),
            '[::1]:9000/api/v1/session?fps=10',
        ),
        (client_grpc.GrpcTlsClientWire(), dataclasses.replace(_ADDRESS, port=443), 'gpu-host:443/api/v1/session'),
    ],
)
def test_a_grpc_session_is_named_by_its_target_and_no_scheme(client_wire, address, spelled):
    assert client_wire.session_url(address) == spelled


class _ManualChannel:
    """A channel whose request consumer advances as gRPC's own does: it takes the next frame only once
    the test has written the one before it."""

    def __init__(self):
        self._taken: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self._writes: queue.SimpleQueue[bool] = queue.SimpleQueue()
        self._responses: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()

    def stream_stream(self, path, request_serializer=None, response_deserializer=None):
        def call(requests, metadata=None) -> '_ManualChannel':
            threading.Thread(target=self._consume, args=(requests,), daemon=True).start()
            return self

        return call

    def _consume(self, requests) -> None:
        for message in requests:
            self._taken.put(message)
            self._writes.get()
        # The client half-closed, so the server ends the stream and the response iterator finishes.
        self._responses.put(None)

    def __iter__(self):
        while (message := self._responses.get()) is not None:
            yield message

    def cancel(self) -> None: ...

    def close(self) -> None:
        self._responses.put(None)

    def taken(self, timeout: float) -> bytes:
        """The frame gRPC has taken from the iterator and not yet written."""
        return self._taken.get(timeout=timeout)

    def write(self) -> None:
        """Finish the write gRPC is on, which is what lets it ask for the next frame."""
        self._writes.put(True)

    def end(self) -> None:
        """End the call, as a dropped connection does."""
        self._responses.put(None)


class _Sender:
    """One ``send`` on a thread of its own, and what it did."""

    def __init__(self, conn: client_grpc.GrpcClientConnection):
        self.outcome: Exception | None = None
        self.returned = threading.Event()
        threading.Thread(target=self._send, args=(conn,), daemon=True).start()

    def _send(self, conn: client_grpc.GrpcClientConnection) -> None:
        try:
            conn.send(b'frame')
        except Exception as e:
            self.outcome = e
        finally:
            self.returned.set()


def _manual_connection() -> tuple[_ManualChannel, client_grpc.GrpcClientConnection]:
    """A connection whose channel the test drives by hand; the fake serves the members the connection uses."""
    channel = _ManualChannel()
    return channel, client_grpc.GrpcClientConnection(cast(grpc.Channel, channel), 'manual', ())


def test_a_send_returns_only_once_grpc_has_written_the_frame():
    channel, conn = _manual_connection()
    sender = _Sender(conn)
    assert channel.taken(timeout=5.0) == b'frame'
    assert not sender.returned.wait(0.3), 'the send returned while the frame was still unwritten'
    channel.write()
    assert sender.returned.wait(5.0), 'the send never returned'
    assert sender.outcome is None
    conn.close()


def test_a_send_on_a_call_that_ends_mid_write_raises_a_lost_peer():
    """Nothing bounds the wait but the call itself, so its end has to release the send."""
    channel, conn = _manual_connection()
    sender = _Sender(conn)
    assert channel.taken(timeout=5.0) == b'frame'
    channel.end()
    assert sender.returned.wait(5.0), 'the send waited on a write the ended call can never make'
    assert isinstance(sender.outcome, wire.PeerDisconnected)


@pytest.mark.timeout(10.0)
def test_a_send_after_a_failed_send_raises_at_once():
    """The failed send records the end, so the next one does not wait for a receipt that never comes."""
    channel, conn = _manual_connection()
    sender = _Sender(conn)
    assert channel.taken(timeout=5.0) == b'frame'
    channel.end()
    assert sender.returned.wait(5.0), 'the send waited on a write the ended call can never make'
    refused = time.monotonic()
    with pytest.raises(wire.PeerDisconnected):
        conn.send(b'again')
    assert time.monotonic() - refused < 1.0, 'the second send waited instead of raising'


# ─── a host pinned to the IP addresses to dial ───────────────────────────────

# A name no resolver answers: a call that reaches a server under it dialled the pin.
_PINNED_HOST = 'policy.test'


def _pinned(port: int, *ips: str) -> wire.HostPortAddress:
    pinned_ips = tuple(ipaddress.ip_address(ip) for ip in ips or ('127.0.0.1',))
    return wire.HostPortAddress(_PINNED_HOST, port, wire.SESSION_PATH, '', pinned_ips=pinned_ips)


def _opened_channel(client_wire: client_grpc.GrpcClientWire, address: wire.HostPortAddress, monkeypatch):
    """The target and the options of the channel a probe of ``address`` opens."""
    opened = []
    monkeypatch.setattr(grpc, 'ssl_channel_credentials', lambda: 'roots')

    def opening(target: str, *_roots: str, options: list) -> MagicMock:
        opened.append((target, options))
        return MagicMock()

    monkeypatch.setattr(grpc, 'insecure_channel', opening)
    monkeypatch.setattr(grpc, 'secure_channel', opening)
    _probing(monkeypatch, None)
    client_wire.probe(address, None, 1.0)
    [(target, options)] = opened
    return target, dict(options)


def test_a_pinned_host_dials_its_pin_and_names_the_host_as_the_authority(monkeypatch):
    target, options = _opened_channel(client_grpc.GrpcClientWire(), _pinned(9000, '10.0.0.2', '10.0.0.3'), monkeypatch)
    assert target == 'ipv4:10.0.0.2:9000,10.0.0.3:9000'
    assert options['grpc.default_authority'] == f'{_PINNED_HOST}:9000'
    assert 'grpc.ssl_target_name_override' not in options


def test_an_ipv6_pin_dials_in_brackets(monkeypatch):
    target, _options = _opened_channel(client_grpc.GrpcClientWire(), _pinned(9000, '2001:db8::1'), monkeypatch)
    assert target == 'ipv6:[2001:db8::1]:9000'


def test_a_pin_of_two_families_dials_the_family_of_its_first_address(monkeypatch):
    """A gRPC target names one address family."""
    address = _pinned(9000, '2001:db8::1', '10.0.0.2', '2001:db8::2')
    target, _options = _opened_channel(client_grpc.GrpcClientWire(), address, monkeypatch)
    assert target == 'ipv6:[2001:db8::1]:9000,[2001:db8::2]:9000'


def test_the_tls_member_checks_the_certificate_of_a_pinned_host_against_the_host(monkeypatch):
    target, options = _opened_channel(client_grpc.GrpcTlsClientWire(), _pinned(443, '10.0.0.2'), monkeypatch)
    assert target == 'ipv4:10.0.0.2:443'
    assert options['grpc.ssl_target_name_override'] == _PINNED_HOST
    assert options['grpc.default_authority'] == f'{_PINNED_HOST}:443'


@pytest.mark.parametrize('client_wire', [client_grpc.GrpcClientWire(), client_grpc.GrpcTlsClientWire()])
def test_an_unpinned_host_leaves_the_name_and_the_authority_to_grpc(client_wire, monkeypatch):
    target, options = _opened_channel(client_wire, _ADDRESS, monkeypatch)
    assert target == 'gpu-host:9000'
    assert 'grpc.default_authority' not in options
    assert 'grpc.ssl_target_name_override' not in options


@pytest.fixture
def bare_server():
    """Starts a gRPC server on loopback with no method, so every call it reads it answers ``UNIMPLEMENTED``."""
    servers: list[grpc.Server] = []

    def start(credentials: grpc.ServerCredentials | None = None) -> int:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        if credentials is None:
            port = server.add_insecure_port('127.0.0.1:0')
        else:
            port = server.add_secure_port('127.0.0.1:0', credentials)
        server.start()
        servers.append(server)
        return port

    yield start
    for server in servers:
        server.stop(grace=None)


def _reaches_the_server(client_wire: client_grpc.GrpcClientWire, address: wire.HostPortAddress) -> None:
    """Every call on ``address`` reaches a server: an ``UNIMPLEMENTED`` answer is the server's own."""
    assert client_wire.probe(address, None, 5.0) is None
    with pytest.raises(wire.KeepaliveUnsupported):
        client_wire.keepalive(address, None, 5.0)
    client_wire.dial(address, None, 5.0).close()


def test_a_pinned_host_reaches_the_server_on_its_pin(bare_server):
    _reaches_the_server(client_grpc.GrpcClientWire(), _pinned(bare_server()))


def _trust_only(monkeypatch, root: bytes) -> None:
    """Verify every channel this test opens against ``root``, in place of the system's own roots."""
    system_roots = grpc.ssl_channel_credentials
    monkeypatch.setattr(grpc, 'ssl_channel_credentials', lambda: system_roots(root))


def test_a_tls_pin_checks_the_certificate_against_the_host(bare_server, certified_for, monkeypatch):
    certified = certified_for(_PINNED_HOST)
    _trust_only(monkeypatch, certified.certificate)
    port = bare_server(grpc.ssl_server_credentials([(certified.key, certified.certificate)]))
    _reaches_the_server(client_grpc.GrpcTlsClientWire(), _pinned(port))


def test_a_tls_pin_to_a_server_whose_certificate_names_another_host_is_refused_for_good(
    bare_server, certified_for, monkeypatch
):
    certified = certified_for('another.test')
    _trust_only(monkeypatch, certified.certificate)
    port = bare_server(grpc.ssl_server_credentials([(certified.key, certified.certificate)]))
    assert client_grpc.GrpcTlsClientWire().probe(_pinned(port), None, 5.0) is wire.Refusal.FINAL
    with pytest.raises(wire.ConnectRefused) as refused:
        client_grpc.GrpcTlsClientWire().dial(_pinned(port), None, 5.0)
    assert refused.value.refusal is wire.Refusal.FINAL
    assert 'Hostname Verification Check failed' in str(refused.value)


def test_a_pin_nothing_answers_on_is_no_answer():
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.close()
    address = _pinned(port)
    assert client_grpc.GrpcClientWire().probe(address, None, 1.0) is wire.Refusal.SILENT
    with pytest.raises(wire.ConnectRefused) as refused:
        client_grpc.GrpcClientWire().dial(address, None, 1.0)
    assert refused.value.refusal is wire.Refusal.SILENT
