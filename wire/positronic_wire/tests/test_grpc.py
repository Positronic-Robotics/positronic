"""The client side of the gRPC wire, driven without a server."""

import dataclasses
import queue
import threading
import time
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
        (grpc.StatusCode.DEADLINE_EXCEEDED, wire.Refusal.COLD),
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


def test_a_port_that_never_answers_is_cold():
    """Nothing listens on port 1; the call never reaches a server."""
    assert (
        client_grpc.GrpcClientWire().probe(dataclasses.replace(_ADDRESS, host='localhost', port=1), None, 0.2)
        is wire.Refusal.COLD
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
