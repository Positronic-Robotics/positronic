"""The client side of the roboarena wire, driven without a server."""

import socket
import ssl
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest
from positronic_wire import roboarena, wire
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, InvalidStatus
from websockets.http11 import Response

_ADDRESS = roboarena.RoboarenaAddress('a-partner-host', 8000)
_ANNOUNCEMENT = b'\x81\xa8endpoint\xa5infer'


def _refused_upgrade(status: HTTPStatus) -> InvalidStatus:
    return InvalidStatus(Response(status, 'refused', Headers()))


@pytest.mark.parametrize(
    ('raised', 'refusal'),
    [
        (TimeoutError('timed out'), wire.Refusal.COLD),
        (ConnectionRefusedError(111, 'Connection refused'), wire.Refusal.COLD),
        (ConnectionClosedError(None, None), wire.Refusal.COLD),
        (InvalidHandshake('dropped'), wire.Refusal.COLD),
        (socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution'), wire.Refusal.COLD),
        (ssl.SSLCertVerificationError('unknown issuer'), wire.Refusal.FINAL),
        (socket.gaierror(socket.EAI_NONAME, 'Name or service not known'), wire.Refusal.FINAL),
    ],
)
def test_a_handshake_that_does_not_open_is_a_refusal_naming_the_root(raised, refusal):
    with (
        patch('positronic_wire.roboarena.connect', side_effect=raised),
        pytest.raises(wire.ConnectRefused, match='ws://a-partner-host:8000') as refused,
    ):
        roboarena.RoboarenaClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is raised


@pytest.mark.parametrize(
    ('status', 'refusal'),
    [
        (HTTPStatus.FORBIDDEN, wire.Refusal.FORBIDDEN),
        (HTTPStatus.SERVICE_UNAVAILABLE, wire.Refusal.COLD),
        (HTTPStatus.UNAUTHORIZED, wire.Refusal.FINAL),
        (HTTPStatus.NOT_FOUND, wire.Refusal.FINAL),
    ],
)
def test_an_edge_in_front_of_the_server_refuses_as_it_does_on_the_websocket_wire(status, refusal):
    """A partner may publish the port behind a proxy, and this wire reads that answer the same way."""
    with (
        patch('positronic_wire.roboarena.connect', side_effect=_refused_upgrade(status)),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        roboarena.RoboarenaClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal


def test_a_dial_opens_the_bare_root_and_leaves_the_first_frame_unread():
    """The server announces its configuration on connect, and ``dial`` leaves it for ``recv``."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = _ANNOUNCEMENT
        connection = roboarena.RoboarenaClientWire().dial(_ADDRESS, None, 3.0)

    assert connect.call_args.args == ('ws://a-partner-host:8000',)
    assert connect.call_args.kwargs['open_timeout'] == 3.0
    assert connect.call_args.kwargs['compression'] is None
    assert connect.call_args.kwargs['max_size'] == wire.MAX_MESSAGE_BYTES
    assert connection.recv() == _ANNOUNCEMENT


def test_a_probe_reads_the_announcement_and_closes():
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = _ANNOUNCEMENT
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 3.0) is None

    assert connect.call_args.args == ('ws://a-partner-host:8000',)
    connect.return_value.close.assert_called_once()


@pytest.mark.parametrize('verb', [roboarena.RoboarenaClientWire.dial, roboarena.RoboarenaClientWire.probe])
def test_a_server_another_party_runs_gets_no_headers_from_the_caller(verb):
    """A caller hands every wire its edge headers, and this wire opens its handshake without them."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = _ANNOUNCEMENT
        verb(roboarena.RoboarenaClientWire(), _ADDRESS, {'Modal-Key': 'k'}, 3.0)
    assert connect.call_args.kwargs.get('additional_headers') is None


def test_a_port_that_accepts_and_announces_nothing_is_cold():
    """The announcement is the whole readiness signal, so a connection that carries none says nothing yet."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.side_effect = TimeoutError('timed out')
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0) is wire.Refusal.COLD
    connect.return_value.close.assert_called_once()


def test_a_probe_reports_the_refusal_a_dial_would_have_raised():
    with patch('positronic_wire.roboarena.connect', side_effect=_refused_upgrade(HTTPStatus.UNAUTHORIZED)):
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0) is wire.Refusal.FINAL


def test_a_probe_of_a_port_nothing_answers_on_is_cold():
    assert roboarena.RoboarenaClientWire().probe(roboarena.RoboarenaAddress('localhost', 1), None, 1.0) is (
        wire.Refusal.COLD
    )


def test_a_text_frame_raises_the_servers_text():
    connection = roboarena.RoboarenaClientConnection(MagicMock(**{'recv.return_value': 'CUDA out of memory'}))
    with pytest.raises(roboarena.TextAnswer, match='CUDA out of memory') as answered:
        connection.recv()
    assert answered.value.text == 'CUDA out of memory'


def test_a_probe_answered_in_text_raises_the_servers_text():
    """A backend that reports a failure is not cold: a retry does not change the text."""
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.return_value = 'CUDA out of memory'
        with pytest.raises(roboarena.TextAnswer, match='CUDA out of memory'):
            roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0)
    connect.return_value.close.assert_called_once()


def test_a_probe_whose_peer_closes_before_the_announcement_is_cold():
    with patch('positronic_wire.roboarena.connect') as connect:
        connect.return_value.recv.side_effect = ConnectionClosedError(None, None)
        assert roboarena.RoboarenaClientWire().probe(_ADDRESS, None, 1.0) is wire.Refusal.COLD
    connect.return_value.close.assert_called_once()


def test_a_read_on_a_closed_connection_says_the_peer_ended_the_session():
    closed = ConnectionClosedError(None, None)
    connection = roboarena.RoboarenaClientConnection(MagicMock(**{'recv.side_effect': closed}))
    with pytest.raises(wire.PeerDisconnected) as ended:
        connection.recv()
    assert ended.value.__cause__ is closed


def test_a_send_on_a_closed_connection_says_the_peer_ended_the_session():
    closed = ConnectionClosedError(None, None)
    connection = roboarena.RoboarenaClientConnection(MagicMock(**{'send.side_effect': closed}))
    with pytest.raises(wire.PeerDisconnected) as ended:
        connection.send(b'')
    assert ended.value.__cause__ is closed


@pytest.mark.parametrize(
    ('address', 'session_url'),
    [
        (roboarena.RoboarenaAddress('a-partner-host', 8000), 'ws://a-partner-host:8000'),
        (roboarena.RoboarenaAddress('127.0.0.1', 9000), 'ws://127.0.0.1:9000'),
        (roboarena.RoboarenaAddress('::1', 9000), 'ws://[::1]:9000'),
    ],
)
def test_the_wire_names_the_root_it_dials_with_the_port_the_partner_published(address, session_url):
    """No port is left out: a roboarena server publishes no default, so every address states one."""
    assert roboarena.RoboarenaClientWire().session_url(address) == session_url


def test_the_address_carries_the_host_and_the_port_alone():
    """The protocol routes on a key inside each frame, so a session names no route and no params."""
    address = roboarena.RoboarenaAddress('a-partner-host', 8000)
    assert (address.path, address.query) == ('', '')
    assert address.at_root() is address
    assert roboarena.RoboarenaClientWire().ADDRESS is roboarena.RoboarenaAddress
