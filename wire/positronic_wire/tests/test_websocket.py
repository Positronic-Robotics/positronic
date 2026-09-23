"""The client side of the websocket wire, driven without a server."""

import dataclasses
import socket
import ssl
from http import HTTPStatus
from pathlib import Path
from unittest.mock import patch

import pytest
from positronic_wire import grpc as client_grpc
from positronic_wire import websocket, wire
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, InvalidStatus
from websockets.http11 import Response

_ADDRESS = wire.HostPortAddress('localhost', 8000, wire.SESSION_PATH, '')


def _refused_upgrade(status: HTTPStatus) -> InvalidStatus:
    return InvalidStatus(Response(status, 'refused', Headers()))


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
    refused_upgrade = _refused_upgrade(status)
    with (
        patch('positronic_wire.websocket.connect', side_effect=refused_upgrade),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        websocket.WebsocketClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is refused_upgrade


@pytest.mark.parametrize(
    ('raised', 'refusal'),
    [
        (TimeoutError('timed out'), wire.Refusal.COLD),
        (ssl.SSLError('reset'), wire.Refusal.COLD),
        (ConnectionClosedError(None, None), wire.Refusal.COLD),
        (InvalidHandshake('dropped'), wire.Refusal.COLD),
        (ConnectionRefusedError(111, 'Connection refused'), wire.Refusal.COLD),
        (socket.gaierror(socket.EAI_AGAIN, 'Temporary failure in name resolution'), wire.Refusal.COLD),
        (ssl.SSLCertVerificationError('unknown issuer'), wire.Refusal.FINAL),
        (socket.gaierror(socket.EAI_NONAME, 'Name or service not known'), wire.Refusal.FINAL),
        (socket.gaierror(socket.EAI_NODATA, 'No address associated with hostname'), wire.Refusal.FINAL),
    ],
)
def test_a_handshake_that_does_not_open_is_a_refusal_naming_the_url(raised, refusal):
    with (
        patch('positronic_wire.websocket.connect', side_effect=raised),
        pytest.raises(wire.ConnectRefused, match='ws://localhost:8000/api/v1/session') as refused,
    ):
        websocket.WebsocketClientWire().dial(_ADDRESS, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is raised


def test_a_probe_asks_the_host_root_with_the_headers():
    with patch('positronic_wire.websocket.connect', side_effect=_refused_upgrade(HTTPStatus.FORBIDDEN)) as connect:
        probed = websocket.WebsocketClientWire().probe(
            dataclasses.replace(_ADDRESS, query='fps=10'), {'Modal-Key': 'k'}, 3.0
        )
    assert probed is None
    assert connect.call_args.args == ('ws://localhost:8000',)
    assert connect.call_args.kwargs == {'open_timeout': 3.0, 'additional_headers': {'Modal-Key': 'k'}}


@pytest.mark.parametrize(
    ('status', 'refusal'),
    [
        (HTTPStatus.FORBIDDEN, None),
        (HTTPStatus.UNAUTHORIZED, wire.Refusal.FINAL),
        (HTTPStatus.NOT_FOUND, wire.Refusal.FINAL),
        (HTTPStatus.BAD_GATEWAY, wire.Refusal.COLD),
        (HTTPStatus.SERVICE_UNAVAILABLE, wire.Refusal.COLD),
        (HTTPStatus.TOO_MANY_REQUESTS, wire.Refusal.COLD),
    ],
)
def test_a_probe_reads_the_servers_own_403_as_the_server_and_every_other_status_as_dial_does(status, refusal):
    with patch('positronic_wire.websocket.connect', side_effect=_refused_upgrade(status)):
        assert websocket.WebsocketClientWire().probe(_ADDRESS, None, 1.0) is refusal


def test_a_probe_reads_a_handshake_that_opened_as_the_server():
    with patch('positronic_wire.websocket.connect') as connect:
        assert websocket.WebsocketClientWire().probe(_ADDRESS, None, 1.0) is None
    connect.return_value.close.assert_called_once()


def test_a_probe_of_a_port_nothing_answers_on_is_cold():
    assert websocket.WebsocketClientWire().probe(dataclasses.replace(_ADDRESS, port=1), None, 1.0) is wire.Refusal.COLD


def test_the_tls_member_dials_wss_and_probes_it():
    with patch('positronic_wire.websocket.connect') as connect:
        websocket.WebsocketTlsClientWire().dial(_ADDRESS, None, 1.0)
        websocket.WebsocketTlsClientWire().probe(_ADDRESS, None, 1.0)
    assert [call.args[0] for call in connect.call_args_list] == [
        'wss://localhost:8000/api/v1/session',
        'wss://localhost:8000',
    ]


@pytest.mark.parametrize(
    ('client_wire', 'address', 'session_url'),
    [
        (websocket.WebsocketClientWire(), _ADDRESS, 'ws://localhost:8000/api/v1/session'),
        (websocket.WebsocketClientWire(), dataclasses.replace(_ADDRESS, port=80), 'ws://localhost/api/v1/session'),
        (websocket.WebsocketTlsClientWire(), dataclasses.replace(_ADDRESS, port=443), 'wss://localhost/api/v1/session'),
        (
            websocket.WebsocketTlsClientWire(),
            dataclasses.replace(_ADDRESS, port=8443),
            'wss://localhost:8443/api/v1/session',
        ),
        (websocket.WebsocketClientWire(), dataclasses.replace(_ADDRESS, host='::1'), 'ws://[::1]:8000/api/v1/session'),
        (
            websocket.WebsocketClientWire(),
            dataclasses.replace(_ADDRESS, host='127.0.0.1'),
            'ws://127.0.0.1:8000/api/v1/session',
        ),
        (
            websocket.WebsocketClientWire(),
            dataclasses.replace(_ADDRESS, path=wire.session_path('10000'), query='codec.fps=10&pad=false'),
            'ws://localhost:8000/api/v1/session/10000?codec.fps=10&pad=false',
        ),
    ],
)
def test_the_member_spells_the_session_and_leaves_out_its_default_port(client_wire, address, session_url):
    assert client_wire.session_url(address) == session_url


def test_the_socket_wire_names_the_socket_it_dials_and_claims_no_authority():
    """A socket names no host and no port, so the log names the socket and the handshake a stand-in."""
    address = wire.UnixSocketAddress(Path('/run/policy.sock'), wire.session_path('10000'), 'fps=10')
    unix = websocket.WebsocketUnixClientWire()

    assert unix.session_url(address) == 'ws+unix:///run/policy.sock/api/v1/session/10000?fps=10'
    assert unix.handshake_url(address) == 'ws://localhost/api/v1/session/10000?fps=10'


def test_each_member_declares_the_address_it_dials():
    """A caller builds the address its wire names, and the other wire's address is a type error."""
    assert websocket.WebsocketClientWire().ADDRESS is wire.HostPortAddress
    assert websocket.WebsocketTlsClientWire().ADDRESS is wire.HostPortAddress
    assert websocket.WebsocketUnixClientWire().ADDRESS is wire.UnixSocketAddress


@pytest.mark.parametrize(
    ('raised', 'refusal'),
    [
        # Reached a live socket: a server that is starting or restarting raises each of these.
        (TimeoutError('handshake timed out'), wire.Refusal.COLD),
        (ConnectionResetError(104, 'Connection reset by peer'), wire.Refusal.COLD),
        (ConnectionAbortedError(103, 'Software caused connection abort'), wire.Refusal.COLD),
        (BrokenPipeError(32, 'Broken pipe'), wire.Refusal.COLD),
        (FileNotFoundError(2, 'No such file or directory'), wire.Refusal.COLD),
        # This process's own, and no retry reaches any of them.
        (PermissionError(13, 'Permission denied'), wire.Refusal.FINAL),
        (OSError(24, 'Too many open files'), wire.Refusal.FINAL),
    ],
)
def test_a_socket_dial_that_did_not_open_says_whether_a_retry_can_reach_it(raised, refusal, tmp_path):
    """A connection-level failure reached the socket, so it reads as it does on a port. An absent path
    is cold because a misspelt one and a socket nobody has bound yet look the same from here."""
    address = wire.UnixSocketAddress(tmp_path / 'absent.sock', wire.session_path(), '')

    with (
        patch('positronic_wire.websocket.unix_connect', side_effect=raised),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        websocket.WebsocketUnixClientWire().dial(address, None, 1.0)
    assert refused.value.refusal is refusal
    assert refused.value.__cause__ is raised


def test_a_refusal_from_a_path_that_is_not_a_socket_is_final(tmp_path):
    """A path holding a regular file is refused in the same words as a socket nobody listens on, so
    the refusal reads the path: only one of the two can ever come good."""
    not_a_socket = tmp_path / 'regular.file'
    not_a_socket.write_text('not a socket')
    address = wire.UnixSocketAddress(not_a_socket, wire.session_path(), '')

    with (
        patch('positronic_wire.websocket.unix_connect', side_effect=ConnectionRefusedError(111, 'Connection refused')),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        websocket.WebsocketUnixClientWire().dial(address, None, 1.0)
    assert refused.value.refusal is wire.Refusal.FINAL


def test_a_socket_address_refuses_a_relative_path():
    """`--policy.address.uds=policy.sock` names a different socket to each caller, so the address refuses it."""
    with pytest.raises(ValueError, match='relative socket path'):
        wire.UnixSocketAddress(Path('policy.sock'), wire.session_path(), '')


_SESSION = wire.SESSION_PATH
# One row per URL component. A tuple is the address the URL names, and a port of ``None`` is the member's own
# ``DEFAULT_PORT``. A string is the refusal.
_HOST_PORT_GRAMMAR = [
    # Whitespace around the URL
    ('  ws://localhost:8000  ', ('localhost', 8000, _SESSION, '')),
    # Scheme
    ('grpcs://localhost:8000', ('localhost', 8000, _SESSION, '')),
    ('localhost:8000', ('localhost', 8000, _SESSION, '')),
    ('localhost:8000/api/v1/session?next=http://peer', ('localhost', 8000, _SESSION, 'next=http://peer')),
    # User
    ('ws://user:pass@localhost:8000/api/v1/session', 'names a user'),
    ('ws://:secret@localhost:8000', 'names a user'),
    ('user@localhost:8000', 'names a user'),
    # Host
    ('ws://GPU-Box:9000', ('gpu-box', 9000, _SESSION, '')),
    # Empty host
    ('ws://:8000', 'no host'),
    ('ws:///api/v1/session', 'no host'),
    # IPv6 literal
    ('ws://[::1]:8000', ('::1', 8000, _SESSION, '')),
    ('[::1]', ('::1', None, _SESSION, '')),
    # Port
    ('ws://localhost', ('localhost', None, _SESSION, '')),
    ('ws://localhost:', ('localhost', None, _SESSION, '')),
    ('ws://localhost:port', 'Port could not be cast'),
    ('ws://localhost:70000', 'Port out of range'),
    # Path
    ('ws://localhost:8000/api/v1/session/org/model', ('localhost', 8000, '/api/v1/session/org/model', '')),
    ('ws://localhost:8000/healthz', 'unexpected path'),
    ('ws://localhost:8000/api/v1/sessions', 'unexpected path'),
    # Trailing slash
    ('ws://localhost:8000/', ('localhost', 8000, _SESSION, '')),
    ('ws://localhost:8000/api/v1/session/', ('localhost', 8000, _SESSION, '')),
    ('ws://localhost:8000/api/v1/session/model/', ('localhost', 8000, '/api/v1/session/model/', '')),
    # A path that repeats the session route
    ('ws://localhost:8000/api/v1/session/api/v1/session', ('localhost', 8000, '/api/v1/session/api/v1/session', '')),
    # Params
    ('ws://localhost:8000/api/v1/session;x', 'unexpected path'),
    ('ws://localhost:8000/api/v1/session/model;v=1', ('localhost', 8000, '/api/v1/session/model;v=1', '')),
    # Query
    (
        'ws://localhost:8000/api/v1/session?codec.fps=10&pad=false',
        ('localhost', 8000, _SESSION, 'codec.fps=10&pad=false'),
    ),
    # Fragment
    ('ws://localhost:8000/api/v1/session/model#revision', 'names a fragment'),
    ('ws://localhost:8000#', 'names a fragment'),
    # Percent-encoding
    (
        'ws://gpu%2dbox:8000/api/v1/session/org%2Fmodel?x=a%20b',
        ('gpu%2dbox', 8000, '/api/v1/session/org%2Fmodel', 'x=a%20b'),
    ),
]


@pytest.mark.parametrize(
    'client_wire',
    [
        websocket.WebsocketClientWire(),
        websocket.WebsocketTlsClientWire(),
        client_grpc.GrpcClientWire(),
        client_grpc.GrpcTlsClientWire(),
    ],
    ids=lambda client_wire: client_wire.NAME,
)
@pytest.mark.parametrize(('url', 'named'), _HOST_PORT_GRAMMAR)
def test_a_host_port_member_reads_a_url_by_the_grammar(client_wire, url, named):
    if isinstance(named, str):
        with pytest.raises(ValueError, match=named):
            client_wire.address_of(url)
        return
    host, port, path, query = named
    address = wire.HostPortAddress(host, client_wire.DEFAULT_PORT if port is None else port, path, query)
    assert client_wire.address_of(url) == address
    assert client_wire.address_of(client_wire.session_url(address)) == address


# One row per URL component. A tuple is the socket, the route and the query the URL names. A string is the
# refusal.
_SOCKET_GRAMMAR = [
    # Whitespace around the URL
    ('  unix:///run/policy.sock  ', ('/run/policy.sock', _SESSION, '')),
    # Scheme
    ('ws+unix:///run/policy.sock', ('/run/policy.sock', _SESSION, '')),
    ('/run/policy.sock', ('/run/policy.sock', _SESSION, '')),
    # User
    ('unix://user@/run/policy.sock', 'names a user'),
    # Host, IPv6 literal, port
    ('unix://localhost/run/policy.sock', 'names no host'),
    ('unix://[::1]/run/policy.sock', 'names no host'),
    ('unix://:8000/run/policy.sock', 'names no host'),
    # Path
    ('unix:///run/policy.sock/api/v1/session/10000', ('/run/policy.sock', '/api/v1/session/10000', '')),
    ('unix:///run/api/v1/session.sock', ('/run/api/v1/session.sock', _SESSION, '')),
    ('unix:///run/api/v1/sessions/policy.sock', ('/run/api/v1/sessions/policy.sock', _SESSION, '')),
    ('unix:///', 'names none'),
    ('unix:///run/', 'names none'),
    ('unix:///api/v1/session', 'names none'),
    # Trailing slash
    ('unix:///run/policy.sock/', 'names none'),
    ('unix:///run/policy.sock%2F', 'names none'),
    ('unix:///run/policy.sock/.', 'names none'),
    ('unix:///run/policy.sock%2F..', 'names none'),
    ('unix:///run/policy.sock/api/v1/session/', ('/run/policy.sock', _SESSION, '')),
    ('unix:///run/policy.sock/api/v1/session/model/', ('/run/policy.sock', '/api/v1/session/model/', '')),
    # A path that repeats the session route
    (
        'unix:///run/policy.sock/api/v1/session/api/v1/session',
        ('/run/policy.sock', '/api/v1/session/api/v1/session', ''),
    ),
    ('unix:///run%2Fapi%2Fv1%2Fsession%2Fpolicy.sock', 'holds the session route'),
    ('unix:///run/policy.sock%2Fapi%2Fv1%2Fsession', 'holds the session route'),
    # Params
    ('unix:///run/policy;v=1.sock/api/v1/session/model;v=1', ('/run/policy;v=1.sock', '/api/v1/session/model;v=1', '')),
    # Query
    ('unix:///run/policy.sock?fps=10', ('/run/policy.sock', _SESSION, 'fps=10')),
    # Fragment
    ('unix:///run/policy#1.sock', 'names a fragment'),
    # Percent-encoding
    ('unix:///run/my%20policy%3F%231.sock?x=a%20b', ('/run/my policy?#1.sock', _SESSION, 'x=a%20b')),
    ('unix:///run/policy%2Fsock.sock', ('/run/policy/sock.sock', _SESSION, '')),
    ('unix:///run/policy%00.sock', 'holds a NUL byte'),
]


@pytest.mark.parametrize(('url', 'named'), _SOCKET_GRAMMAR)
def test_the_socket_member_reads_a_url_by_the_grammar(url, named):
    client_wire = websocket.WebsocketUnixClientWire()
    if isinstance(named, str):
        with pytest.raises(ValueError, match=named):
            client_wire.address_of(url)
        return
    uds, path, query = named
    address = wire.UnixSocketAddress(Path(uds), path, query)
    assert client_wire.address_of(url) == address
    assert client_wire.address_of(client_wire.session_url(address)) == address
