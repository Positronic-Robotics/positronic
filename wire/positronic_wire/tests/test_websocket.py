"""The client side of the websocket wire, driven without a server."""

import socket
import ssl
from http import HTTPStatus
from unittest.mock import patch

import pytest
from positronic_wire import websocket, wire
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidHandshake, InvalidStatus
from websockets.http11 import Response

_ADDRESS = wire.SessionAddress('localhost', 8000, wire.SESSION_PATH, '')


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
        probed = websocket.WebsocketClientWire().probe(_ADDRESS._replace(query='fps=10'), {'Modal-Key': 'k'}, 3.0)
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
    assert websocket.WebsocketClientWire().probe(_ADDRESS._replace(port=1), None, 1.0) is wire.Refusal.COLD


def test_the_tls_member_dials_wss_and_probes_it():
    with patch('positronic_wire.websocket.connect') as connect:
        websocket.WebsocketTlsClientWire().dial(_ADDRESS, None, 1.0)
        websocket.WebsocketTlsClientWire().probe(_ADDRESS, None, 1.0)
    assert [call.args[0] for call in connect.call_args_list] == [
        'wss://localhost:8000/api/v1/session',
        'wss://localhost:8000',
    ]


@pytest.mark.parametrize(
    ('client_wire', 'address', 'session_url', 'api_url'),
    [
        (
            websocket.WebsocketClientWire(),
            _ADDRESS,
            'ws://localhost:8000/api/v1/session',
            'http://localhost:8000/api/v1',
        ),
        (
            websocket.WebsocketClientWire(),
            _ADDRESS._replace(port=80),
            'ws://localhost/api/v1/session',
            'http://localhost/api/v1',
        ),
        (
            websocket.WebsocketTlsClientWire(),
            _ADDRESS._replace(port=443),
            'wss://localhost/api/v1/session',
            'https://localhost/api/v1',
        ),
        (
            websocket.WebsocketTlsClientWire(),
            _ADDRESS._replace(port=8443),
            'wss://localhost:8443/api/v1/session',
            'https://localhost:8443/api/v1',
        ),
        (
            websocket.WebsocketClientWire(),
            _ADDRESS._replace(host='::1'),
            'ws://[::1]:8000/api/v1/session',
            'http://[::1]:8000/api/v1',
        ),
        (
            websocket.WebsocketClientWire(),
            _ADDRESS._replace(host='127.0.0.1'),
            'ws://127.0.0.1:8000/api/v1/session',
            'http://127.0.0.1:8000/api/v1',
        ),
        (
            websocket.WebsocketClientWire(),
            _ADDRESS._replace(path=wire.session_path('10000'), query='codec.fps=10&pad=false'),
            'ws://localhost:8000/api/v1/session/10000?codec.fps=10&pad=false',
            'http://localhost:8000/api/v1',
        ),
    ],
)
def test_the_member_spells_the_session_and_the_api_and_leaves_out_its_default_port(
    client_wire, address, session_url, api_url
):
    assert client_wire.session_url(address) == session_url
    assert client_wire.api_url(address) == api_url
