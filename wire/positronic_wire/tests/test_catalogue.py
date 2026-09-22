"""The model catalogue, read by each wire over the transport it carries sessions on."""

import json
import socket
import socketserver
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from positronic_wire import wire
from positronic_wire.websocket import WebsocketClientWire, WebsocketUnixClientWire

_MODELS = ['checkpoint-10000', 'org/repo']


def _handler(status: int, seen: list[dict[str, str]]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler dispatches on this name
            seen.append({'path': self.path, **{k.lower(): v for k, v in self.headers.items()}})
            body = json.dumps({wire.MODELS_KEY: _MODELS}).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            # This server handles one connection at a time, so a kept-alive one would hold the next
            # request until the client closed, and the descriptor count would read the server's own.
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 — the base class names it this
            """Serve quietly: the test reads what it asked for from ``seen``."""

    return Handler


class _UnixHTTPServer(HTTPServer):
    """``HTTPServer`` on a Unix socket. It binds a path, so it has no host to name and no port."""

    address_family = socket.AF_UNIX

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = 'localhost', 0


@pytest.fixture
def short_tmp_dir():
    """A directory short enough for a socket path: ``sun_path`` holds 104 bytes, and macOS's
    ``tmp_path`` alone spends more than that."""
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
        yield Path(directory)


@pytest.fixture
def catalogue(request, short_tmp_dir):
    """A server answering the catalogue route, on a port or on a socket as the test asks."""
    status, over_a_socket = getattr(request, 'param', (HTTPStatus.OK, False))
    seen: list[dict[str, str]] = []
    if over_a_socket:
        uds = short_tmp_dir / 'p.sock'
        # `HTTPServer` is typed for AF_INET; this subclass binds a path, which is the whole point.
        server: HTTPServer = _UnixHTTPServer(str(uds), _handler(status, seen))  # type: ignore[arg-type]
        address: wire.SessionAddress = wire.UnixSocketAddress(uds, wire.session_path(), '')
        client_wire: wire.ClientWire = WebsocketUnixClientWire()
    else:
        server = HTTPServer(('127.0.0.1', 0), _handler(status, seen))
        address = wire.HostPortAddress('127.0.0.1', server.server_address[1], wire.session_path(), '')
        client_wire = WebsocketClientWire()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield client_wire, address, seen
    server.shutdown()
    server.server_close()
    thread.join(timeout=5.0)


@pytest.mark.parametrize('catalogue', [(HTTPStatus.OK, False), (HTTPStatus.OK, True)], indirect=True)
def test_a_wire_reads_the_catalogue_on_the_transport_it_carries_sessions_on(catalogue):
    """The network member reads it over TCP and the socket member over the socket it dials."""
    client_wire, address, seen = catalogue

    assert client_wire.list_models(address, None, 5.0) == _MODELS
    assert [asked['path'] for asked in seen] == [wire.MODELS_PATH]


@pytest.mark.parametrize('catalogue', [(HTTPStatus.OK, False), (HTTPStatus.OK, True)], indirect=True)
def test_the_catalogue_read_carries_the_headers_a_session_dials_with(catalogue):
    """An edge that authenticates on them lets the read through, as it lets a session through."""
    client_wire, address, seen = catalogue

    client_wire.list_models(address, {'Modal-Key': 'k'}, 5.0)

    assert seen[0]['modal-key'] == 'k'


@pytest.mark.parametrize(
    ('catalogue', 'refusal'),
    [
        ((HTTPStatus.SERVICE_UNAVAILABLE, False), wire.Refusal.COLD),
        ((HTTPStatus.FORBIDDEN, False), wire.Refusal.FORBIDDEN),
        ((HTTPStatus.UNAUTHORIZED, False), wire.Refusal.FINAL),
        ((HTTPStatus.SERVICE_UNAVAILABLE, True), wire.Refusal.COLD),
    ],
    indirect=['catalogue'],
)
def test_a_catalogue_that_does_not_answer_200_refuses_in_the_terms_a_dial_uses(catalogue, refusal):
    client_wire, address, _seen = catalogue

    with pytest.raises(wire.ConnectRefused) as refused:
        client_wire.list_models(address, None, 5.0)
    assert refused.value.refusal is refusal


def test_a_catalogue_on_a_socket_nobody_bound_is_cold(short_tmp_dir):
    """The same reading a dial gets: the path can still become a socket, so a retry can reach it."""
    address = wire.UnixSocketAddress(short_tmp_dir / 'absent.sock', wire.session_path(), '')

    with pytest.raises(wire.ConnectRefused) as refused:
        WebsocketUnixClientWire().list_models(address, None, 1.0)
    assert refused.value.refusal is wire.Refusal.COLD


def test_a_catalogue_on_a_port_nothing_answers_on_is_cold():
    address = wire.HostPortAddress('127.0.0.1', 1, wire.session_path(), '')

    with pytest.raises(wire.ConnectRefused) as refused:
        WebsocketClientWire().list_models(address, None, 1.0)
    assert refused.value.refusal is wire.Refusal.COLD
