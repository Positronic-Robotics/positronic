"""The client side of a Modal Server with sticky sessions."""

import atexit
import email.message
import http.client
import json
import pickle
import socket
import ssl
import threading
import urllib.error
from collections.abc import Iterator, Mapping
from http import HTTPStatus
from unittest.mock import patch

import pytest
from positronic_wire import modal, websocket, wire

_ADDRESS = wire.HostPortAddress('app-server.eu-west.modal.direct', 443, wire.SESSION_PATH, '')
_CREDENTIALS = {'Modal-Key': 'KEY', 'Modal-Secret': 'SECRET'}
_ROOT = 'https://app-server.eu-west.modal.direct'


class FakeEdge:
    """The Modal edge's session calls. A start answers the next token, unless ``refuse`` names a status first."""

    def __init__(self) -> None:
        self.starts: list[Mapping[str, str]] = []
        self.ended: list[str] = []
        self.refuse: list[int] = []
        # A start blocks until this is set, as a start on a cold app waits for a container.
        self.container_free = threading.Event()
        self.container_free.set()
        self._issued = 0

    def post(self, url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        if url.endswith(modal.TERMINATE_PATH):
            self.ended.append(headers[modal.SESSION_TOKEN_HEADER])
            return b''
        assert url.endswith(modal.START_PATH)
        self.starts.append({**headers, 'url': url})
        if self.refuse:
            raise urllib.error.HTTPError(url, self.refuse.pop(0), 'refused', email.message.Message(), None)
        self.container_free.wait()
        self._issued += 1
        return json.dumps({modal.START_ANSWER_TOKEN: f'token-{self._issued}'}).encode()


class FakeConnection(wire.ClientConnection):
    def __init__(self, token: str):
        self.token = token
        self.dropped = False

    def send(self, message: bytes) -> None:
        if self.dropped:
            raise wire.PeerDisconnected('dropped')

    def recv(self, timeout: float | None = None) -> bytes:
        if self.dropped:
            raise wire.PeerDisconnected('dropped')
        return b'frame'

    def close(self) -> str:
        return 'closed'


class FakeWire(wire.ClientWire[wire.HostPortAddress]):
    """A websocket member that records the token on each call. ``ended`` tokens are refused as the edge refuses a
    session that has ended (404); ``refusals`` refuse the next dials."""

    NAME = 'websocket_tls'
    ADDRESS = wire.HostPortAddress

    def __init__(self) -> None:
        self.dialled: list[str] = []
        self.called: list[str] = []
        self.ended: set[str] = set()
        self.refusals: list[wire.Refusal] = []

    @staticmethod
    def _token(headers: Mapping[str, str] | None) -> str:
        assert headers is not None
        return headers[modal.SESSION_HEADER].removeprefix('Bearer ')

    def session_url(self, address: wire.HostPortAddress) -> str:
        return f'wss://{address.host}{address.path}'

    def dial(self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float):
        token = self._token(headers)
        self.dialled.append(token)
        if self.refusals:
            raise wire.ConnectRefused(self.refusals.pop(0), 'refused')
        if token in self.ended:
            raise wire.ConnectRefused(wire.Refusal.FINAL, '404')
        return FakeConnection(token)

    def keepalive(self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, timeout: float) -> int:
        token = self._token(headers)
        self.called.append(token)
        assert headers is not None and headers[modal.IDLE_TIMEOUT_HEADER] == str(modal.IDLE_TIMEOUT)
        if token in self.ended:
            raise wire.ConnectRefused(wire.Refusal.FINAL, '404')
        return 60

    def probe(self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float):
        token = self._token(headers)
        self.called.append(token)
        return wire.Refusal.FINAL if token in self.ended else None


@pytest.fixture
def edge() -> Iterator[FakeEdge]:
    fake = FakeEdge()
    with patch.object(modal.ModalClientWire, '_post', staticmethod(fake.post)):
        yield fake
        fake.container_free.set()


@pytest.fixture
def inner() -> FakeWire:
    return FakeWire()


def _wire(inner: wire.ClientWire[wire.HostPortAddress]) -> modal.ModalClientWire:
    """A wire whose sessions end with the test, while the fake edge answers, and not at the process exit."""
    client_wire = modal.ModalClientWire(inner)
    atexit.unregister(client_wire.end_all)
    return client_wire


@pytest.fixture
def modal_wire(edge: FakeEdge, inner: FakeWire) -> Iterator[modal.ModalClientWire]:
    client_wire = _wire(inner)
    yield client_wire
    client_wire.end_all()


def _drop(conn: wire.ClientConnection) -> None:
    """Drop the connection under ``conn``, and close it as a caller does after the drop."""
    assert isinstance(conn, modal.ModalClientConnection)
    inner_conn = conn._inner
    assert isinstance(inner_conn, FakeConnection)
    inner_conn.dropped = True
    with pytest.raises(wire.PeerDisconnected):
        conn.recv(1.0)
    conn.close()


def test_a_dial_starts_a_session_with_the_credentials_and_dials_its_token(modal_wire, edge, inner):
    conn = modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert conn.recv(1.0) == b'frame'
    assert len(edge.starts) == 1
    assert edge.starts[0]['Modal-Key'] == 'KEY'
    assert edge.starts[0][modal.IDLE_TIMEOUT_HEADER] == str(modal.IDLE_TIMEOUT)
    assert inner.dialled == ['token-1']


def test_a_closed_connection_ends_its_session(modal_wire, edge):
    conn = modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)
    conn.close()
    conn.close()

    assert edge.ended == ['token-1']


def test_a_dial_raises_cold_while_the_start_waits_and_the_next_dial_continues_the_same_start(modal_wire, edge, inner):
    edge.container_free.clear()

    with pytest.raises(wire.ConnectRefused) as refused:
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 0.05)
    assert refused.value.refusal is wire.Refusal.COLD
    with pytest.raises(wire.ConnectRefused):
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 0.05)
    edge.container_free.set()
    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert len(edge.starts) == 1
    assert inner.dialled == ['token-1']


@pytest.mark.parametrize('status', [HTTPStatus.GATEWAY_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS])
def test_a_start_the_edge_answers_with_a_5xx_or_a_429_continues_under_the_same_key(
    modal_wire, edge, monkeypatch, status
):
    monkeypatch.setattr('positronic_wire.modal.time.sleep', lambda seconds: None)
    edge.refuse = [status]

    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    keys = [start[modal.IDEMPOTENCY_KEY_HEADER] for start in edge.starts]
    stamps = [start[modal.TIMESTAMP_HEADER] for start in edge.starts]
    assert len(keys) == 2 and keys[0] == keys[1] and stamps[0] == stamps[1]


@pytest.mark.parametrize(
    ('status', 'refusal'),
    [
        (HTTPStatus.UNAUTHORIZED, wire.Refusal.FORBIDDEN),
        (HTTPStatus.FORBIDDEN, wire.Refusal.FORBIDDEN),
        (HTTPStatus.NOT_FOUND, wire.Refusal.FINAL),
    ],
)
def test_a_start_the_edge_refuses_surfaces_its_refusal_on_the_dial(modal_wire, edge, status, refusal):
    edge.refuse = [status]

    with pytest.raises(wire.ConnectRefused) as refused:
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert refused.value.refusal is refusal


def test_a_dial_after_a_refused_start_starts_again(modal_wire, edge, inner):
    edge.refuse = [HTTPStatus.UNAUTHORIZED]
    with pytest.raises(wire.ConnectRefused):
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert len(edge.starts) == 2
    assert inner.dialled == ['token-1']


def test_a_reconnect_after_a_drop_dials_the_same_session(modal_wire, edge, inner):
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))

    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert inner.dialled == ['token-1', 'token-1']
    assert len(edge.starts) == 1
    assert edge.ended == []


def test_a_dropped_session_waits_for_a_dial_on_its_own_address(modal_wire, edge, inner):
    other_query = wire.HostPortAddress(_ADDRESS.host, _ADDRESS.port, wire.SESSION_PATH, 'fps=10')
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))

    modal_wire.dial(other_query, _CREDENTIALS, 5.0)
    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert inner.dialled == ['token-1', 'token-2', 'token-1']


def test_a_reconnect_refused_for_a_while_keeps_its_session_for_its_own_address(modal_wire, edge, inner):
    other_query = wire.HostPortAddress(_ADDRESS.host, _ADDRESS.port, wire.SESSION_PATH, 'fps=10')
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))
    inner.refusals = [wire.Refusal.COLD]
    with pytest.raises(wire.ConnectRefused):
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    modal_wire.dial(other_query, _CREDENTIALS, 5.0)
    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert inner.dialled == ['token-1', 'token-1', 'token-2', 'token-1']


def test_sessions_under_other_credentials_are_kept_apart(modal_wire, edge, inner):
    other_credentials = {'Modal-Key': 'OTHER', 'Modal-Secret': 'OTHER'}
    modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0)

    conn = modal_wire.dial(_ADDRESS, other_credentials, 5.0)
    conn.close()

    assert inner.dialled == ['token-2']
    assert [start['Modal-Key'] for start in edge.starts] == ['KEY', 'OTHER']
    assert edge.ended == ['token-2']


def test_a_timed_out_connection_is_not_a_drop_and_ends_its_session(modal_wire, edge):
    conn = modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)
    conn.close()

    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert edge.ended == ['token-1']
    assert len(edge.starts) == 2


class FakeTimer:
    """Stands in for ``threading.Timer``: ``fire`` runs the call the wire scheduled, and no time passes."""

    made: list['FakeTimer'] = []

    def __init__(self, interval: float, function, args=()) -> None:
        self.interval, self.function, self.args = interval, function, args
        self.daemon = False
        FakeTimer.made.append(self)

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        pass

    def fire(self) -> None:
        self.function(*self.args)


def test_a_dropped_session_no_dial_takes_ends_after_the_idle_timeout(modal_wire, edge, inner, monkeypatch):
    monkeypatch.setattr(FakeTimer, 'made', [])
    monkeypatch.setattr('positronic_wire.modal.threading.Timer', FakeTimer)
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))
    [timer] = FakeTimer.made

    timer.fire()

    assert timer.interval == modal.IDLE_TIMEOUT
    assert edge.ended == ['token-1']
    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)
    assert inner.dialled == ['token-1', 'token-2']


def test_a_wire_given_an_idle_timeout_starts_and_parks_its_sessions_with_it(edge, inner, monkeypatch):
    monkeypatch.setattr(FakeTimer, 'made', [])
    monkeypatch.setattr('positronic_wire.modal.threading.Timer', FakeTimer)
    client_wire = modal.ModalClientWire(inner, idle_timeout=30)
    atexit.unregister(client_wire.end_all)

    _drop(client_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))

    assert edge.starts[0][modal.IDLE_TIMEOUT_HEADER] == '30'
    assert FakeTimer.made[0].interval == 30
    client_wire.end_all()


def test_a_dropped_session_a_dial_took_is_not_ended_by_its_timer(modal_wire, edge, monkeypatch):
    monkeypatch.setattr(FakeTimer, 'made', [])
    monkeypatch.setattr('positronic_wire.modal.threading.Timer', FakeTimer)
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))
    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    FakeTimer.made[0].fire()

    assert edge.ended == []


def test_a_session_parked_again_is_not_ended_by_the_timer_of_its_first_park(modal_wire, edge, inner, monkeypatch):
    monkeypatch.setattr(FakeTimer, 'made', [])
    monkeypatch.setattr('positronic_wire.modal.threading.Timer', FakeTimer)
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))
    first_park, second_park = FakeTimer.made

    first_park.fire()

    assert edge.ended == []
    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)
    assert inner.dialled == ['token-1', 'token-1', 'token-1']
    second_park.fire()
    assert edge.ended == []


def test_a_dial_on_a_session_that_has_ended_waits_for_a_new_one(modal_wire, edge, inner):
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))
    inner.ended.add('token-1')

    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert inner.dialled == ['token-1', 'token-1', 'token-2']
    assert edge.ended == ['token-1']


def test_a_dial_refused_for_a_while_keeps_its_session_for_the_next_dial(modal_wire, edge, inner):
    inner.refusals = [wire.Refusal.COLD]
    with pytest.raises(wire.ConnectRefused):
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert inner.dialled == ['token-1', 'token-1']
    assert edge.ended == []


def test_a_dial_refused_for_good_on_a_new_session_ends_it_and_surfaces(modal_wire, edge, inner):
    inner.refusals = [wire.Refusal.FINAL]

    with pytest.raises(wire.ConnectRefused) as refused:
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert refused.value.refusal is wire.Refusal.FINAL
    assert edge.ended == ['token-1']


def test_a_keepalive_starts_the_spare_and_the_next_dial_takes_it(modal_wire, edge, inner):
    assert modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0) == 60
    assert modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0) == 60

    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert inner.called == ['token-1', 'token-1']
    assert inner.dialled == ['token-1']
    assert len(edge.starts) == 1


def test_a_keepalive_during_a_session_reaches_its_container(modal_wire, edge, inner):
    modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0)

    assert inner.called == ['token-1']
    assert len(edge.starts) == 1


def test_a_keepalive_on_a_spare_that_has_ended_runs_again_on_a_new_one(modal_wire, edge, inner):
    modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0)
    inner.ended.add('token-1')

    assert modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0) == 60

    assert inner.called == ['token-1', 'token-1', 'token-2']


def test_a_keepalive_raises_cold_while_the_spare_starts(modal_wire, edge):
    edge.container_free.clear()

    with pytest.raises(wire.ConnectRefused) as refused:
        modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 0.05)

    assert refused.value.refusal is wire.Refusal.COLD


def test_a_probe_is_unsupported_and_starts_no_session(modal_wire, edge):
    with pytest.raises(wire.ProbeUnsupported):
        modal_wire.probe(_ADDRESS, _CREDENTIALS, 5.0)

    assert edge.starts == []


def test_a_start_the_edge_drops_before_its_answer_continues_under_the_same_key(modal_wire, edge, monkeypatch):
    monkeypatch.setattr('positronic_wire.modal.time.sleep', lambda seconds: None)
    answer = edge.post

    def drop_first(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        if len(edge.starts) == 0:
            edge.starts.append(dict(headers))
            raise http.client.RemoteDisconnected('Remote end closed connection without response')
        return answer(url, headers, timeout)

    with patch.object(modal.ModalClientWire, '_post', staticmethod(drop_first)):
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert [start[modal.IDEMPOTENCY_KEY_HEADER] for start in edge.starts[:1]] == [
        edge.starts[1][modal.IDEMPOTENCY_KEY_HEADER]
    ]


@pytest.mark.parametrize(
    'reason',
    [socket.gaierror(socket.EAI_NONAME, 'Name or service not known'), ssl.SSLCertVerificationError('unknown issuer')],
)
def test_a_start_no_retry_reaches_surfaces_at_once_as_final(modal_wire, reason):
    def unreachable(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        raise urllib.error.URLError(reason)

    with (
        patch.object(modal.ModalClientWire, '_post', staticmethod(unreachable)),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert refused.value.refusal is wire.Refusal.FINAL


def test_a_terminate_the_edge_drops_is_logged_and_not_raised(modal_wire, edge):
    conn = modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    def dropped(url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        raise http.client.RemoteDisconnected('Remote end closed connection without response')

    with patch.object(modal.ModalClientWire, '_post', staticmethod(dropped)):
        assert conn.close() == 'closed'


def test_ending_all_sessions_ends_the_spare_and_a_dropped_session(modal_wire, edge):
    _drop(modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0))
    modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0)

    modal_wire.end_all()

    assert sorted(edge.ended) == ['token-1', 'token-2']


def test_each_server_holds_its_own_sessions(modal_wire, edge, inner):
    other = wire.HostPortAddress('other-server.eu-west.modal.direct', 443, wire.SESSION_PATH, '')
    modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0)

    modal_wire.dial(other, _CREDENTIALS, 5.0)

    assert inner.dialled == ['token-2']
    assert edge.starts[1]['url'] == 'https://other-server.eu-west.modal.direct' + modal.START_PATH


def test_a_pickled_wire_holds_no_sessions(modal_wire, edge):
    modal_wire.keepalive(_ADDRESS, _CREDENTIALS, 5.0)

    copy = pickle.loads(pickle.dumps(_wire(websocket.WebsocketTlsClientWire())))
    atexit.unregister(copy.end_all)

    assert isinstance(copy, modal.ModalClientWire)
    assert copy._endpoints == {}


@pytest.mark.parametrize('answer', [b'{}', b'not json', b'{"token": ""}', b'{"token": "a b"}', b'{"token": 7}'])
def test_a_start_answer_without_a_token_a_header_holds_is_final(modal_wire, answer):
    with (
        patch.object(modal.ModalClientWire, '_post', staticmethod(lambda url, headers, timeout: answer)),
        pytest.raises(wire.ConnectRefused) as refused,
    ):
        modal_wire.dial(_ADDRESS, _CREDENTIALS, 5.0)

    assert refused.value.refusal is wire.Refusal.FINAL
