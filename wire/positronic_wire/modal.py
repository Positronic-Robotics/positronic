"""The client side of a Modal Server with sticky sessions (``@app.server`` with ``@modal.sessioned``)."""

import atexit
import dataclasses
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from functools import partial
from http import HTTPStatus
from http.client import HTTPException, HTTPMessage
from typing import IO, NamedTuple

from positronic_wire import wire
from positronic_wire.websocket import refusal_of

_logger = logging.getLogger(__name__)

START_PATH = '/_modal/sessions/start'
TERMINATE_PATH = '/_modal/sessions/terminate'
SESSION_HEADER = 'Modal-Authorization'
# A session's idle timeout. A call outside a session carries it too: the edge forwards only the session id.
IDLE_TIMEOUT_HEADER = 'x-modal-server-session-idle-timeout'
IDEMPOTENCY_KEY_HEADER = 'x-idempotency-key'
TIMESTAMP_HEADER = 'x-modal-timestamp'
SESSION_TOKEN_HEADER = 'x-modal-server-session-token'
# The key of the token in the JSON answer to a session start.
START_ANSWER_TOKEN = 'token'
# The default seconds a session lives after its last request, an open websocket included. Short, so a session that
# no connection takes frees its container soon.
IDLE_TIMEOUT = 120
# The edge answers a start that waited this long with a 5xx; the start then continues under its key.
START_REQUEST_TIMEOUT = 330.0
# The default most wall-clock time one start may take, its continues included, before the next dial starts another.
START_DEADLINE = 900.0
TERMINATE_TIMEOUT = 60.0
# The most of an answer to a session call that the client reads: a token in a JSON object fits in it many times.
MAX_ANSWER_BYTES = 16 * 1024


class _Parked(NamedTuple):
    """A dropped connection's session: the full address it dialled, and the timer that ends it."""

    address: wire.HostPortAddress
    timer: threading.Timer


@dataclasses.dataclass
class _Endpoint:
    """The Modal sessions this process holds on one endpoint."""

    root: str
    # The credentials every start and terminate on this endpoint carries; a request in a session carries a token.
    headers: Mapping[str, str] | None
    # The sessions that open connections hold, oldest first.
    held: dict[str, None] = dataclasses.field(default_factory=dict)
    # A session that no connection holds. The next dial takes it.
    spare: str | None = None
    # The sessions of connections that dropped, each until the next dial on its address takes it or its timer ends it.
    parked: dict[str, _Parked] = dataclasses.field(default_factory=dict)
    starting: bool = False
    # Why the last start placed no session. The next wait for a start raises it.
    start_failure: wire.ConnectRefused | None = None


class ModalClientConnection(wire.ClientConnection):
    """A client's end of one session on a Modal Server, and the Modal session it dialled with.

    ``release`` runs once, on ``close``, with whether the server dropped the connection.
    """

    def __init__(self, inner: wire.ClientConnection, release: Callable[[bool], None]):
        self._inner = inner
        self._release: Callable[[bool], None] | None = release
        self._dropped = False

    def send(self, message: bytes) -> None:
        try:
            self._inner.send(message)
        except wire.PeerDisconnected:
            self._dropped = True
            raise

    def recv(self, timeout: float | None = None) -> bytes:
        try:
            return self._inner.recv(timeout)
        except wire.PeerDisconnected:
            self._dropped = True
            raise

    def close(self) -> str:
        try:
            return self._inner.close()
        finally:
            release, self._release = self._release, None
            if release is not None:
                release(self._dropped)


class ModalClientWire(wire.ClientWire[wire.HostPortAddress]):
    """A TLS websocket session on a Modal Server with sticky sessions.

    ``dial`` takes a session for the connection, and dials ``inner`` with its token. A session comes from a
    dropped connection on the address, then the spare, then a new start. A start waits until a container takes it,
    minutes on a cold app, so it runs in its own thread. A dial waits for it up to ``open_timeout``, and raises
    ``COLD`` while it still waits. The next dial waits for the same start. The edge answers a start that waits past
    its own limit with a 5xx, and the start continues under the same idempotency key, so it places one session.

    A normal close ends the connection's session, so its container takes the next one. A dropped connection keeps
    its session for ``idle_timeout``, so a reconnect reaches the same container. ``keepalive`` carries the token
    of a held session or of the spare, and starts the spare when there is none: on a cold app that wakes the
    container the next dial takes. ``probe`` raises ``wire.ProbeUnsupported``. The wire ends every session it holds
    when the process exits.

    A pickled wire holds none of these sessions.
    """

    NAME = 'modal_websocket_tls'
    ADDRESS = wire.HostPortAddress
    DEFAULT_PORT = 443

    def __init__(
        self,
        inner: wire.ClientWire[wire.HostPortAddress],
        idle_timeout: int = IDLE_TIMEOUT,
        start_deadline: float = START_DEADLINE,
    ):
        self._inner = inner
        self._idle_timeout = idle_timeout
        self._start_deadline = start_deadline
        # Keyed by the server and the credentials, so a session never serves another caller's credentials.
        self._endpoints: dict[tuple[wire.HostPortAddress, frozenset[tuple[str, str]]], _Endpoint] = {}
        self._changed = threading.Condition()
        atexit.register(self.end_all)

    def __reduce__(self) -> tuple[type['ModalClientWire'], tuple[wire.ClientWire[wire.HostPortAddress], int, float]]:
        # The sessions belong to this process, and a lock does not pickle.
        return ModalClientWire, (self._inner, self._idle_timeout, self._start_deadline)

    def session_url(self, address: wire.HostPortAddress) -> str:
        return self._inner.session_url(address)

    def dial(
        self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> ModalClientConnection:
        """A client's end of one session on ``address``, on a Modal session of its own.

        A session that the wire already holds can have ended at its idle timeout. The edge answers its dial with
        404, a ``FINAL`` refusal, and the dial then waits for a new session in the same call.
        """
        deadline = time.monotonic() + open_timeout
        endpoint = self._endpoint(address, headers)
        with self._changed:
            token = next((t for t, parked in reversed(endpoint.parked.items()) if parked.address == address), None)
            if token is not None:
                endpoint.parked.pop(token).timer.cancel()
                keep = partial(self._park, endpoint, address)
            else:
                token, endpoint.spare = endpoint.spare, None
                keep = partial(self._keep_as_spare, endpoint)
        if token is not None:
            try:
                return self._dial_on(endpoint, address, token, deadline, keep)
            except wire.ConnectRefused as refused:
                if refused.refusal is not wire.Refusal.FINAL:
                    raise
                _logger.info('The Modal session this dial took has ended (%s); waiting for a new one', refused)
        token = self._wait_for_spare(endpoint, deadline, take=True)
        return self._dial_on(endpoint, address, token, deadline, partial(self._keep_as_spare, endpoint))

    def keepalive(self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, timeout: float) -> int | None:
        """The keepalive of ``inner`` on a Modal session's container.

        The session is the newest that a connection holds, or else the spare, which this starts where there is
        none. Raises ``COLD`` while the spare still waits for a container. A keepalive that names a session which has
        ended (a ``FINAL`` refusal) drops it, and runs once more on the next. A keepalive on a session that this one
        waited for surfaces its refusal.
        """
        deadline = time.monotonic() + timeout
        endpoint = self._endpoint(address, headers)
        with self._changed:
            token = next(reversed(endpoint.held), None) or endpoint.spare
        if token is not None:
            try:
                return self._inner.keepalive(address, self._call_headers(token), max(0.0, deadline - time.monotonic()))
            except wire.ConnectRefused as refused:
                if refused.refusal is not wire.Refusal.FINAL:
                    raise
                _logger.info('The Modal session this keepalive named has ended (%s); waiting for a new one', refused)
            with self._changed:
                if endpoint.spare == token:
                    endpoint.spare = None
        token = self._wait_for_spare(endpoint, deadline, take=False)
        return self._inner.keepalive(address, self._call_headers(token), max(0.0, deadline - time.monotonic()))

    def probe(
        self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        """Raises ``wire.ProbeUnsupported``."""
        raise wire.ProbeUnsupported(f'{self.NAME} carries no probe; ask keepalive whether the server is up')

    def end_all(self) -> None:
        """End every Modal session this wire holds."""
        with self._changed:
            ending = []
            for endpoint in self._endpoints.values():
                for parked in endpoint.parked.values():
                    parked.timer.cancel()
                tokens = [*endpoint.held, *endpoint.parked, *([endpoint.spare] if endpoint.spare else [])]
                ending += [(endpoint, token) for token in tokens]
                endpoint.held, endpoint.parked, endpoint.spare = {}, {}, None
        for endpoint, token in ending:
            self._terminate(endpoint, token)

    def _endpoint(self, address: wire.HostPortAddress, headers: Mapping[str, str] | None) -> _Endpoint:
        """The sessions on ``address``'s server under the credentials ``headers``."""
        root = address.at_root()
        key = (root, frozenset((headers or {}).items()))
        with self._changed:
            endpoint = self._endpoints.get(key)
            if endpoint is None:
                endpoint = _Endpoint(
                    f'https://{wire.netloc(root, self.DEFAULT_PORT)}', dict(headers) if headers else None
                )
                self._endpoints[key] = endpoint
            return endpoint

    def _dial_on(
        self,
        endpoint: _Endpoint,
        address: wire.HostPortAddress,
        token: str,
        deadline: float,
        keep: Callable[[str], None],
    ) -> ModalClientConnection:
        """``inner``'s dial with ``token``. A ``FINAL`` refusal ends the session; ``keep`` takes it back after any
        other failure."""
        try:
            conn = self._inner.dial(address, {SESSION_HEADER: f'Bearer {token}'}, max(0.0, deadline - time.monotonic()))
        except BaseException as failed:
            if isinstance(failed, wire.ConnectRefused) and failed.refusal is wire.Refusal.FINAL:
                self._terminate(endpoint, token)
            else:
                keep(token)
            raise
        with self._changed:
            endpoint.held[token] = None
        return ModalClientConnection(conn, partial(self._release, endpoint, address, token))

    def _release(self, endpoint: _Endpoint, address: wire.HostPortAddress, token: str, dropped: bool) -> None:
        """Park the session of a dropped connection on ``address``, and end that of a closed one."""
        with self._changed:
            endpoint.held.pop(token, None)
        if dropped:
            self._park(endpoint, address, token)
        else:
            self._terminate(endpoint, token)

    def _park(self, endpoint: _Endpoint, address: wire.HostPortAddress, token: str) -> None:
        """Keep ``token`` for the next dial on ``address``, until ``idle_timeout`` passes."""
        timer = threading.Timer(self._idle_timeout, lambda: self._end_parked(endpoint, token, timer))
        timer.daemon = True
        with self._changed:
            endpoint.parked[token] = _Parked(address, timer)
        timer.start()

    def _end_parked(self, endpoint: _Endpoint, token: str, timer: threading.Timer) -> None:
        """End ``token`` where ``timer`` guards its current park."""
        with self._changed:
            parked = endpoint.parked.get(token)
            if parked is None or parked.timer is not timer:
                return
            del endpoint.parked[token]
        self._terminate(endpoint, token)

    def _keep_as_spare(self, endpoint: _Endpoint, token: str) -> None:
        """Keep ``token`` as the spare, or end it where there is one already."""
        with self._changed:
            if endpoint.spare is None:
                endpoint.spare = token
                self._changed.notify_all()
                return
        self._terminate(endpoint, token)

    def _call_headers(self, token: str) -> dict[str, str]:
        return {SESSION_HEADER: f'Bearer {token}', IDLE_TIMEOUT_HEADER: str(self._idle_timeout)}

    def _wait_for_spare(self, endpoint: _Endpoint, deadline: float, *, take: bool) -> str:
        """The spare's token, which this starts where no start runs, waiting up to the ``time.monotonic()`` instant
        ``deadline``. ``take`` removes it from the spare.

        Raises ``ConnectRefused``: ``COLD`` while the start still waits for a container, and the start's own refusal
        where it placed no session.
        """
        with self._changed:
            while endpoint.spare is None:
                if endpoint.start_failure is not None:
                    failure, endpoint.start_failure = endpoint.start_failure, None
                    raise failure
                if not endpoint.starting:
                    endpoint.starting = True
                    threading.Thread(
                        target=self._start, args=(endpoint,), name='modal-session-start', daemon=True
                    ).start()
                left = deadline - time.monotonic()
                if left <= 0:
                    raise wire.ConnectRefused(
                        wire.Refusal.COLD, f'The Modal session start on {endpoint.root} still waits for a container'
                    )
                self._changed.wait(left)
            token = endpoint.spare
            if take:
                endpoint.spare = None
            return token

    def _start(self, endpoint: _Endpoint) -> None:
        """Start a session for the spare. Runs in its own thread, and continues the start under one key until a
        container takes it or ``start_deadline`` passes."""
        token: str | None = None
        failure: wire.ConnectRefused | None = None
        try:
            token = self._start_session(endpoint)
        except wire.ConnectRefused as refused:
            failure = refused
        except Exception as e:
            failure = wire.ConnectRefused(
                wire.Refusal.SILENT, f'The Modal session start on {endpoint.root} failed: {e}'
            )
            failure.__cause__ = e
        extra: str | None = None
        with self._changed:
            endpoint.starting = False
            endpoint.start_failure = failure
            if token is not None and endpoint.spare is None:
                endpoint.spare = token
            else:
                extra = token
            self._changed.notify_all()
        if extra is not None:
            self._terminate(endpoint, extra)

    @staticmethod
    def _session_token(answer: bytes) -> str:
        """The token in the answer to a session start.

        Raises ``wire.ConnectRefused`` with ``FINAL`` where the answer carries none. The token goes into a header, so it
        is one run of visible ASCII characters.
        """
        try:
            token = json.loads(answer)[START_ANSWER_TOKEN]
        except (ValueError, LookupError, TypeError, RecursionError) as e:
            raise wire.ConnectRefused(wire.Refusal.FINAL, 'Modal answered a session start without a token') from e
        if not isinstance(token, str) or not token or not all('!' <= c <= '~' for c in token):
            raise wire.ConnectRefused(wire.Refusal.FINAL, 'Modal answered a session start with a token no header holds')
        return token

    @staticmethod
    def _refusal(failed: OSError | HTTPException) -> wire.Refusal:
        """What a session call that got no HTTP answer says about the edge: ``FINAL`` for a host with no address and for
        a certificate that does not verify, ``SILENT`` for the rest."""
        reason = failed.reason if isinstance(failed, urllib.error.URLError) else failed
        return refusal_of(reason) if isinstance(reason, OSError) else wire.Refusal.SILENT

    def _start_session(self, endpoint: _Endpoint) -> str:
        """The token of a new session. A 5xx, a 429 and a lost answer continue the start under the same key."""
        key, stamp = str(uuid.uuid4()), str(time.time())
        deadline = time.monotonic() + self._start_deadline
        lost: Exception | None = None
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise wire.ConnectRefused(
                    wire.Refusal.SILENT, f'No container took the Modal session start on {endpoint.root} in time'
                ) from lost
            headers = {
                **(endpoint.headers or {}),
                IDLE_TIMEOUT_HEADER: str(self._idle_timeout),
                IDEMPOTENCY_KEY_HEADER: key,
                TIMESTAMP_HEADER: stamp,
            }
            try:
                return self._session_token(
                    self._post(endpoint.root + START_PATH, headers, min(left, START_REQUEST_TIMEOUT))
                )
            except urllib.error.HTTPError as refused:
                # The edge answers a start it rate-limits with 429, and the start continues as after a 5xx.
                if refused.code < HTTPStatus.INTERNAL_SERVER_ERROR and refused.code != HTTPStatus.TOO_MANY_REQUESTS:
                    forbidden = refused.code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN)
                    raise wire.ConnectRefused(
                        wire.Refusal.FORBIDDEN if forbidden else wire.Refusal.FINAL,
                        f'Modal refused a session start on {endpoint.root} with {refused.code}',
                    ) from None
                lost = refused
            except (OSError, HTTPException) as failed:
                if self._refusal(failed) is wire.Refusal.FINAL:
                    raise wire.ConnectRefused(
                        wire.Refusal.FINAL, f'The Modal session start on {endpoint.root} failed: {failed}'
                    ) from failed
                _logger.info('A Modal session start got no answer (%s); asking again under the same key', failed)
                lost = failed
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    def _terminate(self, endpoint: _Endpoint, token: str) -> None:
        """End the session, so its container takes the next one. A session that already ended answers 404.

        A session that this call does not end ends at its idle timeout, so a failure is logged and not raised.
        """
        headers = {**(endpoint.headers or {}), SESSION_TOKEN_HEADER: token}
        try:
            self._post(endpoint.root + TERMINATE_PATH, headers, TERMINATE_TIMEOUT)
        except urllib.error.HTTPError as answer:
            if answer.code != HTTPStatus.NOT_FOUND:
                _logger.error('Modal session terminate answered %s; the session ends at its idle timeout', answer.code)
        except (OSError, HTTPException) as lost:
            _logger.error('Modal session terminate got no answer (%s); the session ends at its idle timeout', lost)

    class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
        """Answers a redirect as an ``HTTPError``. A session call carries the endpoint's credentials, so it goes to the
        endpoint alone. ``urllib`` makes a redirected ``POST`` a ``GET`` without its body, which no session call is."""

        def redirect_request(
            self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
        ) -> None:
            return None

    _SESSION_CALLS = urllib.request.build_opener(_RefuseRedirects)

    @classmethod
    def _post(cls, url: str, headers: Mapping[str, str], timeout: float) -> bytes:
        """The answer to a ``POST`` on ``url``. A refusal and a redirect raise ``urllib.error.HTTPError``."""
        request = urllib.request.Request(url, data=b'', method='POST', headers=dict(headers))
        with cls._SESSION_CALLS.open(request, timeout=timeout) as answer:
            return answer.read(MAX_ANSWER_BYTES)
