import logging
import ssl
import time
import urllib.parse
from enum import Enum
from http import HTTPStatus
from typing import Any

import grpc
import httpx
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect

from . import grpc_wire, protocol, wire
from .protocol import deserialise, serialise, typed_commands

logger = logging.getLogger(__name__)

# A first ``infer`` can include the backend's own startup cost, such as a JAX compilation. Bound each ``recv``
# generously enough to outlast that (still surfacing a stalled/half-open connection), and let callers override
# per use.
DEFAULT_INFER_TIMEOUT = 180.0


class InferenceSession:
    """One session over one open connection, whichever wire carries it."""

    def __init__(self, conn: wire.ClientConnection, infer_timeout: float = DEFAULT_INFER_TIMEOUT):
        self._conn = conn
        self._infer_timeout = infer_timeout
        self._metadata = self._handshake()

    def _handshake(self, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        The server must send an update at least every ``timeout_per_message`` seconds.
        """
        try:
            while True:
                response = deserialise(self._conn.recv(timeout=timeout_per_message))
                if protocol.ERROR in response:
                    raise RuntimeError(f'Server error: {response[protocol.ERROR]}')
                try:
                    status = protocol.ServerStatus(response.get(protocol.STATUS))
                except ValueError:
                    raise RuntimeError(f'Unexpected server response: {response}') from None

                if status is protocol.ServerStatus.READY:
                    return response[protocol.META]
                if status is protocol.ServerStatus.ERROR:
                    raise RuntimeError('Server error: Unknown error')

                message = response.get(protocol.MESSAGE, status)
                logger.info(f'Server status: [{status}] {message}')

        except TimeoutError:
            raise TimeoutError(
                f'Server did not send status update within {timeout_per_message}s. '
                f'Server may have crashed or model loading is taking too long without progress updates.'
            ) from None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> Any:
        """Send an observation and get the served session's result, with every robot-command channel typed.

        ``obs`` must be wire-serializable: plain-data containers and scalars, plus numeric numpy
        arrays/scalars, and no arbitrary Python objects. The result is whatever the server's session
        returned — canonically a list of action dicts, but a bare dict or ``None`` too.
        """
        serialised = serialise(obs)
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)

        self._conn.send(serialised)
        try:
            response = deserialise(self._conn.recv(timeout=self._infer_timeout))
        except TimeoutError:
            # The observation is in flight but unanswered; the server's late response would sit in the socket and
            # the next ``recv`` would pair it with a future observation. Close so the desynced session can't be
            # reused — a subsequent ``infer`` fails loudly on the closed socket instead.
            self._conn.close()
            raise TimeoutError(
                f'No inference response within {self._infer_timeout}s — server stalled or connection half-open'
            ) from None
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and protocol.ERROR in response:
            raise RuntimeError(f'Server error: {response[protocol.ERROR]}')

        return typed_commands(response[protocol.RESULT])

    def close(self):
        logger.info('InferenceSession.close: %s', self._conn.close())


def _session_path(path: str, url: str) -> str:
    """The session path a URL names: ``/api/v1/session``, plus the model id it addresses, if any.

    A URL naming no model — a bare host, or the endpoint with or without a trailing slash — addresses the
    endpoint itself, which serves whatever the server pinned.
    """
    if path.rstrip('/') in ('', wire.SESSION_PATH):
        return wire.SESSION_PATH
    if not path.startswith(f'{wire.SESSION_PATH}/'):
        raise ValueError(f'Unexpected path {path!r} in {url!r}; expected {wire.SESSION_PATH}[/<model_id>]')
    # Kept as written, percent-encoding included, so the server decodes exactly the id whoever handed out
    # the URL meant: a trailing slash is part of that id, and an id may itself be a path (a HuggingFace
    # repo, say), whose own slashes stay separators.
    return path


class _ConnectOutcome(Enum):
    RETRY = 'retry'
    SURFACE = 'surface'


class _Refusal(Enum):
    """What a refused connect says about the server."""

    COLD = 'cold'  # still coming up; retry to the deadline
    FORBIDDEN = 'forbidden'  # a cold backend, or a refused credential; a few attempts, then surface
    FINAL = 'final'  # the endpoint is saying no; surface at once


_COLD_GRPC_CODES = (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.RESOURCE_EXHAUSTED, grpc.StatusCode.DEADLINE_EXCEEDED)


def _refusal(e: Exception) -> _Refusal:
    """How to read a refused connect, over either wire.

    Each gRPC code stands for the HTTP status its wire twin answers: ``PERMISSION_DENIED`` for 403,
    ``UNAVAILABLE`` for 503, ``RESOURCE_EXHAUSTED`` for 429. A TLS edge no client can use answers
    ``UNAVAILABLE`` too, exactly as a cold backend does, so its details tell them apart.
    """
    if isinstance(e, InvalidStatus):
        status = e.response.status_code
        if status == HTTPStatus.FORBIDDEN:
            return _Refusal.FORBIDDEN
        if status >= HTTPStatus.INTERNAL_SERVER_ERROR or status == HTTPStatus.TOO_MANY_REQUESTS:
            return _Refusal.COLD
        return _Refusal.FINAL
    # A gRPC error carries its code as a `Call`; anything else says nothing about the server.
    if isinstance(e, grpc.Call):
        if grpc_wire.edge_is_unusable(e.details() or ''):
            return _Refusal.FINAL
        code = e.code()
        if code is grpc.StatusCode.PERMISSION_DENIED:
            return _Refusal.FORBIDDEN
        return _Refusal.COLD if code in _COLD_GRPC_CODES else _Refusal.FINAL
    return _Refusal.COLD


class _ConnectRetries:
    """The retry policy over one ``new_session``'s connect attempts.

    A refusal both wires answer for a cold backend and for a refused credential — HTTP 403, gRPC
    ``PERMISSION_DENIED`` — gets a few attempts rather than the whole ``connect_deadline``.
    """

    MAX_FORBIDDEN_ATTEMPTS = 3

    def __init__(self) -> None:
        self._forbidden_attempts = 0

    def take(self, e: Exception) -> _ConnectOutcome:
        """Spend a refused connect against the budget."""
        refusal = _refusal(e)
        if refusal is _Refusal.FORBIDDEN:
            self._forbidden_attempts += 1
            again = self._forbidden_attempts < self.MAX_FORBIDDEN_ATTEMPTS
        else:
            again = refusal is _Refusal.COLD
        return _ConnectOutcome.RETRY if again else _ConnectOutcome.SURFACE


# The URL schemes that put a session on the gRPC wire: plaintext, and behind a TLS edge.
_GRPC_SCHEMES = ('grpc', 'grpcs')


class InferenceClient:
    """The wire connection to one inference server, addressed by one URL.

    Accepted URL forms: ``host``, ``host:port``, and ``scheme://host[:port][/api/v1/session[/<model_id>]]``,
    each with an optional ``?query``. ``https``/``wss``/``grpcs`` enable TLS (bare or ``http``/``ws``/``grpc``
    forms don't); the port defaults to the scheme's own, 443 for TLS and 80 otherwise. Everything the URL
    says about the session — the model id it names and the query it carries as session params — reaches
    the server exactly as written, so every session opened here serves that model with those params.

    ``grpc://`` names the same session on the gRPC wire, which the server offers on a port of its own, and
    ``grpcs://`` names that port behind a TLS edge. Either port carries sessions alone, so ``list_models``
    needs the HTTP URL.

    ``headers`` carry auth, whether the server checks it or a proxy in front of it does — credentials stay
    out of the URL, which is meant to be safe to hand around.

    The timeouts describe this connection, not any one session: ``open_timeout`` bounds the TCP/TLS
    handshake alone, ``connect_deadline`` how long a cold backend may take to answer across retries, and
    ``infer_timeout`` one inference round trip.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = 10.0,
        connect_deadline: float = 900.0,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
    ):
        split = urllib.parse.urlsplit(url if '://' in url else f'//{url}')
        if split.scheme not in ('', 'http', 'ws', 'https', 'wss', *_GRPC_SCHEMES):
            raise ValueError(f'Unsupported scheme {split.scheme!r} in {url!r}')
        if not split.hostname:
            raise ValueError(f'No host in {url!r}')
        grpc_wired = split.scheme in _GRPC_SCHEMES
        secure = split.scheme in ('https', 'wss', 'grpcs')
        session_scheme = split.scheme if grpc_wired else ('wss' if secure else 'ws')
        http_scheme = 'https' if secure else 'http'
        default_port = 443 if secure else 80
        # urlsplit strips the brackets an IPv6 host needs back in a netloc.
        host = f'[{split.hostname}]' if ':' in split.hostname else split.hostname
        port = default_port if split.port is None else split.port
        netloc = host if port == default_port else f'{host}:{port}'
        # Forwarded verbatim: the server reads each param value as a JSON literal, and only whoever wrote
        # the URL knows whether `true` means the bool or the string.
        query = f'?{split.query}' if split.query else ''
        self._session_path = _session_path(split.path, url)
        self._query = split.query
        self._grpc_target = f'{host}:{port}' if grpc_wired else None
        self._grpc_secure = secure
        self.session_url = f'{session_scheme}://{netloc}{self._session_path}{query}'
        self.api_url = None if self._grpc_target else f'{http_scheme}://{netloc}/api/v1'
        self.headers = dict(headers) if headers else None
        self.open_timeout = open_timeout
        self.connect_deadline = connect_deadline
        self.infer_timeout = infer_timeout

    def _connect(self) -> wire.ClientConnection:
        """One session's connection, over the wire the URL names."""
        if self._grpc_target is not None:
            return grpc_wire.GrpcClientConnection(
                self._grpc_target,
                self._session_path,
                self._query,
                self.headers,
                self.open_timeout,
                secure=self._grpc_secure,
            )
        # A proxy between here and the server closes a connection it has read nothing from, often
        # after 60s — well inside one ``infer_timeout`` inference, which sends nothing until it
        # answers. The pings keep it open.
        websocket = connect(
            self.session_url,
            open_timeout=self.open_timeout,
            additional_headers=self.headers,
            ping_interval=20.0,
            max_size=wire.MAX_MESSAGE_BYTES,
        )
        return wire.WebsocketClientConnection(websocket)

    def _open_session(self) -> InferenceSession:
        """One attempt at a session, closing the connection whenever the handshake does not finish.

        A refusal the server sends as a protocol frame — an unknown model, a session param it rejects
        — raises past every transport handler, and a gRPC connection holds a reader thread until it
        is closed.
        """
        conn = self._connect()
        try:
            return InferenceSession(conn, infer_timeout=self.infer_timeout)
        except BaseException:
            conn.close()
            raise

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the model the URL names."""
        deadline = time.monotonic() + self.connect_deadline
        backoff = 1.0
        retries = _ConnectRetries()
        while True:
            try:
                return self._open_session()
            # ``SSLCertVerificationError`` is an ``ssl.SSLError``, but a bad certificate is permanent
            # misconfiguration, not a cold start — surface it immediately instead of retrying to the deadline.
            except ssl.SSLCertVerificationError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e
            # Each of these is a backend that is not ready yet — a timed-out connect, a reset TLS
            # handshake, a refused upgrade or gRPC call, a dropped status handshake — so one must not
            # kill the run. ``_ConnectRetries`` decides which of them is the endpoint saying no.
            except (
                TimeoutError,
                ssl.SSLError,
                ConnectionClosed,
                InvalidHandshake,
                grpc.RpcError,
                wire.PeerDisconnected,
            ) as e:
                if retries.take(e) is _ConnectOutcome.SURFACE:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{e} (connecting to {self.session_url})') from e
                logger.info('Server not ready (cold start?): %s; retrying in %.0fs', e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except OSError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e

    def list_models(self) -> list[str]:
        """List available models from the server."""
        if self.api_url is None:
            raise ValueError(f'{self.session_url} names the gRPC session port; list the models over HTTP')
        response = httpx.get(f'{self.api_url}/models', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
