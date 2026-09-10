import logging
import re
import ssl
import time
import urllib.parse
from enum import Enum
from functools import partial
from http import HTTPStatus
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect, unix_connect
from websockets.sync.connection import Connection

from . import protocol
from .protocol import deserialise, serialise, typed_commands

logger = logging.getLogger(__name__)

# A first ``infer`` can include the backend's own startup cost, such as a JAX compilation. Bound each ``recv``
# generously enough to outlast that (still surfacing a stalled/half-open connection), and let callers override
# per use.
DEFAULT_INFER_TIMEOUT = 180.0


class InferenceSession:
    def __init__(self, websocket: Connection, infer_timeout: float = DEFAULT_INFER_TIMEOUT):
        self._websocket = websocket
        self._infer_timeout = infer_timeout
        self._metadata = self._handshake()

    def _handshake(self, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        The server must send an update at least every ``timeout_per_message`` seconds.
        """
        try:
            while True:
                response = deserialise(self._websocket.recv(timeout=timeout_per_message))
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

        self._websocket.send(serialised)
        try:
            response = deserialise(self._websocket.recv(timeout=self._infer_timeout))
        except TimeoutError:
            # The observation is in flight but unanswered; the server's late response would sit in the socket and
            # the next ``recv`` would pair it with a future observation. Close so the desynced session can't be
            # reused — a subsequent ``infer`` fails loudly on the closed socket instead.
            self._websocket.close()
            raise TimeoutError(
                f'No inference response within {self._infer_timeout}s — server stalled or connection half-open'
            ) from None
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and protocol.ERROR in response:
            raise RuntimeError(f'Server error: {response[protocol.ERROR]}')

        return typed_commands(response[protocol.RESULT])

    def close(self):
        state_before_close = self._websocket.state.name
        self._websocket.close()
        # A close that times out still reaches CLOSED locally; only the close code says the server answered.
        logger.info(
            'InferenceSession.close: state %s -> %s, close code %s',
            state_before_close,
            self._websocket.state.name,
            self._websocket.close_code,
        )


def _session_path(path: str, url: str) -> str:
    """The session path a URL names: ``/api/v1/session``, plus the model id it addresses, if any.

    A URL naming no model — a bare host, or the endpoint with or without a trailing slash — addresses the
    endpoint itself, which serves whatever the server pinned.
    """
    if path.rstrip('/') in ('', '/api/v1/session'):
        return '/api/v1/session'
    if not path.startswith('/api/v1/session/'):
        raise ValueError(f'Unexpected path {path!r} in {url!r}; expected /api/v1/session[/<model_id>]')
    # Kept as written, percent-encoding included, so the server decodes exactly the id whoever handed out
    # the URL meant: a trailing slash is part of that id, and an id may itself be a path (a HuggingFace
    # repo, say), whose own slashes stay separators.
    return path


def _socket_and_path(split: urllib.parse.SplitResult, url: str) -> tuple[str, str]:
    """The socket path a ``unix://`` URL names, decoded, and the URL path left over for the server.

    The split runs over the encoded path, so an escaped separator inside a directory name stays part
    of that name. Only the socket path is decoded, and once: it names a file, where the URL path
    reaches the server as written, which is what lets a model id carry its own escapes.
    """
    if split.netloc or not split.path.startswith('/'):
        raise ValueError(f'Socket path must be absolute in {url!r}; write unix:///path/to.sock')
    marker = re.search(r'/api/v1(?=/|$)', split.path)
    if marker is None:
        return urllib.parse.unquote(split.path), ''
    return urllib.parse.unquote(split.path[: marker.start()]), split.path[marker.start() :]


class _ConnectOutcome(Enum):
    RETRY = 'retry'
    SURFACE = 'surface'


class _ConnectRetries:
    """The retry policy over one ``new_session``'s connect attempts.

    403 is both a cold backend and a refused credential, so it gets a few attempts rather than the whole
    ``connect_deadline``.
    """

    MAX_FORBIDDEN_ATTEMPTS = 3

    def __init__(self) -> None:
        self._forbidden_attempts = 0

    def take(self, e: Exception) -> _ConnectOutcome:
        """Spend a refused connect against the budget."""
        if not isinstance(e, InvalidStatus):
            return _ConnectOutcome.RETRY
        status = e.response.status_code
        if status == HTTPStatus.FORBIDDEN:
            self._forbidden_attempts += 1
            again = self._forbidden_attempts < self.MAX_FORBIDDEN_ATTEMPTS
        else:
            again = status >= HTTPStatus.INTERNAL_SERVER_ERROR or status == HTTPStatus.TOO_MANY_REQUESTS
        return _ConnectOutcome.RETRY if again else _ConnectOutcome.SURFACE


class InferenceClient:
    """The wire connection to one inference server, addressed by one URL.

    Accepted URL forms: ``host``, ``host:port``, and ``scheme://host[:port][/api/v1/session[/<model_id>]]``,
    each with an optional ``?query``. ``https``/``wss`` enable TLS (bare or ``http``/``ws`` forms don't); the
    port defaults to the scheme's own, 443 for TLS and 80 otherwise. Everything the URL says about the
    session — the model id it names and the query it carries as session params — reaches the server exactly
    as written, so every session opened here serves that model with those params.

    ``unix://<absolute socket path>[/api/v1/session[/<model_id>]][?query]`` reaches a server on the same
    machine over a Unix domain socket, which needs no network. The socket path runs to the first
    ``/api/v1`` segment, so ``unix:///run/policy.sock`` is the default session and
    ``unix:///run/policy.sock/api/v1/session/10000?fps=10`` names a model and a param. TLS does not apply.

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
        if split.scheme not in ('', 'http', 'ws', 'https', 'wss', 'unix'):
            raise ValueError(f'Unsupported scheme {split.scheme!r} in {url!r}')
        secure = split.scheme in ('https', 'wss')
        if split.scheme == 'unix':
            uds, path = _socket_and_path(split, url)
            # A socket path is not a host. The server reads the path and the query alone, so the
            # handshake asks for them under a host that stands in for the socket.
            netloc = 'localhost'
        else:
            uds = None
            if not split.hostname:
                raise ValueError(f'No host in {url!r}')
            path = split.path
            default_port = 443 if secure else 80
            # urlsplit strips the brackets an IPv6 host needs back in a netloc.
            host = f'[{split.hostname}]' if ':' in split.hostname else split.hostname
            port = default_port if split.port is None else split.port
            netloc = host if port == default_port else f'{host}:{port}'
        ws_scheme = 'wss' if secure else 'ws'
        http_scheme = 'https' if secure else 'http'
        # Forwarded verbatim: the server reads each param value as a JSON literal, and only whoever wrote
        # the URL knows whether `true` means the bool or the string.
        query = f'?{split.query}' if split.query else ''
        session_path = _session_path(path, url)
        self.uds = uds
        # The URL the websocket handshake asks for, and the TCP address to dial when there is no socket.
        self._ws_uri = f'{ws_scheme}://{netloc}{session_path}{query}'
        # What an error names. Over a socket the stand-in host would not say which socket failed.
        self.session_url = self._ws_uri if uds is None else f'unix://{uds}{session_path}{query}'
        self.api_url = f'{http_scheme}://{netloc}/api/v1'
        self.headers = dict(headers) if headers else None
        self.open_timeout = open_timeout
        self.connect_deadline = connect_deadline
        self.infer_timeout = infer_timeout

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the model the URL names."""
        deadline = time.monotonic() + self.connect_deadline
        backoff = 1.0
        retries = _ConnectRetries()
        while True:
            ws = None
            try:
                # A proxy between here and the server closes a connection it has read nothing from, often
                # after 60s — well inside one ``infer_timeout`` inference, which sends nothing until it
                # answers. The pings keep it open.
                dial = (
                    partial(connect, self._ws_uri)
                    if self.uds is None
                    else partial(unix_connect, self.uds, uri=self._ws_uri)
                )
                ws = dial(open_timeout=self.open_timeout, additional_headers=self.headers, ping_interval=20.0)
                return InferenceSession(ws, infer_timeout=self.infer_timeout)
            # ``SSLCertVerificationError`` is an ``ssl.SSLError``, but a bad certificate is permanent
            # misconfiguration, not a cold start — surface it immediately instead of retrying to the deadline.
            except ssl.SSLCertVerificationError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e
            # A cold backend fails before the session is ready in several ways: the connect times out, the edge
            # resets TLS (``SSLError``), it rejects or drops the HTTP upgrade (``InvalidHandshake`` — e.g. a
            # 502/503 while the backend boots), or it accepts the socket and then drops or stalls the status
            # handshake inside ``InferenceSession`` (``ConnectionClosed``/``TimeoutError``). All mean "not ready
            # yet", so retry within the deadline instead of letting one kill the run.
            except (TimeoutError, ssl.SSLError, ConnectionClosed, InvalidHandshake) as e:
                if ws is not None:
                    ws.close()
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
        transport = None if self.uds is None else httpx.HTTPTransport(uds=self.uds)
        with httpx.Client(transport=transport) as client:
            response = client.get(f'{self.api_url}/models', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
