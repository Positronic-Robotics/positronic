import logging
import ssl
import time
import urllib.parse
from collections.abc import Callable
from enum import Enum
from http import HTTPStatus
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from . import protocol
from .protocol import deserialise, serialise, typed_commands

logger = logging.getLogger(__name__)

# A first ``infer`` can include the backend's own startup cost, such as a JAX compilation. Bound each ``recv``
# generously enough to outlast that (still surfacing a stalled/half-open connection), and let callers override
# per use.
DEFAULT_INFER_TIMEOUT = 180.0

# How long ``infer`` may spend rebuilding a dropped connection, across every attempt of one reconnect.
# FOOTGUN: the arm holds its last setpoint throughout, and an attended trial (``timeout_sec=None``) has no
# episode deadline behind this one to end that stall.
DEFAULT_RECONNECT_DEADLINE = 45.0


def _handshake(websocket: Connection, timeout_per_message: float = 30.0) -> dict[str, Any]:
    """Receive status updates until server is ready.

    The server must send an update at least every ``timeout_per_message`` seconds.
    """
    try:
        while True:
            response = deserialise(websocket.recv(timeout=timeout_per_message))
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


class InferenceSession:
    """One websocket to one served session, carrying observations out and trajectories back.

    ``reopen`` returns a fresh socket whose handshake has reached ready, and makes a dropped connection
    recoverable: an ``infer`` that loses the socket reconnects through it and sends the same observation
    again. A session without one raises on a drop, which is all a caller owning no way to reconnect can do.
    """

    def __init__(
        self,
        websocket: Connection,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        *,
        metadata: dict[str, Any] | None = None,
        reopen: Callable[[], Connection] | None = None,
    ):
        self._websocket = websocket
        self._infer_timeout = infer_timeout
        self._reopen = reopen
        self._metadata = _handshake(websocket) if metadata is None else metadata

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict[str, Any]) -> Any:
        """Send an observation and get the served session's result, with every robot-command channel typed.

        ``obs`` must be wire-serializable: plain-data containers and scalars, plus numeric numpy
        arrays/scalars, and no arbitrary Python objects. The result is whatever the server's session
        returned — canonically a list of action dicts, but a bare dict or ``None`` too.

        A dropped socket reconnects and sends the observation once more, within
        ``DEFAULT_RECONNECT_DEADLINE``. Every other failure reaches the caller as it is.
        """
        serialised = serialise(obs)
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)
        try:
            return self._round_trip(serialised)
        # The server may still be computing this observation, so a second send doubles the work on a backend
        # already too slow to answer the first.
        except TimeoutError:
            raise
        # A container recycling under a backend that scales to zero drops the socket, on the send or the recv,
        # as ``ConnectionClosed`` or a bare ``OSError`` (TLS included). Neither says anything about the
        # observation, so it goes again on a new socket.
        except (ConnectionClosed, OSError) as e:
            if self._reopen is None:
                raise
            logger.warning('Inference connection dropped (%s); reconnecting and sending the observation again', e)
            self._websocket.close()
            # The URL pins the model, so the session keeps the metadata it opened under: the episode records
            # that meta once, and it describes the whole episode.
            self._websocket = self._reopen()
            # The server built this session a moment ago, so it holds none of the dropped one's state. A second
            # drop is a backend that cannot serve this observation at all, and reaches the caller.
            return self._round_trip(serialised)

    def _round_trip(self, serialised: bytes) -> Any:
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
        self._websocket.close()


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

    ``headers`` carry auth, whether the server checks it or a proxy in front of it does — credentials stay
    out of the URL, which is meant to be safe to hand around.

    The timeouts describe this connection, not any one session: ``open_timeout`` bounds the TCP/TLS
    handshake alone, ``connect_deadline`` how long a cold backend may take to answer across retries,
    ``infer_timeout`` one inference round trip, and ``reconnect_deadline`` how long a session may spend
    rebuilding a socket dropped mid-inference. The last is much the shortest, because it is spent with the
    caller's robot live — see ``DEFAULT_RECONNECT_DEADLINE``.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = 10.0,
        connect_deadline: float = 900.0,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        reconnect_deadline: float = DEFAULT_RECONNECT_DEADLINE,
    ):
        split = urllib.parse.urlsplit(url if '://' in url else f'//{url}')
        if split.scheme not in ('', 'http', 'ws', 'https', 'wss'):
            raise ValueError(f'Unsupported scheme {split.scheme!r} in {url!r}')
        if not split.hostname:
            raise ValueError(f'No host in {url!r}')
        secure = split.scheme in ('https', 'wss')
        ws_scheme = 'wss' if secure else 'ws'
        http_scheme = 'https' if secure else 'http'
        default_port = 443 if secure else 80
        # urlsplit strips the brackets an IPv6 host needs back in a netloc.
        host = f'[{split.hostname}]' if ':' in split.hostname else split.hostname
        port = default_port if split.port is None else split.port
        netloc = host if port == default_port else f'{host}:{port}'
        # Forwarded verbatim: the server reads each param value as a JSON literal, and only whoever wrote
        # the URL knows whether `true` means the bool or the string.
        query = f'?{split.query}' if split.query else ''
        self.session_url = f'{ws_scheme}://{netloc}{_session_path(split.path, url)}{query}'
        self.api_url = f'{http_scheme}://{netloc}/api/v1'
        self.headers = dict(headers) if headers else None
        self.open_timeout = open_timeout
        self.connect_deadline = connect_deadline
        self.infer_timeout = infer_timeout
        self.reconnect_deadline = reconnect_deadline

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the model the URL names."""
        ws, metadata = self._open(self.connect_deadline)
        return InferenceSession(
            ws,
            infer_timeout=self.infer_timeout,
            metadata=metadata,
            # A drop mid-inference reconnects through the same cold-start retries, on the shorter deadline
            # a live robot can afford.
            reopen=lambda: self._open(self.reconnect_deadline)[0],
        )

    def _open(self, deadline_sec: float) -> tuple[Connection, dict[str, Any]]:
        """A connected socket whose status handshake has reached ready, and the metadata that handshake read.

        Retries a backend that is not up yet, until ``deadline_sec`` of wall clock has passed.
        """
        deadline = time.monotonic() + deadline_sec
        backoff = 1.0
        retries = _ConnectRetries()
        while True:
            ws = None
            try:
                # A proxy between here and the server closes a connection it has read nothing from, often
                # after 60s — well inside one ``infer_timeout`` inference, which sends nothing until it
                # answers. The pings keep it open.
                ws = connect(
                    self.session_url,
                    open_timeout=self.open_timeout,
                    additional_headers=self.headers,
                    ping_interval=20.0,
                )
                return ws, _handshake(ws)
            # ``SSLCertVerificationError`` is an ``ssl.SSLError``, but a bad certificate is permanent
            # misconfiguration, not a cold start — surface it immediately instead of retrying to the deadline.
            except ssl.SSLCertVerificationError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e
            # Each of these is a backend not up yet — a timed-out connect, a TLS reset at the edge, a 502/503
            # on the upgrade, a status handshake dropped or stalled — so retry within the deadline.
            except (TimeoutError, ssl.SSLError, ConnectionClosed, InvalidHandshake) as e:
                if ws is not None:
                    ws.close()
                if retries.take(e) is _ConnectOutcome.SURFACE:
                    raise
                # The sleep is clipped to what is left, so the deadline bounds the wall clock spent here and
                # not merely the instant the last attempt starts at.
                pause = min(backoff, deadline - time.monotonic())
                if pause <= 0:
                    raise TimeoutError(f'{e} (connecting to {self.session_url})') from e
                logger.info('Server not ready (cold start?): %s; retrying in %.0fs', e, pause)
                time.sleep(pause)
                backoff = min(backoff * 2, 30.0)
            except OSError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e

    def list_models(self) -> list[str]:
        """List available models from the server."""
        response = httpx.get(f'{self.api_url}/models', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
