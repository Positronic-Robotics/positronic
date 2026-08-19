import logging
import ssl
import time
import urllib.parse
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

# Overall bound on reaching ``ready``: the per-message timeout bounds SILENCE only, so a server streaming
# ``loading`` frames inside it trips nothing.
DEFAULT_READY_TIMEOUT = 300.0


class ServerNotReady(RuntimeError):
    """A server kept talking but never reached ``ready`` within the time allowed.

    Not retried: a reconnect does not fix it. Carries the last status frame.
    """

    def __init__(self, url: str, status: str, message: str, waited_s: float):
        self.url = url
        self.status = status
        self.message = message
        self.waited_s = waited_s
        last = f'{status}: {message}' if message else status
        super().__init__(f'{url} did not become ready within {waited_s:.0f}s; last status was [{last}]')


class InferenceSession:
    def __init__(
        self,
        websocket: Connection,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
        *,
        url: str = '',
        ready_deadline: float | None = None,
        waiting_since: float | None = None,
    ):
        self._websocket = websocket
        self._infer_timeout = infer_timeout
        self._metadata = self._handshake(url=url, ready_deadline=ready_deadline, waiting_since=waiting_since)

    def _handshake(
        self,
        timeout_per_message: float = 30.0,
        *,
        url: str = '',
        ready_deadline: float | None = None,
        waiting_since: float | None = None,
    ) -> dict[str, Any]:
        """Receive status updates until server is ready.

        Every layer below ``ready`` — connect, TLS, upgrade — completes while the model is still loading.

        Args:
            timeout_per_message: how long the server may stay SILENT between frames (default: 30s).
            url: the endpoint being waited on, for the failure to name.
            ready_deadline: monotonic deadline for reaching ``ready``; ``None`` waits while it keeps talking.
            waiting_since: when the caller started waiting — its first connection attempt, not this
                handshake, so the reported wait covers the same span the deadline bounds.
        """
        started = time.monotonic() if waiting_since is None else waiting_since
        status: str = 'no status frame'
        message = ''
        try:
            while True:
                timeout = timeout_per_message
                if ready_deadline is not None:
                    remaining = ready_deadline - time.monotonic()
                    if remaining <= 0:
                        raise ServerNotReady(url, status, message, time.monotonic() - started)
                    timeout = min(timeout, remaining)
                response = deserialise(self._websocket.recv(timeout=timeout))
                if protocol.ERROR in response:
                    raise RuntimeError(f'Server error: {response[protocol.ERROR]}')
                try:
                    reported = protocol.ServerStatus(response.get(protocol.STATUS))
                except ValueError:
                    raise RuntimeError(f'Unexpected server response: {response}') from None

                if reported is protocol.ServerStatus.READY:
                    return response[protocol.META]
                if reported is protocol.ServerStatus.ERROR:
                    raise RuntimeError('Server error: Unknown error')

                # Everything else the server can report is a session still in flight rather than one that
                # failed: keep the last of them, which is what a give-up names.
                status, message = reported, response.get(protocol.MESSAGE, reported)
                logger.info(f'Server status: [{status}] {message}')

        except TimeoutError:
            # A recv bounded by the deadline expiring is not the server going quiet — report which happened.
            if ready_deadline is not None and time.monotonic() >= ready_deadline:
                raise ServerNotReady(url, status, message, time.monotonic() - started) from None
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

    def new_session(self, ready_deadline: float | None = None) -> InferenceSession:
        """Creates a new inference session on the model the URL names.

        ``ready_deadline`` bounds the whole wait, reconnects included; ``None`` leaves ``connect_deadline``.
        """
        # Before the first attempt, so a refusal reports the wait the deadline bounds rather than the last
        # handshake: an endpoint that spends most of its budget refusing connections waited that long too.
        started = time.monotonic()
        deadline = started + self.connect_deadline
        if ready_deadline is not None:
            deadline = min(deadline, ready_deadline)
        # The handshake gets the same bound as the connects, so a shorter ``connect_deadline`` is not
        # outlived by a server that connects and then streams ``loading``. Left unbounded where the caller
        # named no readiness bound: ``connect_deadline`` covers reaching a server, not loading a checkpoint.
        handshake_deadline = deadline if ready_deadline is not None else None
        backoff = 1.0
        retries = _ConnectRetries()
        while True:
            ws = None
            session = None
            try:
                # Capped by the deadline: an unanswered connect otherwise overruns the caller's bound.
                open_timeout = min(self.open_timeout, max(deadline - time.monotonic(), 0.0))
                # A proxy between here and the server closes a connection it has read nothing from, often
                # after 60s — well inside one ``infer_timeout`` inference, which sends nothing until it
                # answers. The pings keep it open.
                ws = connect(
                    self.session_url, open_timeout=open_timeout, additional_headers=self.headers, ping_interval=20.0
                )
                session = InferenceSession(
                    ws,
                    infer_timeout=self.infer_timeout,
                    url=self.session_url,
                    ready_deadline=handshake_deadline,
                    waiting_since=started,
                )
                return session
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
                # Closed here rather than left to ``finally``, which runs after the backoff below.
                if ws is not None:
                    ws.close()
                    ws = None
                if retries.take(e) is _ConnectOutcome.SURFACE:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{e} (connecting to {self.session_url})') from e
                # Capped by the deadline: a full backoff otherwise sleeps up to 30s past the caller's bound.
                pause = min(backoff, max(deadline - time.monotonic(), 0.0))
                logger.info('Server not ready (cold start?): %s; retrying in %.0fs', e, pause)
                time.sleep(pause)
                backoff = min(backoff * 2, 30.0)
            except OSError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e
            finally:
                # A socket that produced no session is owned by nobody; a returned session owns its own.
                if session is None and ws is not None:
                    ws.close()

    def list_models(self) -> list[str]:
        """List available models from the server."""
        response = httpx.get(f'{self.api_url}/models', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
