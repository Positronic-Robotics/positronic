import logging
import ssl
import time
import urllib.parse
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from positronic.offboard.protocol import ERROR, MESSAGE, META, PENDING_STATUSES, READY, RESULT, STATUS
from positronic.utils.serialization import deserialise, serialise

logger = logging.getLogger(__name__)

# Only the checkpoint pinned at server startup is pre-warmed; a session that requests any other model loads it
# cold, so its first ``infer`` can include the backend's JAX compilation. Bound each ``recv`` generously enough to
# outlast that compile (still surfacing a stalled/half-open connection), and let callers override per use.
DEFAULT_INFER_TIMEOUT = 180.0

# How long a server may take to reach ``ready`` before the wait is given up on. The per-message timeout below
# bounds SILENCE, not the handshake: a server that keeps sending ``loading`` frames inside it never trips one,
# so without an overall bound the wait has none, and a stuck load is indistinguishable from a slow one until
# somebody gives up by hand. Sits between the two bounds already here — longer than one inference round trip
# (``DEFAULT_INFER_TIMEOUT``), well short of the connect deadline a retry cycle may spend — and is a parameter
# because how long a legitimate cold start takes is a fact about a deployment's checkpoint and hardware.
DEFAULT_READY_TIMEOUT = 300.0


class ServerNotReady(RuntimeError):
    """A server did not reach ``ready`` within the time allowed, and what it was doing when the wait ended.

    Distinct from the timeouts around it because it means something different: a per-message timeout is a
    server that went silent, which a reconnect may well fix, while this is a server that kept talking and
    never became able to serve. So it is not retried, and it carries the last status frame — the only
    evidence of what it was doing — for whoever reads the failure.
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
    ):
        self._websocket = websocket
        self._infer_timeout = infer_timeout
        self._metadata = self._handshake(url=url, ready_deadline=ready_deadline)

    def _handshake(
        self, timeout_per_message: float = 30.0, *, url: str = '', ready_deadline: float | None = None
    ) -> dict[str, Any]:
        """Receive status updates until the server reports itself ready.

        ``ready`` is the one frame that means the server can serve: our own server sends it only once the
        checkpoint is loaded and a session reset (``PolicyServer._serve_session``). Everything before it —
        the TCP connect, the TLS handshake, the websocket upgrade — completes while the model is still
        loading, which is why none of them answers whether a policy can answer.

        Args:
            timeout_per_message: how long the server may stay SILENT between frames (default: 30s).
            url: the endpoint being waited on, for the failure to name.
            ready_deadline: monotonic deadline for reaching ``ready`` at all. ``None`` waits as long as
                the server keeps talking, which is right for a caller carrying its own bound.
        """
        started = time.monotonic()
        status, message = 'no status frame', ''
        try:
            while True:
                timeout = timeout_per_message
                if ready_deadline is not None:
                    remaining = ready_deadline - time.monotonic()
                    if remaining <= 0:
                        raise ServerNotReady(url, status, message, time.monotonic() - started)
                    timeout = min(timeout, remaining)
                response = deserialise(self._websocket.recv(timeout=timeout))
                status = response.get(STATUS)

                if status == READY:
                    return response[META]

                if status in PENDING_STATUSES:
                    message = response.get(MESSAGE, status)
                    logger.info(f'Server status: [{status}] {message}')
                    continue

                if status == ERROR or ERROR in response:
                    raise RuntimeError(f'Server error: {response.get(ERROR, "Unknown error")}')

                raise RuntimeError(f'Unexpected server response: {response}')

        except TimeoutError:
            # A recv bounded by the deadline rather than by the per-message allowance is the deadline
            # expiring, not the server going quiet; the two have different answers, so say which happened.
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
        """
        Send an observation and get the served session's result — canonically a list of action
        dicts, but whatever the server's session returned (a bare dict or ``None`` included).

        Both `obs` and the returned action must be wire-serializable: plain-data containers and
        scalars, plus numeric numpy arrays/scalars. Do not pass arbitrary Python objects.
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

        if isinstance(response, dict) and ERROR in response:
            raise RuntimeError(f'Server error: {response[ERROR]}')

        return response[RESULT]

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

        ``ready_deadline`` is a monotonic deadline bounding the whole wait for a servable session,
        reconnects included, not each handshake separately. ``None`` leaves ``connect_deadline`` as
        the only bound.
        """
        deadline = time.monotonic() + self.connect_deadline
        if ready_deadline is not None:
            deadline = min(deadline, ready_deadline)
        backoff = 1.0
        while True:
            ws = None
            session = None
            try:
                # Capped by the deadline: an unanswered connect otherwise spends its own fixed timeout
                # past the bound the caller set, and the bound is the whole point of giving one.
                open_timeout = min(self.open_timeout, max(deadline - time.monotonic(), 0.0))
                # A proxy between here and the server closes a connection it has read nothing from, often
                # after 60s — well inside one ``infer_timeout`` inference, which sends nothing until it
                # answers. The pings keep it open.
                ws = connect(
                    self.session_url,
                    open_timeout=open_timeout,
                    additional_headers=self.headers,
                    ping_interval=20.0,
                )
                session = InferenceSession(
                    ws, infer_timeout=self.infer_timeout, url=self.session_url, ready_deadline=ready_deadline
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
                # A non-101 upgrade response only means "not ready" when it's a 5xx or 429; any other status
                # (401/403/404, …) is permanent misconfiguration and surfaces immediately.
                if isinstance(e, InvalidStatus) and not (
                    e.response.status_code >= 500 or e.response.status_code == 429
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'{e} (connecting to {self.session_url})') from e
                logger.info('Server not ready (cold start?): %s; retrying in %.0fs', e, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except OSError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e
            finally:
                # Every path that leaves without a session leaves the socket open otherwise —
                # including ``ServerNotReady``, which is deliberately not one of the retried
                # exceptions above and so passes straight through them.
                if session is None and ws is not None:
                    ws.close()

    def list_models(self) -> list[str]:
        """List available models from the server."""
        response = httpx.get(f'{self.api_url}/models', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
