import logging
import ssl
import time
import urllib.parse
from typing import Any

import httpx
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.sync.client import connect
from websockets.sync.connection import Connection

from positronic.utils.serialization import deserialise, serialise

logger = logging.getLogger(__name__)

# Only the checkpoint pinned at server startup is pre-warmed; a session that requests any other model loads it
# cold, so its first ``infer`` can include the backend's JAX compilation. Bound each ``recv`` generously enough to
# outlast that compile (still surfacing a truly stalled/half-open connection), and let callers override per use.
DEFAULT_INFER_TIMEOUT = 180.0


class InferenceSession:
    def __init__(self, websocket: Connection, infer_timeout: float = DEFAULT_INFER_TIMEOUT):
        self._websocket = websocket
        self._infer_timeout = infer_timeout
        self._metadata = self._handshake()

    def _handshake(self, timeout_per_message: float = 30.0) -> dict[str, Any]:
        """Receive status updates until server is ready.

        Args:
            timeout_per_message: Timeout for each individual message (default: 30s).
                               Server must send updates at least this frequently.
        """
        try:
            while True:
                response = deserialise(self._websocket.recv(timeout=timeout_per_message))
                status = response.get('status')

                if status == 'ready':
                    return response['meta']

                if status in ('waiting', 'loading'):
                    message = response.get('message', status)
                    logger.info(f'Server status: [{status}] {message}')
                    continue

                if status == 'error' or 'error' in response:
                    raise RuntimeError(f'Server error: {response.get("error", "Unknown error")}')

                raise RuntimeError(f'Unexpected server response: {response}')

        except TimeoutError:
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

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')

        return response['result']

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

    ``headers`` carry auth for an endpoint behind a reverse proxy — credentials stay out of the URL, which
    is meant to be safe to hand around.

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

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the model the URL names."""
        deadline = time.monotonic() + self.connect_deadline
        backoff = 1.0
        while True:
            ws = None
            try:
                ws = connect(self.session_url, open_timeout=self.open_timeout, additional_headers=self.headers)
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

    def list_models(self) -> list[str]:
        """List available models from the server."""
        response = httpx.get(f'{self.api_url}/models', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
