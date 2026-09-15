import logging
import re
import time
import urllib.parse
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Self

import httpx

from positronic import telemetry, telemetry_keys

from . import protocol, wire, wires
from .protocol import deserialise, serialise, typed_commands
from .wire import ClientWire

logger = logging.getLogger(__name__)

# A first ``infer`` can include the backend's own startup cost, such as a JAX compilation. Bound each ``recv``
# generously enough to outlast that (still surfacing a stalled/half-open connection), and let callers override
# per use.
DEFAULT_INFER_TIMEOUT = 180.0
# Opening one session: the connect on either transport, and the handshake on it.
DEFAULT_OPEN_TIMEOUT = 10.0
DEFAULT_CONNECT_DEADLINE = 900.0


class InferenceSession:
    """One session over one open connection, whichever wire carries it."""

    # The timing block of the last decoded inference response; empty when the server sent none, and
    # empty while a round trip is in flight. Declared here so an implementation that skips ``__init__``
    # still carries it.
    served_timing: Mapping[str, float] = MappingProxyType({})

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
        self.served_timing = {}
        serialised = serialise(obs)
        logger.debug('Size of serialised obs: %1.f KiB', len(serialised) / 1024)
        # The pair reads as the uplink and then the wait the server's own time sits inside: each span
        # holds the socket alone. A send outlasting its own bytes is an uplink too slow for the payload.
        wire_bytes = {telemetry_keys.ATTR_WIRE_BYTES: len(serialised)}
        with telemetry.span(telemetry_keys.SPAN_WIRE_SEND, **wire_bytes):
            self._conn.send(serialised)
        try:
            with telemetry.span(telemetry_keys.SPAN_WIRE_RECV):
                received = self._conn.recv(timeout=self._infer_timeout)
        except TimeoutError:
            # The observation is in flight but unanswered; the server's late response would sit in the socket and
            # the next ``recv`` would pair it with a future observation. Close so the desynced session can't be
            # reused — a subsequent ``infer`` fails loudly on the closed socket instead.
            self._conn.close()
            raise TimeoutError(
                f'No inference response within {self._infer_timeout}s — server stalled or connection half-open'
            ) from None
        response = deserialise(received)
        self.served_timing = response.get(protocol.TIMING) or {} if isinstance(response, dict) else {}
        logger.debug('Size of deserialised response: %1.f KiB', len(response) / 1024)

        if isinstance(response, dict) and protocol.ERROR in response:
            raise RuntimeError(f'Server error: {response[protocol.ERROR]}')

        return typed_commands(response[protocol.RESULT])

    def close(self):
        logger.info('InferenceSession.close: %s', self._conn.close())


class _ConnectOutcome(Enum):
    RETRY = 'retry'
    SURFACE = 'surface'


class _ConnectRetries:
    """The retry policy over one ``new_session``'s connect attempts.

    A ``FORBIDDEN`` refusal means a cold backend or a refused credential, and gets ``MAX_FORBIDDEN_ATTEMPTS``
    attempts.
    """

    MAX_FORBIDDEN_ATTEMPTS = 3

    def __init__(self) -> None:
        self._forbidden_attempts = 0

    def take(self, refusal: wire.Refusal) -> _ConnectOutcome:
        """Spend a refused connect against the budget."""
        if refusal is wire.Refusal.FORBIDDEN:
            self._forbidden_attempts += 1
            again = self._forbidden_attempts < self.MAX_FORBIDDEN_ATTEMPTS
        else:
            again = refusal is wire.Refusal.COLD
        return _ConnectOutcome.RETRY if again else _ConnectOutcome.SURFACE


def _socket_and_path(split: urllib.parse.SplitResult, url: str) -> tuple[str, str, str]:
    """The socket path a ``unix://`` URL names — decoded to dial, as written to name — and the URL
    path left over for the server.

    The split runs over the encoded path, so an escaped ``/api/v1`` cannot be read as the marker.
    Decoding follows, and it resolves every escape: ``%2F`` becomes a separator like any other, so a
    socket path cannot hold a directory whose own name carries a slash. Only the socket path is
    decoded, because it names a file; the URL path reaches the server as written, so a model id
    carries its own escapes. The path as written stays for ``session_url``: a decoded ``?`` or ``#``
    would read there as a delimiter, so that URL would name a different socket.
    """
    if split.netloc or not split.path.startswith('/'):
        raise ValueError(f'Socket path must be absolute in {url!r}; write unix:///path/to.sock')
    marker = re.search(r'/api/v1(?=/|$)', split.path)
    written = split.path if marker is None else split.path[: marker.start()]
    if not written:
        raise ValueError(f'No socket path in {url!r}; write unix:///path/to.sock/api/v1/session')
    return urllib.parse.unquote(written), written, ('' if marker is None else split.path[marker.start() :])


def _session_path(path: str, url: str) -> str:
    """The session path a URL names: ``/api/v1/session``, plus the model id it addresses, if any.

    A URL naming no model — a bare host, or the endpoint with or without a trailing slash — addresses the
    endpoint itself, which serves whatever the server pinned.
    """
    if path.rstrip('/') in ('', wire.SESSION_PATH):
        return wire.SESSION_PATH
    if not path.startswith(f'{wire.SESSION_PATH}/'):
        raise ValueError(f'Unexpected path {path!r} in {url!r}; expected {wire.SESSION_PATH}[/<model_id>]')
    # Kept as written, percent-encoding included: a trailing slash is part of the id, and an id that is
    # itself a path (a HuggingFace repo) keeps its slashes as separators.
    return path


class InferenceClient:
    """The connection to one inference server: a wire, a session address, and the settings each session opens with.

    ``from_url`` reads the wire and the address off one URL. ``headers`` carry the credentials; the address
    carries none. ``open_timeout`` bounds opening one session on either transport — the socket or TCP/TLS connect and
    the handshake on it — ``connect_deadline`` the retries until a cold backend answers, and
    ``infer_timeout`` one inference round trip.
    """

    def __init__(
        self,
        client_wire: ClientWire,
        address: wire.SessionAddress,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = DEFAULT_OPEN_TIMEOUT,
        connect_deadline: float = DEFAULT_CONNECT_DEADLINE,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
    ):
        self._wire = client_wire
        self._address = address
        self.session_url = client_wire.session_url(address)
        self.api_url = client_wire.api_url(address)
        self.headers = dict(headers) if headers else None
        self.open_timeout = open_timeout
        self.connect_deadline = connect_deadline
        self.infer_timeout = infer_timeout

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        open_timeout: float = DEFAULT_OPEN_TIMEOUT,
        connect_deadline: float = DEFAULT_CONNECT_DEADLINE,
        infer_timeout: float = DEFAULT_INFER_TIMEOUT,
    ) -> Self:
        """The client one URL names.

        The URL is ``host``, ``host:port`` or ``scheme://host[:port][/api/v1/session[/<model_id>]]``, each
        with an optional ``?query``. The scheme selects the wire and whether the session runs over TLS
        (``wires.BY_SCHEME`` lists them); a URL with no scheme takes the wire that lists the empty scheme,
        without TLS. The port defaults to 443 with TLS and to 80 without. The model id and the query reach
        the server as written, and every session opened here carries them.

        ``unix://<absolute socket path>[/api/v1/session[/<model_id>]][?query]`` reaches a server on the
        same machine over a Unix domain socket, which needs no network. The socket path runs to the first
        ``/api/v1`` segment, so ``unix:///run/policy.sock`` is the default session and
        ``unix:///run/policy.sock/api/v1/session/10000?fps=10`` names a model and a param. TLS does not
        apply.
        """
        split = urllib.parse.urlsplit(url if '://' in url else f'//{url}')
        selected = wires.BY_SCHEME.get(split.scheme)
        if selected is None:
            raise ValueError(f'Unsupported scheme {split.scheme!r} in {url!r}')
        client_wire, scheme = selected
        # The query is forwarded verbatim: the server reads each param value as a JSON literal, and only
        # whoever wrote the URL knows whether `true` means the bool or the string.
        if split.scheme == wire.UNIX_SCHEME:
            uds, written, path = _socket_and_path(split, url)
            # A socket path is not a host. The server reads the path and the query alone, so the handshake
            # asks for them under a host that stands in for the socket.
            address = wire.SessionAddress(
                host='localhost',
                port=wire.default_port(scheme.secure),
                path=_session_path(path, url),
                query=split.query,
                secure=scheme.secure,
                uds=uds,
                uds_as_written=written,
            )
        else:
            if not split.hostname:
                raise ValueError(f'No host in {url!r}')
            address = wire.SessionAddress(
                host=split.hostname,
                port=wire.default_port(scheme.secure) if split.port is None else split.port,
                path=_session_path(split.path, url),
                query=split.query,
                secure=scheme.secure,
            )
        return cls(
            client_wire,
            address,
            headers=headers,
            open_timeout=open_timeout,
            connect_deadline=connect_deadline,
            infer_timeout=infer_timeout,
        )

    @property
    def uds(self) -> str | None:
        """The Unix socket path this client dials, or ``None`` over a network."""
        return self._address.uds

    def _open_session(self) -> InferenceSession:
        """One attempt at a session. The connection closes when the handshake does not finish.

        A refusal sent as a protocol frame (an unknown model, a rejected session param) raises past every
        transport handler, and a connection may hold a reader thread until it is closed.
        """
        conn = self._wire.dial(self._address, self.headers, self.open_timeout)
        try:
            return InferenceSession(conn, infer_timeout=self.infer_timeout)
        except BaseException:
            conn.close()
            raise

    def new_session(self) -> InferenceSession:
        """Creates a new inference session on the model the URL names.

        Raises ``wire.ConnectRefused`` when the wire refuses the session and no retry clears it.
        """
        deadline = time.monotonic() + self.connect_deadline
        backoff = 1.0
        retries = _ConnectRetries()
        while True:
            try:
                return self._open_session()
            except wire.ConnectRefused as e:
                refusal, not_ready = e.refusal, e
            # A status handshake the server did not finish: a backend that is not ready.
            except (TimeoutError, wire.PeerDisconnected) as e:
                refusal, not_ready = wire.Refusal.COLD, e
            except OSError as e:
                raise type(e)(f'{e} (connecting to {self.session_url})') from e
            if retries.take(refusal) is _ConnectOutcome.SURFACE:
                raise not_ready
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{not_ready} (connecting to {self.session_url})') from not_ready
            logger.info('Server not ready (cold start?): %s; retrying in %.0fs', not_ready, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    def list_models(self) -> list[str]:
        """List available models from the server."""
        if self.api_url is None:
            raise ValueError(f'{self.session_url} names a wire that carries sessions alone; list the models over HTTP')
        transport = None if self._address.uds is None else httpx.HTTPTransport(uds=self._address.uds)
        with httpx.Client(transport=transport) as client:
            response = client.get(f'{self.api_url}/{wire.MODELS_ROUTE}', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
