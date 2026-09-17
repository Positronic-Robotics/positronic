import logging
import time
import urllib.parse
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any, Self

import httpx

from positronic import keys, telemetry, telemetry_keys

from . import protocol, wire, wires
from .protocol import deserialise, serialise, typed_commands
from .wire import ClientWire

logger = logging.getLogger(__name__)

# A first ``infer`` can include the backend's own startup cost, such as a JAX compilation. Bound each ``recv``
# generously enough to outlast that (still surfacing a stalled/half-open connection), and let callers override
# per use.
DEFAULT_INFER_TIMEOUT = 180.0
# One TCP/TLS handshake, and the retries until a cold backend answers.
DEFAULT_OPEN_TIMEOUT = 10.0
DEFAULT_CONNECT_DEADLINE = 900.0
# One unary call, which answers off state the server already holds.
DEFAULT_VERB_TIMEOUT = 30.0
# How often ``warm(wait=True)`` reads the count again. A warm is a forward pass, not a load.
WARM_POLL_SEC = 2.0


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


class _ConnectWaits:
    """The log line each wait of one ``new_session`` writes.

    The readiness verb names what the server is doing, where a refused connect names only that it was
    refused. A server too old to answer the verb is asked once.
    """

    def __init__(self, client: 'InferenceClient'):
        self._client = client
        self._answers_ready = True

    def line(self, not_ready: BaseException, budget: float) -> str:
        """The line one wait writes. ``budget`` is what is left of the connect deadline, and the probe
        spends no more than that: a verb timeout outlasting the connect is not the caller's to give."""
        if self._answers_ready:
            if budget <= 0:
                return f'Server not ready: {not_ready}'
            try:
                state = self._client._readiness(min(self._client.verb_timeout, budget))
            except wire.VerbUnsupported:
                self._answers_ready = False
            except (wire.ConnectRefused, OSError, TimeoutError):
                # The probe adds nothing here: ``not_ready`` already says the server answers nothing.
                return f'Server not ready: {not_ready}'
            else:
                return f'Server answers {state.status} on {state.inferences} inferences: {state.message}'
        return f'Server not ready, and too old to say how warm it is: {not_ready}'


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
    carries none. ``open_timeout`` bounds one TCP/TLS handshake, ``connect_deadline`` the retries until a
    cold backend answers, and ``infer_timeout`` one inference round trip.
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
        verb_timeout: float = DEFAULT_VERB_TIMEOUT,
    ):
        self._wire = client_wire
        self._address = address
        self.session_url = client_wire.session_url(address)
        self.api_url = client_wire.api_url(address)
        self.headers = dict(headers) if headers else None
        self.open_timeout = open_timeout
        self.connect_deadline = connect_deadline
        self.infer_timeout = infer_timeout
        self.verb_timeout = verb_timeout

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
        """
        split = urllib.parse.urlsplit(url if '://' in url else f'//{url}')
        selected = wires.BY_SCHEME.get(split.scheme)
        if selected is None:
            raise ValueError(f'Unsupported scheme {split.scheme!r} in {url!r}')
        if not split.hostname:
            raise ValueError(f'No host in {url!r}')
        client_wire, scheme = selected
        address = wire.SessionAddress(
            host=split.hostname,
            port=wire.default_port(scheme.secure) if split.port is None else split.port,
            path=_session_path(split.path, url),
            # Forwarded verbatim: the server reads each param value as a JSON literal, and only whoever
            # wrote the URL knows whether `true` means the bool or the string.
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
        waits = _ConnectWaits(self)
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
            logger.info('%s; retrying in %.0fs', waits.line(not_ready, deadline - time.monotonic()), backoff)
            time.sleep(max(0.0, min(backoff, deadline - time.monotonic())))
            backoff = min(backoff * 2, 30.0)

    def readiness(self) -> protocol.Readiness:
        """What the server says about itself now. Raises ``wire.VerbUnsupported`` where the server serves
        sessions but not the verb."""
        return self._readiness(self.verb_timeout)

    def _readiness(self, timeout: float) -> protocol.Readiness:
        return protocol.Readiness.from_wire(self._wire.call(self._address, wire.READY, {}, self.headers, timeout))

    def warm(self, task: str, wait_deadline: float = 0.0) -> protocol.Readiness:
        """Ask the server to pay the first inference on ``task``, and answer what it says now.

        The server starts the warm and does not wait for it, so the record comes back cold. A
        ``wait_deadline`` above zero reads the count again until it moves, and gives up at that many
        seconds. Raises ``wire.VerbUnsupported`` where the server does not answer the verb.
        """
        payload = {keys.TASK: task}
        started = protocol.Readiness.from_wire(
            self._wire.call(self._address, wire.WARM, payload, self.headers, self.verb_timeout)
        )
        if wait_deadline <= 0:
            return started
        # The count this warm has to pass: a server that answered earlier inferences does not start at zero.
        answered_before = started.inferences
        deadline = time.monotonic() + wait_deadline
        latest = started
        while latest.inferences <= answered_before:
            # A poll interval and a verb timeout that outlast the deadline make `wait_deadline` return
            # well after it. Both waits are the caller's to spend.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(WARM_POLL_SEC, remaining))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            latest = self._readiness(min(self.verb_timeout, remaining))
        return latest

    def list_models(self) -> list[str]:
        """List available models from the server."""
        if self.api_url is None:
            raise ValueError(f'{self.session_url} names a wire that carries sessions alone; list the models over HTTP')
        response = httpx.get(f'{self.api_url}/{wire.MODELS_ROUTE}', headers=self.headers)
        response.raise_for_status()
        return response.json()['models']
