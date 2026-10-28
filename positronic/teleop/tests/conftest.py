import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest
from positronic_wire import registry
from positronic_wire.wire import SESSION_PATH, UnixSocketAddress

from positronic.offboard.client import InferenceClient, InferenceSession


@pytest.fixture
def socket_path():
    # A Unix socket path has a short length limit, so it does not go under the long pytest directory.
    with tempfile.TemporaryDirectory(prefix='teleop-') as directory:
        yield Path(directory) / 'teleop.sock'


class TeleopConnection:
    """A session of the teleop service, which a test drives one frame at a time, as the client stack does.

    The session opens in the background, because its preparation waits for Ready.
    """

    def __init__(self, socket_path: Path, query: str = ''):
        address = UnixSocketAddress(socket_path, SESSION_PATH, query)
        wire = registry.client_wire('websocket_unix')
        client = InferenceClient(wire, address, infer_timeout=5, connect_deadline=5, reconnect_deadline=1)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self.opening: Future[InferenceSession] = self._pool.submit(client.new_session)

    @property
    def session(self) -> InferenceSession:
        return self.opening.result(5)

    def wait_open(self) -> None:
        """Wait until the session is ready, or raise what its opening raised."""
        self.opening.result(5)

    def frame(self, request: dict) -> dict:
        """The one element of the reply to the frame ``request``."""
        [element] = self.session.infer(request)
        return element

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if self.opening.exception() is None:
            self.session.close()


@pytest.fixture
def connect():
    """Open ``TeleopConnection`` sessions, and close them at the end of the test. A test ends each wait for Ready."""
    opened: list[TeleopConnection] = []

    def open_connection(socket_path: Path, query: str = '') -> TeleopConnection:
        opened.append(TeleopConnection(socket_path, query))
        return opened[-1]

    yield open_connection
    for connection in opened:
        connection.close()
