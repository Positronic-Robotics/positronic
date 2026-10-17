import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest
from positronic_wire import registry
from positronic_wire.wire import SESSION_PATH, UnixSocketAddress

from positronic.offboard.client import InferenceClient
from positronic.teleop.protocol import KIND, Reply, Request


@pytest.fixture
def socket_path():
    # A Unix socket path has a short length limit, so it does not go under the long pytest directory.
    with tempfile.TemporaryDirectory(prefix='teleop-') as directory:
        yield Path(directory) / 'teleop.sock'


class TeleopConnection:
    """A session of the teleop service, which a test drives one request at a time, as ``TeleopClient`` does."""

    def __init__(self, socket_path: Path, query: str = ''):
        address = UnixSocketAddress(socket_path, SESSION_PATH, query)
        wire = registry.client_wire('websocket_unix')
        client = InferenceClient(wire, address, infer_timeout=5, connect_deadline=5, reconnect_deadline=1)
        self.session = client.new_session()
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._closing = threading.Event()

    def ask(self, request: dict) -> dict:
        return self.session.infer(request)

    def ready(self) -> Reply:
        """Ask for Ready until the service gives an answer that is not ``WAITING``, or the connection closes."""
        while (reply := Reply(self.ask({KIND: Request.READY})[KIND])) is Reply.WAITING and not self._closing.is_set():
            pass
        return reply

    def ready_in_background(self) -> Future[Reply]:
        return self._pool.submit(self.ready)

    def close(self) -> None:
        self._closing.set()
        self._pool.shutdown(wait=True)
        self.session.close()


@pytest.fixture
def connect():
    """Open ``TeleopConnection`` sessions, and close them at the end of the test."""
    opened: list[TeleopConnection] = []

    def open_connection(socket_path: Path, query: str = '') -> TeleopConnection:
        opened.append(TeleopConnection(socket_path, query))
        return opened[-1]

    yield open_connection
    for connection in opened:
        connection.close()
