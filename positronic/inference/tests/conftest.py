import asyncio
import socket
import threading
import time
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from positronic.inference.server import InferenceServer


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_server_in_thread(server: InferenceServer, loop: asyncio.AbstractEventLoop):
    """Run the async server in a separate thread."""
    asyncio.set_event_loop(loop)
    task = loop.create_task(server.serve())
    try:
        loop.run_forever()
    finally:
        # Safe shutdown sequence
        task.cancel()
        try:
            # Let the task handle cancellation and exit context managers
            loop.run_until_complete(task)
        except (asyncio.CancelledError, RuntimeError):
            pass
        loop.close()


@pytest.fixture
def mock_policy() -> MagicMock:
    """Mock policy for testing."""
    policy = MagicMock()
    policy.select_action.return_value = {'action_data': [1, 2, 3]}
    policy.meta = {'model_name': 'test_model'}
    return policy


@pytest.fixture
def inference_server(mock_policy: MagicMock) -> Generator[tuple[str, int], None, None]:
    """Fixture to start and stop the inference server.

    Returns:
        tuple[str, int]: (host, port)
    """
    port = find_free_port()
    host = 'localhost'
    server = InferenceServer(mock_policy, host, port)

    server_loop = asyncio.new_event_loop()
    server_thread = threading.Thread(target=run_server_in_thread, args=(server, server_loop), daemon=True)
    server_thread.start()

    # Poll for server startup
    start_time = time.time()
    while time.time() - start_time < 5.0:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    else:
        raise RuntimeError('Server failed to start')

    yield host, port

    # Cleanup
    server_loop.call_soon_threadsafe(server_loop.stop)
    server_thread.join(timeout=1.0)
