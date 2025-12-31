import asyncio
import socket
import threading
import time
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from positronic.offboard.basic_server import InferenceServer


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_server_in_thread(server: InferenceServer, loop: asyncio.AbstractEventLoop):
    """Run the async server in a separate thread."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


@pytest.fixture
def mock_policy() -> MagicMock:
    """Mock policy for testing."""
    policy = MagicMock()
    policy.select_action.return_value = {'action_data': [1, 2, 3]}
    policy.meta = {'model_name': 'test_model'}
    return policy


@pytest.fixture
def mock_policy_registry() -> dict[str, MagicMock]:
    policy_alpha = MagicMock()
    policy_alpha.select_action.return_value = {'action_data': ['alpha']}
    policy_alpha.meta = {'model_name': 'alpha'}

    policy_beta = MagicMock()
    policy_beta.select_action.return_value = {'action_data': ['beta']}
    policy_beta.meta = {'model_name': 'beta'}

    return {'alpha': policy_alpha, 'beta': policy_beta}


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
    server_loop.call_soon_threadsafe(server.shutdown)
    server_thread.join(timeout=1.0)


@pytest.fixture
def multi_policy_server(
    mock_policy_registry: dict[str, MagicMock],
) -> Generator[tuple[str, int, dict[str, MagicMock]], None, None]:
    port = find_free_port()
    host = 'localhost'
    registry = {name: (lambda policy=policy: policy) for name, policy in mock_policy_registry.items()}
    server = InferenceServer(registry, host, port)

    server_loop = asyncio.new_event_loop()
    server_thread = threading.Thread(target=run_server_in_thread, args=(server, server_loop), daemon=True)
    server_thread.start()

    start_time = time.time()
    while time.time() - start_time < 5.0:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(0.05)
    else:
        raise RuntimeError('Server failed to start')

    yield host, port, mock_policy_registry

    server_loop.call_soon_threadsafe(server.shutdown)
    server_thread.join(timeout=1.0)
