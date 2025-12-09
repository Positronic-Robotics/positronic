import asyncio
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest

from positronic.inference.client import InferenceClient
from positronic.inference.server import InferenceServer


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def run_server_in_thread(server, loop):
    """Run the async server in a separate thread."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(server.serve())
    except asyncio.CancelledError:
        pass


@pytest.fixture
def mock_policy():
    """Mock policy for testing."""
    policy = MagicMock()
    policy.select_action.return_value = {'action_data': [1, 2, 3]}
    policy.meta = {'model_name': 'test_model'}
    return policy


@pytest.fixture
def inference_server(mock_policy):
    """Fixture to start and stop the inference server."""
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
    # Since it's a daemon thread, it will die when the test process ends.
    # But explicitly stopping the loop is good practice if possible.
    server_loop.call_soon_threadsafe(server_loop.stop)
    server_thread.join(timeout=1.0)


def test_inference_client_connect_and_infer(inference_server, mock_policy):
    """Test standard client connection and inference flow."""
    host, port = inference_server
    client = InferenceClient(host, port)

    with client.start_session() as session:
        # 1. Verify Metadata Handshake
        assert session.metadata['model_name'] == 'test_model'

        # 2. Verify Inference
        obs = {'image': 'test'}
        action = session.infer(obs)

        assert action['action_data'] == [1, 2, 3]
        mock_policy.select_action.assert_called_with(obs)


def test_inference_client_reset(inference_server, mock_policy):
    """Test that starting a new session calls reset on the policy."""
    host, port = inference_server
    client = InferenceClient(host, port)

    # First session (Reset #1)
    with client.start_session():
        pass

    # Second session (Reset #2)
    with client.start_session():
        pass

    assert mock_policy.reset.call_count == 2
