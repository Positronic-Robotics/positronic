from unittest.mock import MagicMock

from positronic.policy import RemotePolicy


def test_remote_policy_lifecycle(inference_server, mock_policy):
    """Test RemotePolicy.reset() and select_action()."""
    host, port = inference_server

    # 1. Initialize RemotePolicy
    policy = RemotePolicy(host, port)

    # 2. Check Metadata (Should auto-connect)
    # This verifies that connection happens implicitly without explicit reset first
    meta = policy.meta
    # Ensure flatten_dict logic works
    assert meta['server.model_name'] == 'test_model'
    assert meta['type'] == 'remote'

    # Check that reset was called implicitly on server
    assert mock_policy.reset.call_count == 1

    # 3. Select Action
    obs = {'dataset': 'test'}
    action = policy.select_action(obs)
    assert action['action_data'] == [1, 2, 3]

    # 4. Reset again (new session)
    policy.reset()
    assert mock_policy.reset.call_count == 2

    # Explicitly close to ensure clean server shutdown
    policy.close()


def test_remote_policy_chunking(inference_server):
    """Test that RemotePolicy handles action chunks correctly."""
    host, port = inference_server

    # Use a separate RemotePolicy and mock the internal session to return a chunk of actions.
    policy = RemotePolicy(host, port)
    policy.reset()

    # Inject a mock session or mock the infer method on the existing session
    # We can trust the integration test for connection, here we want to test the buffering logic.
    mock_session = MagicMock()
    # Simulate first call returning a chunk of 3 actions
    mock_session.infer.return_value = [{'a': 1}, {'a': 2}, {'a': 3}]
    mock_session.metadata = {'model_name': 'test_model'}

    # Replace the real session with our mock
    # Note: policy.session is set by reset(), which uses a context manager.
    # We override it manually for this specific white-box test.
    policy._session = mock_session
    policy._action_queue.clear()  # Ensure clean state

    # 1. First call: Should trigger network call and buffer 3 actions
    action1 = policy.select_action({'obs': 1})
    assert action1 == {'a': 1}
    assert mock_session.infer.call_count == 1

    # 2. Second call: Should come from buffer (no network call)
    action2 = policy.select_action({'obs': 2})
    assert action2 == {'a': 2}
    assert mock_session.infer.call_count == 1

    # 3. Third call: Should come from buffer (no network call)
    action3 = policy.select_action({'obs': 3})
    assert action3 == {'a': 3}
    assert mock_session.infer.call_count == 1

    # Prepare for next chunk
    mock_session.infer.return_value = [{'b': 1}]  # Test list format support too

    # 4. Fourth call: Buffer empty, should trigger network call
    action4 = policy.select_action({'obs': 4})
    assert action4 == {'b': 1}
    assert mock_session.infer.call_count == 2

    policy.close()
