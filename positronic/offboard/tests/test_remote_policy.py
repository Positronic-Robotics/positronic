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
    """Test that RemotePolicy returns action chunks correctly."""
    host, port = inference_server

    # Use a separate RemotePolicy and mock the internal session to return a chunk of actions.
    policy = RemotePolicy(host, port)
    policy.reset()

    # Inject a mock session or mock the infer method on the existing session
    mock_session = MagicMock()
    # Simulate first call returning a chunk of 3 actions
    chunk = [{'a': 1}, {'a': 2}, {'a': 3}]
    mock_session.infer.return_value = chunk
    mock_session.metadata = {'model_name': 'test_model'}

    # Replace the real session with our mock
    policy._RemotePolicy__session = mock_session

    # 1. First call: Should trigger network call and return the entire chunk
    actions = policy.select_action({'obs': 1})
    assert actions == chunk
    assert mock_session.infer.call_count == 1

    # 2. Second call: Should trigger another network call (since policy doesn't buffer)
    new_chunk = [{'b': 1}]
    mock_session.infer.return_value = new_chunk
    actions2 = policy.select_action({'obs': 2})
    assert actions2 == new_chunk
    assert mock_session.infer.call_count == 2

    policy.close()
