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
    policy._close_session()
