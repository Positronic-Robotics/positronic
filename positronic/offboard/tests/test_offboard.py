from types import MappingProxyType

from positronic.offboard.client import InferenceClient
from positronic.utils.serialization import deserialise, serialise


def test_inference_client_connect_and_infer(inference_server, mock_policy):
    """Test standard client connection and inference flow."""
    host, port = inference_server
    client = InferenceClient(host, port)

    session = client.new_session()
    try:
        # 1. Verify Metadata Handshake
        assert session.metadata['model_name'] == 'test_model'

        # 2. Verify Inference
        obs = {'image': 'test'}
        action = session.infer(obs)

        assert action['action_data'] == [1, 2, 3]
        mock_policy.select_action.assert_called_with(obs)
    finally:
        session.close()


def test_inference_client_reset(inference_server, mock_policy):
    """Test that starting a new session calls reset on the policy."""
    host, port = inference_server
    client = InferenceClient(host, port)

    # First session (Reset #1)
    session = client.new_session()
    session.close()

    # Second session (Reset #2)
    session = client.new_session()
    session.close()

    assert mock_policy.reset.call_count == 2


def test_inference_client_selects_model_id(multi_policy_server):
    host, port, policies = multi_policy_server
    client = InferenceClient(host, port)

    default_session = client.new_session()
    try:
        assert default_session.metadata['model_name'] == 'alpha'
        action = default_session.infer({'obs': 'default'})
        assert action['action_data'] == ['alpha']
    finally:
        default_session.close()

    alpha_session = client.new_session('alpha')
    try:
        assert alpha_session.metadata['model_name'] == 'alpha'
        action = alpha_session.infer({'obs': 'alpha'})
        assert action['action_data'] == ['alpha']
    finally:
        alpha_session.close()

    beta_session = client.new_session('beta')
    try:
        assert beta_session.metadata['model_name'] == 'beta'
        action = beta_session.infer({'obs': 'beta'})
        assert action['action_data'] == ['beta']
    finally:
        beta_session.close()

    policies['alpha'].select_action.assert_any_call({'obs': 'alpha'})
    policies['beta'].select_action.assert_any_call({'obs': 'beta'})
    policies['alpha'].select_action.assert_any_call({'obs': 'default'})


def test_wire_serialisation_accepts_mappingproxy():
    backing = {'a': 1, 'b': {'c': 2}}
    frozen = MappingProxyType(backing)
    payload = {'obs': frozen}

    round_trip = deserialise(serialise(payload))

    # mappingproxy is normalized to a plain dict for the wire.
    assert round_trip == {'obs': {'a': 1, 'b': {'c': 2}}}
