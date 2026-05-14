from types import MappingProxyType

import numpy as np

from positronic.drivers.roboarm.command import CartesianPosition, JointDelta, JointPosition, Recover, Reset
from positronic.geom import Rotation, Transform3D
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
        mock_policy._mock_session.assert_called_with(obs)
    finally:
        session.close()


def test_inference_client_new_session(inference_server, mock_policy):
    """Test that starting a new session calls new_session on the policy."""
    host, port = inference_server
    client = InferenceClient(host, port)

    # First session
    session = client.new_session()
    session.close()

    # Second session
    session = client.new_session()
    session.close()

    assert mock_policy.new_session.call_count == 2


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

    policies['alpha']._mock_session.assert_any_call({'obs': 'alpha'})
    policies['beta']._mock_session.assert_any_call({'obs': 'beta'})
    policies['alpha']._mock_session.assert_any_call({'obs': 'default'})


def test_wire_serialisation_accepts_mappingproxy():
    backing = {'a': 1, 'b': {'c': 2}}
    frozen = MappingProxyType(backing)
    payload = {'obs': frozen}

    round_trip = deserialise(serialise(payload))

    # mappingproxy is normalized to a plain dict for the wire.
    assert round_trip == {'obs': {'a': 1, 'b': {'c': 2}}}


class TestCommandRoundtrip:
    """msgpack-level (de)serialization reconstructs Command instances transparently."""

    def test_reset(self):
        assert isinstance(deserialise(serialise(Reset())), Reset)

    def test_recover(self):
        assert isinstance(deserialise(serialise(Recover())), Recover)

    def test_cartesian_position(self):
        pose = Transform3D(translation=np.array([0.1, 0.2, 0.3], dtype=np.float32), rotation=Rotation.identity)
        cmd = CartesianPosition(pose=pose)
        result = deserialise(serialise(cmd))
        assert isinstance(result, CartesianPosition)
        np.testing.assert_allclose(result.pose.translation, [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(result.pose.rotation.as_quat, Rotation.identity.as_quat, atol=1e-6)

    def test_joint_position(self):
        positions = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)
        result = deserialise(serialise(JointPosition(positions=positions)))
        assert isinstance(result, JointPosition)
        np.testing.assert_allclose(result.positions, positions)

    def test_joint_delta(self):
        velocities = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07], dtype=np.float32)
        result = deserialise(serialise(JointDelta(velocities=velocities)))
        assert isinstance(result, JointDelta)
        np.testing.assert_allclose(result.velocities, velocities)

    def test_action_trajectory_payload(self):
        """The actual server→client payload: a list of action dicts with embedded Commands."""
        pose = Transform3D(translation=np.array([0.4, 0.5, 0.6], dtype=np.float32), rotation=Rotation.identity)
        joints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        actions = [
            {'robot_command': CartesianPosition(pose=pose), 'target_grip': 0.5, 'timestamp': 0.0},
            {'robot_command': Reset(), 'timestamp': 0.1},
            {'robot_command': JointPosition(positions=joints), 'target_grip': 0.8, 'timestamp': 0.2},
        ]
        result = deserialise(serialise({'result': actions}))['result']
        assert len(result) == 3
        assert isinstance(result[0]['robot_command'], CartesianPosition)
        assert isinstance(result[1]['robot_command'], Reset)
        assert isinstance(result[2]['robot_command'], JointPosition)
        assert result[0]['target_grip'] == 0.5
        assert result[1]['timestamp'] == 0.1
        np.testing.assert_allclose(result[2]['robot_command'].positions, joints)

    def test_plain_dict_passthrough(self):
        """Dicts without Commands round-trip unchanged."""
        payload = {'action_data': [1, 2, 3], 'meta': {'k': 'v'}}
        assert deserialise(serialise(payload)) == payload
