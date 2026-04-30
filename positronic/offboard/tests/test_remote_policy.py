from unittest.mock import MagicMock

import numpy as np

from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon
from positronic.policy.remote import RemoteSession


def _mock_ws_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """Tests for RemoteSession._prepare_obs image resize logic."""

    def test_server_tuple_resizes_all_images(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': (64, 48)}), resize=None)
        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(240, 320), 'state': np.array([1.0])}
        result = session._prepare_obs(obs)
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (48, 64, 3)
        np.testing.assert_array_equal(result['state'], obs['state'])

    def test_server_dict_resizes_per_key(self):
        sizes = {'cam_a': (64, 48), 'cam_b': (32, 24)}
        session = RemoteSession(_mock_ws_session({'image_sizes': sizes}), resize=None)
        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(480, 640)}
        result = session._prepare_obs(obs)
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (24, 32, 3)

    def test_fallback_resize_scales_by_max_dim(self):
        session = RemoteSession(_mock_ws_session(), resize=160)
        obs = {'cam': _make_image(480, 640)}
        result = session._prepare_obs(obs)
        assert result['cam'].shape == (120, 160, 3)

    def test_no_resize_when_already_correct_size(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': (64, 48)}), resize=None)
        img = _make_image(48, 64)
        result = session._prepare_obs({'cam': img})
        assert result['cam'] is img

    def test_no_resize_without_server_sizes_or_fallback(self):
        session = RemoteSession(_mock_ws_session(), resize=None)
        img = _make_image(480, 640)
        result = session._prepare_obs({'cam': img})
        assert result['cam'] is img

    def test_normalizes_list_to_tuple(self):
        """Wire format (msgpack) turns tuples into lists — must normalize."""
        session = RemoteSession(_mock_ws_session({'image_sizes': [64, 48]}), resize=None)
        assert session._default_image_size == (64, 48)
        assert isinstance(session._default_image_size, tuple)

    def test_normalizes_dict_values(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': {'cam_a': [64, 48], 'cam_b': [32, 24]}}), resize=None)
        assert session._image_sizes == {'cam_a': (64, 48), 'cam_b': (32, 24)}
        assert all(isinstance(v, tuple) for v in session._image_sizes.values())

    def test_non_image_values_pass_through(self):
        session = RemoteSession(_mock_ws_session({'image_sizes': (64, 48)}), resize=None)
        obs = {'state': np.array([1.0, 2.0]), 'task': 'pick cube', 'flag': True}
        result = session._prepare_obs(obs)
        np.testing.assert_array_equal(result['state'], obs['state'])
        assert result['task'] == 'pick cube'
        assert result['flag'] is True


class TestActionHorizonWrapping:
    def test_truncates_action_chunks(self):
        mock_ws = _mock_ws_session()
        mock_ws.infer.return_value = [
            {'a': 1, 'timestamp': 0.0},
            {'a': 2, 'timestamp': 0.25},
            {'a': 3, 'timestamp': 0.5},
            {'a': 4, 'timestamp': 0.75},
        ]
        # Build: ActionHorizon wrapping a RemotePolicy
        policy = RemotePolicy('localhost', 0)
        policy._client = MagicMock()
        policy._client.new_session.return_value = mock_ws
        wrapped = ActionHorizon(0.5).wrap(policy)

        session = wrapped.new_session()
        actions = session({})
        assert len(actions) == 2
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25

    def test_no_truncation_without_horizon(self):
        mock_ws = _mock_ws_session()
        mock_ws.infer.return_value = [{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 1.0}]
        policy = RemotePolicy('localhost', 0)
        policy._client = MagicMock()
        policy._client.new_session.return_value = mock_ws

        session = policy.new_session()
        actions = session({})
        assert len(actions) == 2


def test_remote_policy_lifecycle(inference_server, mock_policy):
    """Test RemotePolicy.new_session() and session call."""
    host, port = inference_server

    policy = RemotePolicy(host, port)
    session = policy.new_session()

    meta = session.meta
    assert meta['server.model_name'] == 'test_model'
    assert meta['type'] == 'remote'

    obs = {'dataset': 'test'}
    action = session(obs)
    assert action['action_data'] == [1, 2, 3]

    session.close()

    # New session
    session2 = policy.new_session()
    session2.close()


def test_remote_policy_chunking(inference_server):
    """Test that RemotePolicy session returns action chunks correctly."""
    host, port = inference_server

    policy = RemotePolicy(host, port)
    session = policy.new_session()

    # Inject a mock ws session
    mock_ws = MagicMock()
    chunk = [{'a': 1}, {'a': 2}, {'a': 3}]
    mock_ws.infer.return_value = chunk
    mock_ws.metadata = {'model_name': 'test_model'}
    session._session = mock_ws

    actions = session({'obs': 1})
    assert actions == chunk
    assert mock_ws.infer.call_count == 1

    new_chunk = [{'b': 1}]
    mock_ws.infer.return_value = new_chunk
    actions2 = session({'obs': 2})
    assert actions2 == new_chunk
    assert mock_ws.infer.call_count == 2

    session.close()


def test_remote_session_meta(inference_server):
    """Session meta must include server metadata."""
    host, port = inference_server
    policy = RemotePolicy(host, port)
    session = policy.new_session()

    meta = session.meta
    assert meta['type'] == 'remote'
    assert meta['server.model_name'] == 'test_model'

    session.close()
