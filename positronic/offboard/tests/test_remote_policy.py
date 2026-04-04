from unittest.mock import MagicMock

import numpy as np

from positronic.policy import RemotePolicy
from positronic.policy.base import SampledPolicy
from positronic.policy.codec import ActionHorizon


def _mock_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """Tests for RemotePolicy._prepare_obs image resize logic."""

    def test_server_tuple_resizes_all_images(self):
        policy = RemotePolicy('localhost', 0)
        policy._RemotePolicy__session = _mock_session({'image_sizes': (64, 48)})
        policy._default_image_size = (64, 48)

        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(240, 320), 'state': np.array([1.0])}
        result = policy._prepare_obs(obs)

        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (48, 64, 3)
        np.testing.assert_array_equal(result['state'], obs['state'])

    def test_server_dict_resizes_per_key(self):
        policy = RemotePolicy('localhost', 0)
        sizes = {'cam_a': (64, 48), 'cam_b': (32, 24)}
        policy._RemotePolicy__session = _mock_session({'image_sizes': sizes})
        policy._image_sizes = sizes

        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(480, 640)}
        result = policy._prepare_obs(obs)

        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (24, 32, 3)

    def test_fallback_resize_scales_by_max_dim(self):
        policy = RemotePolicy('localhost', 0, resize=160)
        policy._RemotePolicy__session = _mock_session()

        obs = {'cam': _make_image(480, 640)}
        result = policy._prepare_obs(obs)

        # scale = min(1, 160/640) = 0.25 → 160x120
        assert result['cam'].shape == (120, 160, 3)

    def test_no_resize_when_already_correct_size(self):
        policy = RemotePolicy('localhost', 0)
        policy._RemotePolicy__session = _mock_session({'image_sizes': (64, 48)})
        policy._default_image_size = (64, 48)

        img = _make_image(48, 64)
        obs = {'cam': img}
        result = policy._prepare_obs(obs)

        assert result['cam'] is img  # same object, no copy

    def test_no_resize_without_server_sizes_or_fallback(self):
        policy = RemotePolicy('localhost', 0)
        policy._RemotePolicy__session = _mock_session()

        img = _make_image(480, 640)
        obs = {'cam': img}
        result = policy._prepare_obs(obs)

        assert result['cam'] is img

    def test_reset_normalizes_list_to_tuple(self):
        """Wire format (msgpack) turns tuples into lists — reset() must normalize."""
        policy = RemotePolicy('localhost', 0)
        mock_client = MagicMock()
        mock_client.new_session.return_value = _mock_session({'image_sizes': [64, 48]})
        policy._client = mock_client

        policy.reset()
        assert policy._default_image_size == (64, 48)
        assert isinstance(policy._default_image_size, tuple)

    def test_reset_normalizes_dict_values(self):
        policy = RemotePolicy('localhost', 0)
        mock_client = MagicMock()
        mock_client.new_session.return_value = _mock_session({'image_sizes': {'cam_a': [64, 48], 'cam_b': [32, 24]}})
        policy._client = mock_client

        policy.reset()
        assert policy._image_sizes == {'cam_a': (64, 48), 'cam_b': (32, 24)}
        assert all(isinstance(v, tuple) for v in policy._image_sizes.values())

    def test_non_image_values_pass_through(self):
        policy = RemotePolicy('localhost', 0)
        policy._RemotePolicy__session = _mock_session({'image_sizes': (64, 48)})
        policy._default_image_size = (64, 48)

        obs = {'state': np.array([1.0, 2.0]), 'task': 'pick cube', 'flag': True}
        result = policy._prepare_obs(obs)

        np.testing.assert_array_equal(result['state'], obs['state'])
        assert result['task'] == 'pick cube'
        assert result['flag'] is True


class TestActionHorizonWrapping:
    def test_truncates_action_chunks(self):
        policy = RemotePolicy('localhost', 0)
        mock = _mock_session()
        mock.infer.return_value = [
            {'a': 1, 'timestamp': 0.0},
            {'a': 2, 'timestamp': 0.25},
            {'a': 3, 'timestamp': 0.5},
            {'a': 4, 'timestamp': 0.75},
        ]
        policy._RemotePolicy__session = mock
        wrapped = ActionHorizon(0.5).wrap(policy)

        actions = wrapped.select_action({})
        assert len(actions) == 2
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25

    def test_no_truncation_without_horizon(self):
        policy = RemotePolicy('localhost', 0)
        mock = _mock_session()
        mock.infer.return_value = [{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 1.0}]
        policy._RemotePolicy__session = mock

        actions = policy.select_action({})
        assert len(actions) == 2


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


class TestMetaRecording:
    """Regression: server metadata must always be captured in RemotePolicy.meta.

    A dirty change on the notebook replaced ``self._session.metadata`` (lazy init via
    property) with ``self.__session is not None`` (bypasses lazy init).  When meta() was
    called before reset(), no server metadata was recorded — 5 episodes got "unknown" model.

    These tests would FAIL with the buggy pattern and PASS with the current code.
    """

    def test_meta_before_reset_returns_server_metadata(self, inference_server):
        """RemotePolicy.meta must return server metadata without prior reset().

        Production flow: DsWriterCommandMetaBridge calls inference.meta() on START,
        which calls policy.meta before any RESET command is issued.
        """
        host, port = inference_server
        policy = RemotePolicy(host, port)

        # No reset() — call meta directly
        meta = policy.meta

        assert meta['type'] == 'remote'
        assert meta['server.model_name'] == 'test_model'

        policy.close()

    def test_sampled_policy_meta_before_reset(self, inference_server):
        """SampledPolicy wrapping RemotePolicy must capture server metadata without reset().

        SampledPolicy.__init__ picks a sub-policy but does NOT call reset() on it.
        When DsWriterCommandMetaBridge calls meta() on the first START, the lazy init
        in RemotePolicy._session must trigger a connection and capture server metadata.
        """
        host, port = inference_server
        remote = RemotePolicy(host, port)
        sampled = SampledPolicy(remote)

        meta = sampled.meta

        assert meta['type'] == 'remote'
        assert meta['server.model_name'] == 'test_model'

        remote.close()

    def test_bypassing_lazy_init_loses_metadata(self):
        """Demonstrates the exact bug: __session check bypasses lazy init.

        The dirty code did:
            if self.__session is not None:
                result['server'] = self.__session.metadata
        Instead of:
            result['server'] = self._session.metadata  # triggers lazy init
        """
        policy = RemotePolicy('localhost', 0)

        # Before any reset(), __session is None
        assert policy._RemotePolicy__session is None

        # Buggy pattern: check __session directly → no metadata captured
        buggy_meta = {'type': 'remote'}
        if policy._RemotePolicy__session is not None:
            buggy_meta['server'] = policy._RemotePolicy__session.metadata

        assert 'server' not in buggy_meta  # Bug: server metadata lost!
