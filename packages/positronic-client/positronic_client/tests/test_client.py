from unittest.mock import MagicMock, patch

import numpy as np

from positronic_client.client import DEFAULT_INFER_TIMEOUT, InferenceClient, InferenceSession
from positronic_client.serialization import serialise


def _ready_ws(metadata=None):
    """A mock websocket whose handshake immediately reports ready with the given metadata."""
    ws = MagicMock()
    ws.recv.return_value = serialise({'status': 'ready', 'meta': metadata or {}})
    return ws


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """Tests for InferenceSession._prepare_obs image resize logic."""

    def test_server_tuple_resizes_all_images(self):
        session = InferenceSession(_ready_ws({'image_sizes': (64, 48)}))
        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(240, 320), 'state': np.array([1.0])}
        result = session._prepare_obs(obs)
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (48, 64, 3)
        np.testing.assert_array_equal(result['state'], obs['state'])

    def test_server_dict_resizes_per_key(self):
        sizes = {'cam_a': (64, 48), 'cam_b': (32, 24)}
        session = InferenceSession(_ready_ws({'image_sizes': sizes}))
        obs = {'cam_a': _make_image(480, 640), 'cam_b': _make_image(480, 640)}
        result = session._prepare_obs(obs)
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (24, 32, 3)

    def test_fallback_resize_scales_by_max_dim(self):
        session = InferenceSession(_ready_ws(), resize=160)
        obs = {'cam': _make_image(480, 640)}
        result = session._prepare_obs(obs)
        assert result['cam'].shape == (120, 160, 3)

    def test_no_resize_when_already_correct_size(self):
        session = InferenceSession(_ready_ws({'image_sizes': (64, 48)}))
        img = _make_image(48, 64)
        result = session._prepare_obs({'cam': img})
        assert result['cam'] is img

    def test_no_resize_without_server_sizes_or_fallback(self):
        session = InferenceSession(_ready_ws())
        img = _make_image(480, 640)
        result = session._prepare_obs({'cam': img})
        assert result['cam'] is img

    def test_normalizes_list_to_tuple(self):
        """Wire format (msgpack) turns tuples into lists — must normalize."""
        session = InferenceSession(_ready_ws({'image_sizes': [64, 48]}))
        assert session._default_image_size == (64, 48)
        assert isinstance(session._default_image_size, tuple)

    def test_normalizes_dict_values(self):
        session = InferenceSession(_ready_ws({'image_sizes': {'cam_a': [64, 48], 'cam_b': [32, 24]}}))
        assert session._image_sizes == {'cam_a': (64, 48), 'cam_b': (32, 24)}
        assert all(isinstance(v, tuple) for v in session._image_sizes.values())

    def test_non_image_values_pass_through(self):
        session = InferenceSession(_ready_ws({'image_sizes': (64, 48)}))
        obs = {'state': np.array([1.0, 2.0]), 'task': 'pick cube', 'flag': True}
        result = session._prepare_obs(obs)
        np.testing.assert_array_equal(result['state'], obs['state'])
        assert result['task'] == 'pick cube'
        assert result['flag'] is True


class TestInferenceClientHeaders:
    def test_default_headers_empty_and_ws_scheme(self):
        client = InferenceClient('localhost', 8000)
        assert client.headers is None
        assert client.base_uri == 'ws://localhost:8000/api/v1/session'
        assert client.api_url == 'http://localhost:8000/api/v1'

    def test_headers_stored_and_copied(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        client = InferenceClient('localhost', 8000, headers=headers)
        assert client.headers == headers
        # Defensive copy — mutating the caller's dict must not affect the client.
        headers['Modal-Key'] = 'mutated'
        assert client.headers['Modal-Key'] == 'k'

    def test_secure_switches_scheme_and_omits_default_port(self):
        client = InferenceClient('example.com', 443, secure=True)
        assert client.base_uri == 'wss://example.com/api/v1/session'
        assert client.api_url == 'https://example.com/api/v1'

    def test_secure_keeps_non_default_port(self):
        client = InferenceClient('example.com', 8443, secure=True)
        assert client.base_uri == 'wss://example.com:8443/api/v1/session'
        assert client.api_url == 'https://example.com:8443/api/v1'

    def test_insecure_omits_default_port(self):
        client = InferenceClient('example.com', 80, secure=False)
        assert client.base_uri == 'ws://example.com/api/v1/session'
        assert client.api_url == 'http://example.com/api/v1'

    def test_new_session_passes_additional_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with (
            patch('positronic_client.client.connect') as mock_connect,
            patch('positronic_client.client.InferenceSession') as mock_session_cls,
        ):
            client = InferenceClient('localhost', 8000, headers=headers)
            client.new_session()

            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs['additional_headers'] == headers
            assert mock_session_cls.call_args.args == (mock_connect.return_value,)
            assert mock_session_cls.call_args.kwargs['infer_timeout'] == DEFAULT_INFER_TIMEOUT

    def test_new_session_without_headers_omits_additional_headers(self):
        with (
            patch('positronic_client.client.connect') as mock_connect,
            patch('positronic_client.client.InferenceSession'),
        ):
            client = InferenceClient('localhost', 8000)
            client.new_session()

            mock_connect.assert_called_once()
            assert 'additional_headers' not in mock_connect.call_args.kwargs

    def test_new_session_uses_pinned_model_id(self):
        with (
            patch('positronic_client.client.connect') as mock_connect,
            patch('positronic_client.client.InferenceSession'),
        ):
            client = InferenceClient('localhost', 8000, model_id='m42')
            client.new_session()
            assert mock_connect.call_args.args[0].endswith('/api/v1/session/m42')
            # A per-call model_id overrides the pinned one.
            client.new_session(model_id='m7')
            assert mock_connect.call_args.args[0].endswith('/api/v1/session/m7')

    def test_list_models_passes_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with patch('positronic_client.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': ['m1']}
            client = InferenceClient('localhost', 8000, headers=headers)

            models = client.list_models()

            assert models == ['m1']
            assert mock_get.call_args.kwargs['headers'] == headers

    def test_list_models_without_headers_passes_none(self):
        with patch('positronic_client.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': []}
            client = InferenceClient('localhost', 8000)
            client.list_models()

            assert mock_get.call_args.kwargs['headers'] is None
