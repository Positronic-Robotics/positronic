import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from positronic import keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT, InferenceClient
from positronic.policy import RemotePolicy
from positronic.policy.codec import ActionHorizon
from positronic.policy.remote import RemoteSession
from positronic.policy.wrappers import ChunkedSchedule

EMPTY_STACK = {'local_stack': {'seq': []}}


def _mock_ws_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _mock_remote_policy(metadata=None, infer_return=None, **kwargs):
    """A RemotePolicy whose wire client is mocked out; returns (policy, mock_ws)."""
    mock_ws = _mock_ws_session(metadata)
    if infer_return is not None:
        mock_ws.infer.return_value = infer_return
    policy = RemotePolicy('localhost:0', **kwargs)
    policy._endpoint._client = MagicMock()
    policy._endpoint._client.new_session.return_value = mock_ws
    return policy, mock_ws


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """Tests for RemoteSession's optional JPEG compression. Image geometry is the declared
    stack's business (see RestrictImageSize) — the session only compresses."""

    def test_images_pass_through_untouched_by_default(self):
        session = RemoteSession(_mock_ws_session())
        obs = {'cam': _make_image(480, 640), 'state': np.array([1.0])}
        assert session._prepare_obs(obs) is obs

    def test_compression_reaches_nested_images(self):
        session = RemoteSession(_mock_ws_session(), compress_images=True)
        result = session._prepare_obs({
            'cam': _make_image(48, 64),
            'video': {'wrist': _make_image(48, 64)},
            'state': np.array([1.0, 2.0]),
            keys.TASK: 'pick cube',
        })
        assert isinstance(result['cam'], dict)
        assert isinstance(result['video']['wrist'], dict)
        np.testing.assert_array_equal(result['state'], np.array([1.0, 2.0]))
        assert result[keys.TASK] == 'pick cube'


class TestInferenceClientHeaders:
    def test_default_headers_empty(self):
        assert InferenceClient('localhost:8000').headers is None

    def test_headers_stored_and_copied(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        client = InferenceClient('localhost:8000', headers=headers)
        assert client.headers == headers
        # Defensive copy — mutating the caller's dict must not affect the client.
        headers['Modal-Key'] = 'mutated'
        assert client.headers is not None and client.headers['Modal-Key'] == 'k'

    def test_new_session_passes_additional_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession') as mock_session_cls,
        ):
            client = InferenceClient('localhost:8000', headers=headers)
            client.new_session()

            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs['additional_headers'] == headers
            mock_session_cls.assert_called_once_with(mock_connect.return_value, infer_timeout=DEFAULT_INFER_TIMEOUT)

    def test_new_session_without_headers_passes_none(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost:8000')
            client.new_session()

            mock_connect.assert_called_once()
            assert mock_connect.call_args.kwargs['additional_headers'] is None

    def test_list_models_passes_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': ['m1']}
            client = InferenceClient('localhost:8000', headers=headers)

            models = client.list_models()

            assert models == ['m1']
            assert mock_get.call_args.kwargs['headers'] == headers

    def test_list_models_without_headers_passes_none(self):
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': []}
            client = InferenceClient('localhost:8000')
            client.list_models()

            assert mock_get.call_args.kwargs['headers'] is None


class TestInferenceClientUrl:
    """One URL carries host, port, TLS, model id, and session params; headers stay their own argument."""

    def test_bare_host_defaults_to_the_scheme_port(self):
        client = InferenceClient('gpu-host')
        assert client.session_url == 'ws://gpu-host/api/v1/session'
        assert client.api_url == 'http://gpu-host/api/v1'

    def test_explicit_port_is_kept(self):
        client = InferenceClient('localhost:8000')
        assert client.session_url == 'ws://localhost:8000/api/v1/session'
        assert client.api_url == 'http://localhost:8000/api/v1'

    def test_query_rides_along_verbatim(self):
        """Nothing re-encodes the query: 'false' stays the JSON literal whoever wrote the URL meant."""
        client = InferenceClient('gpu-host:9000?codec.fps=10&pad=false')
        assert client.session_url == 'ws://gpu-host:9000/api/v1/session?codec.fps=10&pad=false'
        assert client.api_url == 'http://gpu-host:9000/api/v1'

    def test_tls_scheme_defaults_to_443(self):
        """`https://` is the scheme a fronted endpoint hands out; `wss://` names the same connection."""
        for url in ('https://example.com', 'wss://example.com'):
            client = InferenceClient(url)
            assert client.session_url == 'wss://example.com/api/v1/session'
            assert client.api_url == 'https://example.com/api/v1'

    def test_full_url_keeps_model_id_and_query(self):
        client = InferenceClient('https://gpu-host:8443/api/v1/session/10000?fps=2.5')
        assert client.session_url == 'wss://gpu-host:8443/api/v1/session/10000?fps=2.5'
        assert client.api_url == 'https://gpu-host:8443/api/v1'

    @pytest.mark.parametrize('url', ['gpu-host/', 'http://gpu-host/api/v1/session', 'http://gpu-host/api/v1/session/'])
    def test_url_naming_no_model_is_the_bare_endpoint(self, url):
        assert InferenceClient(url).session_url == 'ws://gpu-host/api/v1/session'

    def test_trailing_slash_belongs_to_the_model_id(self):
        """Sources advertise pinned checkpoint dirs verbatim, and `resolve` matches ids exactly."""
        client = InferenceClient('http://gpu-host/api/v1/session/s3%3A//ckpt/checkpoint-500/')
        assert client.session_url == 'ws://gpu-host/api/v1/session/s3%3A//ckpt/checkpoint-500/'

    def test_model_id_keeps_its_slashes(self):
        client = InferenceClient('http://gpu-host:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID')
        assert client.session_url == 'ws://gpu-host:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID'

    def test_percent_encoding_survives_as_written(self):
        """The server decodes the id whoever handed out the URL meant, so the client normalizes nothing."""
        client = InferenceClient('gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231')
        assert client.session_url == 'ws://gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231'

    def test_unexpected_path_rejected(self):
        with pytest.raises(ValueError, match='/api/v1/session'):
            InferenceClient('gpu-host:8000/api/v2/other')
        with pytest.raises(ValueError, match='/api/v1/session'):
            InferenceClient('gpu-host:8000/api/v1/sessions/10000')

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match='scheme'):
            InferenceClient('ftp://gpu-host:8000')

    def test_every_session_dials_the_session_url(self):
        with (
            patch('positronic.offboard.client.connect') as mock_connect,
            patch('positronic.offboard.client.InferenceSession'),
        ):
            client = InferenceClient('localhost:8000/api/v1/session/10000?fps=10')
            client.new_session()
            client.new_session()

            assert mock_connect.call_count == 2
            for call in mock_connect.call_args_list:
                assert call.args[0] == client.session_url == 'ws://localhost:8000/api/v1/session/10000?fps=10'


def test_remote_policy_hands_the_url_and_headers_to_the_client():
    headers = {'Modal-Key': 'k'}
    client = RemotePolicy('https://example.com/api/v1/session/10000', headers=headers)._endpoint._client
    assert client is not None
    assert client.session_url == 'wss://example.com/api/v1/session/10000'
    assert client.headers == headers


class TestActionHorizonWrapping:
    def test_truncates_action_chunks(self):
        actions = [
            {'a': 1, 'timestamp': 0.0},
            {'a': 2, 'timestamp': 0.25},
            {'a': 3, 'timestamp': 0.5},
            {'a': 4, 'timestamp': 0.75},
        ]
        # Build: ActionHorizon wrapping a RemotePolicy with no local stack of its own
        policy, _ = _mock_remote_policy(EMPTY_STACK, infer_return=actions)
        wrapped = ActionHorizon(0.5).wrap(policy)

        session = wrapped.new_session()
        actions = session({'obs_time_ns': 0})
        assert actions is not None
        assert len(actions) == 3  # 2 within-horizon actions + horizon sentinel
        assert actions[0]['timestamp'] == 0.0
        assert actions[1]['timestamp'] == 0.25
        assert actions[2] == {'timestamp': 0.5}  # horizon sentinel (timestamp = horizon_sec)

    def test_no_truncation_without_horizon(self):
        policy, _ = _mock_remote_policy(
            EMPTY_STACK, infer_return=[{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 1.0}]
        )

        session = policy.new_session()
        actions = session({})
        assert actions is not None
        assert len(actions) == 2


def test_remote_session_normalizes_single_dict():
    """Server returning a single action dict (legacy shape) is wrapped into a 1-element list."""
    policy, _ = _mock_remote_policy(EMPTY_STACK, infer_return={'robot_command': 'X', 'timestamp': 0.0})

    session = policy.new_session()
    actions = session({})
    assert actions == [{'robot_command': 'X', 'timestamp': 0.0}]


def test_remote_session_passes_through_none():
    policy, mock_ws = _mock_remote_policy(EMPTY_STACK)
    mock_ws.infer.return_value = None

    session = policy.new_session()
    assert session({}) is None


def test_remote_policy_meta_exposes_server_fields():
    """RemotePolicy.meta must expose server metadata so SampledPolicy._get_keys
    can read e.g. 'server.checkpoint_path' before a session is created."""
    policy, _ = _mock_remote_policy({'checkpoint_path': '/ckpts/abc', 'model_name': 'foo', **EMPTY_STACK})

    meta = policy.meta
    assert meta['type'] == 'remote'
    assert meta['server.checkpoint_path'] == '/ckpts/abc'
    assert meta['server.model_name'] == 'foo'


def test_no_declaration_falls_back_to_chunked_schedule():
    """A server that declares no ``local_stack`` in the handshake gets the standard ChunkedSchedule."""
    clock = [0.0]
    policy, _ = _mock_remote_policy(infer_return=[{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 0.5}])
    session = policy.new_session(now=lambda: clock[0])
    actions = session({'obs_time_ns': 0})
    # ChunkedSchedule anchored the chunk to now=0.0 and gates re-inference until it is consumed.
    assert actions == [{'a': 1, 'timestamp': 0.0}, {'a': 2, 'timestamp': 0.5}]
    clock[0] = 0.2
    assert session({'obs_time_ns': 0}) is None


def test_declared_stack_built_at_session_open():
    """The server-declared local stack runs in front of the connection."""
    declared = {'local_stack': {'name': 'chunked_schedule'}}
    clock = [1.0]
    policy, mock_ws = _mock_remote_policy(declared, infer_return=[{'a': 1, 'timestamp': 0.0}])
    session = policy.new_session(now=lambda: clock[0])
    actions = session({'obs_time_ns': 0})
    assert actions == [{'a': 1, 'timestamp': 1.0}]


def test_unknown_declared_entry_fails_before_motion():
    policy, _ = _mock_remote_policy({'local_stack': {'name': 'run_arbitrary_code'}, 'positronic_version': '9.9.9'})
    with pytest.raises(ValueError, match='9.9.9'):
        policy.new_session()


def test_operator_local_drives_a_server_that_declares_nothing():
    """The deprecated override stands in where the standard ChunkedSchedule would otherwise apply."""
    policy, _ = _mock_remote_policy(infer_return=[{'a': 1}], local=ActionHorizon(10.0))
    session = policy.new_session(now=lambda: 5.0)
    # ActionHorizon leaves the untimestamped action alone, where ChunkedSchedule would have stamped it.
    assert session({'obs_time_ns': 0}) == [{'a': 1}]


def _sent_frame(metadata, local=None):
    """The 'cam' frame as it reached the wire, for a server whose handshake is `metadata`."""
    policy, mock_ws = _mock_remote_policy(metadata, infer_return=[], local=local)
    policy.new_session(now=lambda: 0.0)({'obs_time_ns': 0, 'cam': _make_image(480, 640)})
    return mock_ws.infer.call_args.args[0]['cam']


def test_legacy_image_sizes_bound_frames_on_the_rig(caplog):
    """A server that declares no stack but reports `image_sizes` gets them honoured as a wire bound."""
    with caplog.at_level(logging.WARNING, logger='positronic.policy.remote'):
        sent = _sent_frame({'image_sizes': (224, 224)})
    # 480x640 scaled down to fit 224x224, aspect ratio kept.
    assert sent.shape == (168, 224, 3)
    assert 'image_sizes' in caplog.text


def test_legacy_per_camera_sizes_collapse_to_the_largest():
    """One bound covers every image, so a mapping errs large rather than shrinking a camera too far."""
    sent = _sent_frame({'image_sizes': {'cam': (224, 224), 'wrist': (320, 256)}})
    assert sent.shape == (240, 320, 3)


def test_legacy_image_sizes_bound_an_operator_stack_too():
    """The bound is a wire setting rather than part of the stack, so `--policy.local` does not drop it."""
    assert _sent_frame({'image_sizes': (224, 224)}, local=ChunkedSchedule()).shape == (168, 224, 3)


def test_legacy_state_only_codec_bounds_nothing():
    """A codec with no images reports an empty mapping, which names no geometry to bound by."""
    sent = _sent_frame({'image_sizes': {}})
    assert sent.shape == (480, 640, 3)


def test_declared_stack_wins_over_legacy_image_sizes():
    """`image_sizes` is the codec's geometry; once a server declares a stack, only the stack bounds the wire."""
    sent = _sent_frame({'image_sizes': (224, 224), **EMPTY_STACK})
    assert sent.shape == (480, 640, 3)


def test_operator_local_rejected_when_the_server_declares():
    """Against a declaring server the override is a contradiction, not a preference."""
    declared = {'local_stack': {'name': 'temporal_stack', 'args': {'keys': ['v'], 'offsets_sec': [0.0]}}}
    policy, _ = _mock_remote_policy(declared, local=ChunkedSchedule())
    with pytest.raises(ValueError, match='--policy.local'):
        policy.new_session()


def test_compression_follows_the_server_declaration():
    """A server behind a message-size cap declares ``remote(compress_images=True)`` and the rig obeys."""
    policy, mock_ws = _mock_remote_policy({**EMPTY_STACK, 'compress_images': True}, infer_return=[])
    policy.new_session()({'cam': _make_image(48, 64)})
    assert isinstance(mock_ws.infer.call_args.args[0]['cam'], dict)


def test_frames_stay_raw_where_the_server_declares_no_compression():
    policy, mock_ws = _mock_remote_policy({**EMPTY_STACK, 'compress_images': False}, infer_return=[])
    policy.new_session()({'cam': _make_image(48, 64)})
    assert isinstance(mock_ws.infer.call_args.args[0]['cam'], np.ndarray)


def test_compression_override_drives_a_server_that_declares_nothing():
    policy, mock_ws = _mock_remote_policy(infer_return=[], compress_images=True)
    policy.new_session(now=lambda: 0.0)({'obs_time_ns': 0, 'cam': _make_image(48, 64)})
    assert isinstance(mock_ws.infer.call_args.args[0]['cam'], dict)


def test_compression_override_rejected_when_the_server_declares():
    policy, _ = _mock_remote_policy({**EMPTY_STACK, 'compress_images': False}, compress_images=True)
    with pytest.raises(ValueError, match='--policy.compress_images'):
        policy.new_session()


def test_remote_policy_lifecycle(inference_server, mock_policy):
    """RemotePolicy against a live server whose pipeline declares a chunked_schedule local stack."""
    host, port = inference_server

    policy = RemotePolicy(f'{host}:{port}')
    session = policy.new_session(now=lambda: 0.0)

    meta = session.meta
    assert meta['server.model_name'] == 'test_model'
    assert meta['type'] == 'remote'

    obs = {'dataset': 'test'}
    action = session(obs)
    # Single-dict server response is normalized to a 1-element list (Session contract) and
    # anchored to absolute time by the declared ChunkedSchedule.
    assert action == [{'action_data': [1, 2, 3], 'timestamp': 0.0}]

    session.close()

    # New session
    session2 = policy.new_session(now=lambda: 0.0)
    session2.close()


def test_remote_session_meta(inference_server):
    """Session meta must include server metadata."""
    host, port = inference_server
    policy = RemotePolicy(f'{host}:{port}')
    session = policy.new_session(now=lambda: 0.0)

    meta = session.meta
    assert meta['type'] == 'remote'
    assert meta['server.model_name'] == 'test_model'

    session.close()
