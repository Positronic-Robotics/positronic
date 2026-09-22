import time
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from positronic import keys, telemetry, telemetry_keys
from positronic.cfg import codecs
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.geom import Transform3D
from positronic.offboard import keys as offboard_keys
from positronic.offboard import protocol, websocket_wire, wire
from positronic.offboard.client import (
    DEFAULT_INFER_TIMEOUT,
    DEFAULT_OPEN_TIMEOUT,
    InferenceClient,
    InferenceSession,
    _ConnectRetries,
)
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.offboard.tests.conftest import DictSource
from positronic.policy.base import Obs, Step
from positronic.policy.codec import ChangeEEFrame, Codec, RestrictImageSize
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.policy.remote import RemotePolicy, prepare_obs, round_trip
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec

CHUNKED_STACK = {'local_stack': {'name': 'chunked_schedule', 'args': {'fps': 10}}}


class _FakeWire(wire.ClientWire):
    """A client wire that answers each dial from ``outcomes``: a connection to return, or a refusal to raise."""

    SCHEME = 'fake'
    SECURE_SCHEME = 'fakes'

    def __init__(self, *outcomes: wire.ClientConnection | wire.ConnectRefused):
        self._outcomes = list(outcomes)
        self.dials: list[tuple[wire.SessionAddress, Mapping[str, str] | None, float]] = []

    def api_url(self, address: wire.SessionAddress) -> str:
        return f'http://{address.netloc}{wire.API_PATH}'

    def dial(self, address: wire.SessionAddress, headers: Mapping[str, str] | None, open_timeout: float):
        self.dials.append((address, headers, open_timeout))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, wire.ConnectRefused):
            raise outcome
        return outcome


_ADDRESS = wire.SessionAddress('localhost', 8000, wire.SESSION_PATH, '', secure=False)


def _mock_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _mock_remote_policy(metadata=None, infer_return=None):
    """A RemotePolicy whose wire client is mocked out; returns (policy, mock_session)."""
    mock_session = _mock_session(metadata)
    if infer_return is not None:
        mock_session.infer.return_value = infer_return
    policy = RemotePolicy('localhost:0')
    policy._client = MagicMock()
    policy._client.new_session.return_value = mock_session
    return policy, mock_session


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """The border's own settings. Image geometry is the declared stack's business (see RestrictImageSize)."""

    def test_images_pass_through_untouched_by_default(self):
        obs = {'cam': _make_image(480, 640), 'state': np.array([1.0])}
        prepared = prepare_obs(obs, compress_images=False)
        assert prepared.keys() == obs.keys()
        assert all(prepared[key] is value for key, value in obs.items())

    def test_compression_reaches_nested_images(self):
        result = prepare_obs(
            {
                'cam': _make_image(48, 64),
                'video': {'wrist': _make_image(48, 64)},
                'state': np.array([1.0, 2.0]),
                keys.TASK: 'pick cube',
            },
            compress_images=True,
        )
        assert isinstance(result['cam'], dict)
        assert isinstance(result['video']['wrist'], dict)
        np.testing.assert_array_equal(result['state'], np.array([1.0, 2.0]))
        assert result[keys.TASK] == 'pick cube'


class TestInferenceClientHeaders:
    def test_default_headers_empty(self):
        assert InferenceClient.from_url('localhost:8000').headers is None

    def test_headers_stored_and_copied(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        client = InferenceClient.from_url('localhost:8000', headers=headers)
        assert client.headers == headers
        # Defensive copy — mutating the caller's dict must not affect the client.
        headers['Modal-Key'] = 'mutated'
        assert client.headers is not None and client.headers['Modal-Key'] == 'k'

    def test_new_session_dials_with_the_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        conn = MagicMock()
        fake = _FakeWire(conn)
        with patch('positronic.offboard.client.InferenceSession') as mock_session_cls:
            InferenceClient(fake, _ADDRESS, headers=headers).new_session()

        assert fake.dials == [(_ADDRESS, headers, DEFAULT_OPEN_TIMEOUT)]
        assert mock_session_cls.call_args.args[0] is conn
        assert mock_session_cls.call_args.kwargs['infer_timeout'] == DEFAULT_INFER_TIMEOUT

    def test_new_session_without_headers_dials_with_none(self):
        fake = _FakeWire(MagicMock())
        with patch('positronic.offboard.client.InferenceSession'):
            InferenceClient(fake, _ADDRESS).new_session()

        assert fake.dials == [(_ADDRESS, None, DEFAULT_OPEN_TIMEOUT)]

    def test_list_models_passes_headers(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': ['m1']}
            client = InferenceClient.from_url('localhost:8000', headers=headers)

            models = client.list_models()

            assert models == ['m1']
            assert mock_get.call_args.kwargs['headers'] == headers

    def test_list_models_without_headers_passes_none(self):
        with patch('positronic.offboard.client.httpx.get') as mock_get:
            mock_get.return_value.json.return_value = {'models': []}
            client = InferenceClient.from_url('localhost:8000')
            client.list_models()

            assert mock_get.call_args.kwargs['headers'] is None


class TestInferenceClientUrl:
    """One URL carries host, port, TLS, model id, and session params; headers stay their own argument."""

    def test_bare_host_defaults_to_the_scheme_port(self):
        client = InferenceClient.from_url('gpu-host')
        assert client.session_url == 'ws://gpu-host/api/v1/session'
        assert client.api_url == 'http://gpu-host/api/v1'

    def test_explicit_port_is_kept(self):
        client = InferenceClient.from_url('localhost:8000')
        assert client.session_url == 'ws://localhost:8000/api/v1/session'
        assert client.api_url == 'http://localhost:8000/api/v1'

    def test_an_ipv6_host_is_bracketed_in_the_url(self):
        """A caller that builds the address itself passes the host raw, brackets included by the wire."""
        address = wire.SessionAddress('::1', 8000, wire.SESSION_PATH, '', secure=False)
        client = InferenceClient(websocket_wire.WebsocketClientWire(), address)
        assert client.session_url == 'ws://[::1]:8000/api/v1/session'
        assert client.api_url == 'http://[::1]:8000/api/v1'

    @pytest.mark.parametrize('host', ['127.0.0.1', 'gpu-host'])
    def test_a_host_that_is_no_ipv6_literal_reaches_the_url_unchanged(self, host):
        address = wire.SessionAddress(host, 8000, wire.SESSION_PATH, '', secure=False)
        client = InferenceClient(websocket_wire.WebsocketClientWire(), address)
        assert client.session_url == f'ws://{host}:8000/api/v1/session'
        assert client.api_url == f'http://{host}:8000/api/v1'

    def test_query_rides_along_verbatim(self):
        """Nothing re-encodes the query: 'false' stays the JSON literal whoever wrote the URL meant."""
        client = InferenceClient.from_url('gpu-host:9000?codec.fps=10&pad=false')
        assert client.session_url == 'ws://gpu-host:9000/api/v1/session?codec.fps=10&pad=false'
        assert client.api_url == 'http://gpu-host:9000/api/v1'

    def test_tls_scheme_defaults_to_443(self):
        """`https://` is the scheme a fronted endpoint hands out; `wss://` names the same connection."""
        for url in ('https://example.com', 'wss://example.com'):
            client = InferenceClient.from_url(url)
            assert client.session_url == 'wss://example.com/api/v1/session'
            assert client.api_url == 'https://example.com/api/v1'

    def test_full_url_keeps_model_id_and_query(self):
        client = InferenceClient.from_url('https://gpu-host:8443/api/v1/session/10000?fps=2.5')
        assert client.session_url == 'wss://gpu-host:8443/api/v1/session/10000?fps=2.5'
        assert client.api_url == 'https://gpu-host:8443/api/v1'

    @pytest.mark.parametrize('url', ['gpu-host/', 'http://gpu-host/api/v1/session', 'http://gpu-host/api/v1/session/'])
    def test_url_naming_no_model_is_the_bare_endpoint(self, url):
        assert InferenceClient.from_url(url).session_url == 'ws://gpu-host/api/v1/session'

    def test_trailing_slash_belongs_to_the_model_id(self):
        """Sources advertise pinned checkpoint dirs verbatim, and `resolve` matches ids exactly."""
        client = InferenceClient.from_url('http://gpu-host/api/v1/session/s3%3A//ckpt/checkpoint-500/')
        assert client.session_url == 'ws://gpu-host/api/v1/session/s3%3A//ckpt/checkpoint-500/'

    def test_model_id_keeps_its_slashes(self):
        client = InferenceClient.from_url('http://gpu-host:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID')
        assert client.session_url == 'ws://gpu-host:8000/api/v1/session/GEAR-Dreams/DreamZero-DROID'

    def test_percent_encoding_survives_as_written(self):
        """The server decodes the id whoever handed out the URL meant, so the client normalizes nothing."""
        client = InferenceClient.from_url('gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231')
        assert client.session_url == 'ws://gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231'

    def test_unexpected_path_rejected(self):
        with pytest.raises(ValueError, match='/api/v1/session'):
            InferenceClient.from_url('gpu-host:8000/api/v2/other')
        with pytest.raises(ValueError, match='/api/v1/session'):
            InferenceClient.from_url('gpu-host:8000/api/v1/sessions/10000')

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match='scheme'):
            InferenceClient.from_url('ftp://gpu-host:8000')

    def test_every_session_dials_the_same_address(self):
        address = wire.SessionAddress('localhost', 8000, f'{wire.SESSION_PATH}/10000', 'fps=10', secure=False)
        fake = _FakeWire(MagicMock(), MagicMock())
        with patch('positronic.offboard.client.InferenceSession'):
            client = InferenceClient(fake, address)
            client.new_session()
            client.new_session()

        assert [dialed for dialed, _headers, _timeout in fake.dials] == [address, address]
        assert client.session_url == 'fake://localhost:8000/api/v1/session/10000?fps=10'


def _refused(refusal: wire.Refusal) -> wire.ConnectRefused:
    return wire.ConnectRefused(refusal, 'refused')


class TestNewSessionRetriesRefusedConnects:
    """Which refusals are a backend still coming up, and which are the endpoint saying no."""

    def test_a_forbidden_refusal_retries_and_the_session_that_follows_is_returned(self):
        fake = _FakeWire(_refused(wire.Refusal.FORBIDDEN), MagicMock())
        with (
            patch('positronic.offboard.client.InferenceSession') as mock_session_cls,
            patch('positronic.offboard.client.time.sleep'),
        ):
            session = InferenceClient(fake, _ADDRESS).new_session()

        assert len(fake.dials) == 2
        assert session is mock_session_cls.return_value

    def test_a_forbidden_refusal_gives_up_once_its_attempts_are_spent(self):
        fake = _FakeWire(*[_refused(wire.Refusal.FORBIDDEN)] * (_ConnectRetries.MAX_FORBIDDEN_ATTEMPTS + 5))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(wire.ConnectRefused),
        ):
            InferenceClient(fake, _ADDRESS).new_session()

        assert len(fake.dials) == _ConnectRetries.MAX_FORBIDDEN_ATTEMPTS

    def test_a_final_refusal_is_raised_at_once(self):
        fake = _FakeWire(_refused(wire.Refusal.FINAL))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(wire.ConnectRefused) as refused,
        ):
            InferenceClient(fake, _ADDRESS).new_session()

        assert len(fake.dials) == 1
        assert refused.value.refusal is wire.Refusal.FINAL

    def test_a_cold_refusal_retries_to_the_deadline(self):
        fake = _FakeWire(_refused(wire.Refusal.COLD))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(TimeoutError, match='fake://localhost:8000'),
        ):
            InferenceClient(fake, _ADDRESS, connect_deadline=0.0).new_session()

        assert len(fake.dials) == 1

    def test_each_session_opens_on_a_full_budget(self):
        """A client that spent forbidden refusals opening one session still gets all of them for the next."""
        one_session = [_refused(wire.Refusal.FORBIDDEN)] * (_ConnectRetries.MAX_FORBIDDEN_ATTEMPTS - 1) + [MagicMock()]
        fake = _FakeWire(*one_session * 2)
        with patch('positronic.offboard.client.InferenceSession'), patch('positronic.offboard.client.time.sleep'):
            client = InferenceClient(fake, _ADDRESS)
            client.new_session()
            client.new_session()

        assert len(fake.dials) == 2 * len(one_session)


def test_remote_policy_hands_the_url_and_headers_to_the_client():
    headers = {'Modal-Key': 'k'}
    client = RemotePolicy('https://example.com/api/v1/session/10000', headers=headers)._client
    assert client is not None
    assert client.session_url == 'wss://example.com/api/v1/session/10000'
    assert client.headers == headers


class FixedModel(Model):
    def __init__(self):
        self.observations = []
        self.session_ids = []
        self.ended_sessions = []

    def __call__(self, obs: Obs, *, session_id: str):
        self.observations.append(obs)
        self.session_ids.append(session_id)
        if obs.get('fail'):
            raise ValueError('model failed')
        return [{'value': index} for index in range(4)]

    def meta(self):
        return {'model_name': 'fixed'}

    def end_session(self, session_id: str) -> None:
        self.ended_sessions.append(session_id)


@pytest.fixture
def served(start_server):
    def start(*, codec=None, local=None, transport='websocket', model=None):
        model = FixedModel() if model is None else model
        pipeline = PolicyDeployment(
            DictSource({'050000': model}),
            local if local is not None else Sequential(StopOnFault(), ChunkedSchedule(fps=10, horizon_sec=0.2)),
            codec=codec,
        )
        server = start_server(pipeline, grpc=transport == 'grpc')
        port = server.port if transport == 'websocket' else server.grpc_port
        scheme = 'http' if transport == 'websocket' else 'grpc'
        return f'{scheme}://{server.host}:{port}/api/v1/session/050000', model, pipeline

    return start


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
@pytest.mark.parametrize('resize_first', [False, True])
def test_remote_chunk_cadence_and_fresh_episode_state(served, transport, resize_first):
    resize = RestrictImageSize(8, 8)
    schedule = ChunkedSchedule(fps=10, horizon_sec=0.2)
    local = Sequential(resize, StopOnFault(), schedule) if resize_first else Sequential(StopOnFault(), schedule, resize)
    url, model, pipeline = served(local=local, transport=transport)
    policy = RemotePolicy(url)
    assert policy.meta()['server.model_name'] == 'fixed'
    assert policy.meta()['server.action_fps'] == 10
    assert policy.meta()['server.action_horizon_sec'] == 0.2
    assert len(model.ended_sessions) == 1  # The metadata probe also ends its session.
    obs = {'image': np.zeros((16, 16, 3), dtype=np.uint8)}
    for episode in range(2):
        now = [0]
        runtime = Executor(lambda now=now: now[0], simulated=True, charge_inference_time=False)
        run = runtime.start(policy)
        try:
            first = run.send(obs)
            assert isinstance(first, Step)
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            step = run.send(obs)
            assert isinstance(step, Step)
            assert [commands for commands in (first.commands, step.commands) if commands] == [{'value': 0}]
            now[0] = 100_000_000
            step = run.send(obs)
            assert isinstance(step, Step) and step.commands == {'value': 1}
            assert len(model.observations) == 2 * episode + 1
            assert model.observations[-1]['image'].shape == (8, 8, 3)
            now[0] = 200_000_000
            first = run.send(obs)
            assert isinstance(first, Step)
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            step = run.send(obs)
            assert isinstance(step, Step)
            assert [commands for commands in (first.commands, step.commands) if commands] == [{'value': 0}]
            assert len(model.observations) == 2 * episode + 2
        finally:
            runtime.close()
            run.close()
        assert len(model.ended_sessions) == episode + 2
        assert model.session_ids[-2:] == [model.ended_sessions[-1]] * 2
    assert len(set(model.ended_sessions)) == 3
    assert from_spec(pipeline.local.to_spec()).to_spec() == pipeline.local.to_spec()


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
@pytest.mark.parametrize('payload', [{protocol.OBSERVATION: {}}, {protocol.END_SESSION: True}], ids=['infer', 'end'])
def test_wrong_session_id_closes_only_the_requesting_session(served, transport, payload):
    url, model, _ = served(transport=transport)
    client = InferenceClient.from_url(url)
    first, second = client.new_session(), client.new_session()
    try:
        assert first.session_id != second.session_id
        assert protocol.SESSION_ID not in first.metadata
        first._conn.send(protocol.serialise({protocol.SESSION_ID: second.session_id, **payload}))
        response = protocol.deserialise(first._conn.recv(timeout=5))
        assert response[protocol.STATUS] == protocol.ServerStatus.ERROR
        assert 'session ID' in response[protocol.ERROR]
        with pytest.raises(wire.PeerDisconnected):
            first._conn.recv(timeout=5)
        assert model.observations == []
        assert model.ended_sessions == [first.session_id]
        with pytest.raises(wire.PeerDisconnected):
            first.infer({})
        assert second.infer({})[0] == {'value': 0}
        assert model.session_ids == [second.session_id]
        first.close()
        first.close()
        assert model.ended_sessions == [first.session_id]
        with pytest.raises(wire.PeerDisconnected, match='closed'):
            first.infer({})
    finally:
        first.close()
        second.close()
    assert model.ended_sessions == [first.session_id, second.session_id]


def test_fatal_server_error_closes_client_without_masking_the_error():
    conn = MagicMock(spec=wire.ClientConnection)
    conn.recv.side_effect = [
        protocol.serialise({protocol.STATUS: protocol.ServerStatus.READY, protocol.META: {}, protocol.SESSION_ID: 's'}),
        protocol.serialise({protocol.STATUS: protocol.ServerStatus.ERROR, protocol.ERROR: 'session ID mismatch'}),
    ]
    session = InferenceSession(conn)
    with pytest.raises(RuntimeError, match='session ID mismatch'):
        session.infer({})
    session.close()
    assert conn.send.call_count == 1
    conn.close.assert_called_once()


@pytest.mark.parametrize('failure', [TimeoutError(), wire.PeerDisconnected('connection lost')])
def test_failed_round_trip_closes_without_sending_end_on_the_broken_connection(failure):
    conn = MagicMock(spec=wire.ClientConnection)
    conn.recv.side_effect = [
        protocol.serialise({protocol.STATUS: protocol.ServerStatus.READY, protocol.META: {}, protocol.SESSION_ID: 's'}),
        failure,
    ]
    session = InferenceSession(conn)
    with pytest.raises(type(failure)):
        session.infer({})
    session.close()
    assert conn.send.call_count == 1
    conn.close.assert_called_once()


class OffsetCodec(Codec):
    def encode(self, data):
        return {**data, 'encoded': 42}

    def _decode_single(self, data):
        return {'value': data['value'] + 10}


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
def test_server_codec_wraps_model_and_errors_leave_connection_usable(served, transport):
    url, model, pipeline = served(codec=OffsetCodec(), transport=transport)
    session = InferenceClient.from_url(url).new_session()
    try:
        assert session.metadata[offboard_keys.LOCAL_STACK] == pipeline.local.to_spec()
        assert session.infer({}) == [{'value': index + 10} for index in range(4)]
        assert model.observations[-1]['encoded'] == 42
        assert session.served_timing[protocol.TIMING_INFER] >= session.served_timing[protocol.TIMING_MODEL] >= 0
        with pytest.raises(RuntimeError, match='model failed'):
            session.infer({'fail': True})
        assert session.served_timing == {}
        assert session.infer({})[0] == {'value': 10}
    finally:
        session.close()


def test_model_timing_excludes_codec_work_and_belongs_to_each_request(served, monkeypatch):
    now_ns = 0
    monkeypatch.setattr('positronic.offboard.server.time.time_ns', lambda: now_ns)

    class TimedModel(FixedModel):
        def __call__(self, obs: Obs, *, session_id: str):
            nonlocal now_ns
            now_ns += obs['duration_ns']
            return super().__call__(obs, session_id=session_id)

    class TimedCodec(Codec):
        def encode(self, data):
            nonlocal now_ns
            now_ns += 3_000_000
            return data

        def decode(self, data):
            nonlocal now_ns
            now_ns += 5_000_000
            return data

    url, _, _ = served(model=TimedModel(), codec=TimedCodec())
    session = InferenceClient.from_url(url).new_session()
    try:
        for model_ms in (2, 7):
            assert session.infer({'duration_ns': model_ms * 1_000_000}) == [{'value': index} for index in range(4)]
            assert session.served_timing[protocol.TIMING_MODEL] == model_ms
            assert session.served_timing[protocol.TIMING_INFER] == model_ms + 8
            assert session.served_timing[protocol.timing_key('timed_codec')] == model_ms + 8
            assert session.served_timing[protocol.timing_key(telemetry_keys.SPAN_POLICY_ENCODE)] == 3
            with pytest.raises(RuntimeError, match='model failed'):
                session.infer({'duration_ns': 11_000_000, 'fail': True})
            assert session.served_timing == {}
    finally:
        session.close()


def test_act_codec_matches_data_conversions_without_timing():
    config = {'obs': codecs.eepose_obs, 'action': codecs.absolute_pos_action, 'flip_grip': True}
    data_codec = codecs.compose_data.override(**config).instantiate()
    timed_codec = codecs.compose.override(**config, fps=15.0, horizon=1.0).instantiate()
    obs = {
        keys.EE_POSE: np.array([0.1, 0.2, 0.3, 1, 0, 0, 0]),
        keys.GRIP: 0.25,
        keys.WRIST_IMAGE: np.full((224, 224, 3), 32, dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.full((224, 224, 3), 64, dtype=np.uint8),
        keys.TASK: 'stack',
    }
    encoded = data_codec.encode(obs)
    expected = timed_codec.encode(obs)
    for key in expected:
        np.testing.assert_array_equal(encoded[key], expected[key])
    actions = [{'action': np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 0.25])} for _ in range(50)]
    decoded = data_codec.decode(actions)
    timed = timed_codec.decode(actions)
    assert len(decoded) == 50
    assert len(timed) == 16
    for actual, reference in zip(decoded[:15], timed[:15], strict=True):
        assert keys.ACTION_TIMESTAMP not in actual
        assert actual.keys() == reference.keys() - {keys.ACTION_TIMESTAMP}
        assert protocol.serialise(actual) == protocol.serialise({key: reference[key] for key in actual})
    rebuilt = from_spec(data_codec.to_spec())
    assert isinstance(rebuilt, Codec)
    assert rebuilt.to_spec() == data_codec.to_spec()


def test_act_codec_can_run_on_either_side_of_the_connection(served):
    class EchoStateModel(FixedModel):
        def __call__(self, obs: Obs, *, session_id: str):
            self.observations.append(obs)
            return [{'action': obs['observation.state']}]

    codec = codecs.compose_data.override(
        obs=codecs.eepose_obs, action=codecs.absolute_pos_action, flip_grip=True
    ).instantiate()
    obs = {
        keys.EE_POSE: np.array([0.1, 0.2, 0.3, 1, 0, 0, 0]),
        keys.GRIP: 0.25,
        keys.WRIST_IMAGE: np.full((224, 224, 3), 32, dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.full((224, 224, 3), 64, dtype=np.uint8),
        keys.TASK: 'stack',
    }
    outputs = []
    inputs = []
    for placement in (
        {'codec': codec},
        {'local': Sequential(StopOnFault(), ChunkedSchedule(fps=10), codec)},
        {'local': Sequential(StopOnFault(), codec, ChunkedSchedule(fps=10))},
    ):
        url, model, _ = served(model=EchoStateModel(), **placement)
        runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
        run = runtime.start(RemotePolicy(url))
        try:
            first = run.send(obs)
            assert isinstance(first, Step)
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            completed = run.send(obs)
            assert isinstance(completed, Step)
            outputs.append(protocol.serialise(dict(first.commands) | dict(completed.commands)))
            inputs.append(protocol.serialise(model.observations[0]))
        finally:
            runtime.close()
            run.close()
    assert inputs[0] == inputs[1] == inputs[2]
    assert outputs[0] == outputs[1] == outputs[2]


def test_pipeline_rejects_frame_conversion_on_both_sides():
    local = Sequential(ChangeEEFrame(Transform3D.identity), ChunkedSchedule(fps=10))
    with pytest.raises(ValueError, match='Only one side'):
        PolicyDeployment(DictSource({'050000': FixedModel()}), local, codec=ChangeEEFrame(Transform3D.identity))


@pytest.fixture
def runtime():
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.mark.parametrize('declaration', [None, {'seq': []}, {'name': 'run_arbitrary_code'}, {'name': 'flip_grip'}])
def test_invalid_declaration_fails_before_inference_and_closes_connection(runtime, declaration):
    metadata = {} if declaration is None else {offboard_keys.LOCAL_STACK: declaration}
    policy, session = _mock_remote_policy(metadata)
    with pytest.raises(ValueError):
        runtime.start(policy)
    session.infer.assert_not_called()
    session.close.assert_called_once()


@pytest.mark.parametrize('compressed', [False, True])
def test_compression_follows_the_handshake(runtime, compressed):
    policy, session = _mock_remote_policy(
        {**CHUNKED_STACK, offboard_keys.COMPRESS_IMAGES: compressed}, infer_return=[{'value': 42}]
    )
    run = runtime.start(policy)
    try:
        run.send({'image': _make_image(48, 64)})
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        sent = session.infer.call_args.args[0]['image']
        assert isinstance(sent, dict if compressed else np.ndarray)
    finally:
        runtime.close()
        run.close()
    session.close.assert_called_once()


@pytest.mark.parametrize('fails', [False, True])
def test_inference_telemetry_excludes_image_preparation_and_records_failures(tmp_path, monkeypatch, fails):
    session = _mock_session()
    session.served_timing = {}
    if fails:
        session.infer.side_effect = TimeoutError('server stalled')
    encoded_at = []

    def encode(image):
        encoded_at.append(time.time_ns())
        return {'jpeg': b''}

    monkeypatch.setattr('positronic.policy.remote.encode_jpeg', encode)
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'test-infer'):
        if fails:
            with pytest.raises(TimeoutError, match='server stalled'):
                round_trip(session, {'image': _make_image(48, 64)}, compress_images=True)
        else:
            assert round_trip(session, {'image': _make_image(48, 64)}, compress_images=True) == {'action': 'test'}
    spans = {s.name: s for s in telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))}
    assert set(spans) == {telemetry_keys.SPAN_POLICY_PREPARE, telemetry_keys.SPAN_POLICY_INFER}
    prepare = spans[telemetry_keys.SPAN_POLICY_PREPARE]
    assert prepare.start_ns <= encoded_at[0] <= prepare.end_ns
    assert prepare.end_ns <= spans[telemetry_keys.SPAN_POLICY_INFER].start_ns


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
def test_bare_commands_cross_the_wire_as_typed_commands(start_server, make_mock_model, runtime, transport, tmp_path):
    pose = [0.4, 0.0, 0.6, 1, 0, 0, 0, 1, 0, 0, 0, 1]
    model = make_mock_model([{keys.ROBOT_COMMAND: {'type': 'cartesian_pos', 'pose': pose}}], {})
    server = start_server(
        PolicyDeployment(DictSource({'default': model}), ChunkedSchedule(fps=10)), grpc=transport == 'grpc'
    )
    url = (
        f'http://{server.host}:{server.port}'
        if transport == 'websocket'
        else f'grpc://{server.host}:{server.grpc_port}'
    )
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'remote-stack'):
        run = runtime.start(RemotePolicy(url))
        try:
            first = run.send({})
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            completed = run.send({})
            assert isinstance(first, Step) and isinstance(completed, Step)
            commands = dict(first.commands) | dict(completed.commands)
            command = commands[keys.ROBOT_COMMAND]
            assert isinstance(command, CartesianPosition)
            np.testing.assert_allclose(command.pose.translation, [0.4, 0.0, 0.6])
        finally:
            runtime.close()
            run.close()
    spans = {s.span_id: s for s in telemetry.read_spans(telemetry.spans_path(tmp_path, telemetry_keys.HARNESS_PROCESS))}
    [span] = [s for s in spans.values() if s.name == telemetry_keys.SPAN_WIRE_RECV]
    for parent_name in (
        telemetry_keys.SPAN_POLICY_INFER,
        telemetry_keys.SPAN_POLICY_SUBMIT,
        'chunked_schedule',
        'remote_policy',
    ):
        assert span.parent_id is not None
        span = spans[span.parent_id]
        assert span.name == parent_name
    assert span.parent_id is None
