import dataclasses
import pathlib
import threading
import time
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from positronic_wire import registry, websocket, wire

from positronic import keys, telemetry, telemetry_keys
from positronic.cfg import codecs
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.geom import Transform3D
from positronic.offboard import keys as offboard_keys
from positronic.offboard import protocol
from positronic.offboard.client import (
    DEFAULT_INFER_TIMEOUT,
    DEFAULT_OPEN_TIMEOUT,
    InferenceClient,
    InferenceSession,
    _ConnectRetries,
)
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.offboard.tests.conftest import DictSource
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs, Step
from positronic.policy.codec import ChangeEEFrame, Codec, RestrictImageSize
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.remote import RemotePolicy, prepare_obs, round_trip
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec

CHUNKED_STACK = {'local_stack': {'name': 'chunked_schedule', 'version': 2, 'args': {'fps': 10}}}


class _FakeWire(wire.ClientWire[wire.HostPortAddress]):
    """A client wire that answers each dial from ``outcomes``: a connection to return, or a refusal to raise."""

    NAME = 'fake'
    ADDRESS = wire.HostPortAddress

    def __init__(self, *outcomes: wire.ClientConnection | wire.ConnectRefused):
        self._outcomes = list(outcomes)
        self.dials: list[tuple[wire.HostPortAddress, Mapping[str, str] | None, float]] = []
        self.catalogue_reads: list[tuple[wire.HostPortAddress, Mapping[str, str] | None, float]] = []
        self.models: list[str] = []

    def session_url(self, address: wire.HostPortAddress) -> str:
        query = f'?{address.query}' if address.query else ''
        return f'fake://{wire.netloc(address, 0)}{address.path}{query}'

    def list_models(self, address: wire.HostPortAddress, headers, open_timeout: float) -> list[str]:
        self.catalogue_reads.append((address, headers, open_timeout))
        return self.models

    def probe(
        self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        return None

    def dial(self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float):
        self.dials.append((address, headers, open_timeout))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, wire.ConnectRefused):
            raise outcome
        return outcome


def _address(host: str, port: int, model: str = '', query: str = '') -> wire.HostPortAddress:
    """Where a network wire opens a session."""
    return wire.HostPortAddress(host, port, wire.session_path(model), query)


_ADDRESS = wire.HostPortAddress('localhost', 8000, wire.SESSION_PATH, '')


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
    policy = RemotePolicy('websocket', _address('localhost', 0))
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
        assert InferenceClient(websocket.WebsocketClientWire(), _ADDRESS).headers is None

    def test_headers_stored_and_copied(self):
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        client = InferenceClient(websocket.WebsocketClientWire(), _ADDRESS, headers=headers)
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

    def test_the_catalogue_read_hands_the_wire_the_headers_and_the_open_timeout(self):
        """The client asks the wire for the catalogue, with the headers and the timeout it dials with."""
        headers = {'Modal-Key': 'k', 'Modal-Secret': 's'}
        fake = _FakeWire()
        fake.models = ['m1']

        client = InferenceClient(fake, _ADDRESS, headers=headers, open_timeout=3.0)

        assert client.list_models() == ['m1']
        assert fake.catalogue_reads == [(_ADDRESS, headers, 3.0)]


def test_every_session_dials_the_same_address():
    address = wire.HostPortAddress('localhost', 8000, wire.session_path('10000'), 'fps=10')
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


def test_remote_policy_hands_the_wire_the_server_the_model_and_the_headers_to_the_client():
    headers = {'Modal-Key': 'k'}
    policy = RemotePolicy(
        'websocket_tls', _address('example.com', 443, model='10000', query='fps=2.5'), headers=headers
    )
    client = policy._client
    assert client.session_url == 'wss://example.com/api/v1/session/10000?fps=2.5'
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
        address = server.ws(model='050000')[1] if transport == 'websocket' else server.grpc(model='050000')[1]
        return address, model, pipeline

    return start


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
@pytest.mark.parametrize('resize_first', [False, True])
def test_remote_chunk_cadence_and_fresh_episode_state(served, transport, resize_first):
    resize = RestrictImageSize(8, 8)
    schedule = ChunkedSchedule(fps=10, horizon_sec=0.2)
    local = Sequential(resize, StopOnFault(), schedule) if resize_first else Sequential(StopOnFault(), schedule, resize)
    address, model, pipeline = served(local=local, transport=transport)
    policy = RemotePolicy(transport, address)
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
    address, model, _ = served(transport=transport)
    client = InferenceClient(registry.client_wire(transport), address)
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
        protocol.serialise({
            protocol.STATUS: protocol.ServerStatus.READY,
            protocol.META: {},
            protocol.SESSION_ID: 's',
            protocol.PROTOCOL_VERSION: 2,
        }),
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
        protocol.serialise({
            protocol.STATUS: protocol.ServerStatus.READY,
            protocol.META: {},
            protocol.SESSION_ID: 's',
            protocol.PROTOCOL_VERSION: 2,
        }),
        failure,
    ]
    session = InferenceSession(conn)
    with pytest.raises(type(failure)):
        session.infer({})
    session.close()
    assert conn.send.call_count == 1
    conn.close.assert_called_once()


@pytest.mark.parametrize(
    'response, error',
    [
        (wire.PeerDisconnected('no acknowledgement'), wire.PeerDisconnected),
        (protocol.serialise({protocol.SESSION_ID: 'wrong', protocol.END_SESSION: True}), RuntimeError),
        (protocol.serialise({protocol.ERROR: 'cleanup failed'}), RuntimeError),
    ],
    ids=['no-ack', 'wrong-session', 'cleanup-error'],
)
def test_close_still_requires_a_valid_ack_when_the_final_write_reports_disconnect(response, error):
    conn = MagicMock(spec=wire.ClientConnection)
    conn.recv.side_effect = [
        protocol.serialise({
            protocol.STATUS: protocol.ServerStatus.READY,
            protocol.META: {},
            protocol.SESSION_ID: 's',
            protocol.PROTOCOL_VERSION: 2,
        }),
        response,
    ]
    conn.send.side_effect = wire.PeerDisconnected('stream ended')
    session = InferenceSession(conn)
    with pytest.raises(error):
        session.close()
    conn.close.assert_called_once()


class OffsetCodec(Codec):
    def encode(self, data):
        return {**data, 'encoded': 42}

    def _decode_single(self, data):
        return {'value': data['value'] + 10}


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
def test_server_codec_wraps_model_and_errors_leave_connection_usable(served, transport):
    address, model, pipeline = served(codec=OffsetCodec(), transport=transport)
    session = InferenceClient(registry.client_wire(transport), address).new_session()
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
    transport = 'websocket'
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

    address, _, _ = served(model=TimedModel(), codec=TimedCodec())
    session = InferenceClient(registry.client_wire(transport), address).new_session()
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


def test_training_metadata_does_not_change_inference_data():
    config = {'obs': codecs.eepose_obs, 'action': codecs.absolute_pos_action, 'flip_grip': True}
    data_codec = codecs.compose_data.override(**config).instantiate()
    training_codec = codecs.compose.override(**config, training_fps=15.0).instantiate()
    obs = {
        keys.EE_POSE: np.array([0.1, 0.2, 0.3, 1, 0, 0, 0]),
        keys.GRIP: 0.25,
        keys.WRIST_IMAGE: np.full((224, 224, 3), 32, dtype=np.uint8),
        keys.EXTERIOR_IMAGE: np.full((224, 224, 3), 64, dtype=np.uint8),
        keys.TASK: 'stack',
    }
    encoded = data_codec.encode(obs)
    expected = training_codec.encode(obs)
    for key in expected:
        np.testing.assert_array_equal(encoded[key], expected[key])
    actions = [{'action': np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 0.25])} for _ in range(50)]
    decoded = data_codec.decode(actions)
    trained = training_codec.decode(actions)
    assert len(decoded) == 50
    assert len(trained) == 50
    for actual, reference in zip(decoded, trained, strict=True):
        assert 'timestamp' not in actual
        assert protocol.serialise(actual) == protocol.serialise(reference)
    assert training_codec.training_encoder.meta[policy_keys.ACTION_FPS] == 15
    rebuilt = from_spec(data_codec.to_spec())
    assert isinstance(rebuilt, Codec)
    assert rebuilt.to_spec() == data_codec.to_spec()


def test_act_codec_can_run_on_either_side_of_the_connection(served):
    transport = 'websocket'

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
        address, model, _ = served(model=EchoStateModel(), **placement)
        runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
        run = runtime.start(RemotePolicy(transport, address))
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


def test_stack_failure_finishes_active_inference_before_closing_session(runtime):
    started, release = threading.Event(), threading.Event()
    order = []
    stack = Sequential(TemporalStack(('image',), (0.0,)), ChunkedSchedule(fps=10))
    policy, session = _mock_remote_policy({offboard_keys.LOCAL_STACK: stack.to_spec()})

    def infer(obs):
        started.set()
        assert release.wait(5), 'inference was not released'
        order.append('inference finished')
        return [{}]

    session.infer.side_effect = infer
    session.close.side_effect = lambda: order.append('session closed')
    run = runtime.start(policy)
    releaser = threading.Timer(0.05, release.set)
    try:
        run.send({'image': _make_image(8, 8)})
        assert started.wait(5)
        releaser.start()
        with pytest.raises(KeyError, match='image'):
            run.send({})
    finally:
        release.set()
        releaser.cancel()
        runtime.close()
        run.close()
    assert order == ['inference finished', 'session closed']


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
    address = server.ws()[1] if transport == 'websocket' else server.grpc()[1]
    with telemetry.bind(tmp_path, telemetry_keys.HARNESS_PROCESS, 'remote-stack'):
        run = runtime.start(RemotePolicy(transport, address))
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


def test_a_client_refuses_a_wire_handed_the_other_wire_address():
    """The registry answers by name, so the type cannot catch this one; the client does, before it dials."""
    socket_address = wire.UnixSocketAddress(pathlib.Path('/run/policy.sock'), wire.SESSION_PATH, '')

    with pytest.raises(ValueError, match='websocket dials a HostPortAddress'):
        InferenceClient(websocket.WebsocketClientWire(), socket_address)

    with pytest.raises(ValueError, match='websocket_unix dials a UnixSocketAddress'):
        InferenceClient(websocket.WebsocketUnixClientWire(), _ADDRESS)


def test_a_websocket_port_that_never_answers_is_named_at_the_deadline():
    """Nothing listens on port 1; the refused connect is a backend that is not ready, and the deadline ends it."""
    address = dataclasses.replace(_ADDRESS, port=1)
    client = InferenceClient(websocket.WebsocketClientWire(), address, open_timeout=0.2, connect_deadline=0.0)
    with pytest.raises(TimeoutError, match='ws://localhost:1'):
        client.new_session()


def test_a_wire_no_registry_member_carries_is_refused():
    with pytest.raises(ValueError, match="No wire is called 'ws'"):
        RemotePolicy('ws', _address('localhost', 8000))
