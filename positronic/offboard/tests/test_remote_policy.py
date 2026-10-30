import asyncio
import dataclasses
import http.server
import json
import logging
import pathlib
import re
import threading
import time
from collections.abc import Mapping
from contextlib import suppress
from unittest.mock import MagicMock, patch

import mujoco as mj
import numpy as np
import pytest
from positronic_model_server import keys as offboard_keys
from positronic_model_server import protocol, serialization, spec, websocket_wire
from positronic_wire import grpc, registry, websocket, wire

from positronic import keys, telemetry, telemetry_keys
from positronic.cfg import codecs
from positronic.cfg import codecs as cfg_codecs
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.command import CartesianPosition, to_wire
from positronic.drivers.roboarm.ik import _prepare_spec
from positronic.drivers.roboarm.models import (
    EE_LINK,
    FRANKA_HAND_FRAME,
    GRIPPER_SITE,
    EEFrame,
    bundled_franka_model,
    ee_frames_entry,
    parse_ee_frames,
)
from positronic.drivers.roboarm.tests.fakes import FakeRobotState
from positronic.eval import keys as eval_keys
from positronic.geom import Rotation, Transform3D
from positronic.gui.station import Outcome, Phase
from positronic.gui.web import EndBody
from positronic.offboard import frame_stream as frame_stream_module
from positronic.offboard import protocol as legacy_protocol
from positronic.offboard import server as server_module
from positronic.offboard.client import (
    DEFAULT_INFER_TIMEOUT,
    DEFAULT_OPEN_TIMEOUT,
    FRAME_MISS,
    ConnectRetries,
    FrameMiss,
    InferenceClient,
    InferenceSession,
)
from positronic.offboard.frame_stream import FRAME_STREAM, REF, ROW, StreamedRows
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import keys as policy_keys
from positronic.policy import remote as remote_module
from positronic.policy.base import Obs, Policy, PolicyEnd, Step
from positronic.policy.codecs import ChangeEEFrame, Codec, EncodeImages, EndCommand, ResizeMode, TransformImages
from positronic.policy.codecs.image import ResizeCache
from positronic.policy.compatibility import WALL_TIME_NS
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.keys import ACTION_FPS
from positronic.policy.processors import (
    ChunkedSchedule,
    ConstantDuration,
    DelayEstimate,
    EndOnField,
    PauseOnUnavailable,
    RTCSchedule,
    TemporalStack,
)
from positronic.policy.remote import RemotePolicy, declared_stack, prepare_obs, round_trip
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec
from positronic.policy.tests.test_harness import legacy_point_warnings, run_robot_rollout
from positronic.tests.testing_inference import post, read_status, run_attended, wait
from positronic.utils import flatten_dict

CHUNKED_STACK = {'local_stack': {'name': 'chunked_schedule', 'version': 2, 'args': {'fps': 10}}}


class _FakeWire(wire.ClientWire[wire.HostPortAddress]):
    """A client wire that answers each dial from ``outcomes``: a connection to return, or a refusal to raise."""

    NAME = 'fake'
    ADDRESS = wire.HostPortAddress

    def __init__(self, *outcomes: wire.ClientConnection | wire.ConnectRefused):
        self._outcomes = list(outcomes)
        self.dials: list[tuple[wire.HostPortAddress, Mapping[str, str] | None, float]] = []

    def session_url(self, address: wire.HostPortAddress) -> str:
        query = f'?{address.query}' if address.query else ''
        return f'fake://{wire.netloc(address, 0)}{address.path}{query}'

    def probe(
        self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float
    ) -> wire.Refusal | None:
        return None

    def keepalive(self, address, headers, timeout):
        raise wire.KeepaliveUnsupported('this wire answers sessions alone')

    def dial(self, address: wire.HostPortAddress, headers: Mapping[str, str] | None, open_timeout: float):
        self.dials.append((address, headers, open_timeout))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, wire.ConnectRefused):
            raise outcome
        return outcome


def _address(host: str, port: int, query: str = '') -> wire.HostPortAddress:
    """Where a network wire opens a session."""
    return wire.HostPortAddress(host, port, wire.SESSION_PATH, query)


_ADDRESS = wire.HostPortAddress('localhost', 8000, wire.SESSION_PATH, '')


def _mock_session(metadata=None):
    session = MagicMock()
    session.metadata = metadata or {}
    session.infer.return_value = {'action': 'test'}
    return session


def _mock_remote_policy(metadata=None, infer_return=None, **policy_kwargs):
    """A RemotePolicy whose wire client is mocked out; returns (policy, mock_session)."""
    mock_session = _mock_session(metadata)
    if infer_return is not None:
        mock_session.infer.return_value = infer_return
    policy = RemotePolicy('websocket', _address('localhost', 0), **policy_kwargs)
    policy._client = MagicMock()
    policy._client.new_session.return_value = mock_session
    return policy, mock_session


def _make_image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestPrepareObs:
    """The border's own settings. Image geometry is the declared stack's business (see TransformImages)."""

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


def test_every_session_dials_the_same_address():
    address = wire.HostPortAddress('localhost', 8000, wire.SESSION_PATH, 'fps=10')
    fake = _FakeWire(MagicMock(), MagicMock())
    with patch('positronic.offboard.client.InferenceSession'):
        client = InferenceClient(fake, address)
        client.new_session()
        client.new_session()

    assert [dialed for dialed, _headers, _timeout in fake.dials] == [address, address]
    assert client.session_url == 'fake://localhost:8000/api/v1/session?fps=10'


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
        fake = _FakeWire(*[_refused(wire.Refusal.FORBIDDEN)] * (ConnectRetries.MAX_FORBIDDEN_ATTEMPTS + 5))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(wire.ConnectRefused),
        ):
            InferenceClient(fake, _ADDRESS).new_session()

        assert len(fake.dials) == ConnectRetries.MAX_FORBIDDEN_ATTEMPTS

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

    @pytest.mark.parametrize('refusal', [wire.Refusal.COLD, wire.Refusal.SILENT])
    def test_a_server_not_up_yet_retries_to_the_deadline(self, refusal):
        fake = _FakeWire(_refused(refusal))
        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.sleep'),
            pytest.raises(TimeoutError, match='fake://localhost:8000'),
        ):
            InferenceClient(fake, _ADDRESS, connect_deadline=0.0).new_session()

        assert len(fake.dials) == 1

    def test_each_session_opens_on_a_full_budget(self):
        """A client that spent forbidden refusals opening one session still gets all of them for the next."""
        one_session = [_refused(wire.Refusal.FORBIDDEN)] * (ConnectRetries.MAX_FORBIDDEN_ATTEMPTS - 1) + [MagicMock()]
        fake = _FakeWire(*one_session * 2)
        with patch('positronic.offboard.client.InferenceSession'), patch('positronic.offboard.client.time.sleep'):
            client = InferenceClient(fake, _ADDRESS)
            client.new_session()
            client.new_session()

        assert len(fake.dials) == 2 * len(one_session)

    def test_no_attempt_begins_past_the_connect_deadline(self):
        """An attempt that begins past the deadline runs a whole `open_timeout`, which the caller never granted."""
        deadline = 1.0
        fake = _FakeWire(*[_refused(wire.Refusal.COLD)] * 5)
        clock = [0.0]

        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.monotonic', side_effect=lambda: clock[0]),
            patch(
                'positronic.offboard.client.time.sleep',
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ),
            pytest.raises(TimeoutError),
        ):
            InferenceClient(fake, _ADDRESS, connect_deadline=deadline).new_session()

        assert len(fake.dials) == 1, f'the loop dialled {len(fake.dials)} times inside a {deadline}s deadline'

    def test_a_wait_that_leaves_budget_still_retries(self):
        deadline = 3.0
        fake = _FakeWire(*[_refused(wire.Refusal.COLD)] * 5)
        clock = [0.0]

        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.monotonic', side_effect=lambda: clock[0]),
            patch(
                'positronic.offboard.client.time.sleep',
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ),
            pytest.raises(TimeoutError),
        ):
            InferenceClient(fake, _ADDRESS, connect_deadline=deadline).new_session()

        assert len(fake.dials) == 2, f'a {deadline}s deadline took {len(fake.dials)} attempt(s)'

    def test_a_connect_backoff_does_not_sleep_past_the_deadline(self):
        deadline = 0.5
        fake = _FakeWire(*[_refused(wire.Refusal.COLD)] * 5)
        clock = [0.0]
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            clock[0] += seconds

        with (
            patch('positronic.offboard.client.InferenceSession'),
            patch('positronic.offboard.client.time.monotonic', side_effect=lambda: clock[0]),
            patch('positronic.offboard.client.time.sleep', side_effect=sleep),
            pytest.raises(TimeoutError),
        ):
            InferenceClient(fake, _ADDRESS, connect_deadline=deadline).new_session()

        assert slept == [deadline], f'a {deadline}s connect deadline slept {slept}'


def _ready(
    session_id: str, meta: dict | None = None, *, version: protocol.ProtocolVersion = protocol.ProtocolVersion.V2
) -> bytes:
    return legacy_protocol.serialise({
        protocol.STATUS: protocol.ServerStatus.READY,
        protocol.PROTOCOL_VERSION: version.value,
        protocol.META: meta if meta is not None else {'model_name': 'test'},
        protocol.SESSION_ID: session_id,
    })


def _answer(result) -> bytes:
    return legacy_protocol.serialise({protocol.RESULT: result})


def _connection(*received: bytes | Exception) -> MagicMock:
    """A connection whose ``recv`` returns or raises each of ``received`` in turn."""
    conn = MagicMock(spec=wire.ClientConnection)
    conn.recv.side_effect = list(received)
    return conn


def _observations_sent(conn: MagicMock) -> list:
    return [
        request[protocol.OBSERVATION]
        for request in (legacy_protocol.deserialise(call.args[0]) for call in conn.send.call_args_list)
        if protocol.OBSERVATION in request
    ]


class TestADroppedConnectionReconnects:
    """A connection that drops before the server's first answer opens a new session and sends the observation
    again. A new session holds no state, so this is safe only while the dropped one had answered nothing."""

    def test_a_drop_before_the_first_answer_sends_the_observation_on_a_new_session(self):
        dropped = _connection(_ready('first'), wire.PeerDisconnected('dropped'))
        reopened = _connection(_ready('second'), _answer({'action_data': [1, 2, 3]}))
        session = InferenceClient(_FakeWire(dropped, reopened), _ADDRESS).new_session()

        assert session.infer({'image': 'test'}) == {'action_data': [1, 2, 3]}

        dropped.close.assert_called_once()
        assert _observations_sent(reopened) == [{'image': 'test'}]
        assert legacy_protocol.deserialise(reopened.send.call_args.args[0])[protocol.SESSION_ID] == 'second'
        assert session.session_id == 'second'

    @pytest.mark.parametrize('version', list(protocol.ProtocolVersion))
    def test_numpy_metadata_survives_a_reconnect_and_reordered_mapping(self, version):
        stats = {'mean': np.array([1.0, 2.0]), 'scale': np.float32(0.5)}
        first_meta = {'model': {'stats': [stats]}, 'name': 'test'}
        second_meta = {'name': 'test', 'model': {'stats': [dict(reversed(stats.items()))]}}
        dropped = _connection(_ready('first', first_meta, version=version), wire.PeerDisconnected('dropped'))
        reopened = _connection(_ready('second', second_meta, version=version), _answer({'action_data': [1, 2]}))
        session = InferenceClient(_FakeWire(dropped, reopened), _ADDRESS).new_session()

        assert session.infer({'image': 'test'}) == {'action_data': [1, 2]}
        dropped.close.assert_called_once()
        assert session.session_id == (None if version is protocol.ProtocolVersion.V1 else 'second')

    @pytest.mark.parametrize('version', list(protocol.ProtocolVersion))
    @pytest.mark.parametrize(
        'changed',
        [np.array([1.0, 3.0]), np.array([[1.0, 2.0]]), np.array([1.0, 2.0], dtype=np.float32)],
        ids=['values', 'shape', 'dtype'],
    )
    def test_changed_numpy_metadata_refuses_a_reconnect(self, version, changed):
        first_meta = {'model': {'stats': [np.array([1.0, 2.0])]}}
        second_meta = {'model': {'stats': [changed]}}
        dropped = _connection(_ready('first', first_meta, version=version), wire.PeerDisconnected('dropped'))
        end_ack = legacy_protocol.serialise({protocol.SESSION_ID: 'second', protocol.END_SESSION: True})
        reopened = _connection(_ready('second', second_meta, version=version), end_ack)
        session = InferenceClient(_FakeWire(dropped, reopened), _ADDRESS).new_session()

        with pytest.raises(wire.PeerDisconnected, match='other metadata'):
            session.infer({'image': 'test'})

        assert _observations_sent(reopened) == []
        reopened.close.assert_called_once()

    def test_a_drop_after_an_answer_reaches_the_caller(self):
        """The server's session held state for the episode, and a new session would start without it."""
        conn = _connection(_ready('first'), _answer({'action_data': [1]}), wire.PeerDisconnected('dropped'))
        fake = _FakeWire(conn, _connection(_ready('second')))
        session = InferenceClient(fake, _ADDRESS).new_session()
        session.infer({'step': 0})

        with pytest.raises(wire.PeerDisconnected, match='dropped'):
            session.infer({'step': 1})

        assert len(fake.dials) == 1

    def test_a_second_drop_reaches_the_caller(self):
        fake = _FakeWire(
            _connection(_ready('first'), wire.PeerDisconnected('dropped')),
            _connection(_ready('second'), wire.PeerDisconnected('dropped again')),
        )
        session = InferenceClient(fake, _ADDRESS).new_session()

        with pytest.raises(wire.PeerDisconnected, match='dropped again'):
            session.infer({'image': 'test'})

        assert len(fake.dials) == 2

    def test_a_stall_is_not_sent_again(self):
        """The server may still compute the observation, so a second send doubles the work of a slow backend."""
        fake = _FakeWire(_connection(_ready('first'), TimeoutError()), _connection(_ready('second')))
        session = InferenceClient(fake, _ADDRESS).new_session()

        with pytest.raises(TimeoutError):
            session.infer({'image': 'test'})

        assert len(fake.dials) == 1

    def test_a_new_session_that_declares_other_metadata_is_refused(self):
        """The caller built its stack from the first handshake, and the episode records it."""
        end_ack = legacy_protocol.serialise({protocol.SESSION_ID: 'second', protocol.END_SESSION: True})
        reopened = _connection(_ready('second', {'model_name': 'another'}), end_ack)
        fake = _FakeWire(_connection(_ready('first'), wire.PeerDisconnected('dropped')), reopened)
        session = InferenceClient(fake, _ADDRESS).new_session()

        with pytest.raises(wire.PeerDisconnected, match='other metadata'):
            session.infer({'image': 'test'})

        assert _observations_sent(reopened) == []
        reopened.close.assert_called_once()

    def test_a_reconnect_retries_a_refused_connect(self):
        """A restarted backend refuses connects for a moment, and the reconnect waits it out."""
        fake = _FakeWire(
            _connection(_ready('first'), wire.PeerDisconnected('dropped')),
            _refused(wire.Refusal.COLD),
            _connection(_ready('second'), _answer({'action_data': [1]})),
        )
        with patch('positronic.offboard.client.time.sleep'):
            session = InferenceClient(fake, _ADDRESS).new_session()
            assert session.infer({'image': 'test'}) == {'action_data': [1]}

        assert len(fake.dials) == 3

    def test_the_reconnect_deadline_bounds_a_handshake_that_never_reaches_ready(self):
        """A server that sends status updates while it loads keeps the arm waiting only for the deadline."""
        clock = [0.0]
        loading = legacy_protocol.serialise({
            protocol.STATUS: protocol.ServerStatus.LOADING,
            legacy_protocol.MESSAGE: 'loading',
        })

        def recv_loading(timeout: float | None = None) -> bytes:
            clock[0] += 5.0
            assert clock[0] < 100.0, 'the handshake outlived its deadline'
            return loading

        loads_for_ever = MagicMock(spec=wire.ClientConnection)
        loads_for_ever.recv.side_effect = recv_loading
        fake = _FakeWire(_connection(_ready('first'), wire.PeerDisconnected('dropped')), loads_for_ever)
        with (
            patch('positronic.offboard.client.time.monotonic', side_effect=lambda: clock[0]),
            patch('positronic.offboard.client.time.sleep', side_effect=lambda s: clock.__setitem__(0, clock[0] + s)),
        ):
            session = InferenceClient(fake, _ADDRESS, reconnect_deadline=20.0).new_session()
            started = clock[0]
            with pytest.raises(TimeoutError):
                session.infer({'image': 'test'})

        assert clock[0] - started <= 20.0


class TestWaitForServer:
    """Waiting for a server opens no session."""

    class _ProbedWire(_FakeWire):
        def __init__(self, *answers: wire.Refusal | None):
            super().__init__()
            self._answers = list(answers)
            self.probes = 0

        def probe(self, address, headers, open_timeout):
            self.probes += 1
            return self._answers.pop(0)

    def test_a_cold_server_is_probed_until_it_answers(self):
        fake = self._ProbedWire(wire.Refusal.COLD, wire.Refusal.SILENT, None)
        with patch('positronic.offboard.client.time.sleep'):
            InferenceClient(fake, _ADDRESS).wait_for_server()

        assert fake.probes == 3
        assert fake.dials == []

    def test_a_final_refusal_is_raised_at_once(self):
        fake = self._ProbedWire(wire.Refusal.FINAL)
        with pytest.raises(wire.ConnectRefused) as refused:
            InferenceClient(fake, _ADDRESS).wait_for_server()

        assert refused.value.refusal is wire.Refusal.FINAL
        assert fake.probes == 1


def test_a_frame_miss_answer_raises_frame_miss_with_the_rows_the_server_does_not_have():
    missing = [['image', 2], ['image', 1]]
    conn = _connection(
        _ready('s', version=protocol.ProtocolVersion.V1), legacy_protocol.serialise({FRAME_MISS: missing})
    )
    with pytest.raises(FrameMiss) as raised:
        InferenceSession(conn).infer({'image': 'test'})
    assert raised.value.missing == missing


def test_a_message_without_an_answer_goes_out_alone_and_reads_nothing():
    conn = _connection(_ready('s', version=protocol.ProtocolVersion.V1))
    InferenceSession(conn).send_message({'row': 1})
    assert [legacy_protocol.deserialise(call.args[0]) for call in conn.send.call_args_list] == [{'row': 1}]
    assert conn.recv.call_count == 1  # the handshake


def test_a_message_without_an_answer_goes_out_without_the_v2_envelope():
    conn = _connection(_ready('s'))
    InferenceSession(conn).send_message({'row': 1})
    assert [legacy_protocol.deserialise(call.args[0]) for call in conn.send.call_args_list] == [{'row': 1}]


def test_a_frame_miss_answer_raises_frame_miss_on_a_v2_session():
    missing = [['image', 3]]
    conn = _connection(_ready('s'), legacy_protocol.serialise({FRAME_MISS: missing}))
    with pytest.raises(FrameMiss) as raised:
        InferenceSession(conn).infer({'image': 'test'})
    assert raised.value.missing == missing


def test_a_message_without_an_answer_needs_a_protocol_v1_or_v2_session():
    conn = _connection(_ready('s', version=protocol.ProtocolVersion.V3))
    with pytest.raises(RuntimeError, match='protocol v1 or v2'):
        InferenceSession(conn).send_message({'row': 1})
    conn.send.assert_not_called()


def test_remote_policy_hands_the_wire_the_server_and_the_headers_to_the_client():
    headers = {'Modal-Key': 'k'}
    policy = RemotePolicy('websocket_tls', _address('example.com', 443, query='fps=2.5'), headers=headers)
    client = policy._client
    assert client.session_url == 'wss://example.com/api/v1/session?fps=2.5'
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
            local if local is not None else Sequential(PauseOnUnavailable(), ChunkedSchedule(fps=10, horizon_sec=0.2)),
            codec=codec,
        )
        server = start_server(model, pipeline, grpc=transport == 'grpc')
        address = server.ws()[1] if transport == 'websocket' else server.grpc()[1]
        return address, model, pipeline

    return start


@pytest.mark.timeout(120)
@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
def test_background_console_sends_remote_policy_rollouts_and_receives_their_results(served, make_mock_model, transport):
    model = make_mock_model([{}], {'model_name': 'console-policy'})
    address, _, _ = served(model=model, transport=transport)
    policy = RemotePolicy(transport, address)
    statuses = []

    def script(base, clock):
        yield from wait(lambda: read_status(base), 'the console to answer', clock, seconds=60)
        for verdict in (Outcome.PASS, Outcome.FAIL):
            calls = model.call_count
            assert post(base, '/episode/start') == 200
            yield from wait(lambda calls=calls: model.call_count > calls, 'a model call', clock)
            assert post(base, '/episode/end', EndBody(verdict=verdict)) == 200

            def finished():
                status = read_status(base)
                return status is not None and status.run.phase is Phase.READY

            yield from wait(finished, 'the episode to finish', clock)
        statuses.append(read_status(base))

    run_attended(policy, script)

    [status] = statuses
    assert status is not None
    assert [episode.outcome for episode in status.run.episodes] == [Outcome.PASS, Outcome.FAIL]
    assert model.end_session.call_count == 2


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
@pytest.mark.parametrize('resize_first', [False, True])
@pytest.mark.parametrize('lead_sec', [0.0, 0.05])
def test_remote_chunk_cadence_and_fresh_episode_state(served, transport, resize_first, lead_sec):
    resize = TransformImages(8, 8, mode=ResizeMode.BOUND)
    schedule = ChunkedSchedule(fps=10, horizon_sec=0.2, inference_lead=ConstantDuration(lead_sec))
    local = (
        Sequential(resize, PauseOnUnavailable(), schedule)
        if resize_first
        else Sequential(PauseOnUnavailable(), schedule, resize)
    )
    address, model, pipeline = served(local=local, transport=transport)
    policy = RemotePolicy(transport, address)
    assert policy.meta() == {policy_keys.TYPE: 'remote'}
    assert model.ended_sessions == []
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
            now[0] = round((0.2 - lead_sec) * 1e9)
            first = run.send(obs)
            assert isinstance(first, Step)
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            step = run.send(obs)
            assert isinstance(step, Step)
            if lead_sec:
                assert first.commands == step.commands == {}
                assert step.resume_at_ns == 200_000_000
                now[0] = 200_000_000
                step = run.send(obs)
                assert isinstance(step, Step) and step.commands == {'value': 0}
            else:
                assert [commands for commands in (first.commands, step.commands) if commands] == [{'value': 0}]
            assert len(model.observations) == 2 * episode + 2
        finally:
            runtime.close()
            run.close()
        assert len(model.ended_sessions) == episode + 1
        assert model.session_ids[-2:] == [model.ended_sessions[-1]] * 2
        episode_meta = flatten_dict(runtime.metadata_snapshot())
        assert episode_meta['server.model_name'] == 'fixed'
        assert episode_meta['server.action_fps'] == 10
        assert episode_meta['server.action_horizon_sec'] == 0.2
    assert len(set(model.ended_sessions)) == 2
    assert from_spec(pipeline.local.to_spec()).to_spec() == pipeline.local.to_spec()


class DeclaredPromptModel(FixedModel):
    def meta(self):
        return {'model_name': 'fixed', 'prompt': 'A prompt this deployment declares.'}


def test_a_declared_prompt_is_in_episode_meta_and_the_task_reaches_the_model_unchanged(served):
    """A ``prompt`` in the handshake appears unchanged in the episode metadata, under the server block. The task an
    episode sends reaches the model unchanged, with no prompt beside it."""
    address, model, _ = served(model=DeclaredPromptModel())
    policy = RemotePolicy('websocket', address)

    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    run = runtime.start(policy)
    try:
        run.send({keys.TASK: 'the task this episode sends'})
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
    finally:
        runtime.close()
        run.close()
    server_meta = runtime.metadata_snapshot()[policy_keys.SERVER]
    assert server_meta['prompt'] == 'A prompt this deployment declares.'
    assert model.observations[-1][keys.TASK] == 'the task this episode sends'
    assert 'prompt' not in model.observations[-1]


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
@pytest.mark.parametrize('payload', [{protocol.OBSERVATION: {}}, {protocol.END_SESSION: True}], ids=['infer', 'end'])
def test_wrong_session_id_closes_only_the_requesting_session(served, transport, payload, monkeypatch):
    # The websocket server ends a refused session when its client does not close it within the close timeout.
    monkeypatch.setattr(websocket_wire.WebsocketWire, 'CLOSE_TIMEOUT_SEC', 0.5)
    address, model, _ = served(transport=transport)
    client = InferenceClient(registry.client_wire(transport), address)
    first, second = client.new_session(), client.new_session()
    try:
        assert first.session_id != second.session_id
        assert protocol.SESSION_ID not in first.metadata
        # The server answers this frame by ending the call, which can come before gRPC confirms the write.
        with suppress(wire.PeerDisconnected):
            first._conn.send(legacy_protocol.serialise({protocol.SESSION_ID: second.session_id, **payload}))
        response = legacy_protocol.deserialise(first._conn.recv(timeout=5))
        # A drop now must not reconnect: the server answered on the session's connection, past ``infer``.
        first._answered = True
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
        legacy_protocol.serialise({
            protocol.STATUS: protocol.ServerStatus.READY,
            protocol.META: {},
            protocol.SESSION_ID: 's',
            protocol.PROTOCOL_VERSION: 2,
        }),
        legacy_protocol.serialise({
            protocol.STATUS: protocol.ServerStatus.ERROR,
            protocol.ERROR: 'session ID mismatch',
        }),
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
        legacy_protocol.serialise({
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
        (legacy_protocol.serialise({protocol.SESSION_ID: 'wrong', protocol.END_SESSION: True}), RuntimeError),
        (legacy_protocol.serialise({protocol.ERROR: 'cleanup failed'}), RuntimeError),
    ],
    ids=['no-ack', 'wrong-session', 'cleanup-error'],
)
def test_close_still_requires_a_valid_ack_when_the_final_write_reports_disconnect(response, error):
    conn = MagicMock(spec=wire.ClientConnection)
    conn.recv.side_effect = [
        legacy_protocol.serialise({
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
        assert legacy_protocol.serialise(actual) == legacy_protocol.serialise(reference)
    assert training_codec.training_encoder.meta[ACTION_FPS] == 15
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
        {'local': Sequential(PauseOnUnavailable(), ChunkedSchedule(fps=10), codec)},
        {'local': Sequential(PauseOnUnavailable(), codec, ChunkedSchedule(fps=10))},
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
            outputs.append(legacy_protocol.serialise(dict(first.commands) | dict(completed.commands)))
            inputs.append(legacy_protocol.serialise(model.observations[0]))
        finally:
            runtime.close()
            run.close()
    assert inputs[0] == inputs[1] == inputs[2]
    assert outputs[0] == outputs[1] == outputs[2]


def test_pipeline_rejects_frame_conversion_on_both_sides():
    local = Sequential(ChangeEEFrame(EEFrame(site=GRIPPER_SITE)), ChunkedSchedule(fps=10))
    with pytest.raises(ValueError, match='Only one side'):
        PolicyDeployment(local, codec=ChangeEEFrame(EEFrame(site=GRIPPER_SITE)))


def test_pipeline_takes_frame_conversions_of_different_signals_on_each_side():
    local = Sequential(ChangeEEFrame(EEFrame(site=GRIPPER_SITE), keys=(keys.EE_POSE,)), ChunkedSchedule(fps=10))
    PolicyDeployment(local, codec=ChangeEEFrame(EEFrame(site=GRIPPER_SITE), keys=(keys.TARGET_EE_POSE,)))


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


def test_a_failed_session_close_reaches_the_caller(runtime):
    failure = RuntimeError('Unexpected end-session response')
    policy, session = _mock_remote_policy(CHUNKED_STACK)
    session.close.side_effect = failure
    run = runtime.start(policy)
    with pytest.raises(RuntimeError) as raised:
        run.close()
    assert raised.value is failure


def test_a_session_close_past_the_bound_lets_the_run_end_and_logs_a_late_failure(runtime, monkeypatch, caplog):
    monkeypatch.setattr('positronic.policy.remote._CLOSE_TIMEOUT_S', 0.05)
    release = threading.Event()
    late_failure = TimeoutError('No end-session acknowledgement')
    policy, session = _mock_remote_policy(CHUNKED_STACK)

    def close():
        assert release.wait(5), 'the close was not released'
        raise late_failure

    session.close.side_effect = close
    run = runtime.start(policy)
    threads_before = set(threading.enumerate())
    with caplog.at_level(logging.WARNING, logger='positronic.policy.remote'):
        run.close()
        assert 'did not answer the session close' in caplog.text
        release.set()
        for closer in set(threading.enumerate()) - threads_before:
            closer.join(5)
    assert 'The session close failed after the run continued' in caplog.text
    assert str(late_failure) in caplog.text


@pytest.mark.parametrize('compressed', [False, True])
def test_compression_follows_the_handshake(runtime, compressed):
    policy, session = _mock_remote_policy(
        {**CHUNKED_STACK, policy_keys.COMPRESS_IMAGES: compressed}, infer_return=[{'value': 42}]
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


def test_the_configured_jpeg_quality_reaches_the_encoder(runtime, monkeypatch):
    qualities = []
    monkeypatch.setattr(
        'positronic.policy.remote.encode_jpeg', lambda image, quality: qualities.append(quality) or {'jpeg': b''}
    )
    session = _mock_session({**CHUNKED_STACK, policy_keys.COMPRESS_IMAGES: True})
    session.infer.return_value = [{'value': 42}]
    policy = RemotePolicy('websocket', _address('localhost', 0), jpeg_quality=75)
    policy._client = MagicMock()
    policy._client.new_session.return_value = session
    run = runtime.start(policy)
    try:
        run.send({'image': _make_image(48, 64)})
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
    finally:
        runtime.close()
        run.close()
    assert qualities == [75]


@pytest.mark.parametrize('compressed', [False, True])
def test_a_run_records_the_jpeg_quality_only_when_images_are_compressed(runtime, compressed):
    policy = RemotePolicy('websocket', _address('localhost', 0), jpeg_quality=75)
    policy._client = MagicMock()
    policy._client.new_session.return_value = _mock_session({**CHUNKED_STACK, policy_keys.COMPRESS_IMAGES: compressed})
    runtime.start(policy).close()
    meta = runtime.metadata_snapshot()
    if compressed:
        assert meta[policy_keys.JPEG_QUALITY] == 75
    else:
        assert policy_keys.JPEG_QUALITY not in meta


def test_a_streamed_run_records_the_jpeg_quality_its_requests_carry(runtime):
    meta = _stream_meta({'name': 'encode_images', 'args': {'quality': 75}})
    policy, session = _mock_remote_policy(meta, frame_stream='jpeg')
    session.protocol_version = protocol.ProtocolVersion.V1
    runtime.start(policy).close()
    assert runtime.metadata_snapshot()[policy_keys.JPEG_QUALITY] == 75


def _frames_meta(meta):
    return {name: frame.site for name, frame in parse_ee_frames(meta[roboarm_keys.EE_FRAMES]).items()}


def _recorded_frames(policy, runtime):
    """The site of each pose signal that one run of ``policy`` records as it starts."""
    runtime.start(policy).close()
    return _frames_meta(runtime.metadata_snapshot())


def test_a_run_records_the_frames_the_server_codec_states(runtime):
    server_frames = ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)})
    policy, session = _mock_remote_policy({**CHUNKED_STACK, **server_frames})
    session.protocol_version = protocol.ProtocolVersion.V3
    assert _recorded_frames(policy, runtime) == {keys.EE_POSE: GRIPPER_SITE}


def test_a_run_records_the_frames_both_sides_of_a_split_deployment_state(served, runtime):
    local = Sequential(
        ChangeEEFrame(EEFrame(site=GRIPPER_SITE), keys=(keys.EE_POSE,)),
        PauseOnUnavailable(),
        ChunkedSchedule(fps=10, horizon_sec=0.2),
    )
    codec = ChangeEEFrame(EEFrame(site=GRIPPER_SITE), keys=(keys.TARGET_EE_POSE,))
    address, _, _ = served(local=local, codec=codec)
    frames = _recorded_frames(RemotePolicy('websocket', address), runtime)
    assert frames == {keys.EE_POSE: GRIPPER_SITE, keys.TARGET_EE_POSE: GRIPPER_SITE}


def test_a_server_built_for_the_franka_hand_frame_states_its_poses_at_the_franka_hand_point(served, runtime):
    local = Sequential(
        ChangeEEFrame(cfg_codecs.franka_hand_frame.instantiate()),
        PauseOnUnavailable(),
        ChunkedSchedule(fps=10, horizon_sec=0.2),
    )
    address, _, _ = served(local=local)
    runtime.start(RemotePolicy('websocket', address)).close()
    frame = parse_ee_frames(runtime.metadata_snapshot()[roboarm_keys.EE_FRAMES])[keys.EE_POSE]
    assert (frame.site, frame.offset) == (GRIPPER_SITE, FRANKA_HAND_FRAME)


def test_a_run_records_no_frames_for_a_server_that_states_none(runtime):
    policy, session = _mock_remote_policy(CHUNKED_STACK)
    session.protocol_version = protocol.ProtocolVersion.V3
    runtime.start(policy).close()
    assert roboarm_keys.EE_FRAMES not in runtime.metadata_snapshot()


def test_a_frame_codec_around_a_server_that_states_none_states_the_frames_for_it():
    """A server that knows nothing of ``ee_frames`` reads poses at a frame the client side states for it."""
    policy, session = _mock_remote_policy(CHUNKED_STACK)
    session.protocol_version = protocol.ProtocolVersion.V3
    framed = Sequential(ChangeEEFrame(EEFrame(site=GRIPPER_SITE)), policy)
    assert _frames_meta(framed.meta())[keys.EE_POSE] == GRIPPER_SITE


@pytest.mark.parametrize('side', ['client_stack', 'server_meta'])
def test_a_remote_server_that_states_its_frames_gets_the_pose_with_no_fallback_on_one_session(caplog, side):
    if side == 'client_stack':
        stack = Sequential(ChangeEEFrame(EEFrame(site=GRIPPER_SITE), keys=(keys.EE_POSE,)), ChunkedSchedule(fps=10))
        metadata = {offboard_keys.LOCAL_STACK: stack.to_spec()}
    else:
        metadata = {**CHUNKED_STACK, **ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)})}
    policy, session = _mock_remote_policy(metadata, infer_return=[{keys.TARGET_GRIP: 0.5}])
    session.protocol_version = protocol.ProtocolVersion.V3

    assert run_robot_rollout(policy, ticks=5).result()[eval_keys.TERMINATED] is True
    np.testing.assert_allclose(session.infer.call_args_list[0].args[0][keys.EE_POSE][:3], [0.1, 0.2, 0.3])
    assert legacy_point_warnings(caplog) == []
    client = policy._client
    assert isinstance(client, MagicMock)
    client.new_session.assert_called_once()


def test_a_remote_server_that_states_another_site_is_refused_before_it_gets_an_observation():
    policy, session = _mock_remote_policy({**CHUNKED_STACK, **ee_frames_entry({keys.EE_POSE: EEFrame(site=EE_LINK)})})
    session.protocol_version = protocol.ProtocolVersion.V3

    with pytest.raises(ValueError, match=EE_LINK):
        run_robot_rollout(policy).result()
    session.infer.assert_not_called()
    session.close.assert_called_once()


def test_a_remote_server_that_states_two_frames_for_one_pose_is_refused_before_it_gets_an_observation():
    offset = EEFrame(site=GRIPPER_SITE, offset=Transform3D(np.array([0.0, 0.0, 0.1])))
    stack = Sequential(ChangeEEFrame(offset, keys=(keys.EE_POSE,)), ChunkedSchedule(fps=10))
    metadata = {
        offboard_keys.LOCAL_STACK: stack.to_spec(),
        **ee_frames_entry({keys.EE_POSE: EEFrame(site=GRIPPER_SITE)}),
    }
    policy, session = _mock_remote_policy(metadata)
    session.protocol_version = protocol.ProtocolVersion.V3

    with pytest.raises(ValueError, match=re.escape(keys.EE_POSE)):
        run_robot_rollout(policy).result()
    session.infer.assert_not_called()
    session.close.assert_called_once()


def test_a_remote_server_that_repeats_its_client_stack_frame_in_its_metadata_runs(runtime):
    frame = cfg_codecs.droid_frame.instantiate()
    stack = Sequential(ChangeEEFrame(frame, keys=(keys.EE_POSE,)), ChunkedSchedule(fps=10))
    policy, session = _mock_remote_policy({
        offboard_keys.LOCAL_STACK: stack.to_spec(),
        **ee_frames_entry({keys.EE_POSE: frame}),
    })
    session.protocol_version = protocol.ProtocolVersion.V3
    assert _recorded_frames(policy, runtime) == {keys.EE_POSE: GRIPPER_SITE}


# The handshake of a protocol v1 server that states no frames: an unversioned client stack, and no ``ee_frames``.
FRAMELESS_V1_HANDSHAKE = {
    offboard_keys.LOCAL_STACK: {
        'seq': [
            {'name': 'stop_on_fault'},
            {
                'name': 'temporal_stack',
                'args': {
                    'keys': [keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE, keys.EE_POSE, keys.GRIP],
                    'offsets_sec': [(step - 24) / 15 for step in range(25)],
                    'pad_start': False,
                },
            },
            {'name': 'chunked_schedule'},
            {'name': 'restrict_image_size', 'args': {'width': 1024, 'height': 288}},
        ]
    }
}


def test_a_v1_server_that_states_no_frames_observes_and_commands_the_fr3_at_its_old_end_effector(caplog):
    model = bundled_franka_model()
    fr3 = _prepare_spec(model[roboarm_keys.URDF], GRIPPER_SITE, EE_LINK).compile()
    joints = np.array([0.1, -0.4, 0.2, -2.0, 0.1, 1.8, 0.6])
    data = mj.MjData(fr3)
    data.qpos[[fr3.joint(name).qposadr.item() for name in model[roboarm_keys.JOINT_NAMES]]] = joints
    mj.mj_forward(fr3, data)
    gripper, end_effector = (
        Transform3D(data.site(site).xpos.copy(), Rotation.from_rotation_matrix(data.site(site).xmat.reshape(3, 3)))
        for site in (GRIPPER_SITE, EE_LINK)
    )
    np.testing.assert_allclose((gripper * FRANKA_HAND_FRAME).as_matrix, end_effector.as_matrix, atol=1e-9)
    target = CartesianPosition(pose=Transform3D(translation=np.array([0.4, 0.5, 0.6]), rotation=Rotation.identity))
    rows = [{keys.ROBOT_COMMAND: to_wire(target), keys.TARGET_GRIP: 0.5, 'timestamp': step / 15} for step in range(4)]
    replies = iter([_ready('v1', FRAMELESS_V1_HANDSHAKE, version=protocol.ProtocolVersion.V1)])
    conn = MagicMock(spec=wire.ClientConnection)
    conn.recv.side_effect = lambda timeout=None: next(replies, _answer(rows))
    policy = RemotePolicy('websocket', _ADDRESS)
    policy._client = InferenceClient(_FakeWire(conn), _ADDRESS)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    sensors = {keys.WRIST_IMAGE: image, keys.EXTERIOR_IMAGE: image, keys.GRIP: np.float32(0.0)}
    state = FakeRobotState(joints, np.zeros_like(joints), gripper, RobotStatus.AVAILABLE)
    commands = []

    answer = run_robot_rollout(policy, robot_meta=model, ticks=50, commands=commands, sensors=sensors, state=state)

    assert answer.result()[eval_keys.TERMINATED] is True
    (observed,) = legacy_protocol.deserialise(conn.send.call_args_list[0].args[0])[keys.EE_POSE]
    np.testing.assert_allclose(observed, Serializers.transform_3d(gripper * FRANKA_HAND_FRAME), atol=1e-9)
    assert len(legacy_point_warnings(caplog)) == 1
    assert commands and all(isinstance(command, CartesianPosition) for command in commands)
    np.testing.assert_allclose(commands[0].pose.as_matrix, (target.pose * FRANKA_HAND_FRAME.inv).as_matrix, atol=1e-9)


def test_policy_meta_holds_only_client_values_and_makes_no_call():
    policy = RemotePolicy('websocket', _address('localhost', 0))
    client = MagicMock()
    policy._client = client

    assert policy.meta() == {policy_keys.TYPE: 'remote'}
    assert client.method_calls == []


def test_waiting_for_the_server_opens_no_session():
    policy = RemotePolicy('websocket', _address('localhost', 0))
    client = MagicMock()
    policy._client = client

    policy.wait_for_server()
    client.wait_for_server.assert_called_once()
    client.new_session.assert_not_called()


def test_a_run_with_an_invalid_v3_stack_records_no_metadata(runtime):
    policy, session = _mock_remote_policy({offboard_keys.LOCAL_STACK: spec.component('unknown_processor')})
    session.protocol_version = protocol.ProtocolVersion.V3

    with pytest.raises(ValueError, match='Unknown local-stack entry'):
        runtime.start(policy)
    assert runtime.metadata_snapshot() == {}
    assert session.close.call_count == 1

    session.metadata = CHUNKED_STACK
    runtime.start(policy).close()
    assert runtime.metadata_snapshot()[ACTION_FPS] == 10
    assert session.close.call_count == 2


def _stream_meta(*parts: dict, **changes) -> dict:
    """A protocol v1 handshake that stacks ``image`` at 0.1 s ticks and declares a frame stream."""
    temporal = {'name': 'temporal_stack', 'args': {'keys': ['image'], 'offsets_sec': [-0.25, -0.15, 0.0]}}
    meta = {
        offboard_keys.LOCAL_STACK: {'seq': [temporal, *parts]},
        ACTION_FPS: 10.0,
        policy_keys.COMPRESS_IMAGES: True,
        'frame_stream': {'version': 1, 'codecs': ['jpeg', 'h264']},
    }
    return {**meta, **changes}


def _stream_frame(index: int) -> np.ndarray:
    return np.full((8, 12, 3), 10 * index, dtype=np.uint8)


def _frame_index(image: np.ndarray) -> int:
    return round(float(image.mean()) / 10)


def _run_on_mock_session(
    meta: dict,
    ticks: int,
    *,
    frame_stream: str | None,
    replies: list | None = None,
    version: protocol.ProtocolVersion = protocol.ProtocolVersion.V1,
    extra: Mapping[int, dict] | None = None,
    frame=_stream_frame,
    **policy_kwargs,
) -> MagicMock:
    """Run a ``RemotePolicy`` on a mocked session, one tick every 0.1 s, and return the session.

    Each reply is a chunk or an exception; without ``replies`` every call answers a chunk of 5 rows. ``version``
    is the session's protocol version, ``extra`` adds fields to the observation of a tick index, and ``frame``
    gives the image of each tick index.
    """
    policy, session = _mock_remote_policy(meta, frame_stream=frame_stream, **policy_kwargs)
    session.protocol_version = version
    pending = list(replies or [])
    tick_ended = threading.Event()

    def infer(obs):
        # The answer lands after the tick that asked for it, so the schedule reads it at the next tick.
        assert tick_ended.wait(5)
        reply = pending.pop(0) if pending else [{'value': i} for i in range(5)]
        if isinstance(reply, Exception):
            raise reply
        return reply

    session.infer.side_effect = infer
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(policy)
    try:
        for index in range(ticks):
            now[0] = index * 100_000_000
            tick_ended.clear()
            run.send({'image': frame(index), **(extra or {}).get(index, {})})
            tick_ended.set()
            runtime.wait(timeout_sec=5)
    finally:
        tick_ended.set()
        runtime.close()
        run.close()
    return session


def test_an_unknown_frame_stream_codec_is_refused():
    with pytest.raises(ValueError, match='vp9'):
        RemotePolicy('websocket', _address('localhost', 0), frame_stream='vp9')


def test_a_negative_frame_stream_snap_is_refused():
    with pytest.raises(ValueError, match='frame_stream_snap_sec'):
        RemotePolicy('websocket', _address('localhost', 0), frame_stream='jpeg', frame_stream_snap_sec=-0.01)


@pytest.mark.parametrize(
    'parts, ticks, streamed, stacked',
    [
        # Read at 0.1 s, 5 rows at 10 per second: the next request goes at 0.6 s, its rows at 0.35 s and 0.45 s.
        pytest.param((), 8, [3, 4], [3, 4, 6], id='the-handshake-rate'),
        # The declared rate wins over the handshake's: the next request goes at 1.1 s.
        pytest.param(({'name': 'action_timestamp', 'args': {'fps': 5}},), 13, [8, 9], [8, 9, 11], id='a-declared-rate'),
    ],
)
def test_a_frame_stream_sends_the_rows_of_the_next_request_and_the_request_refers_to_them(
    parts, ticks, streamed, stacked
):
    session = _run_on_mock_session(_stream_meta(*parts), ticks, frame_stream='jpeg')

    rows = [call.args[0][ROW] for call in session.send_message.call_args_list]
    assert [(row['call'], row['back']) for row in rows] == [(1, 2), (1, 1)]
    images = [serialization.deserialise(serialization.serialise(row['images']))['image'] for row in rows]
    assert [_frame_index(image) for image in images] == streamed
    first, second = (call.args[0]['image'] for call in session.infer.call_args_list)
    assert REF not in first
    assert [_frame_index(frame) for frame in serialization.deserialise(serialization.serialise(first))] == [0, 0, 0]
    assert (second[REF], second[b'call'], second[b'rows']) == (True, 1, [2, 1])
    (tail,) = serialization.deserialise(serialization.serialise(second[b'tail']))
    assert _frame_index(tail) == stacked[-1]


def test_a_declared_encode_images_leaves_the_stack_for_the_frame_stream_to_refer_to(monkeypatch):
    qualities = []
    for module in (frame_stream_module, remote_module):
        encode = module.encode_jpeg
        monkeypatch.setattr(
            module, 'encode_jpeg', lambda image, q=90, *a, _e=encode: qualities.append(q) or _e(image, q)
        )
    session = _run_on_mock_session(
        _stream_meta({'name': 'encode_images', 'args': {'quality': 75}}), 8, frame_stream='jpeg'
    )

    rows = [call.args[0][ROW] for call in session.send_message.call_args_list]
    assert [(row['call'], row['back']) for row in rows] == [(1, 2), (1, 1)]
    first, second = (call.args[0]['image'] for call in session.infer.call_args_list)
    assert REF not in first
    # The request compares its stack with the streamed rows before any frame is encoded, so it refers to them.
    assert (second[REF], second[b'call'], second[b'rows']) == (True, 1, [2, 1])
    # The rows and the frames each request carries go at the declared quality.
    assert qualities and set(qualities) == {75}


def test_a_protocol_v2_session_streams_the_rows_as_a_v1_session_does():
    v1 = _run_on_mock_session(_stream_meta(), 8, frame_stream='jpeg')
    temporal = {**_stream_meta()[offboard_keys.LOCAL_STACK]['seq'][0], 'version': 2}
    # A protocol v2 stack names its versions and declares its schedule.
    meta = _stream_meta(ChunkedSchedule(fps=10.0).to_spec())
    meta[offboard_keys.LOCAL_STACK]['seq'][0] = temporal
    v2 = _run_on_mock_session(meta, 8, frame_stream='jpeg', version=protocol.ProtocolVersion.V2)

    def rows(session: MagicMock) -> list[dict]:
        return [
            {k: v for k, v in call.args[0][ROW].items() if k != 't_ns'} for call in session.send_message.call_args_list
        ]

    assert [(row['call'], row['back']) for row in rows(v2)] == [(1, 2), (1, 1)]
    assert rows(v2) == rows(v1)
    second = v2.infer.call_args_list[1].args[0]['image']
    assert (second[REF], second[b'call'], second[b'rows']) == (True, 1, [2, 1])


def test_a_frame_stream_targets_the_request_that_the_inference_lead_brings_forward():
    temporal = {**_stream_meta()[offboard_keys.LOCAL_STACK]['seq'][0], 'version': 2}
    meta = _stream_meta(ChunkedSchedule(fps=10.0, inference_lead=ConstantDuration(0.2)).to_spec())
    meta[offboard_keys.LOCAL_STACK]['seq'][0] = temporal
    session = _run_on_mock_session(meta, 6, frame_stream='jpeg', version=protocol.ProtocolVersion.V2)

    # Read at 0.1 s, 5 rows at 10 per second, a lead of 0.2 s: the next request goes at 0.4 s, its rows at 0.15 s
    # and 0.25 s.
    rows = [call.args[0][ROW] for call in session.send_message.call_args_list]
    images = [serialization.deserialise(serialization.serialise(row['images']))['image'] for row in rows]
    assert [(row['call'], row['back'], _frame_index(image)) for row, image in zip(rows, images, strict=True)] == [
        (1, 2, 1),
        (1, 1, 2),
    ]
    second = session.infer.call_args_list[1].args[0]['image']
    assert (second[b'call'], second[b'rows']) == (1, [2, 1])


def test_a_replan_after_an_arm_error_streams_the_rows_of_the_call_that_the_schedule_plans_next():
    temporal = {**_stream_meta()[offboard_keys.LOCAL_STACK]['seq'][0], 'version': 2}
    meta = _stream_meta(ChunkedSchedule(fps=10.0, inference_lead=ConstantDuration(0.3)).to_spec())
    meta[offboard_keys.LOCAL_STACK]['seq'][0] = temporal
    error = {5: {keys.ROBOT_STATUS: RobotStatus.ERROR.value}}
    session = _run_on_mock_session(meta, 9, frame_stream='jpeg', version=protocol.ProtocolVersion.V2, extra=error)

    # Calls at 0.0 s and 0.3 s; the answer read at 0.4 s plans its chunk from 0.6 s to 1.1 s. The error at 0.5 s
    # clears that plan and calls at once. Its answer, read at 0.6 s, starts its chunk then, so the next call goes at
    # 0.8 s with the rows of 0.55 s and 0.65 s: ticks 5 and 6.
    rows = [call.args[0][ROW] for call in session.send_message.call_args_list]
    images = [serialization.deserialise(serialization.serialise(row['images']))['image'] for row in rows]
    assert [(row['call'], row['back'], _frame_index(image)) for row, image in zip(rows, images, strict=True)][-2:] == [
        (3, 2, 5),
        (3, 1, 6),
    ]
    fourth = session.infer.call_args_list[3].args[0]['image']
    assert (fourth[b'call'], fourth[b'rows']) == (3, [2, 1])


def test_a_frame_miss_sends_the_request_again_in_full(caplog):
    chunk = [{'value': i} for i in range(5)]
    with caplog.at_level(logging.WARNING, logger='positronic.policy.remote'):
        session = _run_on_mock_session(
            _stream_meta(), 7, frame_stream='jpeg', replies=[chunk, FrameMiss([['image', 2]]), chunk]
        )

    requests = [call.args[0]['image'] for call in session.infer.call_args_list]
    assert len(requests) == 3
    assert REF in requests[1]
    full = serialization.deserialise(serialization.serialise(requests[2]))
    assert [_frame_index(frame) for frame in full] == [3, 4, 6]
    assert 'does not have the streamed rows' in caplog.text


def _history_stack_spec() -> dict:
    """A temporal stack of ``image`` whose offsets stop before the current frame."""
    return {'name': 'temporal_stack', 'args': {'keys': ['image'], 'offsets_sec': [-0.25, -0.15]}}


@pytest.mark.parametrize(
    'frame_stream, meta',
    [
        pytest.param(None, _stream_meta(), id='the-policy-does-not-ask'),
        pytest.param('jpeg', {**_stream_meta(), 'frame_stream': None}, id='the-server-declares-none'),
        pytest.param(
            'h264', _stream_meta(frame_stream={'version': 1, 'codecs': ['jpeg']}), id='the-server-declares-other-codecs'
        ),
        pytest.param('jpeg', _stream_meta(**{policy_keys.COMPRESS_IMAGES: False}), id='the-server-takes-raw-images'),
        pytest.param(
            'jpeg',
            _stream_meta(**{offboard_keys.LOCAL_STACK: {'seq': [{'name': 'stop_on_fault'}]}}),
            id='the-stack-holds-no-temporal-stack',
        ),
        pytest.param(
            'jpeg',
            _stream_meta(**{offboard_keys.LOCAL_STACK: {'seq': [_history_stack_spec()]}}),
            id='the-stack-holds-no-current-frame',
        ),
    ],
)
def test_requests_go_in_full_unless_the_policy_asks_for_a_frame_stream_the_session_offers(frame_stream, meta, caplog):
    with caplog.at_level(logging.WARNING, logger='positronic.policy.remote'):
        session = _run_on_mock_session(meta, 8, frame_stream=frame_stream)

    session.send_message.assert_not_called()
    requests = [call.args[0]['image'] for call in session.infer.call_args_list]
    assert len(requests) == 2
    assert all(not isinstance(image, dict) or REF not in image for image in requests)
    assert ('frame stream' in caplog.text) == (frame_stream is not None)


@pytest.mark.parametrize('frame_stream, cached', [(None, False), ('jpeg', True)], ids=['no-stream', 'a-stream'])
@pytest.mark.parametrize(
    'resize',
    [
        pytest.param({'name': 'restrict_image_size', 'args': {'width': 6, 'height': 4}}, id='restrict_image_size'),
        pytest.param(
            TransformImages(6, 4, mode=ResizeMode.BOUND, paths=[['image']]).to_spec(), id='transform_images-with-paths'
        ),
    ],
)
def test_only_a_session_that_streams_frames_caches_its_resizes(monkeypatch, frame_stream, cached, resize):
    sizes = []
    put = ResizeCache.put

    def put_and_count(cache, *args):
        put(cache, *args)
        sizes.append(len(cache._entries))

    monkeypatch.setattr(ResizeCache, 'put', put_and_count)
    session = _run_on_mock_session(_stream_meta(resize), 8, frame_stream=frame_stream)

    first = serialization.deserialise(serialization.serialise(session.infer.call_args_list[0].args[0]['image']))
    assert first.shape == (3, 4, 6, 3)  # the declared stack resizes either way
    assert bool(sizes) is cached
    if frame_stream is not None:
        second = session.infer.call_args_list[1].args[0]['image']
        assert second[b'rows'] == [2, 1]  # the rows went out sized as the stack sizes its frames


def _streamed_stack(*parts: Policy | Codec) -> Sequential:
    """A client stack that stacks ``image`` at 0.1 s ticks for a frame stream."""
    return Sequential(TemporalStack(('image',), (-0.25, -0.15, 0.0)), *parts)


@pytest.mark.parametrize(
    'local, compress_images, declared',
    [
        pytest.param(_streamed_stack(ChunkedSchedule(fps=10)), True, True, id='a-temporal-stack'),
        pytest.param(_streamed_stack(ChunkedSchedule(fps=10)), False, False, id='raw-images'),
        pytest.param(ChunkedSchedule(fps=10), True, False, id='no-temporal-stack'),
        pytest.param(
            Sequential(TemporalStack(('image',), (-0.25, -0.15)), ChunkedSchedule(fps=10)),
            True,
            False,
            id='no-current-frame',
        ),
        pytest.param(
            Sequential(
                TemporalStack(('image',), tuple(-0.01 * i for i in range(StreamedRows.MAX_BACK + 1, -1, -1))),
                ChunkedSchedule(fps=10),
            ),
            True,
            False,
            id='more-rows-than-a-stream-carries',
        ),
    ],
)
def test_a_server_declares_a_frame_stream_for_a_client_that_stacks_compressed_frames(
    start_server, local, compress_images, declared
):
    server = start_server(FixedModel(), PolicyDeployment(local, compress_images=compress_images))
    session = InferenceClient(*server.ws()).new_session()
    try:
        assert (FRAME_STREAM in session.metadata) is declared
        if declared:
            assert session.metadata[FRAME_STREAM] == {'version': 1, 'codecs': ['jpeg', 'h264']}
    finally:
        session.close()


def _gradient_frame(index: int) -> np.ndarray:
    """A smooth 32 x 48 frame that changes every tick, which JPEG and H.264 both keep close."""
    ramp = np.add.outer(np.arange(32), np.arange(48)).astype(np.uint8) * 2
    return np.repeat(ramp[..., None], 3, axis=2) + np.uint8(5 * index)


def _served_run(address: wire.HostPortAddress, frame_stream: str | None, ticks: int) -> None:
    """Run a ``RemotePolicy`` on a served session, one tick every 0.1 s; each answer lands before the next tick."""
    policy = RemotePolicy('websocket', address, frame_stream=frame_stream)
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(policy)
    try:
        for index in range(ticks):
            now[0] = index * 100_000_000
            run.send({'image': _gradient_frame(index)})
            runtime.wait(timeout_sec=5)
    finally:
        runtime.close()
        run.close()


@pytest.mark.timeout(120)
@pytest.mark.parametrize('frame_stream, atol', [('jpeg', 0), ('h264', 12)])
def test_a_served_frame_stream_gives_the_model_the_observations_of_full_requests(
    served, monkeypatch, frame_stream, atol
):
    resolved: list[list[int]] = []
    resolve, add = StreamedRows.resolve, StreamedRows.add

    def recording_resolve(rows, obs):
        resolved.append([len(value[b'rows']) for value in obs.values() if isinstance(value, Mapping) and REF in value])
        return resolve(rows, obs)

    def dropping_add(rows, row):
        add(rows, row)
        if row['call'] == 2:
            rows._rows.pop(row['back'], None)  # the server lost the rows of call 2: that request misses them

    monkeypatch.setattr(StreamedRows, 'resolve', recording_resolve)
    monkeypatch.setattr(StreamedRows, 'add', dropping_add)
    local = _streamed_stack(ChunkedSchedule(fps=10), TransformImages(24, 16, mode=ResizeMode.BOUND))
    observed = {}
    for stream in (None, frame_stream):
        resolved.clear()
        address, model, _ = served(local=local)
        _served_run(address, stream, 22)
        observed[stream] = [obs['image'] for obs in model.observations]

    full, streamed = observed[None], observed[frame_stream]
    assert len(full) == len(streamed) == 5
    for expected, got in zip(full, streamed, strict=True):
        assert got.shape == expected.shape == (3, 16, 24, 3)
        np.testing.assert_allclose(got, expected, atol=atol)
    # Calls 1, 3 and 4 refer to both rows; call 2 misses them and goes again in full.
    assert resolved == [[], [2], [2], [], [2], [2]]


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@pytest.mark.timeout(60)
def test_a_server_decodes_each_message_of_a_streamed_session_off_its_event_loop(served, monkeypatch):
    on_loop: list[bool] = []
    decode = server_module.deserialise

    def recording(message):
        on_loop.append(_has_running_loop())
        return decode(message)

    monkeypatch.setattr(server_module, 'deserialise', recording)
    address, model, _ = served(local=_streamed_stack(ChunkedSchedule(fps=10)))
    _served_run(address, 'jpeg', 8)
    assert len(model.observations) >= 2
    assert on_loop and not any(on_loop)


def _rtc_stream_meta(*parts: dict) -> dict:
    """A protocol v2 handshake with an ``rtc_schedule``: ``image`` at 0.1 s ticks, a call 0.4 s after the previous
    one, and a declared ``encode_images`` at quality 75."""
    temporal = TemporalStack(('image',), (-0.25, -0.15, 0.0), pad_start=False).to_spec()
    rtc = RTCSchedule(fps=10.0, call_after_sec=0.4, latency_estimate=DelayEstimate.max(last=5, max_sec=0.3))
    encode = EncodeImages(quality=75).to_spec()
    return {
        offboard_keys.LOCAL_STACK: {'seq': [temporal, rtc.to_spec(), *parts, encode]},
        ACTION_FPS: 10.0,
        policy_keys.COMPRESS_IMAGES: True,
        FRAME_STREAM: {'version': 1, 'codecs': ['jpeg', 'h264']},
    }


def _run_v2_rtc(meta: dict, ticks: int, frame=_stream_frame, **policy_kwargs) -> MagicMock:
    """Run a ``RemotePolicy`` with a frame stream on a mocked protocol v2 rtc session, one tick every 0.1 s.

    ``frame`` gives the image of each tick index."""
    policy, session = _mock_remote_policy(meta, frame_stream='jpeg', **policy_kwargs)
    session.protocol_version = protocol.ProtocolVersion.V2
    tick_ended = threading.Event()

    def infer(obs):
        assert tick_ended.wait(5)
        return [{'value': i} for i in range(12)]

    session.infer.side_effect = infer
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(policy)
    try:
        for index in range(ticks):
            now[0] = index * 100_000_000
            tick_ended.clear()
            run.send({'image': frame(index)})
            tick_ended.set()
            runtime.wait(timeout_sec=5)
    finally:
        tick_ended.set()
        runtime.close()
        run.close()
    return session


def test_an_rtc_frame_stream_targets_its_rows_from_the_call_time_plus_call_after(monkeypatch):
    qualities = []
    for module in (frame_stream_module, remote_module):
        encode = module.encode_jpeg
        monkeypatch.setattr(
            module, 'encode_jpeg', lambda image, q=90, *a, _e=encode: qualities.append(q) or _e(image, q)
        )
    session = _run_v2_rtc(_rtc_stream_meta(), 11)

    # Call 0 at 0.0 s, read at 0.1 s: the first chunk counts from its answer, so call 1 goes at 0.5 s (rows at 0.25 s
    # and 0.35 s: ticks 2 and 3). Call 1 is read at 0.6 s: call 2 goes at 0.9 s, its rows on ticks 6 and 7.
    rows = [call.args[0][ROW] for call in session.send_message.call_args_list]
    assert [(row['call'], row['back']) for row in rows] == [(1, 2), (1, 1), (2, 2), (2, 1)]
    images = [serialization.deserialise(serialization.serialise(row['images']))['image'] for row in rows]
    assert [_frame_index(image) for image in images] == [2, 3, 6, 7]
    requests = [call.args[0]['image'] for call in session.infer.call_args_list]
    assert len(requests) == 3
    assert [(r[b'call'], r[b'rows']) for r in requests[1:]] == [(1, [2, 1]), (2, [2, 1])]
    (tail,) = serialization.deserialise(serialization.serialise(requests[2][b'tail']))
    assert _frame_index(tail) == 9
    # The declared encode_images leaves the stack as arrays; the rows and the requests go at its quality.
    assert qualities and set(qualities) == {75}


def test_a_session_without_a_frame_stream_keeps_the_declared_encode_images():
    stack = declared_stack(_rtc_stream_meta(), protocol.ProtocolVersion.V2)
    assert isinstance(stack, Sequential)
    assert type(stack._components[-1]) is EncodeImages


def test_a_frame_stream_logs_its_frame_miss_answers_at_close(caplog):
    chunk = [{'value': i} for i in range(5)]
    with caplog.at_level(logging.INFO, logger=frame_stream_module.__name__):
        _run_on_mock_session(_stream_meta(), 7, frame_stream='jpeg', replies=[chunk, FrameMiss([['image', 2]]), chunk])
    assert '1 frame_miss answers' in caplog.text


def _drop_the_rows_of_call(monkeypatch, call: int) -> list[list[int]]:
    """Make every served session lose the rows of ``call``; returns, per resolved request, the rows each of its stacks
    refers to."""
    resolved: list[list[int]] = []
    resolve, add = StreamedRows.resolve, StreamedRows.add

    def recording_resolve(rows, obs):
        resolved.append([len(value[b'rows']) for value in obs.values() if isinstance(value, Mapping) and REF in value])
        return resolve(rows, obs)

    def dropping_add(rows, row):
        add(rows, row)
        if row['call'] == call:
            rows._rows.pop(row['back'], None)

    monkeypatch.setattr(StreamedRows, 'resolve', recording_resolve)
    monkeypatch.setattr(StreamedRows, 'add', dropping_add)
    return resolved


class _EightActionModel(FixedModel):
    def __call__(self, obs: Obs, *, session_id: str):
        super().__call__(obs, session_id=session_id)
        return [{'value': index} for index in range(8)]


def _rtc_context(obs: Mapping[str, object]) -> tuple:
    """The RTC fields of an observation as the model receives them: the remaining plan, its two offsets and the
    estimated and measured latencies."""
    prefix = obs[policy_keys.RTC.ACTION_PREFIX]
    assert isinstance(prefix, Mapping)
    return (
        prefix[policy_keys.RTC.PREFIX_ACTIONS],
        prefix[policy_keys.RTC.CHUNK_OFFSET_SEC],
        prefix[policy_keys.RTC.OBSERVATION_OFFSET_SEC],
        obs[policy_keys.RTC.ESTIMATED_LATENCY_SEC],
        obs[policy_keys.RTC.MEASURED_LATENCY_SEC],
    )


@pytest.mark.timeout(120)
@pytest.mark.parametrize('frame_stream, atol', [('jpeg', 0), ('h264', 12)])
def test_a_served_rtc_frame_stream_gives_the_model_the_observations_of_full_requests(
    served, monkeypatch, frame_stream, atol
):
    resolved = _drop_the_rows_of_call(monkeypatch, 2)
    rtc = RTCSchedule(fps=10.0, call_after_sec=0.4, latency_estimate=DelayEstimate.max(last=5, max_sec=0.3))
    local = _streamed_stack(rtc, TransformImages(24, 16, mode=ResizeMode.BOUND), EncodeImages(quality=80))
    observed = {}
    for stream in (None, frame_stream):
        resolved.clear()
        address, model, _ = served(local=local, model=_EightActionModel())
        _served_run(address, stream, 19)
        observed[stream] = [(obs['image'], _rtc_context(obs)) for obs in model.observations]

    full, streamed = observed[None], observed[frame_stream]
    # Calls at 0.0 s, then (the first chunk counts from its answer) 0.5, 0.9, 1.3 and 1.7 s.
    assert len(full) == len(streamed) == 5
    for (expected, expected_context), (got, context) in zip(full, streamed, strict=True):
        assert got.shape == expected.shape == (3, 16, 24, 3)
        np.testing.assert_allclose(got, expected, atol=atol)
        assert context == expected_context
    # The model gets the remaining plan from the second call on, with its offsets and the latencies.
    assert full[0][1] == ([], 0.0, 0.0, 0.0, None)
    # The model gets the remaining plan of each 8-action chunk from the second call on: the call 0.4 s into a chunk
    # leaves its last 4 actions, and each answer took one 0.1 s tick.
    assert full[0][1] == ([], 0.0, 0.0, 0.0, None)
    assert [context for _, context in full[1:]] == [([{'value': i} for i in range(4, 8)], 0.4, 0.0, 0.1, 0.1)] * 4
    # Calls 1, 3 and 4 refer to both rows; call 2 misses them and goes again in full.
    assert resolved == [[], [2], [2], [], [2], [2]]


def _noise_frame(index: int) -> np.ndarray:
    """A 16 x 24 frame of noise that changes every other tick, as a camera slower than the control loop gives."""
    return np.random.default_rng(index // 2).integers(0, 256, (16, 24, 3), dtype=np.uint8)


def _wire_bytes(session: MagicMock) -> tuple[list[bytes], list[bytes]]:
    """The serialised requests and rows that ``session`` got, each without its wall-clock time."""
    requests = [
        serialization.serialise({k: v for k, v in call.args[0].items() if k != WALL_TIME_NS})
        for call in session.infer.call_args_list
    ]
    rows = [
        serialization.serialise({k: v for k, v in call.args[0][ROW].items() if k != 't_ns'})
        for call in session.send_message.call_args_list
    ]
    return requests, rows


_RESIZE = {'name': 'restrict_image_size', 'args': {'width': 12, 'height': 8}}


@pytest.mark.parametrize('frame_stream', ['jpeg', 'h264'])
def test_a_chunked_stream_fed_by_its_temporal_stack_sends_the_bytes_of_the_stream_that_compares_pixels(
    monkeypatch, frame_stream
):
    chunk = [{'value': i} for i in range(5)]
    replies = [chunk, chunk, FrameMiss([['image', 2]]), chunk, chunk]
    fed = _run_on_mock_session(
        _stream_meta(_RESIZE), 26, frame_stream=frame_stream, replies=replies, frame=_noise_frame
    )
    # No component passes StackRows on, so the stack gives arrays and the stream compares their pixels.
    monkeypatch.setattr(RemotePolicy, '_ROWS_PASS_THROUGH', frozenset())
    compared = _run_on_mock_session(
        _stream_meta(_RESIZE), 26, frame_stream=frame_stream, replies=replies, frame=_noise_frame
    )

    assert _wire_bytes(fed) == _wire_bytes(compared)
    requests = [call.args[0]['image'] for call in fed.infer.call_args_list]
    assert sum(isinstance(r, dict) and REF in r and len(r[b'rows']) == 2 for r in requests) >= 2


@pytest.mark.parametrize('snap', [0.0, 0.05])
def test_an_rtc_stream_fed_by_its_temporal_stack_sends_the_bytes_of_the_stream_that_compares_pixels(monkeypatch, snap):
    meta = _rtc_stream_meta(TransformImages(12, 8, mode=ResizeMode.BOUND).to_spec())
    fed = _run_v2_rtc(meta, 15, frame=_noise_frame, frame_stream_snap_sec=snap)
    monkeypatch.setattr(RemotePolicy, '_ROWS_PASS_THROUGH', frozenset())
    compared = _run_v2_rtc(meta, 15, frame=_noise_frame, frame_stream_snap_sec=snap)

    assert _wire_bytes(fed) == _wire_bytes(compared)
    requests = [call.args[0]['image'] for call in fed.infer.call_args_list]
    assert sum(isinstance(r, dict) and REF in r and len(r[b'rows']) > 0 for r in requests) >= 2


def test_a_fed_stream_transforms_on_the_call_only_the_frames_that_the_request_carries(monkeypatch):
    on_call = []
    transform = TransformImages._transform_frame

    def counted(codec, frame):
        if threading.current_thread().name.startswith('policy-fn'):
            on_call.append(frame.shape)
        return transform(codec, frame)

    monkeypatch.setattr(TransformImages, '_transform_frame', counted)
    session = _run_on_mock_session(_stream_meta(_RESIZE), 8, frame_stream='jpeg', frame=_noise_frame)

    first, second = (call.args[0]['image'] for call in session.infer.call_args_list)
    assert REF not in first and (second[b'rows'], len(second[b'tail'][b'frames'])) == ([2, 1], 1)
    assert len(on_call) == 3 + 1  # the first request goes in full; the second carries its current frame


@pytest.mark.parametrize(
    'parts, fed',
    [
        pytest.param((_RESIZE,), True, id='known-components'),
        pytest.param((_RESIZE, {'name': 'pack_observation_fields', 'args': {'layout': {}}}), False, id='another-codec'),
    ],
)
def test_only_a_stack_of_known_components_feeds_the_streamer_its_entries(parts, fed):
    policy, session = _mock_remote_policy(_stream_meta(*parts), frame_stream='jpeg')
    session.protocol_version = protocol.ProtocolVersion.V1
    *_, streamer, _, feeds = policy._stack_and_streamer(session, True, threading.Lock(), MagicMock())
    try:
        assert feeds is fed
    finally:
        assert streamer is not None
        streamer.close()


def test_an_answer_reports_its_result_once_at_its_first_read(runtime):
    reads = []
    answer = RemotePolicy._ReportingRuntime(runtime, reads.append).submit(lambda: [1, 2, 3])
    assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
    assert answer.done()
    assert reads == []
    assert answer.result() == answer.result() == [1, 2, 3]
    assert reads == [[1, 2, 3]]


@pytest.mark.parametrize('fails', [False, True])
def test_inference_telemetry_excludes_image_preparation_and_records_failures(tmp_path, monkeypatch, fails):
    session = _mock_session()
    session.served_timing = {}
    if fails:
        session.infer.side_effect = TimeoutError('server stalled')
    encoded_at = []

    def encode(image, quality):
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
    server = start_server(model, PolicyDeployment(ChunkedSchedule(fps=10)), grpc=transport == 'grpc')
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


class _KeepaliveAnswer(http.server.BaseHTTPRequestHandler):
    """Answers a keepalive after ``delay_s``, as a server with no idle timeout does."""

    delay_s = 0.0

    def do_POST(self):
        time.sleep(self.delay_s)
        body = json.dumps({wire.ALIVE_SECONDS: None}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def keepalive_server():
    """Starts an HTTP server whose keepalive answers after the delay a test gives; yields its address."""
    servers: list[http.server.ThreadingHTTPServer] = []

    def start(delay_s: float) -> wire.HostPortAddress:
        handler = type('Handler', (_KeepaliveAnswer,), {'delay_s': delay_s})
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return wire.HostPortAddress('127.0.0.1', server.server_address[1], wire.SESSION_PATH, '')

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


class TestEveryWireSpendsTheCallersBudgetOnce:
    """A caller's timeout is one budget: each step of the call takes what the steps before it left."""

    @staticmethod
    def _clock_after_a_connect_that_took(spent: float):
        """A clock that reads 0 once, then ``spent``: the connect took ``spent`` seconds."""
        readings = iter([0.0])
        return lambda: next(readings, spent)

    def test_the_websocket_wire_times_out_an_answer_the_connect_left_no_time_for(self, keepalive_server):
        budget = 2.0
        address = keepalive_server(delay_s=0.5)
        with (
            patch('positronic_wire.websocket.time.monotonic', side_effect=self._clock_after_a_connect_that_took(1.8)),
            pytest.raises(wire.ConnectRefused),
        ):
            websocket.WebsocketClientWire().keepalive(address, None, budget)

    def test_the_websocket_wire_takes_an_answer_inside_what_the_connect_left(self, keepalive_server):
        budget = 2.0
        address = keepalive_server(delay_s=0.0)
        with patch('positronic_wire.websocket.time.monotonic', side_effect=self._clock_after_a_connect_that_took(1.0)):
            assert websocket.WebsocketClientWire().keepalive(address, None, budget) is None

    def test_the_grpc_wire_gives_the_call_what_the_channel_left(self):
        budget, on_the_channel = 4.0, 1.0
        unary = MagicMock(return_value=json.dumps({wire.ALIVE_SECONDS: None}).encode())
        channel = MagicMock(**{'unary_unary.return_value': unary})

        def a_channel_that_took_its_time(*_args, **_kwargs):
            time.sleep(on_the_channel)
            return channel

        with (
            patch.object(grpc.GrpcClientWire, 'channel', return_value=channel),
            patch('positronic_wire.grpc._ready_channel', side_effect=a_channel_that_took_its_time),
        ):
            grpc.GrpcClientWire().keepalive(_ADDRESS, None, budget)

        given = unary.call_args.kwargs['timeout']
        assert given <= budget - on_the_channel, f'the channel spent {on_the_channel}s and the call still got {given}s'


class _EndingModel(FixedModel):
    def __call__(self, obs: Obs, *, session_id: str):
        self.observations.append(obs)
        pose = np.array([0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5])
        return [{'action': pose}, {'action': pose, policy_keys.END: 'complete'}]


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
@pytest.mark.parametrize('side', ['server', 'client'])
def test_an_end_command_in_the_codec_ends_the_client_run_when_its_element_plays(served, transport, side):
    codec = codecs.absolute_pos_action.override(tgt_ee_pose_key='pose', tgt_grip_key='grip').instantiate()
    codec = codec & EndCommand()
    if side == 'server':  # openpi, galaxea, lerobot: the action codec is the deployment codec
        placement = {
            'codec': codec,
            'local': Sequential(
                EndOnField(), ChunkedSchedule(fps=10), TransformImages(640, 640, mode=ResizeMode.BOUND)
            ),
        }
    else:  # dreamzero: the action codec sits in the client stack, inside the schedule
        placement = {'local': Sequential(EndOnField(), ChunkedSchedule(fps=10), codec)}
    address, _, _ = served(model=_EndingModel(), transport=transport, **placement)
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(RemotePolicy(transport, address))
    try:
        run.send({})
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        first = run.send({})
        assert first is not None
        assert keys.ROBOT_COMMAND in first.commands and policy_keys.END not in first.commands
        now[0] = 100_000_000
        last = run.send({})
        assert last is not None
        assert keys.ROBOT_COMMAND in last.commands and policy_keys.END not in last.commands
        with pytest.raises(StopIteration) as end:
            run.send({})
        assert end.value.value is PolicyEnd.COMPLETE
    finally:
        runtime.close()
        run.close()
