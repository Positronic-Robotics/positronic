"""Unversioned server contracts, exact component selection, and deprecation at connection time."""

import logging
import threading
from concurrent.futures import Future
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
from positronic_wire import wire

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.offboard import protocol
from positronic.offboard.client import InferenceClient, InferenceSession
from positronic.offboard.server import PolicyServer
from positronic.offboard.spec import PolicyDeployment
from positronic.offboard.tests.conftest import DictSource
from positronic.policy import keys as policy_keys
from positronic.policy import spec
from positronic.policy.base import Step
from positronic.policy.codec import RestrictImageSize
from positronic.policy.compatibility import V1_SERVER_DEFAULT_ACTION_FPS, from_v1_spec
from positronic.policy.executor import Executor, WaitStatus, _UnchargedAnswer
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.remote import RemotePolicy
from positronic.utils import flatten_dict
from positronic.utils.versions import Deprecation, Version


@pytest.fixture
def controlled_runtime(monkeypatch):
    now = [1_000_000_000]
    calls = []
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)

    def submit(function, obs):
        future = Future()
        calls.append((future, obs, function))
        return _UnchargedAnswer(future)

    monkeypatch.setattr(runtime, 'submit', submit)
    yield runtime, now, calls
    runtime.close()


@pytest.mark.parametrize(
    'declared, handshake, answer',
    [
        pytest.param(
            {
                'seq': [
                    {'name': 'stop_on_fault'},
                    {
                        'seq': [
                            {'name': 'chunked_schedule'},
                            {'name': 'action_horizon', 'args': {'horizon_sec': 0.15}},
                            {'name': 'action_timestamp', 'args': {'fps': 10}},
                        ]
                    },
                ]
            },
            {},
            [{'value': i} for i in range(4)],
            id='declared-timing',
        ),
        pytest.param(
            {
                'seq': [
                    {'name': 'stop_on_fault'},
                    {'name': 'chunked_schedule'},
                    {'name': 'restrict_image_size', 'args': {'width': 32, 'height': 32}},
                ]
            },
            {'action_fps': 10, 'action_horizon_sec': 0.15},
            [{'value': 0, 'timestamp': 0.0}, {'value': 1, 'timestamp': 0.1}, {'timestamp': 0.15}],
            id='server-stamped',
        ),
    ],
)
def test_v1_timing_preserves_horizon_and_chunk_boundary(controlled_runtime, declared, handshake, answer):
    runtime, now, calls = controlled_runtime
    run = runtime.start(from_v1_spec(declared, handshake), MagicMock(return_value=answer))
    try:
        assert run.send({}).commands == {}
        future, obs, function = calls[0]
        future.set_result(function(obs))
        assert run.send({}) == Step({'value': 0}, 1_100_000_000)
        now[0] = 1_100_000_000
        assert run.send({}) == Step({'value': 1}, 1_150_000_000)
        assert len(calls) == 1
        now[0] = 1_150_000_000
        assert run.send({}).commands == {}
        assert len(calls) == 2
        assert calls[0][1]['obs_time_ns'] == 1_000_000_000
        assert calls[0][1]['wall_time_ns'] > 0
    finally:
        run.close()


def test_v1_server_without_action_fps_runs_at_the_default_rate_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger='positronic.policy.compatibility'):
        stack = from_v1_spec({'seq': [{'name': 'stop_on_fault'}, {'name': 'chunked_schedule'}]}, {})
    assert flatten_dict(stack.meta())[policy_keys.ACTION_FPS] == V1_SERVER_DEFAULT_ACTION_FPS
    assert 'Rebuild the server on current positronic' in caplog.text


def test_v1_fault_pauses_and_keeps_the_pending_answer_and_history(controlled_runtime):
    runtime, now, calls = controlled_runtime
    stack = from_v1_spec(
        {
            'seq': [
                {'name': 'stop_on_fault'},
                {'name': 'temporal_stack', 'args': {'keys': ['image'], 'offsets_sec': [-0.3, 0], 'pad_start': False}},
                {'name': 'chunked_schedule'},
            ]
        },
        {'action_fps': 10},
    )
    run = runtime.start(stack, MagicMock(return_value=[{'value': 'kept', 'timestamp': 0}, {'timestamp': 0.1}]))
    try:
        run.send({'image': np.array([1])})
        assert len(calls) == 1
        now[0] += 100_000_000
        assert run.send({keys.ROBOT_STATUS: RobotStatus.ERROR, 'image': np.array([2])}).commands == {}
        future, obs, function = calls[0]
        future.set_result(function(obs))
        now[0] += 100_000_000
        assert run.send({'image': np.array([3])}).commands == {'value': 'kept'}
        assert len(calls) == 1
        now[0] += 100_000_000
        run.send({'image': np.array([4])})
        assert len(calls) == 2
        np.testing.assert_array_equal(calls[1][1]['image'], [[1], [4]])
    finally:
        run.close()


def test_v1_ticks_that_send_nothing_encode_no_image(controlled_runtime, monkeypatch):
    runtime, now, calls = controlled_runtime
    encoded = []
    monkeypatch.setattr(RestrictImageSize, 'encode', lambda self, data: encoded.append(data) or data)
    stack = from_v1_spec(
        {
            'seq': [
                {'name': 'chunked_schedule'},
                {'name': 'restrict_image_size', 'args': {'width': 32, 'height': 32}},
                {'name': 'action_timestamp', 'args': {'fps': 10}},
            ]
        },
        {},
    )
    run = runtime.start(stack, MagicMock(return_value=[{'value': 1}]))
    try:
        for _ in range(3):
            now[0] += 5_000_000
            assert run.send({'image': np.zeros((4, 4, 3), np.uint8)}).commands == {}
        assert len(calls) == 1 and encoded == []
        future, obs, function = calls[0]
        future.set_result(function(obs))
        assert len(encoded) == 1
    finally:
        run.close()


def test_v1_codec_above_a_layer_still_encodes_on_every_tick(controlled_runtime, monkeypatch):
    runtime, now, calls = controlled_runtime
    encoded = []
    monkeypatch.setattr(RestrictImageSize, 'encode', lambda self, data: encoded.append(data) or data)
    stack = from_v1_spec(
        {
            'seq': [
                {'name': 'restrict_image_size', 'args': {'width': 32, 'height': 32}},
                {'name': 'chunked_schedule'},
                {'name': 'action_timestamp', 'args': {'fps': 10}},
            ]
        },
        {},
    )
    run = runtime.start(stack, MagicMock(return_value=[{'value': 1}]))
    try:
        for _ in range(3):
            now[0] += 5_000_000
            run.send({'image': np.zeros((4, 4, 3), np.uint8)})
        assert len(calls) == 1 and len(encoded) == 3
    finally:
        run.close()


def test_v1_single_action_answer_holds_for_one_period(controlled_runtime):
    runtime, now, calls = controlled_runtime
    stack = from_v1_spec({'seq': [{'name': 'chunked_schedule'}, {'name': 'action_timestamp', 'args': {'fps': 10}}]}, {})
    run = runtime.start(stack, MagicMock(return_value={'value': 1}))
    try:
        run.send({})
        future, obs, function = calls[0]
        future.set_result(function(obs))
        assert run.send({}) == Step({'value': 1}, 1_100_000_000)
        now[0] += 50_000_000
        run.send({})
        assert len(calls) == 1
        now[0] += 50_000_000
        run.send({})
        assert len(calls) == 2
    finally:
        run.close()


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
@pytest.mark.parametrize('scheduled', [True, False], ids=['scheduled-chunk', 'codec-only'])
def test_new_client_runs_an_unversioned_server(start_server, make_mock_model, monkeypatch, transport, scheduled):
    observations = []
    disconnected = threading.Event()
    declared = (
        {'seq': [{'name': 'stop_on_fault'}, {'name': 'chunked_schedule'}]}
        if scheduled
        else {'name': 'binarize_grip_training', 'args': {'keys': []}}
    )
    actions = (
        [{'value': 10, 'timestamp': 0}, {'value': 20, 'timestamp': 0.1}, {'timestamp': 0.2}]
        if scheduled
        else {'value': 10}
    )

    async def v1_session(server, conn, model_id):
        await conn.send(protocol.serialise({'status': 'ready', 'meta': {'local_stack': declared, 'action_fps': 10}}))
        try:
            while True:
                observation = protocol.deserialise(await conn.receive())
                assert 'session_id' not in observation
                observations.append(observation)
                await conn.send(protocol.serialise({'result': actions}))
        except wire.PeerDisconnected:
            disconnected.set()

    monkeypatch.setattr(PolicyServer, '_serve_session', v1_session)
    served = start_server(
        PolicyDeployment(DictSource({'default': make_mock_model([], {})}), ChunkedSchedule(fps=10)),
        grpc=transport == 'grpc',
    )
    address = served.ws()[1] if transport == 'websocket' else served.grpc()[1]
    now = [1_000_000_000]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(RemotePolicy(transport, address))
    try:
        first = run.send({'image': np.array([1, 2])})
        assert isinstance(first, Step) and first.commands == {}
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        assert run.send({'image': np.array([1, 2])}) == Step({'value': 10}, 1_100_000_000)
        if scheduled:
            now[0] = 1_100_000_000
            assert run.send({}) == Step({'value': 20}, 1_200_000_000)
        assert len(observations) == 1
        assert observations[0]['obs_time_ns'] == 1_000_000_000
        np.testing.assert_array_equal(observations[0]['image'], [1, 2])
    finally:
        runtime.close()
        run.close()
    assert disconnected.wait(5)
    assert len(observations) == 1  # V1 closes the transport without an end-session request.


def test_unknown_protocol_closes_before_any_request(start_server, make_mock_model, monkeypatch):
    monkeypatch.setattr(protocol, 'CURRENT_VERSION', 99)
    model = make_mock_model([], {})
    served = start_server(PolicyDeployment(DictSource({'default': model}), ChunkedSchedule(fps=10)))
    with pytest.raises(ValueError, match='Unsupported policy protocol version 99'):
        InferenceClient(*served.ws()).new_session()
    model.assert_not_called()


def test_component_versions_are_independent_and_exact():
    stack = spec.from_spec({
        'seq': [
            {'name': 'chunked_schedule', 'version': 2, 'args': {'fps': 10}},
            {'name': 'restrict_image_size', 'version': 1, 'args': {'width': 32, 'height': 32}},
        ]
    })
    assert [part['version'] for part in stack.to_spec()['seq']] == [2, 1]
    for version in (1, 99):
        with pytest.raises(ValueError, match=f'Unsupported.*version {version}'):
            spec.from_spec({'name': 'chunked_schedule', 'version': version, 'args': {'fps': 10}})


@pytest.mark.parametrize(
    'timing',
    [{'name': 'action_timestamp', 'args': {'fps': 10}}, {'name': 'action_horizon', 'args': {'horizon_sec': 0.1}}],
)
@pytest.mark.parametrize('group', [None, 'seq', 'par'])
def test_v2_stack_rejects_v1_timing_codecs(timing, group):
    codec = {group: [timing, {'name': 'flip_grip'}]} if group else timing
    with pytest.raises(ValueError, match=f"Unknown local-stack entry '{timing['name']}'"):
        spec.from_spec({'seq': [{'name': 'chunked_schedule', 'version': 2, 'args': {'fps': 10}}, codec]})


def test_deprecated_component_warns_once_per_stack_and_removed_one_fails(monkeypatch):
    current = spec.COMPONENTS['flip_grip'][1].implementation
    notice = Deprecation(date(2020, 1, 1), date(2020, 7, 1), 'Serve flip_grip v2 instead.')
    monkeypatch.setitem(spec.COMPONENTS, 'flip_grip', {1: Version(current, notice)})
    with pytest.warns(FutureWarning, match='flip_grip.*2020-07-01') as warnings:
        spec.from_spec({'seq': [{'name': 'flip_grip'}, {'name': 'flip_grip'}]})
    assert len(warnings) == 1
    monkeypatch.setitem(spec.COMPONENTS, 'flip_grip', {1: Version(None, notice, removed_on=date(2020, 7, 1))})
    with pytest.raises(ValueError, match='has been removed.*flip_grip v2'):
        spec.from_spec({'name': 'flip_grip'})


def test_deprecated_protocol_warns_and_keeps_raw_observations(monkeypatch):
    notice = Deprecation(date(2020, 1, 1), date(2020, 7, 1), 'Upgrade the server to protocol v2.')
    monkeypatch.setitem(protocol.VERSIONS, 1, Version(protocol.ProtocolVersion.V1, notice))
    conn = MagicMock(spec=wire.ClientConnection)
    conn.recv.side_effect = [protocol.serialise({'status': 'ready', 'meta': {}}), protocol.serialise({'result': []})]
    with pytest.warns(FutureWarning, match='policy protocol v1.*2020-07-01'):
        session = InferenceSession(conn)
    assert session.session_id is None
    session.infer({'value': 42})
    session.close()
    assert protocol.deserialise(conn.send.call_args.args[0]) == {'value': 42}
    assert conn.send.call_count == 1
    conn.close.assert_called_once()
