"""The Positronic client selects v3 without interpreting native results as robot commands."""

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial

import numpy as np
import pytest
from positronic_model_server import grpc_wire, keys, protocol, serialization, server_wire, spec, websocket_wire
from positronic_model_server.server import Model, ModelServer, Session
from positronic_wire import grpc, websocket, wire

from positronic import keys as obs_keys
from positronic.drivers.roboarm import command
from positronic.offboard import protocol as legacy_protocol
from positronic.offboard.client import InferenceClient
from positronic.policy import keys as policy_keys
from positronic.policy.codecs import Codec, EncodeImages, PolicyCommandError
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.keys import ACTION_FPS
from positronic.policy.processors import (
    ConstantDuration,
    InferenceSchedule,
    Periodic,
    RequestTiming,
    TemporalStack,
    TrajectoryMerge,
    TrajectoryPlayer,
)
from positronic.policy.remote import RemotePolicy
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec
from positronic.utils import flatten_dict
from positronic.utils.versions import resolve_version


class EchoSession(Session):
    def infer(self, observation):
        return observation


class EchoModel(Model):
    def __init__(self, session_metadata=None):
        super().__init__(parameters={'fps': 20})
        self._session_metadata = session_metadata

    def prepare_session(self, params):
        return EchoSession(
            spec.component('chunked_schedule', version=2, fps=params['fps']), metadata=self._session_metadata
        )


@pytest.fixture(
    params=[(websocket_wire.WebsocketWire, websocket.WebsocketClientWire), (grpc_wire.GrpcWire, grpc.GrpcClientWire)]
)
def serving(request):
    @contextmanager
    def serve(load_model, **kwargs):
        server = ModelServer(load_model, **kwargs)
        transport_type, client_type = request.param
        transport = transport_type(server_wire.ServedHostPort('127.0.0.1', 0))
        ready = threading.Event()
        thread = threading.Thread(target=server.serve, args=([transport], ready.set), daemon=True)
        thread.start()
        try:
            assert ready.wait(5)
            bound = transport.served_address
            yield client_type(), wire.HostPortAddress(bound.host, bound.port, wire.SESSION_PATH, '')
        finally:
            server.shutdown()
            thread.join(10)
            assert not thread.is_alive()

    return serve


def test_v3_client_preserves_native_markers_in_metadata_and_results(serving):
    native = {'robot_command': {'vendor': 'value'}, b'__cmd__': 'not a robot command', 'array': np.arange(12)}

    with serving(partial(EchoModel, session_metadata={'native': native})) as endpoint:
        session = InferenceClient(*endpoint).new_session()
        try:
            assert session.protocol_version is protocol.ProtocolVersion.V3
            assert session.metadata['native'][b'__cmd__'] == native[b'__cmd__']
            result = session.infer(native)
            assert result['robot_command'] == native['robot_command']
            assert result[b'__cmd__'] == native[b'__cmd__']
            np.testing.assert_array_equal(result['array'], native['array'])
        finally:
            session.close()


@pytest.fixture
def watch_sends(monkeypatch):
    def watch(session):
        sent = threading.Semaphore(0)
        send = session._conn.send

        def send_and_notify(data):
            send(data)
            sent.release()

        monkeypatch.setattr(session._conn, 'send', send_and_notify)
        return sent

    return watch


def test_calls_pipeline_while_model_execution_and_replies_stay_ordered(serving, watch_sends):
    release = threading.Event()
    observed = []

    class PipelineSession(Session):
        def infer(self, observation):
            index = observation['index']
            observed.append(index)
            if index == 1:
                assert release.wait(5)
                return None
            if index == 2:
                raise ValueError('Rejected sample')
            return index

    class PipelineModel(Model):
        def prepare_session(self, params):
            return PipelineSession(spec.component('chunked_schedule', version=2, fps=10))

    with serving(PipelineModel) as endpoint:
        session = InferenceClient(*endpoint, infer_timeout=5).new_session()
        assert session.infer({'index': 0}) == 0
        sent = watch_sends(session)
        with ThreadPoolExecutor(max_workers=3) as pool:
            try:
                answers = []
                for index in (1, 2, 3):
                    answers.append(pool.submit(session.infer, {'index': index}))
                    assert sent.acquire(timeout=2), 'A send waited for an earlier prediction'
                assert not any(answer.done() for answer in answers)
                release.set()
                assert answers[0].result(timeout=5) is None
                with pytest.raises(RuntimeError, match='Rejected sample'):
                    answers[1].result(timeout=5)
                assert answers[2].result(timeout=5) == 3
                assert observed == [0, 1, 2, 3]
            finally:
                release.set()
        session.close()


def test_pipeline_timeout_fails_unanswered_calls_and_closes_session(serving, watch_sends):
    release = threading.Event()

    class StalledSession(Session):
        def infer(self, observation):
            if observation['index']:
                assert release.wait(5)
            return observation['index']

    class StalledModel(Model):
        def prepare_session(self, params):
            return StalledSession(spec.component('chunked_schedule', version=2, fps=10))

    with serving(StalledModel) as endpoint:
        session = InferenceClient(*endpoint, infer_timeout=0.5).new_session()
        assert session.infer({'index': 0}) == 0
        sent = watch_sends(session)
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                answers = []
                for index in (1, 2):
                    answers.append(pool.submit(session.infer, {'index': index}))
                    assert sent.acquire(timeout=2)
                for answer in answers:
                    with pytest.raises((TimeoutError, wire.PeerDisconnected)):
                        answer.result(timeout=3)
                with pytest.raises(wire.PeerDisconnected):
                    session.infer({'index': 3})
            finally:
                release.set()
        session.close()


def test_remote_calls_match_responses_when_encoding_reorders_observations(serving, monkeypatch):
    encoding, release = threading.Event(), threading.Event()
    encoded, observed = [], []

    class SlowCodec(Codec):
        def encode(self, data):
            if data['index'] == 1:
                encoding.set()
                assert release.wait(5)
            encoded.append(data['index'])
            return data

        def _decode_single(self, data):
            return data

    stack = Sequential(
        TrajectoryPlayer(),
        Periodic(10),
        InferenceSchedule(
            1,
            request_timing=RequestTiming.every(0.1),
            merge=TrajectoryMerge.at_observation(),
            latency_estimate=ConstantDuration(),
            max_in_flight=3,
        ),
        SlowCodec(),
        from_spec(
            spec.component(
                'pack_fields', encode_layout={'index': ['index'], policy_keys.REPLAN_NOW: [policy_keys.REPLAN_NOW]}
            )
        ),
    )

    class ObservingSession(Session):
        def infer(self, observation):
            observed.append(observation)
            return [{'value': observation['index']}]

    class ObservingModel(Model):
        def prepare_session(self, params):
            return ObservingSession(spec.component('chunked_schedule', version=2, fps=10))

    monkeypatch.setattr('positronic.policy.remote.declared_stack', lambda *_: stack)
    now = [0]
    with serving(ObservingModel) as (client, address):
        runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
        run = runtime.start(RemotePolicy(client.NAME, address))
        try:
            run.send({'index': 0, policy_keys.REPLAN_NOW: False})
            assert runtime.wait(timeout_sec=3).status is WaitStatus.ANSWERS_READY
            run.send({'index': 0, policy_keys.REPLAN_NOW: False})
            now[0] = 100_000_000
            run.send({'index': 1, policy_keys.REPLAN_NOW: True})
            assert encoding.wait(3)
            now[0] = 200_000_000
            run.send({'index': 2, policy_keys.REPLAN_NOW: False})
            assert runtime.wait(timeout_sec=3).status is WaitStatus.ANSWERS_READY
            assert observed[-1]['index'] == 2
            release.set()
            assert runtime.wait(timeout_sec=3).status is WaitStatus.ANSWERS_READY
            for value in (1, 2):
                step = run.send({'index': 2})
                assert step is not None and step.commands == {'value': value}
            assert encoded == [0, 2, 1]
            assert observed == [
                {'index': 0, policy_keys.REPLAN_NOW: False},
                {'index': 2, policy_keys.REPLAN_NOW: False},
                {'index': 1, policy_keys.REPLAN_NOW: True},
            ]
        finally:
            release.set()
            runtime.close()
            run.close()


def test_stack_failure_drains_sent_inference_before_closing_session(serving):
    started, release = threading.Event(), threading.Event()
    order = []
    stack = spec.sequence(
        TemporalStack(('image',), (0.0,)).to_spec(), spec.component('chunked_schedule', version=2, fps=10)
    )

    class BlockingSession(Session):
        def infer(self, observation):
            started.set()
            assert release.wait(5)
            order.append('inference finished')

        def close(self):
            order.append('session closed')

    class BlockingModel(Model):
        def prepare_session(self, params):
            return BlockingSession(stack)

    with serving(BlockingModel) as (client, address):
        runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
        run = runtime.start(RemotePolicy(client.NAME, address))
        releaser = threading.Timer(0.05, release.set)
        try:
            run.send({'image': np.zeros((8, 8, 3), dtype=np.uint8)})
            assert started.wait(5)
            releaser.start()
            with pytest.raises(PolicyCommandError, match='image'):
                run.send({})
            assert order == ['inference finished', 'session closed']
        finally:
            release.set()
            releaser.cancel()
            runtime.close()
            run.close()


@pytest.mark.parametrize('image_args', [{}, {'paths': [['video', 'camera']]}], ids=['automatic', 'explicit'])
def test_images_use_client_selection_and_explicit_output_paths(serving, image_args):
    image = np.full((2, 3, 8, 12, 3), 140, dtype=np.uint8)
    state = np.full(image.shape, 1.25, dtype=np.float32)
    description = spec.sequence(
        spec.component('chunked_schedule', version=2, fps=20), spec.component('encode_images', quality=95, **image_args)
    )
    seen = []

    class ImageSession(Session):
        def infer(self, observation):
            seen.append(observation)
            return {'images': [observation['video']['camera']], 'state': observation['state']}

    class ImageModel(Model):
        def prepare_session(self, params):
            return ImageSession(description, output_images=[serialization.JpegEncoding(('images', 0), 95)])

    with serving(ImageModel) as endpoint:
        session = InferenceClient(*endpoint).new_session()
        try:
            codec = from_spec(description[spec.SEQ][-1])
            assert isinstance(codec, EncodeImages)
            result = codec.wrap(session.infer)({'video': {'camera': image}, 'state': state})
            assert seen[0]['video']['camera'].shape == image.shape
            assert result['images'][0].shape == image.shape
            np.testing.assert_allclose(result['images'][0], image, atol=2)
            np.testing.assert_array_equal(result['state'], state)
        finally:
            session.close()


def test_remote_policy_derives_client_metadata_without_changing_the_server_report(serving):
    with serving(EchoModel) as (client, address):
        policy = RemotePolicy(client.NAME, address)
        runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
        try:
            runtime.start(policy).close()
        finally:
            runtime.close()
        metadata = flatten_dict(runtime.metadata_snapshot())
        assert metadata[ACTION_FPS] == 20
        assert f'{policy_keys.SERVER}.{ACTION_FPS}' not in metadata
        assert metadata[f'{policy_keys.SERVER}.{keys.EFFECTIVE_PARAMS}.fps'] == 20


# The seven joint targets of one action, as a native model answers them.
JOINT_TARGET = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]


def answering(answer, description):
    """A model whose every session answers ``answer`` and declares ``description`` as its client stack."""

    class AnswerSession(Session):
        def infer(self, observation):
            return answer

    class AnswerModel(Model):
        def prepare_session(self, params):
            return AnswerSession(description)

    return AnswerModel


def first_step(client, address):
    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    run = runtime.start(RemotePolicy(client.NAME, address))
    try:
        run.send({})
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        return run.send({})
    finally:
        runtime.close()
        run.close()


def test_a_v3_command_mapping_that_the_declared_stack_does_not_decode_fails_as_a_policy_command_error(serving):
    answer = [{obs_keys.ROBOT_COMMAND: {'type': 'joint_pos', 'positions': JOINT_TARGET}, obs_keys.TARGET_GRIP: 0.0}]
    with serving(answering(answer, spec.component('chunked_schedule', version=2, fps=20))) as endpoint:
        with pytest.raises(PolicyCommandError, match=f'{obs_keys.ROBOT_COMMAND} carries a dict, not a command'):
            first_step(*endpoint)


def test_a_v3_joint_list_that_the_declared_stack_decodes_reaches_the_caller_as_a_command(serving):
    answer = [{obs_keys.ROBOT_COMMAND: JOINT_TARGET, obs_keys.TARGET_GRIP: 0.0}]
    description = spec.sequence(
        spec.component('chunked_schedule', version=2, fps=20),
        spec.parallel(
            spec.component('rename_observation_fields', mapping={}),
            spec.component('joint_position_action', joints_key=obs_keys.ROBOT_COMMAND, grip_key=obs_keys.TARGET_GRIP),
        ),
    )
    with serving(answering(answer, description)) as endpoint:
        step = first_step(*endpoint)
    assert step is not None
    assert isinstance(step.commands[obs_keys.ROBOT_COMMAND], command.JointPosition)


@pytest.mark.parametrize('version', [None, 1, 2])
def test_remote_rtc_preserves_prefix_offsets_and_decodes_commands(serving, version):
    schedule = spec.component('rtc_schedule', version=version or 1, fps=10, call_after_sec=0.2, latency_estimate=0.1)
    if version is None:
        del schedule[spec.VERSION]
    if version == 2:
        schedule = spec.sequence(
            spec.component('trajectory_player', plan_key=policy_keys.PLAN, record_stats=False), schedule
        )
    description = spec.sequence(
        schedule,
        spec.parallel(
            spec.component(
                'select_observation_fields',
                mapping={
                    name: name for name in (policy_keys.RTC.ESTIMATED_LATENCY_SEC, policy_keys.RTC.MEASURED_LATENCY_SEC)
                },
            ),
            spec.component(
                'joint_position_action',
                joints_key=obs_keys.ROBOT_COMMAND,
                grip_key=obs_keys.TARGET_GRIP,
                commands_path=[policy_keys.RTC.ACTION_PREFIX, policy_keys.RTC.PREFIX_ACTIONS],
            ),
        ),
    )
    seen = []

    class RTCSession(Session):
        def infer(self, observation):
            seen.append(observation)
            return [{obs_keys.ROBOT_COMMAND: JOINT_TARGET, obs_keys.TARGET_GRIP: float(i)} for i in range(5)]

    class RTCModel(Model):
        def prepare_session(self, params):
            return RTCSession(description)

    with serving(RTCModel) as (client, address):
        now = [1_000_000_000]
        runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
        run = runtime.start(RemotePolicy(client.NAME, address))
        try:
            run.send({})
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            first = run.send({})
            assert first is not None
            assert isinstance(first.commands[obs_keys.ROBOT_COMMAND], command.JointPosition)
            now[0] += 250_000_000
            step = run.send({})
            assert step is not None
            assert step.commands[obs_keys.TARGET_GRIP] == 2.0
            assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
            prefix = seen[1][policy_keys.RTC.ACTION_PREFIX]
            assert [action[obs_keys.TARGET_GRIP] for action in prefix[policy_keys.RTC.PREFIX_ACTIONS]] == [2, 3, 4]
            assert prefix[policy_keys.RTC.CHUNK_OFFSET_SEC] == pytest.approx(0.2)
            assert prefix[policy_keys.RTC.OBSERVATION_OFFSET_SEC] == pytest.approx(-0.05)
            assert policy_keys.PLAN not in seen[1]
            assert seen[1][policy_keys.RTC.ESTIMATED_LATENCY_SEC] == 0.1
            assert seen[1][policy_keys.RTC.MEASURED_LATENCY_SEC] == 0.0
        finally:
            runtime.close()
            run.close()


def test_legacy_protocol_catalog_does_not_advertise_v3():
    with pytest.raises(ValueError, match='Unsupported policy protocol version 3'):
        resolve_version(legacy_protocol.VERSIONS, 3, 'policy protocol')
