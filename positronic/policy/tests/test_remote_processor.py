"""The callable inference boundary exercised through the public offboard client and server."""

import threading
from collections.abc import Callable

import numpy as np
import pytest

from positronic import keys
from positronic.cfg import codecs
from positronic.offboard import grpc_wire, protocol, websocket_wire
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import PolicyServer
from positronic.policy.base import Obs, Sequential, Step
from positronic.policy.codec import Codec, RestrictImageSize
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.policy.remote import RemotePolicy
from positronic.policy.spec import Model, ModelSource, Pipeline, from_spec


class FixedModel(Model):
    def __init__(self):
        self.observations = []
        self.closed = False

    def __call__(self, obs: Obs):
        self.observations.append(obs)
        if obs.get('fail'):
            raise ValueError('model failed')
        return [{'value': index} for index in range(4)]

    def meta(self):
        return {'model_name': 'fixed'}

    def close(self):
        self.closed = True


class FixedSource(ModelSource):
    def __init__(self, model: Model):
        self.model = model

    def get_models(self):
        return ['050000']

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Model:
        return self.model


@pytest.fixture
def served():
    running = []

    def start(*, codec=None, local_codec=None, transport='websocket', model=None):
        if model is None:
            model = FixedModel()
        pipeline = Pipeline(
            FixedSource(model),
            Sequential(StopOnFault(), ChunkedSchedule(fps=10, horizon_sec=0.2)),
            codec=codec,
            local_codec=local_codec,
        )
        server = PolicyServer(pipeline)
        wire = (
            websocket_wire.WebsocketWire('127.0.0.1', 0, server.api)
            if transport == 'websocket'
            else grpc_wire.GrpcWire('127.0.0.1', 0)
        )
        ready = threading.Event()
        thread = threading.Thread(target=server.serve, args=([wire], ready.set), daemon=True)
        thread.start()
        running.append((server, thread, model))
        assert ready.wait(timeout=5), 'server did not start'
        scheme = 'http' if transport == 'websocket' else 'grpc'
        return f'{scheme}://127.0.0.1:{wire.endpoint.port}/api/v1/session/050000', model, pipeline

    yield start
    for server, thread, model in running:
        server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert model.closed


@pytest.mark.parametrize('transport', ['websocket', 'grpc'])
def test_remote_chunk_cadence_and_fresh_episode_state(served, transport):
    url, model, pipeline = served(local_codec=RestrictImageSize(8, 8), transport=transport)
    policy = RemotePolicy(url)
    assert policy.meta()['server.model_name'] == 'fixed'
    assert policy.meta()['server.action_fps'] == 10
    assert policy.meta()['server.action_horizon_sec'] == 0.2
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
    assert from_spec(pipeline.local.to_spec()).to_spec() == pipeline.local.to_spec()


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
        def __call__(self, obs: Obs):
            nonlocal now_ns
            now_ns += obs['duration_ns']
            return super().__call__(obs)

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
            with pytest.raises(RuntimeError, match='model failed'):
                session.infer({'duration_ns': 11_000_000, 'fail': True})
            assert session.served_timing == {}
    finally:
        session.close()


def test_codec_work_is_inside_submit():
    caller_thread = threading.get_ident()
    threads = []

    class ThreadCodec(OffsetCodec):
        def encode(self, data):
            threads.append(threading.get_ident())
            return super().encode(data)

        def _decode_single(self, data):
            threads.append(threading.get_ident())
            return super()._decode_single(data)

    runtime = Executor(lambda: 0, simulated=True, charge_inference_time=False)
    try:
        answer = runtime.submit(ThreadCodec().wrap(lambda obs: {'value': obs['encoded']}), {})
        assert runtime.wait(timeout_sec=5).status is WaitStatus.ANSWERS_READY
        assert answer.result() == {'value': 52}
        assert len(threads) == 2
        assert all(thread != caller_thread for thread in threads)
    finally:
        runtime.close()


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


def test_sequential_combines_component_metadata():
    class NamedSchedule(ChunkedSchedule):
        def meta(self):
            return {'config': {'fps': 10}}

    class NamedStop(StopOnFault):
        def meta(self):
            return {'config': {'fault_handling': True, 'fps': 20}}

    assert Sequential(NamedStop(), NamedSchedule(fps=10)).meta() == {'config.fault_handling': True, 'config.fps': 10}


def test_act_codec_can_run_on_either_side_of_the_connection(served):
    class EchoStateModel(FixedModel):
        def __call__(self, obs: Obs):
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
    for placement in ({'codec': codec}, {'local_codec': codec}):
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
    assert inputs[0] == inputs[1]
    assert outputs[0] == outputs[1]


def test_spec_rejects_mixed_codec_and_processor_sequence():
    with pytest.raises(ValueError, match='Declare codecs separately'):
        from_spec({'seq': [RestrictImageSize(8, 8).to_spec(), ChunkedSchedule(10).to_spec()]})
