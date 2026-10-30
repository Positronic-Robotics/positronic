"""The Positronic client selects v3 without interpreting native results as robot commands."""

import threading
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
from positronic.policy.codecs import EncodeImages, PolicyCommandError
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.keys import ACTION_FPS
from positronic.policy.remote import RemotePolicy
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


def test_legacy_protocol_catalog_does_not_advertise_v3():
    with pytest.raises(ValueError, match='Unsupported policy protocol version 3'):
        resolve_version(legacy_protocol.VERSIONS, 3, 'policy protocol')
