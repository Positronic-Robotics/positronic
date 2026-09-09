"""The gRPC wire: one session runs over it exactly as it runs over the websocket."""

from unittest.mock import ANY, MagicMock

import configuronic as cfn
import grpc
import pytest

from positronic.offboard import grpc_wire, wire
from positronic.offboard.client import InferenceClient, _ConnectRetries
from positronic.offboard.server import AUTH_HEADER, PolicyServer, bearer
from positronic.offboard.tests.conftest import DictSource, StartServer
from positronic.policy.layers import ChunkedSchedule, TemporalStack
from positronic.policy.spec import ModelSource, PolicySource, remote

_TOKEN = 'test-secret-token'


def grpc_url(server: PolicyServer, path: str = '') -> str:
    return f'grpc://{server.host}:{server.grpc_port}{path}'


@pytest.fixture
def both_wires(start_server: StartServer, make_mock_policy) -> tuple[PolicyServer, MagicMock]:
    """A server offering both wires over one policy."""
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    _host, _port, server = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True)
    return server, policy


def test_a_grpc_session_handshakes_and_infers(both_wires):
    server, policy = both_wires
    session = InferenceClient(grpc_url(server)).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        obs = {'image': 'test'}
        assert session.infer(obs) == [{'action': [1, 2, 3]}]
        policy._mock_session.assert_called_with(obs, ANY)
    finally:
        session.close()


def test_both_wires_answer_one_observation_alike(both_wires):
    server, _policy = both_wires
    obs = {'image': 'test'}
    over_ws = InferenceClient(f'{server.host}:{server.port}').new_session()
    over_grpc = InferenceClient(grpc_url(server)).new_session()
    try:
        assert over_grpc.metadata == over_ws.metadata
        assert over_grpc.infer(obs) == over_ws.infer(obs)
    finally:
        over_ws.close()
        over_grpc.close()


def test_closing_a_session_ends_it_on_the_server(both_wires):
    """``close`` half-closes the stream and waits, so the server releases the session before it returns."""
    server, _policy = both_wires
    session = InferenceClient(grpc_url(server)).new_session()
    assert server._active_sessions == 1
    session.close()
    assert server._active_sessions == 0


def test_a_failed_inference_reaches_the_client_as_an_exception(both_wires):
    server, policy = both_wires
    session = InferenceClient(grpc_url(server)).new_session()
    try:
        policy._mock_session.side_effect = RuntimeError('no such joint')
        with pytest.raises(RuntimeError, match='no such joint'):
            session.infer({'image': 'test'})
    finally:
        session.close()


def test_a_session_that_cannot_open_reaches_the_client_as_an_exception(start_server, make_mock_policy):
    """A model the source refuses fails in the handshake, before the session serves anything."""
    policies = {'alpha': make_mock_policy([{'action': [1]}], {'model_name': 'alpha'})}
    _host, _port, server = start_server(ChunkedSchedule() | remote | DictSource(policies), grpc=True)
    with pytest.raises(RuntimeError, match='Unknown model'):
        InferenceClient(grpc_url(server, f'{wire.SESSION_PATH}/beta')).new_session()


def test_the_session_path_names_the_model(start_server, make_mock_policy):
    policies = {
        'alpha': make_mock_policy([{'action': ['alpha']}], {'model_name': 'alpha'}),
        'beta': make_mock_policy([{'action': ['beta']}], {'model_name': 'beta'}),
    }
    _host, _port, server = start_server(ChunkedSchedule() | remote | DictSource(policies), grpc=True)
    session = InferenceClient(grpc_url(server, f'{wire.SESSION_PATH}/beta')).new_session()
    try:
        assert session.metadata['model_name'] == 'beta'
        assert session.infer({'obs': 'beta'}) == [{'action': ['beta']}]
    finally:
        session.close()


def _tunable_pipe(source: ModelSource, offsets: tuple[float, ...] = (-0.1, 0.0)):
    return TemporalStack(keys=('x',), offsets_sec=offsets) | ChunkedSchedule() | remote | source


def test_the_query_carries_the_session_params(start_server, make_mock_policy):
    policies = {'alpha': make_mock_policy([{'action': ['alpha']}], {'model_name': 'alpha'})}
    pipe = cfn.Config(_tunable_pipe, source=cfn.Config(DictSource, policies=policies))
    _host, _port, server = start_server(pipe, grpc=True)
    session = InferenceClient(grpc_url(server, f'{wire.SESSION_PATH}?offsets=[-0.5, 0.0]')).new_session()
    try:
        assert session.metadata['local_stack']['seq'][0]['args']['offsets_sec'] == [-0.5, 0.0]
    finally:
        session.close()


@pytest.fixture
def authed_server(start_server: StartServer, make_mock_policy) -> PolicyServer:
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    _host, _port, server = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True, auth_token=_TOKEN)
    return server


def test_the_grpc_wire_gates_on_the_bearer_token(authed_server):
    session = InferenceClient(grpc_url(authed_server), headers={AUTH_HEADER: bearer(_TOKEN)}).new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
    finally:
        session.close()


@pytest.mark.parametrize('header', [None, bearer('wrong'), _TOKEN])
def test_the_grpc_wire_refuses_a_session_without_the_token(authed_server, header, monkeypatch):
    # A refused credential and a cold backend answer alike, so the client spends attempts on it; one
    # is enough to see the refusal.
    monkeypatch.setattr(_ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    headers = None if header is None else {AUTH_HEADER: header}
    with pytest.raises(grpc.RpcError) as refused:
        InferenceClient(grpc_url(authed_server), headers=headers).new_session()
    assert refused.value.code() is grpc.StatusCode.PERMISSION_DENIED


@pytest.mark.parametrize('url', ['grpcs://gpu-host:9000', 'tcp://gpu-host:9000'])
def test_an_unknown_scheme_is_refused(url):
    with pytest.raises(ValueError, match='Unsupported scheme'):
        InferenceClient(url)


def test_a_grpc_url_names_the_session_port_alone():
    client = InferenceClient('grpc://gpu-host:9000')
    assert client.session_url == 'grpc://gpu-host:9000/api/v1/session'
    with pytest.raises(ValueError, match='gRPC session port'):
        client.list_models()


@pytest.mark.parametrize(
    ('session_path', 'model_id'),
    [
        (wire.SESSION_PATH, None),
        (f'{wire.SESSION_PATH}/10000', '10000'),
        (f'{wire.SESSION_PATH}/GEAR-Dreams/DreamZero-DROID', 'GEAR-Dreams/DreamZero-DROID'),
        (f'{wire.SESSION_PATH}/s3%3A//bucket/ckpt-1', 's3://bucket/ckpt-1'),
    ],
)
def test_the_session_path_decodes_as_the_websocket_route_does(session_path, model_id):
    assert grpc_wire.model_id_of(session_path) == model_id


def test_a_path_outside_the_session_route_is_refused():
    with pytest.raises(ValueError, match='Unexpected session path'):
        grpc_wire.model_id_of('/api/v2/session/10000')


def test_a_port_that_never_answers_is_named_at_the_deadline():
    """Nothing listens on port 1, so the channel never becomes ready and the connect deadline passes."""
    client = InferenceClient('grpc://localhost:1', open_timeout=0.2, connect_deadline=0.0)
    with pytest.raises(TimeoutError, match='grpc://localhost:1'):
        client.new_session()


def test_an_ipv6_host_binds_in_brackets(start_server: StartServer, make_mock_policy):
    """gRPC's target syntax brackets an IPv6 literal, so a bare '::1' would bind ':::<port>' and fail."""
    assert grpc_wire._bind_target('::', 9000) == '[::]:9000'
    assert grpc_wire._bind_target('0.0.0.0', 9000) == '0.0.0.0:9000'

    policy = make_mock_policy([{'action': [4]}], {'model_name': 'stub'})
    _host, _port, server = start_server(ChunkedSchedule() | remote | PolicySource(policy), grpc=True, host='::1')
    session = InferenceClient(f'grpc://[{server.host}]:{server.grpc_port}').new_session()
    try:
        assert session.infer({'image': 'test'}) == [{'action': [4]}]
    finally:
        session.close()
