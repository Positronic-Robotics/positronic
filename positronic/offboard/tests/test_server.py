import os
import socket
import time
import urllib.parse
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import ANY, MagicMock

import configuronic as cfn
import httpx
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from positronic import keys
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import InferenceClient, InferenceSession, _ConnectRetries
from positronic.offboard.protocol import deserialise
from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV, PolicyServer, bearer
from positronic.offboard.server_utils import warmup
from positronic.offboard.tests.conftest import round_trip
from positronic.policy import Codec, Policy, RemotePolicy, Session
from positronic.policy.base import Runtime
from positronic.policy.codec import ActionTimestamp
from positronic.policy.layers import ChunkedSchedule, TemporalStack
from positronic.policy.spec import ModelSource, PolicySource, inline, remote


class _StubSource(ModelSource):
    """Serves one ready policy under any requested id, so route-supplied checkpoints resolve as-is."""

    def __init__(self, policy: Policy, name: str = 'stub'):
        self._policy = policy
        self._name = name

    def get_models(self) -> list[str]:
        return [self._name]

    def resolve(self, model_id: str | None) -> str:
        return model_id if model_id is not None else self._name

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        return self._policy

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'type': 'stub'}


@pytest.fixture
def stub_server(start_server, make_mock_policy) -> tuple[str, int, PolicyServer, MagicMock]:
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, server = start_server(ChunkedSchedule() | remote | _StubSource(policy))
    return host, port, server, policy


def test_full_inference_cycle(stub_server):
    host, port, _server, policy = stub_server
    client = InferenceClient(f'{host}:{port}')
    session = client.new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        assert session.metadata['type'] == 'stub'
        assert session.metadata['local_stack'] == {'name': 'chunked_schedule'}
        assert offboard_keys.POSITRONIC_VERSION in session.metadata

        obs = {'image': 'test'}
        result = session.infer(obs)
        assert result == [{'action': [1, 2, 3]}]
        policy._mock_session.assert_called_with(obs, ANY)
    finally:
        session.close()


def test_no_codec(stub_server):
    host, port, _server, _policy = stub_server
    client = InferenceClient(f'{host}:{port}')
    session = client.new_session()
    try:
        result = session.infer({'obs': 'data'})
        assert result == [{'action': [1, 2, 3]}]
    finally:
        session.close()


@pytest.mark.parametrize(
    'checkpoint_id',
    [
        'my_checkpoint',
        'GEAR-Dreams/DreamZero-DROID',
        's3://bucket/ckpt-1',
        's3://bucket/checkpoint-500/',
        'weird?x#y',
        '100%done',
    ],
)
def test_checkpoint_id_in_route(stub_server, checkpoint_id):
    host, port, _server, _policy = stub_server
    # ``safe='/'`` keeps a path-shaped id's separators as path segments, and encodes the characters that
    # would otherwise end the path (``?``, ``#``) or be decoded away (``%``).
    quoted = urllib.parse.quote(checkpoint_id, safe='/')
    client = InferenceClient(f'{host}:{port}/api/v1/session/{quoted}')
    session = client.new_session()
    try:
        assert session.metadata['checkpoint_id'] == checkpoint_id
    finally:
        session.close()


class _LatestSource(ModelSource):
    """Source whose 'latest' checkpoint can change after startup; ``resolve(None)``
    returns the current latest, mirroring real vendor sources."""

    def __init__(self, policy: Policy):
        self._policy = policy
        self.latest = '100'

    def get_models(self) -> list[str]:
        return [self.latest]

    def resolve(self, model_id: str | None) -> str:
        return model_id if model_id is not None else self.latest

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        return self._policy


def test_latest_checkpoint_pinned_once_at_startup(start_server, make_mock_policy):
    source = _LatestSource(make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'}))
    host, port, _server = start_server(ChunkedSchedule() | remote | source)
    # A newer checkpoint lands after startup (e.g. a training job writes it)...
    source.latest = '200'
    client = InferenceClient(f'{host}:{port}')
    # ...but a default session still serves the checkpoint pinned at startup.
    session = client.new_session()
    try:
        assert session.metadata['checkpoint_id'] == '100'
    finally:
        session.close()
    # Explicit requests still load the named checkpoint.
    session = InferenceClient(f'{host}:{port}/api/v1/session/200').new_session()
    try:
        assert session.metadata['checkpoint_id'] == '200'
    finally:
        session.close()


class _ProgressSource(_StubSource):
    """Reports load progress, so switching models exercises the ``loading`` frame stream."""

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        if on_progress is not None:
            on_progress('halfway there')
        return self._policy


def test_load_progress_frames_reach_the_client(start_server, make_mock_policy):
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, _server = start_server(ChunkedSchedule() | remote | _ProgressSource(policy))
    # Requesting a non-pinned id forces a load inside the handshake; the source's progress
    # callbacks must arrive as ``loading`` frames before ``ready``.
    ws = connect(f'ws://{host}:{port}/api/v1/session/other')
    try:
        frames = []
        while not any(f.get('status') == 'ready' for f in frames):
            frames.append(deserialise(ws.recv(timeout=10)))
        messages = [f.get('message', '') for f in frames if f.get('status') == 'loading']
        assert any('halfway there' in m for m in messages)
    finally:
        ws.close()


class _IdentityCodec(Codec):
    def encode(self, data):
        return data

    def _decode_single(self, data):
        return data

    @property
    def meta(self):
        return {'codec': 'identity'}


@pytest.fixture
def codec_server(start_server, make_mock_policy) -> tuple[str, int, MagicMock]:
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, _server = start_server(ChunkedSchedule() | remote | _IdentityCodec() | _StubSource(policy))
    return host, port, policy


def test_codec_wrapping(codec_server):
    host, port, _policy = codec_server
    client = InferenceClient(f'{host}:{port}')
    session = client.new_session()
    try:
        assert session.metadata['codec'] == 'identity'
        result = session.infer({'obs': 'data'})
        assert result == [{'action': [1, 2, 3]}]
    finally:
        session.close()


def test_warmup_runs_one_inference_and_ends_its_session(make_mock_policy):
    policy = make_mock_policy([{'action': [1, 2, 3]}], {})
    obs = {'obs': 'zeros'}

    warmup(policy, obs)

    policy._mock_session.assert_called_once_with(obs, ANY)
    policy._mock_session.close.assert_called_once()


def test_a_backend_that_cannot_answer_its_warmup_raises_and_still_ends_its_session(make_mock_policy):
    policy = make_mock_policy([], {})
    policy._mock_session.side_effect = RuntimeError('shape mismatch')

    with pytest.raises(RuntimeError, match='shape mismatch'):
        warmup(policy, {})

    policy._mock_session.close.assert_called_once()


def test_local_stack_declared_in_handshake(start_server, make_mock_policy):
    stub = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    pipeline = ChunkedSchedule() | remote | _IdentityCodec() | _StubSource(stub)
    host, port, _server = start_server(pipeline)
    client = InferenceClient(f'{host}:{port}')
    session = client.new_session()
    try:
        assert session.metadata['local_stack'] == {'name': 'chunked_schedule'}
    finally:
        session.close()


def test_pipeline_with_no_rig_side_half_refused_at_startup(make_mock_policy):
    """Nothing left of the marker leaves the rig nothing to run, so the server refuses to serve it."""
    stub = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    with pytest.raises(ValueError, match='no rig-side stack'):
        PolicyServer(remote | _StubSource(stub))


_INFER = 'infer'


class _ScriptedSession(Session):
    def __init__(self, rt: Runtime):
        self._rt = rt
        self._answer = None

    def __call__(self, obs, time_ns):
        if self._answer is None:
            self._answer = self._rt.fns[_INFER](obs)
            return None
        if not self._answer.done():
            return None
        answer, self._answer = self._answer, None
        return answer.result()


class _ScriptedPolicy(Policy):
    """Deterministic base policy: every session serves the same untimestamped chunk from its runtime."""

    def new_session(self, context=None, rt=None) -> Session:
        assert rt is not None
        return _ScriptedSession(rt)

    @property
    def functions(self):
        return {_INFER: lambda obs: [{'a': 1.0}, {'a': 2.0}, {'a': 3.0}]}


def test_in_process_equals_remote_for_same_pipeline(start_server, open_session):
    """The same pipeline must behave identically served in-process and over the wire."""

    def pipeline():
        return ChunkedSchedule() | remote | ActionTimestamp(fps=10.0) | PolicySource(_ScriptedPolicy())

    host, port, _server = start_server(pipeline())
    remote_session, rt = open_session(RemotePolicy(f'{host}:{port}'))

    local_session, local_rt = open_session(inline(pipeline()))

    remote_actions = round_trip(remote_session, rt, {keys.OBS_TIME_NS: 0}, int(100e9))
    local_actions = round_trip(local_session, local_rt, {keys.OBS_TIME_NS: 0}, int(100e9))
    assert remote_actions == local_actions
    # Three scripted actions plus the chunk-closing validity sentinel ActionTimestamp appends.
    assert local_actions == [
        {'a': 1.0, 'timestamp': 100.0},
        {'a': 2.0, 'timestamp': 100.1},
        {'a': 3.0, 'timestamp': 100.2},
        {'timestamp': 100.3},
    ]

    # Both gate identically while the chunk plays out.
    assert remote_session({keys.OBS_TIME_NS: 0}, int(100.15e9)) is None
    assert local_session({keys.OBS_TIME_NS: 0}, int(100.15e9)) is None

    remote_session.close()


def _tunable_pipe(source: ModelSource, offsets: tuple[float, ...] = (-0.1, 0.0), pad_start: bool = True):
    return TemporalStack(keys=('x',), offsets_sec=offsets, pad_start=pad_start) | ChunkedSchedule() | remote | source


def _param_session(host: str, port: int, query: list[tuple[str, str]]) -> InferenceSession:
    uri = f'ws://{host}:{port}/api/v1/session?' + urllib.parse.urlencode(query)
    return InferenceSession(connect(uri))


@pytest.fixture
def param_server(start_server, make_mock_policy) -> Generator[tuple[str, int], None, None]:
    stub = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    pipe_cfg = cfn.Config(_tunable_pipe, source=cfn.Config(_StubSource, policy=stub))
    host, port, _server = start_server(pipe_cfg)
    yield host, port


def test_session_params_override_declared_local_stack(param_server):
    host, port = param_server
    session = _param_session(host, port, [('offsets', '[-0.5, 0.0]')])
    try:
        stack = session.metadata['local_stack']['seq'][0]
        assert stack['name'] == 'temporal_stack'
        assert stack['args']['offsets_sec'] == [-0.5, 0.0]
    finally:
        session.close()


def test_session_params_coerce_json_values(param_server):
    host, port = param_server
    session = _param_session(host, port, [('pad_start', 'false')])
    try:
        assert session.metadata['local_stack']['seq'][0]['args']['pad_start'] is False
    finally:
        session.close()


def _fps_pipe(source: ModelSource, fps: float = 10.0):
    return ChunkedSchedule() | remote | ActionTimestamp(fps=fps) | source


def test_session_param_retunes_the_served_remote_half(start_server):
    pipe_cfg = cfn.Config(_fps_pipe, source=cfn.Config(PolicySource, policy=_ScriptedPolicy()))
    host, port, _server = start_server(pipe_cfg)

    # The wire carries the server-side half's output: relative timestamps spaced 1/fps.
    default_session = _param_session(host, port, [])
    tuned_session = _param_session(host, port, [('fps', '5')])
    try:
        assert [a['timestamp'] for a in default_session.infer({})] == pytest.approx([0.0, 0.1, 0.2, 0.3])
        assert [a['timestamp'] for a in tuned_session.infer({})] == pytest.approx([0.0, 0.2, 0.4, 0.6])
    finally:
        default_session.close()
        tuned_session.close()


def test_model_id_is_named_by_path_not_query(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='model_id'):
        _param_session(host, port, [('model_id', 'other')])

    session = InferenceSession(connect(f'ws://{host}:{port}/api/v1/session/other?pad_start=false'))
    try:
        assert session.metadata['checkpoint_id'] == 'other'
        assert session.metadata['local_stack']['seq'][0]['args']['pad_start'] is False
    finally:
        session.close()


def test_unknown_session_param_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='nonexistent'):
        _param_session(host, port, [('nonexistent', '1')])


def test_import_string_session_params_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='import syntax'):
        _param_session(host, port, [('pad_start', '"@os.system"')])
    # The relative form is no safer: leading dots walk up the module tree from the key's current
    # value, and `source` holds a config, so they resolve.
    with pytest.raises(RuntimeError, match='import syntax'):
        _param_session(host, port, [('source', '".....os.system"')])
    # Nested values are refused too, and the error names the position inside the value.
    with pytest.raises(RuntimeError, match=r'offsets\[0\]'):
        _param_session(host, port, [('offsets', '["@os.getcwd"]')])


def test_dotted_param_is_data_where_it_could_not_import(param_server):
    """A leading-dot value stays a plain string on a key that gives imports no base to resolve
    against — so ordinary path-ish values are usable as data."""
    host, port = param_server
    session = _param_session(host, port, [('pad_start', '"./data"')])
    try:
        assert session.metadata['local_stack']['seq'][0]['args']['pad_start'] == './data'
    finally:
        session.close()


def test_source_touching_session_param_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='fixed at launch'):
        _param_session(host, port, [('source.name', '"other"')])


def test_plain_pipe_server_rejects_session_params(start_server, make_mock_policy):
    stub = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, _server = start_server(_tunable_pipe(_StubSource(stub)))
    with pytest.raises(RuntimeError, match='config-launched'):
        _param_session(host, port, [('pad_start', 'false')])


def test_duplicate_session_param_keys_rejected(param_server):
    host, port = param_server
    with pytest.raises(RuntimeError, match='[Dd]uplicate'):
        _param_session(host, port, [('pad_start', 'false'), ('pad_start', 'true')])


_TOKEN = 'test-secret-token'

# The deployed endpoint the ``endpoint`` marker's tests address. Unset, those tests serve their own server
# and prove its behaviour; set, the same assertions run through whatever ingress fronts that deployment,
# which is the only place the two can disagree.
ENDPOINT_URL_ENV = 'POSITRONIC_ENDPOINT_URL'
_LIVE_ENDPOINT = os.environ.get(ENDPOINT_URL_ENV)


@pytest.fixture
def authed_endpoint(start_server, make_mock_policy) -> tuple[str, str]:
    """An authenticated server's URL, and the token gating it."""
    if _LIVE_ENDPOINT:
        return _LIVE_ENDPOINT, os.environ[AUTH_TOKEN_ENV]
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, _server = start_server(ChunkedSchedule() | remote | _StubSource(policy), auth_token=_TOKEN)
    return f'{host}:{port}', _TOKEN


@pytest.mark.endpoint
@pytest.mark.parametrize(
    'make_header',
    [
        pytest.param(lambda token: None, id='absent'),
        pytest.param(lambda token: bearer(f'not-{token}'), id='wrong-token'),
        pytest.param(lambda token: token, id='no-bearer-prefix'),
    ],
)
def test_auth_rejects_requests_without_the_token(authed_endpoint, make_header, monkeypatch):
    # A 403 buys retries for a backend that may be merely cold. This one is refusing, so those attempts and
    # the waits between them are dead time; `TestNewSessionRetriesRefusedUpgrades` is what tests the budget.
    monkeypatch.setattr(_ConnectRetries, 'MAX_FORBIDDEN_ATTEMPTS', 1)
    url, token = authed_endpoint
    header = make_header(token)
    client = InferenceClient(url, headers=None if header is None else {AUTH_HEADER: header})
    with pytest.raises(InvalidStatus):
        client.new_session()
    with pytest.raises(httpx.HTTPStatusError):
        client.list_models()


@pytest.mark.endpoint
def test_auth_accepts_the_token(authed_endpoint):
    url, token = authed_endpoint
    client = InferenceClient(url, headers={AUTH_HEADER: bearer(token)})
    assert client.list_models()
    session = client.new_session()
    try:
        # Reaching the handshake metadata means the upgrade completed and the server's first frame arrived.
        # An ingress that drops ``Upgrade`` never gets that far: it answers the handshake with a plain 200.
        assert offboard_keys.POSITRONIC_VERSION in session.metadata
    finally:
        session.close()


# A managed ingress closes a connection it has read nothing from — Nebius' does after ~90s, shorter than one
# cold inference. Nothing crosses the wire while a session waits for actions except the client's keepalive
# pings, so idling past that window and still getting a pong is those pings doing their job.
_IDLE_WINDOW_SEC = 120.0


@pytest.mark.endpoint
@pytest.mark.skipif(not _LIVE_ENDPOINT, reason=f'no ingress to idle against; set {ENDPOINT_URL_ENV}')
def test_session_outlives_an_idle_ingress_window(authed_endpoint):
    url, token = authed_endpoint
    session = InferenceClient(url, headers={AUTH_HEADER: bearer(token)}).new_session()
    try:
        time.sleep(_IDLE_WINDOW_SEC)
        assert session._websocket.ping().wait(timeout=30.0)
    finally:
        session.close()


def test_server_without_a_token_serves_open(stub_server):
    host, port, _server, _policy = stub_server
    assert InferenceClient(f'{host}:{port}').list_models() == ['stub']


@pytest.mark.parametrize(
    'token',
    [
        pytest.param('', id='empty'),
        pytest.param('tökén', id='non-ascii'),
        # What a secret read from a file that ends in one looks like.
        pytest.param('a-token\n', id='trailing-newline'),
        pytest.param('a token', id='space'),
    ],
)
def test_a_token_that_could_never_gate_fails_closed_at_startup(make_mock_policy, token):
    with pytest.raises(ValueError, match='ASCII'):
        PolicyServer(ChunkedSchedule() | remote | _StubSource(make_mock_policy([], {})), auth_token=token)


def test_a_non_ascii_authorization_header_is_refused_rather_than_crashing(start_server, make_mock_policy):
    """A header carries bytes, and Starlette hands them over latin-1 decoded, so a peer can put a
    non-ASCII ``str`` in front of the token comparison."""
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, _server = start_server(ChunkedSchedule() | remote | _StubSource(policy), auth_token=_TOKEN)
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall(
            b'GET /api/v1/models HTTP/1.1\r\nHost: localhost\r\n'
            b'Authorization: Bearer t\xf6ken\r\nConnection: close\r\n\r\n'
        )
        status = sock.recv(64).split(b' ')[1]
    assert status == b'401'
