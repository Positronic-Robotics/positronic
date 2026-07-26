import urllib.parse
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import MagicMock

import configuronic as cfn
import pytest
from websockets.sync.client import connect

from positronic.offboard.client import InferenceClient, InferenceSession
from positronic.offboard.server import PolicyServer
from positronic.policy import Codec, Policy, RemotePolicy, Session
from positronic.policy.codec import ActionTimestamp
from positronic.policy.spec import ModelSource, PolicySource, inline, remote
from positronic.policy.wrappers import ChunkedSchedule, TemporalStack
from positronic.utils.serialization import deserialise


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
    host, port, server = start_server(remote | _StubSource(policy))
    return host, port, server, policy


def test_full_inference_cycle(stub_server):
    host, port, _server, policy = stub_server
    client = InferenceClient(f'{host}:{port}')
    session = client.new_session()
    try:
        assert session.metadata['model_name'] == 'stub'
        assert session.metadata['type'] == 'stub'
        assert session.metadata['local_stack'] == {'seq': []}
        assert 'positronic_version' in session.metadata

        obs = {'image': 'test'}
        result = session.infer(obs)
        assert result == [{'action': [1, 2, 3]}]
        policy._mock_session.assert_called_with(obs)
    finally:
        session.close()


def test_no_codec(stub_server):
    host, port, server, _policy = stub_server
    assert server._remote is None

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
    host, port, _server = start_server(remote | source)
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
    host, port, _server = start_server(remote | _ProgressSource(policy))
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

    def _decode_single(self, data, context):
        return data

    @property
    def meta(self):
        return {'codec': 'identity'}

    def dummy_encoded(self, data=None):
        return data or {}


@pytest.fixture
def codec_server(start_server, make_mock_policy) -> tuple[str, int, MagicMock]:
    policy = make_mock_policy([{'action': [1, 2, 3]}], {'model_name': 'stub'})
    host, port, _server = start_server(remote | _IdentityCodec() | _StubSource(policy))
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


def test_warmup_runs_dummy_inference_at_startup(codec_server):
    _host, _port, policy = codec_server
    # Startup warmed the pinned model up through the codec's dummy_encoded() before serving.
    policy._mock_session.assert_called_once_with({})


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


class _ScriptedSession(Session):
    def __call__(self, obs):
        return [{'a': 1.0}, {'a': 2.0}, {'a': 3.0}]


class _ScriptedPolicy(Policy):
    """Deterministic base policy: every session returns the same untimestamped chunk."""

    def new_session(self, context=None, now=None) -> Session:
        return _ScriptedSession()


def test_in_process_equals_remote_for_same_pipeline(start_server):
    """The same pipeline must behave identically served in-process and over the wire."""

    def pipeline():
        return ChunkedSchedule() | remote | ActionTimestamp(fps=10.0) | PolicySource(_ScriptedPolicy())

    clock = [100.0]

    host, port, _server = start_server(pipeline())
    remote_session = RemotePolicy(f'{host}:{port}').new_session(now=lambda: clock[0])

    local_session = inline(pipeline()).new_session(now=lambda: clock[0])

    remote_actions = remote_session({'obs_time_ns': 0})
    local_actions = local_session({'obs_time_ns': 0})
    assert remote_actions == local_actions
    # Three scripted actions plus the chunk-closing validity sentinel ActionTimestamp appends.
    assert local_actions == [
        {'a': 1.0, 'timestamp': 100.0},
        {'a': 2.0, 'timestamp': 100.1},
        {'a': 3.0, 'timestamp': 100.2},
        {'timestamp': 100.3},
    ]

    # Both gate identically while the chunk plays out.
    clock[0] = 100.15
    assert remote_session({'obs_time_ns': 0}) is None
    assert local_session({'obs_time_ns': 0}) is None

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
