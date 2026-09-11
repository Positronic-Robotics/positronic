import runpy
import threading
from dataclasses import dataclass, field
from unittest.mock import Mock

import msgpack
import numpy as np
import pos3
import pytest
from websockets.sync.server import serve

from positronic import keys
from positronic.policy import keys as policy_keys
from positronic.policy.executor import Executor, blocking
from positronic.policy.spec import split
from positronic.utils.serialization import deserialize
from positronic.vendors.galaxea import codecs, protocol, server


def test_cli_supports_recording_directory(tmp_path, monkeypatch):
    output = tmp_path / 'recordings'

    def record(_commands):
        local = pos3.sync(str(output))
        local.mkdir(parents=True, exist_ok=True)
        (local / 'recording.txt').write_text('recorded')

    monkeypatch.setattr(server.cfn, 'cli', record)
    monkeypatch.setattr('pimm.logging.init_logging', lambda: None)
    runpy.run_path(server.__file__, run_name='__main__')
    assert (output / 'recording.txt').read_text() == 'recorded'


@dataclass
class Backend:
    url: str = ''
    requests: list = field(default_factory=list)
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    metadata: dict = field(default_factory=lambda: {protocol.PROTOCOL: protocol.FULL_CHUNK_V1})
    response: dict = field(
        default_factory=lambda: protocol.chunk_response({
            protocol.RIGHT_ARM: np.arange(32 * 7).reshape(32, 7),
            protocol.RIGHT_GRIPPER: np.zeros((32, 1)),
        })
    )

    def stop(self):
        pass

    def handle(self, connection):
        connection.send(msgpack.packb(self.metadata))
        for message in connection:
            self.requests.append(deserialize(message))
            self.entered.set()
            assert self.release.wait(5)
            connection.send(msgpack.packb(self.response))


@pytest.fixture
def backend():
    backend = Backend()
    backend.release.set()
    with serve(backend.handle, '127.0.0.1', 0) as websocket_server:
        backend.url = f'ws://127.0.0.1:{websocket_server.socket.getsockname()[1]}'
        thread = threading.Thread(target=websocket_server.serve_forever)
        thread.start()
        try:
            yield backend
        finally:
            backend.release.set()
            websocket_server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def test_live_adapter_returns_full_chunk_with_server_side_gripper_conversion(backend):
    codec = codecs.droid()
    policy = codec.wrap(server.GalaxeaPolicy(backend, 5))
    session = blocking(policy).new_session()
    obs = {
        keys.JOINTS: np.zeros(7),
        keys.GRIP: 0.3,
        keys.EXTERIOR_IMAGE: np.zeros((2, 3, 3), dtype=np.uint8),
        keys.WRIST_IMAGE: np.zeros((2, 3, 3), dtype=np.uint8),
        keys.TASK: 'pick towel',
    }
    try:
        assert session.meta == codec.meta
        actions = session(obs, 0)
        assert len(actions) == 33
        assert actions[-1] == {keys.ACTION_TIMESTAMP: 32 / 15}
        np.testing.assert_array_equal(actions[-2][keys.ROBOT_COMMAND].positions, np.arange(217, 224))
        assert all(step[keys.TARGET_GRIP] == 1 for step in actions[:-1])
        assert len(backend.requests) == 1
        np.testing.assert_allclose(backend.requests[0][protocol.STATE][protocol.RIGHT_GRIPPER], [0.7])
    finally:
        session.close()


def test_stock_step_server_is_rejected(backend):
    backend.metadata = {'action_steps': 16}
    policy = blocking(server.GalaxeaPolicy(backend, 5))
    with pytest.raises(ValueError, match='full-chunk backend'):
        policy.new_session()


@pytest.mark.parametrize('response', [{protocol.ERROR: 'missing arm'}, {protocol.ACTIONS: []}, {protocol.ACTIONS: [0]}])
def test_backend_errors_surface(backend, response):
    backend.response = response
    session = blocking(server.GalaxeaPolicy(backend, 5)).new_session()
    try:
        with pytest.raises((RuntimeError, ValueError)):
            session({}, 0)
    finally:
        session.close()


def test_cancellation_discards_pending_chunk_and_next_call_recomputes(backend):
    policy = server.GalaxeaPolicy(backend, 5)
    rt = Executor(policy.functions)
    session = policy.new_session(rt=rt)
    backend.release.clear()
    try:
        assert session({protocol.TASK: 'old'}, 0) is None
        assert backend.entered.wait(5)
        session.cancel()
        backend.release.set()
        rt.wait(timeout=5)
        assert session({protocol.TASK: 'new'}, 1) is None
        assert not rt.owes_an_answer
        assert session({protocol.TASK: 'new'}, 2) is None
        rt.wait(timeout=5)
        assert len(session({}, 3)) == 32
        assert [obs[protocol.TASK] for obs in backend.requests] == ['old', 'new']
    finally:
        backend.release.set()
        rt.close()
        session.close()


def test_inference_timeout_closes_connection():
    class Connection:
        closed = False

        def send(self, data):
            pass

        def recv(self, timeout):
            raise TimeoutError('late response')

        def close(self):
            self.closed = True

    connection = Connection()
    with pytest.raises(TimeoutError):
        server._GalaxeaSession.infer(connection, {}, 1)
    assert connection.closed


def test_vendor_codec_stays_on_server_side():
    pipeline = server.pipeline()
    local, border, remote_half = split(pipeline)
    assert isinstance(pipeline.source, server.GalaxeaSource)
    assert remote_half is not None
    assert 'DroidCodec' not in str(local.to_spec())
    assert (
        remote_half.encode({
            keys.JOINTS: np.zeros(7),
            keys.GRIP: 0,
            keys.EXTERIOR_IMAGE: np.zeros((1, 1, 3), dtype=np.uint8),
            keys.WRIST_IMAGE: np.zeros((1, 1, 3), dtype=np.uint8),
            keys.TASK: '',
        })[protocol.STATE][protocol.RIGHT_GRIPPER]
        == 1
    )


def test_model_load_failure_stops_child(monkeypatch):
    backend = Mock()
    backend.start.side_effect = RuntimeError('model failed to load')
    monkeypatch.setattr(server, '_BackendProcess', Mock(return_value=backend))
    with pytest.raises(RuntimeError, match='model failed to load'):
        server.GalaxeaSource().load(protocol.MODEL_ID)
    backend.stop.assert_called_once()


def test_loaded_policy_metadata_and_cleanup(tmp_path, monkeypatch):
    backend = Mock()
    monkeypatch.setattr(server, '_BackendProcess', Mock(return_value=backend))
    progress = Mock()
    checkpoint = tmp_path / 'model_state_dict.pt'
    source = server.GalaxeaSource(checkpoint_path=str(checkpoint))
    policy = source.load(protocol.MODEL_ID, progress)
    assert source.meta(protocol.MODEL_ID)[policy_keys.CHECKPOINT_PATH] == str(checkpoint)
    backend.start.assert_called_once_with(progress)
    backend.stop.assert_not_called()
    policy.close()
    backend.stop.assert_called_once()


def test_subprocess_preserves_venv_and_checkpoint_symlinks(tmp_path, monkeypatch):
    executable = tmp_path / 'python'
    executable.touch()
    interpreter = tmp_path / '.venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(executable)
    weights = tmp_path / 'blob'
    weights.touch()
    checkpoint = tmp_path / 'model_state_dict.pt'
    checkpoint.symlink_to(weights)
    popen = Mock()
    monkeypatch.setattr(server.subprocess, 'Popen', popen)
    monkeypatch.setattr(server, 'wait_for_subprocess_ready', Mock())
    backend = server._BackendProcess(tmp_path, checkpoint, 'cuda', 0)
    backend.start(None)
    command = popen.call_args.args[0]
    assert command[0] == str(interpreter)
    assert command[command.index('--checkpoint') + 1] == str(checkpoint)
    assert popen.call_args.kwargs['cwd'] == tmp_path
    assert popen.call_args.kwargs['env'][server.VIRTUAL_ENV] == str(tmp_path / '.venv')
    backend.stop()
    popen.return_value.terminate.assert_called_once()
    popen.return_value.wait.assert_called_once_with(timeout=10)
