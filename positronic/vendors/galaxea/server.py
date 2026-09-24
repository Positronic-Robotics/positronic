"""Positronic policy server for internal, non-commercial G0.5-DROID evaluation only."""

import os
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import msgpack
import pos3
from websockets.sync.client import ClientConnection, connect

from pimm.logging import init_logging
from positronic.offboard.server import serve
from positronic.offboard.server_utils import wait_for_subprocess_ready
from positronic.offboard.spec import Model, ModelSource, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable
from positronic.utils.serialization import serialize
from positronic.vendors.galaxea import codecs, protocol

PYTHONPATH = 'PYTHONPATH'
VIRTUAL_ENV = 'VIRTUAL_ENV'
PATH = 'PATH'


class _BackendProcess:
    """Own the full-chunk model process in Galaxea's isolated interpreter."""

    def __init__(self, root: Path, checkpoint: Path, device: str, port: int):
        self._root = root.resolve(strict=True)
        self._checkpoint = checkpoint.absolute()
        self._python = self._root / '.venv/bin/python'
        for path in (self._checkpoint, self._python):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._device = device
        self._port = port
        self.url = f'ws://127.0.0.1:{port}'
        self._process: subprocess.Popen | None = None

    def _ready(self) -> bool:
        try:
            with connect(self.url, compression=None, open_timeout=1) as connection:
                metadata = msgpack.unpackb(connection.recv(timeout=1))
        except (OSError, TimeoutError):
            return False
        if metadata.get(protocol.PROTOCOL) != protocol.FULL_CHUNK_V1:
            raise ValueError('Backend port is occupied by an incompatible server')
        return True

    def _crashed(self) -> tuple[bool, int | None]:
        assert self._process is not None
        code = self._process.poll()
        return code is not None, code

    def start(self, on_progress: Callable[[str], None] | None):
        # Refuse an occupied port before launching a second model into GPU memory.
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', self._port))
        env = os.environ.copy()
        env[PYTHONPATH] = os.pathsep.join((str(self._root / 'src'), str(self._root), str(Path(__file__).parents[1])))
        env[VIRTUAL_ENV] = str(self._root / '.venv')
        env[PATH] = str(self._root / '.venv/bin') + os.pathsep + env.get(PATH, '')
        self._process = subprocess.Popen(
            [
                str(self._python),
                '-m',
                'galaxea.backend',
                '--checkpoint',
                str(self._checkpoint),
                '--device',
                self._device,
                '--port',
                str(self._port),
            ],
            cwd=self._root,
            env=env,
        )
        wait_for_subprocess_ready(self._ready, self._crashed, 'Galaxea model', on_progress, max_wait=1800)

    def stop(self):
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None


class GalaxeaModel(Model):
    """Own the full-chunk backend and each session's connection."""

    def __init__(self, backend: _BackendProcess, infer_timeout: float, meta: dict[str, Any]):
        self._backend = backend
        self._timeout = infer_timeout
        self._meta = meta
        self._connections: dict[str, ClientConnection] = {}

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        if session_id not in self._connections:
            connection = connect(self._backend.url, compression=None, max_size=None)
            try:
                handshake = msgpack.unpackb(connection.recv(timeout=self._timeout))
                if handshake.get(protocol.PROTOCOL) != protocol.FULL_CHUNK_V1:
                    raise ValueError('Expected the Galaxea full-chunk backend; see vendors/galaxea/README.md')
            except Exception:
                connection.close()
                raise
            self._connections[session_id] = connection
        connection = self._connections[session_id]
        try:
            connection.send(serialize(obs))
            response = msgpack.unpackb(connection.recv(timeout=self._timeout))
        except Exception:
            self.end_session(session_id)
            raise
        if protocol.ERROR in response:
            raise RuntimeError(f'G0.5 inference failed: {response[protocol.ERROR]}')
        actions = response[protocol.ACTIONS]
        if not isinstance(actions, list) or not actions or any(not isinstance(step, dict) for step in actions):
            raise ValueError('G0.5 must return a nonempty list of action dictionaries')
        return actions

    def end_session(self, session_id: str) -> None:
        connection = self._connections.pop(session_id, None)
        if connection is not None:
            connection.close()

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self):
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()
        self._backend.stop()


class GalaxeaSource(ModelSource):
    """Load G0.5-DROID in its own Python 3.10 process and expose full-chunk inference."""

    def __init__(
        self,
        checkpoint_path: str = '/galaxea/checkpoints/g05-droid/checkpoints/model_state_dict.pt',
        galaxea_root: str = '/galaxea',
        device: str = 'cuda',
        backend_port: int = 9000,
        infer_timeout: float = 120.0,
    ):
        self._checkpoint = Path(checkpoint_path)
        self._root = Path(galaxea_root)
        self._device = device
        self._port = backend_port
        self._timeout = infer_timeout

    def checkpoint_id(self) -> str:
        return protocol.MODEL_ID

    def load(self, checkpoint_id: str, on_progress: Callable[[str], None] | None = None) -> Model:
        backend = _BackendProcess(self._root, self._checkpoint, self._device, self._port)
        try:
            backend.start(on_progress)
        except Exception:
            backend.stop()
            raise
        return GalaxeaModel(
            backend,
            self._timeout,
            {policy_keys.CHECKPOINT_PATH: str(self._checkpoint), 'usage': 'internal non-commercial evaluation only'},
        )


@cfn.config(codec=cfn.Config(codecs.DroidCodec), source=cfn.Config(GalaxeaSource))
def pipeline(codec: codecs.DroidCodec, source: ModelSource, execution_steps: int = 16):
    return PolicyDeployment(
        source,
        Sequential(PauseOnUnavailable(), ChunkedSchedule(codec.fps, execution_steps / codec.fps), RestrictImageSize()),
        codecs.droid(action=codec),
    )


COMMANDS = {name: serve.override(pipeline=pipeline) for name in ('', 'serve', 'droid')}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
