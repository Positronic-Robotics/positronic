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
from websockets.sync.client import connect

from pimm.logging import init_logging
from positronic.offboard.server import serve
from positronic.offboard.server_utils import wait_for_subprocess_ready
from positronic.policy import Codec, Policy, Session
from positronic.policy import keys as policy_keys
from positronic.policy.base import Answer, Runtime
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.policy.spec import ModelSource, remote
from positronic.utils.serialization import serialize
from positronic.vendors.galaxea import codecs, protocol


class _GalaxeaSession(Session):
    def __init__(self, url: str, timeout: float, rt: Runtime):
        self._rt = rt
        self._timeout = timeout
        self._answer: Answer | None = None
        self._cancelled = False
        self._connection = connect(url, compression=None, max_size=None)
        try:
            self._meta = msgpack.unpackb(self._connection.recv(timeout=timeout))
            if self._meta.get(protocol.PROTOCOL) != protocol.FULL_CHUNK_V1:
                raise ValueError('Expected the Galaxea full-chunk backend; see vendors/galaxea/README.md')
        except Exception:
            self._connection.close()
            raise

    @staticmethod
    def infer(connection, obs, timeout: float) -> list[dict[str, Any]]:
        try:
            connection.send(serialize(obs))
            response = msgpack.unpackb(connection.recv(timeout=timeout))
        except Exception:
            connection.close()
            raise
        if protocol.ERROR in response:
            raise RuntimeError(f'G0.5 inference failed: {response[protocol.ERROR]}')
        actions = response[protocol.ACTIONS]
        if not isinstance(actions, list) or not actions or any(not isinstance(step, dict) for step in actions):
            raise ValueError('G0.5 must return a nonempty list of action dictionaries')
        return actions

    def __call__(self, obs, time_ns):
        if self._answer is None:
            self._answer = self._rt.fns['infer'](self._connection, obs, self._timeout)
            return None
        if not self._answer.done():
            return None
        answer, cancelled = self._answer, self._cancelled
        self._answer, self._cancelled = None, False
        actions = answer.result()
        return None if cancelled else actions

    def cancel(self):
        self._cancelled = self._answer is not None

    @property
    def meta(self):
        return self._meta

    def close(self):
        assert self._answer is None or self._answer.done(), 'Close the runtime before its session'
        self._connection.close()


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


class GalaxeaPolicy(Policy):
    def __init__(self, backend: _BackendProcess, infer_timeout: float):
        self._backend = backend
        self._timeout = infer_timeout

    @property
    def functions(self):
        return {'infer': _GalaxeaSession.infer}

    def new_session(self, context=None, rt=None):
        if rt is None:
            raise ValueError('GalaxeaPolicy requires a runtime for inference')
        return _GalaxeaSession(self._backend.url, self._timeout, rt)

    def close(self):
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

    def get_models(self) -> list[str]:
        return [protocol.MODEL_ID]

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        if model_id != protocol.MODEL_ID:
            raise ValueError(f'Unknown Galaxea model: {model_id}')
        backend = _BackendProcess(self._root, self._checkpoint, self._device, self._port)
        try:
            backend.start(on_progress)
        except Exception:
            backend.stop()
            raise
        return GalaxeaPolicy(backend, self._timeout)

    def meta(self, model_id: str) -> dict[str, Any]:
        return {
            'model_id': model_id,
            policy_keys.CHECKPOINT_PATH: str(self._checkpoint),
            'usage': 'internal non-commercial evaluation only',
        }


@cfn.config(codec=codecs.droid, source=cfn.Config(GalaxeaSource))
def pipeline(codec: Codec, source: ModelSource):
    # TODO: Add an opt-in local layer for RoboArena's missing-gripper behavior. Capture the measured
    # grip per inference request, with state isolated per session, and fill omitted targets in its chunk.
    # Keep preserving the previous target as the default; this state belongs in the layer, not the codec.
    return StopOnFault() | ChunkedSchedule() | RestrictImageSize() | remote | codec | source


COMMANDS = {name: serve.override(pipeline=pipeline) for name in ('', 'serve', 'droid')}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
