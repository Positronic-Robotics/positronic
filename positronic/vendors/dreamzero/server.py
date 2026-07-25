"""DreamZero inference server: the roboarena subprocess model source and its serving pipes."""

import logging
import os
import socket
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import pos3
import websockets.sync.client
from huggingface_hub import snapshot_download
from websockets.exceptions import ConnectionClosed

from positronic.offboard.server import PolicyServer
from positronic.offboard.server_utils import run_with_progress, wait_for_subprocess_ready
from positronic.policy import Codec, Policy, PolicyWrapper, Session
from positronic.policy.codec import RestrictImageSize
from positronic.policy.spec import ModelSource, remote
from positronic.utils.checkpoints import get_latest_checkpoint
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialize, serialize
from positronic.vendors.dreamzero import codecs

logger = logging.getLogger(__name__)


def _dreamzero_root():
    return Path(__file__).parents[4] / 'dreamzero'


def _download_checkpoint(model_path: str) -> Path:
    """Local checkpoint dir for ``model_path``: an ``s3://`` URL or local path via pos3, else a HuggingFace repo."""
    local = os.path.expanduser(model_path)
    if model_path.startswith('s3://') or os.path.exists(local):
        return pos3.download(local)
    return Path(snapshot_download(model_path))


def _resolve_checkpoint_path(model_path: str) -> str:
    """Latest ``checkpoint-N`` under an ``s3://`` run directory; a pinned checkpoint dir, a HuggingFace repo,
    or a local path is returned unchanged."""
    last = model_path.rstrip('/').split('/')[-1]
    if not model_path.startswith('s3://') or last.startswith('checkpoint-'):
        return model_path
    return f'{model_path.rstrip("/")}/{get_latest_checkpoint(model_path, "checkpoint-")}'


# TODO: Extract RoboarenaClient to positronic/offboard/ — roboarena is a cross-vendor
# standard (used by DreamZero, potentially GR00T N2, etc.) and other vendors may need it.
class RoboarenaClient:
    """Client for DreamZero's roboarena WebSocket server.

    Protocol (from eval_utils/policy_server.py + policy_client.py):
    - On connect: server sends PolicyServerConfig as first msgpack message
    - Client sends obs dict with obs["endpoint"] = "infer" or "reset"
    - Server responds with action as raw numpy array (N, 8) via msgpack
    - Uses positronic.utils.serialization for msgpack+numpy wire format
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        self._host = host
        self._port = port
        self._ws = None
        self._server_metadata: dict | None = None

    def connect(self):
        self._ws = websockets.sync.client.connect(
            f'ws://{self._host}:{self._port}', compression=None, max_size=None, ping_interval=60, ping_timeout=600
        )
        # First message from server is PolicyServerConfig metadata
        self._server_metadata = deserialize(self._ws.recv())
        logger.info(f'Connected to roboarena server, metadata: {self._server_metadata}')

    def ping(self) -> bool:
        """Check if the roboarena server port is accepting connections.

        The eval_utils.policy_server.WebsocketPolicyServer has no HTTP health
        endpoint, so we use a raw TCP connect check instead.
        """
        try:
            with socket.create_connection((self._host, self._port), timeout=2):
                return True
        except OSError:
            return False

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        if self._ws is None:
            self.connect()
        observation['endpoint'] = 'infer'
        self._ws.send(serialize(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Server error: {response}')
        return deserialize(response)

    def reset(self, session_id: str | None = None):
        if self._ws is None:
            return
        msg: dict[str, Any] = {'endpoint': 'reset'}
        if session_id is not None:
            msg['session_id'] = session_id
        self._ws.send(serialize(msg))
        self._ws.recv(timeout=10.0)  # Consume "reset successful" response

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None


class DreamZeroSubprocess:
    # wan2.1 (14B): socket_test_optimized_AR.py — uses --enable-dit-cache
    # wan2.2 (5B):  eval_utils/serve_dreamzero_wan22.py — causal chunked inference
    _BACKBONE_SCRIPTS = {'wan2.1': 'socket_test_optimized_AR.py', 'wan2.2': 'eval_utils/serve_dreamzero_wan22.py'}

    def __init__(
        self,
        model_path: str,
        dreamzero_venv: Path,
        backbone: str = 'wan2.1',
        num_gpus: int = 1,
        roboarena_port: int = 9000,
        enable_dit_cache: bool = True,
    ):
        self.model_path = model_path
        self.dreamzero_venv = dreamzero_venv
        self.backbone = backbone
        self.num_gpus = num_gpus
        self.roboarena_port = roboarena_port
        self.enable_dit_cache = enable_dit_cache
        self.process: subprocess.Popen | None = None

    def _build_command(self) -> list[str]:
        root = _dreamzero_root()
        torchrun = str(self.dreamzero_venv / 'bin' / 'torchrun')
        script = self._BACKBONE_SCRIPTS.get(self.backbone, self._BACKBONE_SCRIPTS['wan2.1'])
        command = [
            torchrun,
            f'--nproc_per_node={self.num_gpus}',
            str(root / script),
            '--port',
            str(self.roboarena_port),
            '--model-path',
            self.model_path,
        ]
        if self.backbone == 'wan2.1' and self.enable_dit_cache:
            command.append('--enable-dit-cache')
        return command

    def _launch(self):
        command = self._build_command()
        logger.info(f'Starting DreamZero subprocess: {" ".join(command)}')
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(self.dreamzero_venv)
        env['PATH'] = f'{self.dreamzero_venv / "bin"}:{env.get("PATH", "")}'
        env['TORCH_COMPILE_DISABLE'] = '1'
        self.process = subprocess.Popen(command, env=env, cwd=str(_dreamzero_root()))

    def _check_crashed(self) -> tuple[bool, int | None]:
        if self.process is None:
            return False, None
        exit_code = self.process.poll()
        return exit_code is not None, exit_code

    def start(self, on_progress: Callable[[str], None] | None = None):
        self._launch()
        client = RoboarenaClient(port=self.roboarena_port)
        wait_for_subprocess_ready(
            check_ready=client.ping,
            check_crashed=self._check_crashed,
            description='DreamZero subprocess',
            on_progress=on_progress,
            max_wait=1200.0,
        )

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class _DreamZeroSession(Session):
    def __init__(self, client: RoboarenaClient, session_id: str):
        self._client = client
        self._session_id = session_id

    def __call__(self, obs):
        obs = dict(obs)
        obs['session_id'] = self._session_id
        action_array = np.asarray(self._client.infer(obs))

        # Response is (N, 8) — 7 joints + 1 gripper
        if action_array.ndim == 1:
            return [{'action': action_array}]
        return [{'action': action_array[i]} for i in range(action_array.shape[0])]

    def close(self):
        try:
            self._client.reset(session_id=self._session_id)
        except (OSError, TimeoutError, ConnectionClosed):
            logger.info('DreamZero session reset skipped: backend connection already gone')
        finally:
            self._client.close()


class DreamZeroPolicy(Policy):
    """Owns the DreamZero subprocess; every session talks to it over its own roboarena connection."""

    def __init__(self, sp: DreamZeroSubprocess):
        self._subprocess = sp

    def new_session(self, context=None, now=None):
        client = RoboarenaClient(port=self._subprocess.roboarena_port)
        client.connect()
        return _DreamZeroSession(client, str(uuid.uuid4()))

    def close(self):
        self._subprocess.stop()


class DreamZeroSource(ModelSource):
    """DreamZero checkpoints served through a torchrun subprocess speaking the roboarena protocol.

    ``model_path`` is an ``s3://`` run directory (served at its latest ``checkpoint-N``), a pinned
    checkpoint dir, a HuggingFace repo, or a local path.
    """

    def __init__(
        self,
        model_path: str,
        dreamzero_venv: str = '/.venv/',
        backbone: str = 'wan2.1',
        num_gpus: int = 1,
        roboarena_port: int = 1234,
        enable_dit_cache: bool = True,
    ):
        self._model_path = model_path
        self._dreamzero_venv = Path(dreamzero_venv)
        self._backbone = backbone
        self._num_gpus = num_gpus
        self._roboarena_port = roboarena_port
        self._enable_dit_cache = enable_dit_cache

    def get_models(self) -> list[str]:
        return [_resolve_checkpoint_path(self._model_path)]

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        local_path = run_with_progress(
            lambda: _download_checkpoint(model_id), 'Downloading DreamZero checkpoint', on_progress
        )
        logger.info(f'Starting DreamZero subprocess with {self._num_gpus} GPUs')
        sp = DreamZeroSubprocess(
            model_path=str(local_path),
            dreamzero_venv=self._dreamzero_venv,
            backbone=self._backbone,
            num_gpus=self._num_gpus,
            roboarena_port=self._roboarena_port,
            enable_dit_cache=self._enable_dit_cache,
        )
        try:
            sp.start(on_progress)
        except Exception:
            sp.stop()
            raise
        return DreamZeroPolicy(sp)

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'type': 'dreamzero', 'backbone': self._backbone, 'model_path': model_id, 'num_gpus': self._num_gpus}


dreamzero_source = cfn.Config(DreamZeroSource)


@cfn.config(local=codecs.dreamzero_wrappers, codec=codecs.joints, source=dreamzero_source, width=320, height=176)
def pipe(local: PolicyWrapper, codec: Codec, source: ModelSource, width: int, height: int):
    """One DreamZero serving pipe: the rig-side AR video context, the codec, the subprocess-backed source.

    ``width``/``height`` bound frames on the rig and follow the codec's own geometry.
    """
    return local | RestrictImageSize(width, height) | remote | codec | source


PIPES = {
    'joints': pipe,
    'joints_traj': pipe.override(codec=codecs.joints_traj),
    'joints_ik': pipe.override(codec=codecs.joints_ik),
    'joints_ik_sim': pipe.override(codec=codecs.joints_ik_sim),
    # The pretrained DROID model (wan2.1) asserts exactly 320x180 frames.
    'droid': pipe.override(codec=codecs.droid, height=180),
}


@cfn.config(
    pipe='joints',
    dreamzero_venv='/.venv/',
    backbone='wan2.1',
    num_gpus=1,
    host='0.0.0.0',
    port=8000,
    enable_dit_cache=True,
    recording_dir=None,
    idle_timeout_min=None,
)
def main(
    pipe: str,
    model_path: str,
    dreamzero_venv: str,
    backbone: str,
    num_gpus: int,
    host: str,
    port: int,
    enable_dit_cache: bool,
    recording_dir: str | None,
    idle_timeout_min: float | None,
):
    """Starts the DreamZero inference server."""
    cfg = PIPES[pipe].override(**{
        'source.model_path': model_path,
        'source.dreamzero_venv': dreamzero_venv,
        'source.backbone': backbone,
        'source.num_gpus': num_gpus,
        'source.enable_dit_cache': enable_dit_cache,
    })
    with pos3.mirror():
        PolicyServer(cfg, host=host, port=port, recording_dir=recording_dir, idle_timeout_min=idle_timeout_min).serve()


# Public pretrained DROID checkpoint: wan2.1 backbone (the base default) paired with the DROID
# pipe whose codec feeds its required 320x180 frames.
droid = main.override(pipe='droid', model_path='GEAR-Dreams/DreamZero-DROID')


if __name__ == '__main__':
    init_logging()
    cfn.cli({'serve': main, 'droid': droid})
