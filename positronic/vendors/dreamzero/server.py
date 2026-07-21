"""DreamZero inference server: subprocess management + FastAPI bridge."""

import asyncio
import logging
import os
import socket
import subprocess
import uuid
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import pos3
from fastapi import WebSocket
from huggingface_hub import snapshot_download

from positronic.cfg import codecs as cfg_codecs
from positronic.offboard.server_utils import monitor_async_task, wait_for_subprocess_ready
from positronic.offboard.vendor_server import VendorServer
from positronic.policy import Policy, PolicyWrapper, Session
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
        import websockets.sync.client

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
        self._ws.recv()  # Consume "reset successful" response

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

    async def start_async(self, on_progress=None):
        self._launch()
        client = RoboarenaClient(port=self.roboarena_port)
        await wait_for_subprocess_ready(
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


class DreamZeroPolicy(Policy):
    def __init__(self, client: RoboarenaClient):
        self._client = client
        self._prev_session_id: str | None = None

    def new_session(self, context=None, now=None):
        if self._prev_session_id is not None:
            self._client.reset(session_id=self._prev_session_id)
        session_id = str(uuid.uuid4())
        self._prev_session_id = session_id
        return _DreamZeroSession(self._client, session_id)


class InferenceServer(VendorServer):
    def __init__(
        self,
        definition: PolicyWrapper,
        model_path: str,
        dreamzero_venv: str = '/.venv/',
        backbone: str = 'wan2.1',
        num_gpus: int = 1,
        host: str = '0.0.0.0',
        port: int = 8000,
        roboarena_port: int = 1234,
        enable_dit_cache: bool = True,
        recording_dir: str | None = None,
        idle_timeout_min: float | None = None,
    ):
        super().__init__(
            definition=definition, host=host, port=port, recording_dir=recording_dir, idle_timeout_min=idle_timeout_min
        )
        self.model_path = _resolve_checkpoint_path(model_path)
        self.dreamzero_venv = Path(dreamzero_venv)
        self.backbone = backbone
        self.num_gpus = num_gpus
        self.roboarena_port = roboarena_port
        self.enable_dit_cache = enable_dit_cache
        self.subprocess: DreamZeroSubprocess | None = None
        self._subprocess_lock = asyncio.Lock()

        self.metadata = {
            'host': host,
            'port': port,
            'type': 'dreamzero',
            'backbone': backbone,
            'model_path': self.model_path,
            'num_gpus': num_gpus,
        }

    async def resolve_model(self, model_id: str | None, websocket: WebSocket | None) -> tuple[Any, dict]:
        if model_id is not None and model_id != self.model_path:
            raise ValueError(f'Unknown model: {model_id}. Available: {self.model_path}')

        send_progress = self._progress_sender(websocket)

        async with self._subprocess_lock:
            if self.subprocess is not None:
                return self.subprocess, {}

            download_task = asyncio.create_task(asyncio.to_thread(_download_checkpoint, self.model_path))
            await monitor_async_task(
                download_task, description='Downloading DreamZero checkpoint', on_progress=send_progress
            )

            logger.info(f'Starting DreamZero subprocess with {self.num_gpus} GPUs')
            sp = DreamZeroSubprocess(
                model_path=str(download_task.result()),
                dreamzero_venv=self.dreamzero_venv,
                backbone=self.backbone,
                num_gpus=self.num_gpus,
                roboarena_port=self.roboarena_port,
                enable_dit_cache=self.enable_dit_cache,
            )
            await sp.start_async(on_progress=send_progress)
            self.subprocess = sp
            return sp, {}

    def create_policy(self, model_handle: Any) -> Policy:
        client = RoboarenaClient(port=model_handle.roboarena_port)
        client.connect()
        return DreamZeroPolicy(client)

    async def get_models(self) -> dict:
        return {'models': [self.model_path]}

    def shutdown_model(self):
        if self.subprocess is not None:
            self.subprocess.stop()


@cfn.config(
    definition=cfg_codecs.definition.override(local=codecs.dreamzero_wrappers, codec=codecs.joints),
    dreamzero_venv='/.venv/',
    backbone='wan2.1',
    num_gpus=1,
    port=8000,
    enable_dit_cache=True,
    recording_dir=None,
    idle_timeout_min=None,
)
def server(
    definition: PolicyWrapper,
    model_path: str,
    dreamzero_venv: str,
    backbone: str,
    num_gpus: int,
    port: int,
    enable_dit_cache: bool,
    recording_dir: str | None,
    idle_timeout_min: float | None,
):
    """Starts the DreamZero inference server."""
    with pos3.mirror():
        InferenceServer(
            definition=definition,
            model_path=model_path,
            dreamzero_venv=dreamzero_venv,
            backbone=backbone,
            num_gpus=num_gpus,
            port=port,
            enable_dit_cache=enable_dit_cache,
            recording_dir=recording_dir,
            idle_timeout_min=idle_timeout_min,
        ).serve()


# Public pretrained DROID checkpoint: wan2.1 backbone (the base server default) paired with the DROID
# codec that feeds its required 320x180 frames.
droid = server.override(model_path='GEAR-Dreams/DreamZero-DROID', **{'definition.codec': codecs.droid})


if __name__ == '__main__':
    init_logging()
    cfn.cli({'serve': server, 'droid': droid})
