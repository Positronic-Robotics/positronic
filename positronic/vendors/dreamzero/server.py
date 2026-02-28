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
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from positronic.offboard.server_utils import monitor_async_task, wait_for_subprocess_ready
from positronic.policy import Codec, Policy
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise
from positronic.vendors.dreamzero import codecs

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = 'GEAR-Dreams/DreamZero-DROID'

DREAMZERO_ROOT = Path('/dreamzero')
DREAMZERO_SCRIPT = DREAMZERO_ROOT / 'socket_test_optimized_AR.py'
DREAMZERO_VENV = Path('/.venv')


def _download_checkpoint(model_path: str) -> Path:
    if '/' in model_path and not Path(model_path).exists():
        logger.info(f'Downloading checkpoint from HuggingFace: {model_path}')
        # huggingface_hub lives in the DreamZero base venv, not the positronic venv
        result = subprocess.run(
            [
                str(DREAMZERO_VENV / 'bin' / 'python'),
                '-c',
                f'from huggingface_hub import snapshot_download; print(snapshot_download("{model_path}"))',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    return Path(model_path)


class RoboarenaClient:
    """Client for DreamZero's roboarena WebSocket server.

    Protocol (from eval_utils/policy_server.py + policy_client.py):
    - On connect: server sends PolicyServerConfig as first msgpack message
    - Client sends obs dict with obs["endpoint"] = "infer" or "reset"
    - Server responds with action as raw numpy array (N, 8) via msgpack
    - Uses openpi_client.msgpack_numpy for serialization
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 9000):
        self._host = host
        self._port = port
        self._ws = None
        self._packer = None
        self._server_metadata: dict | None = None

    def connect(self):
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()
        self._unpackb = msgpack_numpy.unpackb
        self._ws = websockets.sync.client.connect(
            f'ws://{self._host}:{self._port}', compression=None, max_size=None, ping_interval=60, ping_timeout=600
        )
        # First message from server is PolicyServerConfig metadata
        self._server_metadata = self._unpackb(self._ws.recv())
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
        self._ws.send(self._packer.pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f'Server error: {response}')
        return self._unpackb(response)

    def reset(self, session_id: str | None = None):
        if self._ws is None:
            return
        msg: dict[str, Any] = {'endpoint': 'reset'}
        if session_id is not None:
            msg['session_id'] = session_id
        self._ws.send(self._packer.pack(msg))
        self._ws.recv()  # Consume "reset successful" response

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None


class DreamZeroSubprocess:
    def __init__(self, model_path: str, num_gpus: int = 1, roboarena_port: int = 9000, enable_dit_cache: bool = True):
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.roboarena_port = roboarena_port
        self.enable_dit_cache = enable_dit_cache
        self.process: subprocess.Popen | None = None

    def _build_command(self) -> list[str]:
        torchrun = str(DREAMZERO_VENV / 'bin' / 'torchrun')
        command = [
            torchrun,
            f'--nproc_per_node={self.num_gpus}',
            str(DREAMZERO_SCRIPT),
            '--port',
            str(self.roboarena_port),
            '--model-path',
            self.model_path,
        ]
        if self.enable_dit_cache:
            command.append('--enable-dit-cache')
        return command

    def _launch(self):
        command = self._build_command()
        logger.info(f'Starting DreamZero subprocess: {" ".join(command)}')
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(DREAMZERO_VENV)
        env['PATH'] = f'{DREAMZERO_VENV / "bin"}:{env.get("PATH", "")}'
        env['TORCH_COMPILE_DISABLE'] = '1'
        self.process = subprocess.Popen(command, env=env, cwd=str(DREAMZERO_ROOT))

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
            max_wait=600.0,
        )

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


class DreamZeroPolicy(Policy):
    def __init__(self, client: RoboarenaClient):
        self._client = client
        self._session_id = str(uuid.uuid4())

    def select_action(self, obs):
        obs['session_id'] = self._session_id
        action_array = np.asarray(self._client.infer(obs))

        # Response is (N, 8) — 7 joints + 1 gripper
        if action_array.ndim == 1:
            return [{'action': action_array}]
        return [{'action': action_array[i]} for i in range(action_array.shape[0])]

    def reset(self):
        self._client.reset(session_id=self._session_id)
        self._session_id = str(uuid.uuid4())


class InferenceServer:
    def __init__(
        self,
        codec: Codec,
        model_path: str,
        num_gpus: int = 1,
        host: str = '0.0.0.0',
        port: int = 8000,
        roboarena_port: int = 1234,
        enable_dit_cache: bool = True,
    ):
        self.codec = codec
        self.model_path = model_path
        self.num_gpus = num_gpus
        self.host = host
        self.port = port
        self.roboarena_port = roboarena_port
        self.enable_dit_cache = enable_dit_cache
        self.subprocess: DreamZeroSubprocess | None = None

        self.metadata = {
            'host': host,
            'port': port,
            'type': 'dreamzero',
            'model_path': model_path,
            'num_gpus': num_gpus,
        }

        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)

    async def _get_subprocess(self, websocket: WebSocket | None = None) -> DreamZeroSubprocess:
        async def send_progress(msg: str):
            if websocket is not None:
                await websocket.send_bytes(serialise({'status': 'loading', 'message': msg}))

        if self.subprocess is not None:
            return self.subprocess

        download_task = asyncio.create_task(asyncio.to_thread(_download_checkpoint, self.model_path))
        await monitor_async_task(
            download_task, description='Downloading DreamZero checkpoint', on_progress=send_progress
        )

        logger.info(f'Starting DreamZero subprocess with {self.num_gpus} GPUs')
        sp = DreamZeroSubprocess(
            model_path=str(download_task.result()),
            num_gpus=self.num_gpus,
            roboarena_port=self.roboarena_port,
            enable_dit_cache=self.enable_dit_cache,
        )
        await sp.start_async(on_progress=send_progress)
        self.subprocess = sp
        return sp

    async def get_models(self):
        return {'models': [self.model_path]}

    async def websocket_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        logger.info(f'Connected to {websocket.client}')

        try:
            subprocess_obj = await self._get_subprocess(websocket)
            meta = {**self.metadata, **self.codec.meta}
            await websocket.send_bytes(serialise({'status': 'ready', 'meta': meta}))
        except Exception as e:
            logger.error(f'Failed to start subprocess: {e}', exc_info=True)
            await websocket.send_bytes(serialise({'status': 'error', 'error': str(e)}))
            await websocket.close(code=1008, reason=str(e)[:100])
            return

        client = RoboarenaClient(port=subprocess_obj.roboarena_port)
        client.connect()
        policy = self.codec.wrap(DreamZeroPolicy(client))
        policy.reset()

        try:
            while True:
                message = await websocket.receive_bytes()
                try:
                    raw_obs = deserialise(message)
                    actions = policy.select_action(raw_obs)
                    await websocket.send_bytes(serialise({'result': actions}))
                except Exception as e:
                    logger.error(f'Error processing message: {e}', exc_info=True)
                    await websocket.send_bytes(serialise({'error': str(e)}))
        except WebSocketDisconnect:
            logger.info('Client disconnected')

    async def _startup(self):
        await self._get_subprocess(websocket=None)
        await self._warmup()

    async def _warmup(self):
        logger.info('Running warmup inference...')
        client = RoboarenaClient(port=self.subprocess.roboarena_port)
        client.connect()
        policy = DreamZeroPolicy(client)
        policy.reset()
        await asyncio.to_thread(policy.select_action, self.codec.dummy_encoded())
        logger.info('Warmup inference complete')

    def _shutdown(self):
        if self.subprocess is not None:
            self.subprocess.stop()

    def serve(self):
        async def _run():
            await self._startup()
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
            await uvicorn.Server(config).serve()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            logger.info('Server stopped by user')
        finally:
            self._shutdown()


@cfn.config(codec=codecs.joints, model_path=DEFAULT_HF_REPO, num_gpus=1, port=8000, enable_dit_cache=True)
def server(codec: Codec, model_path: str, num_gpus: int, port: int, enable_dit_cache: bool):
    """Starts the DreamZero inference server."""
    InferenceServer(
        codec=codec, model_path=model_path, num_gpus=num_gpus, port=port, enable_dit_cache=enable_dit_cache
    ).serve()


if __name__ == '__main__':
    init_logging()
    cfn.cli(server)
