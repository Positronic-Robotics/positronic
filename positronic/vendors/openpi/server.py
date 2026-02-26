import asyncio
import logging
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from positronic.offboard.server_utils import monitor_async_task, wait_for_subprocess_ready
from positronic.policy import Codec, Policy, RecordingCodec
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise
from positronic.vendors.openpi import codecs, ensure_paligemma_tokenizer

logger = logging.getLogger(__name__)


###########################################################################################
# Subprocess manager for OpenPI WebSocket server
###########################################################################################


class OpenpiSubprocess:
    """Manages the OpenPI serve_policy.py subprocess."""

    def __init__(
        self,
        checkpoint_dir: str,
        config_name: str,
        openpi_root: Path | None = None,
        ws_port: int = 8001,
        uv_path: str | None = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.config_name = config_name
        self.openpi_root = openpi_root or Path(__file__).parents[4] / 'openpi'
        self.ws_port = ws_port
        self.uv_path = uv_path or 'uv'
        self.process: subprocess.Popen | None = None
        self._client: WebsocketClientPolicy | None = None

    def _build_command(self) -> list[str]:
        """Build the command to start serve_policy.py."""
        return [
            self.uv_path,
            'run',
            '--frozen',
            '--project',
            str(self.openpi_root),
            '--',
            'python',
            'scripts/serve_policy.py',
            '--port',
            str(self.ws_port),
            'policy:checkpoint',
            '--policy.config',
            self.config_name,
            '--policy.dir',
            str(self.checkpoint_dir),
        ]

    def start(self):
        """Start the OpenPI serve_policy.py subprocess."""
        command = self._build_command()
        logger.info(f'Starting OpenPI subprocess: {" ".join(command)}')
        # Don't pipe stdout/stderr so we can see the output
        self.process = subprocess.Popen(command, env=os.environ.copy(), cwd=str(self.openpi_root))

        self._wait_for_ready()

    def _check_ready(self) -> bool:
        """Check if OpenPI subprocess is ready by checking if port is accepting connections."""
        try:
            with socket.create_connection(('127.0.0.1', self.ws_port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            return False

    def _check_crashed(self) -> tuple[bool, int | None]:
        """Check if subprocess has crashed."""
        if self.process is None:
            return False, None
        exit_code = self.process.poll()
        return exit_code is not None, exit_code

    def _wait_for_ready(self, timeout: float = 300.0, poll_interval: float = 1.0):
        """Wait for the OpenPI server to be ready by connecting to it."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                # Process exited, try to read stderr
                stderr = self.process.stderr.read().decode() if self.process.stderr else ''
                raise RuntimeError(f'OpenPI subprocess exited with code {self.process.returncode}. stderr: {stderr}')

            try:
                # Try to connect
                WebsocketClientPolicy(host='127.0.0.1', port=self.ws_port)
                logger.info('OpenPI subprocess is ready')
                return
            except Exception:
                time.sleep(poll_interval)

        raise RuntimeError(f'OpenPI subprocess did not become ready within {timeout}s')

    async def start_async(self, on_progress=None):
        """Start the OpenPI subprocess asynchronously with optional progress reporting.

        Args:
            on_progress: Optional async callback for progress updates.
        """
        command = self._build_command()
        logger.info(f'Starting OpenPI subprocess: {" ".join(command)}')
        self.process = subprocess.Popen(command, env=os.environ.copy(), cwd=str(self.openpi_root))

        await wait_for_subprocess_ready(
            check_ready=self._check_ready,
            check_crashed=self._check_crashed,
            description='OpenPI subprocess',
            on_progress=on_progress,
            max_wait=300.0,
        )

    @property
    def client(self) -> WebsocketClientPolicy:
        """Get or create WebSocket client for inference."""
        if self._client is None:
            logger.info(f'Creating WebSocket client to OpenPI subprocess on port {self.ws_port}...')
            self._client = WebsocketClientPolicy(host='127.0.0.1', port=self.ws_port)
            logger.info('WebSocket client created successfully')
        return self._client

    def stop(self):
        """Stop the OpenPI subprocess."""
        self._client = None

        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning('OpenPI subprocess did not terminate, killing it')
                self.process.kill()
            self.process = None


###########################################################################################
# Policy wrapper
###########################################################################################


class OpenpiPolicy(Policy):
    """Wraps an OpenPI WebsocketClientPolicy as a Policy."""

    def __init__(self, client: WebsocketClientPolicy):
        self._client = client

    def select_action(self, obs):
        response = self._client.infer(obs)
        actions = response['actions']
        return [{'action': a} for a in actions]

    def reset(self):
        self._client.reset()


###########################################################################################
# FastAPI Inference Server
###########################################################################################


class InferenceServer:
    """FastAPI server that wraps OpenPI subprocess and provides unified API."""

    def __init__(
        self,
        codec: Codec | None,
        checkpoints_dir: str | Path,
        config_name: str = 'pi05_positronic_lowmem',
        checkpoint: str | None = None,
        host: str = '0.0.0.0',
        port: int = 8000,
        openpi_ws_port: int = 8001,
        metadata: dict[str, Any] | None = None,
        recording_dir: str | None = None,
    ):
        self.codec = codec
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/')
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.host = host
        self.port = port
        self.openpi_ws_port = openpi_ws_port
        if recording_dir:
            self.codec = RecordingCodec(self.codec, pos3.sync(recording_dir))

        self.metadata = metadata or {}
        self.metadata.update(
            type='openpi',
            host=host,
            port=port,
            config_name=config_name,
            checkpoint_path=str(checkpoints_dir),
            experiment_name=str(checkpoints_dir).rstrip('/').split('/')[-1] or '',
        )

        # Active subprocess per checkpoint
        self._subprocesses: dict[str, OpenpiSubprocess] = {}
        self._subprocess_lock = asyncio.Lock()

        # Initialize FastAPI
        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{checkpoint_id}')(self.websocket_endpoint)

    async def get_models(self):
        """Return list of available checkpoints."""
        try:
            checkpoints = list_checkpoints(self.checkpoints_dir)
            normalized = [int(cp) for cp in checkpoints if cp.isdigit()]
            normalized.sort()
            return {'models': [str(n) for n in normalized]}
        except Exception:
            logger.exception('Failed to list checkpoints.')
            return {'models': []}

    async def _resolve_checkpoint_id(self, websocket: WebSocket, checkpoint_id: str | None) -> str | None:
        """Resolve checkpoint ID from parameter, config, or latest."""
        if checkpoint_id:
            available = list_checkpoints(self.checkpoints_dir)

            if checkpoint_id.isdigit():
                target_int = int(checkpoint_id)
                for cp in available:
                    if cp.isdigit() and int(cp) == target_int:
                        return cp

            error_msg = f'Checkpoint not found or invalid ID: {checkpoint_id}.'
            logger.error(error_msg)
            await websocket.send_bytes(serialise({'status': 'error', 'error': error_msg}))
            await websocket.close(code=1008, reason='Checkpoint not found')
            return None

        # Use configured checkpoint or latest
        if self.checkpoint:
            return self.checkpoint

        try:
            return get_latest_checkpoint(self.checkpoints_dir)
        except Exception as e:
            error_msg = f'No checkpoint available in {self.checkpoints_dir}: {e}'
            logger.exception(error_msg)
            await websocket.send_bytes(serialise({'status': 'error', 'error': error_msg}))
            await websocket.close(code=1008, reason='No checkpoint available')
            return None

    async def _get_subprocess(self, checkpoint_id: str, websocket: WebSocket) -> OpenpiSubprocess:
        """Get or create subprocess for checkpoint, sending status updates."""

        async def send_progress(msg: str):
            await websocket.send_bytes(serialise({'status': 'loading', 'message': msg}))

        async with self._subprocess_lock:
            if checkpoint_id not in self._subprocesses:
                # Download checkpoint in thread with periodic progress updates
                checkpoint_path = f'{self.checkpoints_dir}/{checkpoint_id}'
                download_task = asyncio.create_task(asyncio.to_thread(pos3.download, checkpoint_path))
                await monitor_async_task(
                    download_task, description=f'Downloading checkpoint {checkpoint_id}', on_progress=send_progress
                )
                checkpoint_dir = download_task.result()

                logger.info(f'Starting OpenPI subprocess for checkpoint {checkpoint_id}')
                subprocess_obj = OpenpiSubprocess(
                    checkpoint_dir=checkpoint_dir, config_name=self.config_name, ws_port=self.openpi_ws_port
                )

                # Start subprocess with periodic progress updates
                await subprocess_obj.start_async(on_progress=send_progress)

                self._subprocesses[checkpoint_id] = subprocess_obj

            return self._subprocesses[checkpoint_id]

    async def websocket_endpoint(self, websocket: WebSocket, checkpoint_id: str | None = None):
        """WebSocket endpoint for inference sessions."""
        await websocket.accept()
        logger.info(f'Connected to {websocket.client} requesting {checkpoint_id or "default"}')

        # Resolve checkpoint
        resolved_checkpoint_id = await self._resolve_checkpoint_id(websocket, checkpoint_id)
        if resolved_checkpoint_id is None:
            return

        try:
            # Get or start subprocess (sends status updates internally)
            subprocess_obj = await self._get_subprocess(resolved_checkpoint_id, websocket)
            base_policy = OpenpiPolicy(subprocess_obj.client)
            policy = self.codec.wrap(base_policy) if self.codec else base_policy
            policy.reset()
            meta = {**self.metadata, 'checkpoint_id': resolved_checkpoint_id, **policy.meta}
            await websocket.send_bytes(serialise({'status': 'ready', 'meta': meta}))

            # Inference loop
            async for message in websocket.iter_bytes():
                try:
                    raw_obs = deserialise(message)
                    actions = policy.select_action(raw_obs)
                    await websocket.send_bytes(serialise({'result': actions}))
                except Exception as e:
                    logger.error(f'Error processing message: {e}')
                    logger.error(traceback.format_exc())
                    await websocket.send_bytes(serialise({'error': str(e)}))

        except (WebSocketDisconnect, Exception) as e:
            logger.info(f'Connection closed: {e}')
            logger.debug(traceback.format_exc())

    async def _startup(self):
        """Pre-load the default checkpoint and start the subprocess before accepting connections."""
        checkpoint_id = self.checkpoint
        if not checkpoint_id:
            checkpoint_id = get_latest_checkpoint(self.checkpoints_dir)

        checkpoint_path = f'{self.checkpoints_dir}/{checkpoint_id}'
        logger.info(f'Pre-loading checkpoint {checkpoint_id}')
        checkpoint_dir = await asyncio.to_thread(pos3.download, checkpoint_path)

        subprocess_obj = OpenpiSubprocess(
            checkpoint_dir=checkpoint_dir, config_name=self.config_name, ws_port=self.openpi_ws_port
        )
        await subprocess_obj.start_async()
        self._subprocesses[checkpoint_id] = subprocess_obj

        await self._warmup(subprocess_obj)

    async def _warmup(self, subprocess_obj: OpenpiSubprocess):
        """Run one warmup inference to trigger JIT compilation."""
        try:
            logger.info('Running warmup inference...')
            dummy = self.codec.dummy_encoded() if self.codec else {}
            await asyncio.to_thread(subprocess_obj.client.infer, dummy)
            logger.info('Warmup inference complete')
        except Exception:
            logger.warning('Warmup inference failed (non-fatal)', exc_info=True)

    def _shutdown(self):
        """Shutdown all subprocesses."""
        for subprocess_obj in self._subprocesses.values():
            subprocess_obj.stop()
        self._subprocesses.clear()

    def serve(self):
        """Start the server: pre-load model, warm up, then serve requests."""

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


###########################################################################################
# Server config
###########################################################################################


@cfn.config(
    codec=codecs.ee,
    checkpoints_dir='',
    config_name='pi05_positronic_lowmem',
    checkpoint=None,
    host='0.0.0.0',
    port=8000,
    openpi_ws_port=8001,
    recording_dir=None,
)
def server(
    codec,
    checkpoints_dir: str,
    config_name: str,
    checkpoint: str | None,
    host: str,
    port: int,
    openpi_ws_port: int,
    recording_dir: str | None,
):
    """OpenPI inference server.

    Args:
        codec: Codec config for observation encoding and action decoding.
            Available codecs:
            - @positronic.vendors.openpi.codecs.ee (default, EE pose + grip)
            - @positronic.vendors.openpi.codecs.ee_joints (EE pose + grip + joints)
            - @positronic.vendors.openpi.codecs.droid (for pretrained DROID models)
        checkpoints_dir: Directory containing model checkpoints.
        config_name: OpenPI config name (default: pi05_positronic_lowmem).
        checkpoint: Specific checkpoint to load (optional, defaults to latest).
        host: Server host address.
        port: Server port.
        openpi_ws_port: Internal WebSocket port for OpenPI subprocess.
        recording_dir: Directory for recording .rrd files (optional, supports S3 paths).
    """
    InferenceServer(
        codec=codec,
        checkpoints_dir=checkpoints_dir,
        config_name=config_name,
        checkpoint=checkpoint,
        host=host,
        port=port,
        openpi_ws_port=openpi_ws_port,
        recording_dir=recording_dir,
    ).serve()


if __name__ == '__main__':
    init_logging()
    ensure_paligemma_tokenizer()
    with pos3.mirror():
        cfn.cli(server)
