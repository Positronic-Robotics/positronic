import asyncio
import logging
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openpi_client import WebsocketClientPolicy

from positronic.policy import Codec
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise
from positronic.vendors.openpi import codecs

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

    def start(self):
        """Start the OpenPI serve_policy.py subprocess."""
        command = [self.uv_path, 'run', '--frozen', '--project', str(self.openpi_root), '--']
        command.extend(['python', 'scripts/serve_policy.py'])
        command.extend(['policy:checkpoint'])
        command.extend(['--policy.config', self.config_name])
        command.extend(['--policy.dir', str(self.checkpoint_dir)])
        command.extend(['--port', str(self.ws_port)])

        env = os.environ.copy()
        logger.info(f'Starting OpenPI subprocess: {" ".join(command)}')
        self.process = subprocess.Popen(
            command, env=env, cwd=str(self.openpi_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        self._wait_for_ready()

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

    @property
    def client(self) -> WebsocketClientPolicy:
        """Get or create WebSocket client for inference."""
        if self._client is None:
            self._client = WebsocketClientPolicy(host='127.0.0.1', port=self.ws_port)
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
# FastAPI Inference Server
###########################################################################################


class InferenceServer:
    """FastAPI server that wraps OpenPI subprocess and provides unified API."""

    def __init__(
        self,
        codec: Codec,
        checkpoints_dir: str | Path,
        config_name: str = 'pi05_positronic_lowmem',
        checkpoint: str | None = None,
        host: str = '0.0.0.0',
        port: int = 8000,
        openpi_ws_port: int = 8001,
        metadata: dict[str, Any] | None = None,
    ):
        self.codec = codec
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/') + '/checkpoints'
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.host = host
        self.port = port
        self.openpi_ws_port = openpi_ws_port

        self.metadata = metadata or {}
        self.metadata.update(host=host, port=port, config_name=config_name)

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
            return {'models': list_checkpoints(self.checkpoints_dir)}
        except Exception:
            logger.exception('Failed to list checkpoints.')
            return {'models': []}

    async def _resolve_checkpoint_id(self, websocket: WebSocket, checkpoint_id: str | None) -> str | None:
        """Resolve checkpoint ID from parameter, config, or latest."""
        if checkpoint_id:
            available = list_checkpoints(self.checkpoints_dir)
            if checkpoint_id not in available:
                logger.error('Checkpoint not found: %s', checkpoint_id)
                await websocket.send_bytes(serialise({'error': 'Checkpoint not found'}))
                await websocket.close(code=1008, reason='Checkpoint not found')
                return None
            return checkpoint_id

        # Use configured checkpoint or latest
        if self.checkpoint:
            return self.checkpoint

        try:
            return get_latest_checkpoint(self.checkpoints_dir)
        except Exception as e:
            logger.exception('Failed to resolve checkpoint')
            await websocket.send_bytes(serialise({'error': f'No checkpoint available: {e}'}))
            await websocket.close(code=1008, reason='No checkpoint available')
            return None

    async def _get_subprocess(self, checkpoint_id: str) -> OpenpiSubprocess:
        """Get or create subprocess for checkpoint."""
        async with self._subprocess_lock:
            if checkpoint_id not in self._subprocesses:
                # Download checkpoint if needed
                with pos3.mirror():
                    checkpoint_dir = pos3.download(f'{self.checkpoints_dir}/{checkpoint_id}')

                logger.info(f'Starting OpenPI subprocess for checkpoint {checkpoint_id}')
                subprocess_obj = OpenpiSubprocess(
                    checkpoint_dir=checkpoint_dir, config_name=self.config_name, ws_port=self.openpi_ws_port
                )
                subprocess_obj.start()
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
            # Get or start subprocess
            subprocess_obj = await self._get_subprocess(resolved_checkpoint_id)

            # Send metadata
            meta = {**self.metadata, 'checkpoint_id': resolved_checkpoint_id}
            # Add codec metadata if available
            if hasattr(self.codec.observation, 'meta'):
                meta.update(self.codec.observation.meta)
            if hasattr(self.codec.action, 'meta'):
                meta.update(self.codec.action.meta)

            await websocket.send_bytes(serialise({'meta': meta}))

            # Inference loop
            async for message in websocket.iter_bytes():
                try:
                    # Deserialize observation from client
                    raw_obs = deserialise(message)

                    # Encode observation using codec
                    encoded_obs = self.codec.observation.encode(raw_obs)

                    # Forward to OpenPI subprocess
                    openpi_response = subprocess_obj.client.infer(encoded_obs)

                    # Decode action using codec
                    decoded_action = self.codec.action.decode(openpi_response['action'], raw_obs)

                    # Send to client
                    await websocket.send_bytes(serialise({'result': decoded_action}))

                except Exception as e:
                    logger.error(f'Error processing message: {e}')
                    logger.debug(traceback.format_exc())
                    error_response = {'error': str(e)}
                    await websocket.send_bytes(serialise(error_response))

        except (WebSocketDisconnect, Exception) as e:
            logger.info(f'Connection closed: {e}')
            logger.debug(traceback.format_exc())

    def serve(self):
        """Start the uvicorn server."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
        server = uvicorn.Server(config)
        return server.serve()

    def shutdown(self):
        """Shutdown all subprocesses."""
        for subprocess_obj in self._subprocesses.values():
            subprocess_obj.stop()
        self._subprocesses.clear()


###########################################################################################
# Pre-configured server variants
###########################################################################################


@cfn.config(
    checkpoints_dir='',
    config_name='pi05_positronic_lowmem',
    checkpoint=None,
    host='0.0.0.0',
    port=8000,
    openpi_ws_port=8001,
)
def server(
    codec, checkpoints_dir: str, config_name: str, checkpoint: str | None, host: str, port: int, openpi_ws_port: int
):
    """Main OpenPI inference server config."""
    return InferenceServer(
        codec=codec,
        checkpoints_dir=checkpoints_dir,
        config_name=config_name,
        checkpoint=checkpoint,
        host=host,
        port=port,
        openpi_ws_port=openpi_ws_port,
    )


# Pre-configured server variants using .copy() and .override()
eepose_absolute = server.copy().override(codec=codecs.eepose_absolute)
openpi_positronic = server.copy().override(codec=codecs.openpi_positronic)
droid = server.copy().override(codec=codecs.droid)
eepose_q = server.copy().override(codec=codecs.eepose_q)
joints = server.copy().override(codec=codecs.joints)


if __name__ == '__main__':
    init_logging()
    cfn.cli({
        'eepose_absolute': eepose_absolute,
        'openpi_positronic': openpi_positronic,
        'droid': droid,
        'eepose_q': eepose_q,
        'joints': joints,
    })
