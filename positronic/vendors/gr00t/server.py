import asyncio
import io
import logging
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import configuronic as cfn
import msgpack
import numpy as np
import pos3
import uvicorn
import zmq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from positronic.offboard.server_utils import monitor_async_task, wait_for_subprocess_ready
from positronic.policy import Codec
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise
from positronic.vendors.gr00t import MODALITY_CONFIGS, codecs

logger = logging.getLogger(__name__)


###########################################################################################
# ZMQ client code for communicating with gr00t N1.6 server
# Adapted from gr00t/policy/server_client.py
###########################################################################################


class MsgSerializer:
    """Message serializer for ZMQ communication (N1.6 format)."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if '__ndarray_class__' in obj:
            return np.load(io.BytesIO(obj['as_npy']), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {'__ndarray_class__': True, 'as_npy': output.getvalue()}
        return obj


class PolicyClient:
    """Client for communicating with GR00T N1.6 PolicyServer via ZMQ."""

    def __init__(self, host: str = 'localhost', port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f'tcp://{self.host}:{self.port}')

    def ping(self) -> bool:
        try:
            self.call_endpoint('ping', requires_input=False)
            return True
        except (zmq.error.ZMQError, RuntimeError):
            self._init_socket()
            return False

    def call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        request: dict = {'endpoint': endpoint}
        if requires_input:
            request['data'] = data

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.Again as err:
            raise RuntimeError(
                f'Timeout after {self.timeout_ms}ms calling endpoint "{endpoint}" at {self.host}:{self.port}'
            ) from err

        if message == b'ERROR':
            raise RuntimeError('Server error. Make sure the correct policy server is running.')
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and 'error' in response:
            raise RuntimeError(f'Server error: {response["error"]}')
        return response

    def get_action(self, observation: dict[str, Any]) -> tuple[dict, dict]:
        response = self.call_endpoint('get_action', {'observation': observation, 'options': None})
        return tuple(response)

    def reset(self) -> dict[str, Any]:
        return self.call_endpoint('reset', {'options': None})

    def close(self):
        self.socket.close()
        self.context.term()


###########################################################################################
# Subprocess manager for gr00t ZMQ server
###########################################################################################


class Gr00tSubprocess:
    """Manages the gr00t ZMQ server subprocess."""

    def __init__(self, checkpoint_dir: str, modality_config_path: str, groot_venv_path: str, zmq_port: int = 5555):
        self.checkpoint_dir = checkpoint_dir
        self.modality_config_path = modality_config_path
        self.groot_venv_path = groot_venv_path
        self.zmq_port = zmq_port
        self.process: subprocess.Popen | None = None
        self._client: PolicyClient | None = None

    def start(self):
        groot_root = Path(__file__).parents[4] / 'gr00t'
        python_bin = str(Path(self.groot_venv_path) / 'bin' / 'python')

        command = [python_bin, 'gr00t/eval/run_gr00t_server.py']
        command.extend(['--model_path', str(self.checkpoint_dir)])
        command.extend(['--embodiment_tag', 'NEW_EMBODIMENT'])
        command.extend(['--modality_config_path', self.modality_config_path])
        command.extend(['--host', '127.0.0.1'])
        command.extend(['--port', str(self.zmq_port)])

        env = os.environ.copy()
        logger.info(f'Starting gr00t subprocess: {" ".join(command)}')
        self.process = subprocess.Popen(command, env=env, cwd=str(groot_root))

        self._wait_for_ready()

    def _check_crashed(self) -> tuple[bool, int | None]:
        """Check if subprocess has crashed."""
        if self.process is None:
            return False, None
        exit_code = self.process.poll()
        return exit_code is not None, exit_code

    def _wait_for_ready(self, timeout: float = 120.0, poll_interval: float = 1.0):
        """Wait for the gr00t server to be ready by polling with ping."""
        client = PolicyClient(host='127.0.0.1', port=self.zmq_port, timeout_ms=2000)
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f'gr00t subprocess exited with code {self.process.returncode}')

            if client.ping():
                logger.info('gr00t subprocess is ready')
                client.close()
                return

            time.sleep(poll_interval)

        client.close()
        raise RuntimeError(f'gr00t subprocess did not become ready within {timeout}s')

    async def start_async(self, on_progress=None):
        """Start the gr00t subprocess asynchronously with optional progress reporting.

        Args:
            on_progress: Optional async callback for progress updates.
        """
        groot_root = Path(__file__).parents[4] / 'gr00t'
        python_bin = str(Path(self.groot_venv_path) / 'bin' / 'python')

        command = [python_bin, 'gr00t/eval/run_gr00t_server.py']
        command.extend(['--model_path', str(self.checkpoint_dir)])
        command.extend(['--embodiment_tag', 'NEW_EMBODIMENT'])
        command.extend(['--modality_config_path', self.modality_config_path])
        command.extend(['--host', '127.0.0.1'])
        command.extend(['--port', str(self.zmq_port)])

        env = os.environ.copy()
        logger.info(f'Starting gr00t subprocess: {" ".join(command)}')
        self.process = subprocess.Popen(command, env=env, cwd=str(groot_root))

        # Use a temporary client for readiness check
        client = PolicyClient(host='127.0.0.1', port=self.zmq_port, timeout_ms=2000)
        try:
            await wait_for_subprocess_ready(
                check_ready=client.ping,
                check_crashed=self._check_crashed,
                description='GR00T subprocess',
                on_progress=on_progress,
                max_wait=120.0,
            )
        finally:
            client.close()

    @property
    def client(self) -> PolicyClient:
        if self._client is None:
            self._client = PolicyClient(host='127.0.0.1', port=self.zmq_port, timeout_ms=15000)
        return self._client

    def stop(self):
        if self._client is not None:
            self._client.close()
            self._client = None

        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


###########################################################################################
# FastAPI Inference Server
###########################################################################################


class InferenceServer:
    def __init__(
        self,
        codec: Codec,
        checkpoints_dir: str,
        checkpoint: str | None,
        modality_config: str,
        groot_venv_path: str,
        host: str = '0.0.0.0',
        port: int = 8000,
        zmq_port: int = 5555,
    ):
        self.codec = codec
        self.checkpoints_dir = checkpoints_dir.rstrip('/')
        self.checkpoint = checkpoint
        self.modality_config = modality_config
        self.modality_config_path = MODALITY_CONFIGS.get(modality_config, modality_config)
        self.groot_venv_path = groot_venv_path
        self.host = host
        self.port = port
        self.zmq_port = zmq_port

        self.subprocess: Gr00tSubprocess | None = None
        self.current_checkpoint_id: str | None = None
        self.current_checkpoint_dir: str | None = None

        self.metadata = {
            'host': host,
            'port': port,
            'type': 'groot',
            'modality_config': modality_config,
            'experiment_name': checkpoints_dir.rstrip('/').split('/')[-1] or '',
        }
        if hasattr(self.codec.observation, 'meta'):
            self.metadata.update({f'observation.{k}': v for k, v in self.codec.observation.meta.items()})
        if hasattr(self.codec.action, 'meta'):
            self.metadata.update({f'action.{k}': v for k, v in self.codec.action.meta.items()})

        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{checkpoint_id}')(self.websocket_endpoint)

    def _resolve_checkpoint_id(self, checkpoint_id: str | None) -> str:
        if checkpoint_id:
            # API ID is a number, map to directory name
            return f'checkpoint-{checkpoint_id}'

        if self.checkpoint:
            return 'checkpoint-' + str(self.checkpoint).strip('/')

        return get_latest_checkpoint(self.checkpoints_dir, 'checkpoint-')

    async def _get_subprocess(
        self, checkpoint_id: str | None, websocket: WebSocket | None = None
    ) -> tuple[Gr00tSubprocess, dict]:
        """Ensure subprocess is running with the specified checkpoint.

        Args:
            checkpoint_id: Checkpoint to load (must be numeric ID)
            websocket: Optional WebSocket to send status updates to

        Returns:
            Tuple of (subprocess, metadata_dict)

        Raises:
            ValueError: If checkpoint_id is invalid or not found
        """

        async def send_progress(msg: str):
            if websocket is not None:
                await websocket.send_bytes(serialise({'status': 'loading', 'message': msg}))

        resolved_path_id = self._resolve_checkpoint_id(checkpoint_id)

        available = list_checkpoints(self.checkpoints_dir, prefix='checkpoint-')
        if resolved_path_id not in available:
            raise ValueError(f'Checkpoint not found: {resolved_path_id}')

        # Normalize ID for metadata if it's a checkpoint path
        normalized_id = resolved_path_id.replace('checkpoint-', '')
        if normalized_id.isdigit():
            normalized_id = str(int(normalized_id))

        if self.current_checkpoint_id == resolved_path_id and self.subprocess is not None:
            return self.subprocess, {'checkpoint_id': normalized_id, 'checkpoint_path': self.current_checkpoint_dir}

        if self.subprocess is not None:
            logger.info(f'Stopping subprocess for checkpoint {self.current_checkpoint_id}')
            self.subprocess.stop()

        logger.info(f'Loading checkpoint {resolved_path_id}')
        checkpoint_path = f'{self.checkpoints_dir}/{resolved_path_id}'

        # Download checkpoint in thread with periodic progress updates
        download_task = asyncio.create_task(asyncio.to_thread(pos3.download, checkpoint_path, exclude=['optimizer.pt']))
        await monitor_async_task(
            download_task, description=f'Downloading checkpoint {resolved_path_id}', on_progress=send_progress
        )
        checkpoint_dir = download_task.result()

        logger.info(f'Starting subprocess for checkpoint {resolved_path_id}')
        subprocess_obj = Gr00tSubprocess(
            checkpoint_dir=str(checkpoint_dir),
            modality_config_path=self.modality_config_path,
            groot_venv_path=self.groot_venv_path,
            zmq_port=self.zmq_port,
        )

        # Start subprocess with periodic progress updates (if websocket provided)
        await subprocess_obj.start_async(on_progress=send_progress)

        self.subprocess = subprocess_obj
        self.current_checkpoint_id = resolved_path_id
        self.current_checkpoint_dir = str(checkpoint_dir)
        return subprocess_obj, {'checkpoint_id': normalized_id, 'checkpoint_path': str(checkpoint_dir)}

    async def get_models(self):
        checkpoints = list_checkpoints(self.checkpoints_dir, prefix='checkpoint-')
        normalized = sorted(
            int(cp.replace('checkpoint-', '')) for cp in checkpoints if cp.replace('checkpoint-', '').isdigit()
        )
        return {'models': [str(n) for n in normalized]}

    async def websocket_endpoint(self, websocket: WebSocket, checkpoint_id: str | None = None):
        await websocket.accept()
        logger.info(f'Connected to {websocket.client} requesting {checkpoint_id or "default"}')

        try:
            # Get or start subprocess (sends status updates internally)
            subprocess, checkpoint_meta = await self._get_subprocess(checkpoint_id, websocket)

            # Send ready with metadata
            meta = {**self.metadata, **checkpoint_meta}
            await websocket.send_bytes(serialise({'status': 'ready', 'meta': meta}))
        except Exception as e:
            logger.error(f'Failed to load checkpoint: {e}')
            await websocket.send_bytes(serialise({'status': 'error', 'error': str(e)}))
            await websocket.close(code=1008, reason=str(e)[:100])
            return

        try:
            subprocess.client.reset()
            try:
                while True:
                    message = await websocket.receive_bytes()
                    try:
                        raw_obs = deserialise(message)
                        encoded_obs = self.codec.observation.encode(raw_obs)
                        action_response, _info = subprocess.client.get_action(encoded_obs)

                        action = {k: v[0] for k, v in action_response.items()}
                        lengths = {len(v) for v in action.values()}
                        assert len(lengths) == 1, f'All values in action must have the same length, got {lengths}'
                        time_horizon = lengths.pop()
                        if self.codec.action.action_horizon is not None:
                            time_horizon = min(time_horizon, self.codec.action.action_horizon)

                        decoded_actions = []
                        for i in range(time_horizon):
                            step_action = {k: v[i] for k, v in action.items()}
                            decoded = self.codec.action.decode(step_action, raw_obs)
                            decoded_actions.append(decoded)

                        await websocket.send_bytes(serialise({'result': decoded_actions}))
                    except Exception as e:
                        logger.error(f'Error processing message: {e}')
                        logger.debug(traceback.format_exc())
                        await websocket.send_bytes(serialise({'error': str(e)}))
            except WebSocketDisconnect:
                logger.info('Client disconnected')

        except Exception as e:
            logger.error(f'Connection error: {e}')
            logger.debug(traceback.format_exc())
            try:
                await websocket.send_bytes(serialise({'error': str(e)}))
            except Exception:
                pass

    async def _startup(self):
        """Start the subprocess on server startup."""
        await self._get_subprocess(None, websocket=None)
        await self._warmup()

    async def _warmup(self):
        """Run one warmup inference to trigger JIT compilation."""
        try:
            logger.info('Running warmup inference...')
            dummy = self.codec.observation.dummy_input()
            encoded = self.codec.observation.encode(dummy)
            await asyncio.to_thread(self.subprocess.client.reset)
            await asyncio.to_thread(self.subprocess.client.get_action, encoded)
            logger.info('Warmup inference complete')
        except Exception:
            logger.warning('Warmup inference failed (non-fatal)', exc_info=True)

    def _shutdown(self):
        """Clean up subprocess on server shutdown."""
        if self.subprocess is not None:
            self.subprocess.stop()

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


@cfn.config(codec=codecs.ee_absolute, checkpoint=None, port=8000, groot_venv_path='/.venv/', modality_config='ee')
def server(
    codec: Codec, checkpoints_dir: str, checkpoint: str | None, port: int, groot_venv_path: str, modality_config: str
):
    """Starts the GR00T inference server with encoding/decoding."""

    with pos3.mirror():
        InferenceServer(
            codec=codec,
            checkpoints_dir=checkpoints_dir,
            checkpoint=checkpoint,
            modality_config=modality_config,
            groot_venv_path=groot_venv_path,
            port=port,
        ).serve()


# Pre-configured server variants matching GR00T modality configs
ee = server.copy()  # Uses default codec=codecs.ee_absolute, modality='ee'
ee_joints = server.override(codec=codecs.ee_joints, modality_config='ee_q')
ee_rot6d = server.override(codec=codecs.ee_rot6d, modality_config='ee_rot6d')
ee_rot6d_joints = server.override(codec=codecs.ee_rot6d_joints, modality_config='ee_rot6d_q')
ee_rot6d_rel = server.override(codec=codecs.ee_rot6d, modality_config='ee_rot6d_rel')
ee_rot6d_joints_rel = server.override(codec=codecs.ee_rot6d_joints, modality_config='ee_rot6d_q_rel')


if __name__ == '__main__':
    init_logging()
    cfn.cli({
        'server': server,
        'ee': ee,
        'ee_joints': ee_joints,
        'ee_rot6d': ee_rot6d,
        'ee_rot6d_joints': ee_rot6d_joints,
        'ee_rot6d_rel': ee_rot6d_rel,
        'ee_rot6d_joints_rel': ee_rot6d_joints_rel,
    })
