import asyncio
import io
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import configuronic as cfn
import msgpack
import numpy as np
import pos3
import zmq
from fastapi import WebSocket

from positronic.offboard.server_utils import monitor_async_task, wait_for_subprocess_ready
from positronic.offboard.vendor_server import VendorServer
from positronic.policy import Codec, Policy
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
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
# Policy wrapper
###########################################################################################


class Gr00tPolicy(Policy):
    """Wraps a GR00T ZMQ PolicyClient as a Policy."""

    def __init__(self, client: PolicyClient):
        self._client = client

    def select_action(self, obs):
        action_response, _info = self._client.get_action(obs)
        action = {k: v[0] for k, v in action_response.items()}
        lengths = {len(v) for v in action.values()}
        assert len(lengths) == 1, f'All values in action must have the same length, got {lengths}'
        time_horizon = lengths.pop()
        return [{k: v[i] for k, v in action.items()} for i in range(time_horizon)]

    def reset(self):
        self._client.reset()


###########################################################################################
# FastAPI Inference Server
###########################################################################################


class InferenceServer(VendorServer):
    def __init__(
        self,
        codec: Codec | None,
        checkpoints_dir: str,
        checkpoint: str | None,
        modality_config: str,
        groot_venv_path: str,
        host: str = '0.0.0.0',
        port: int = 8000,
        zmq_port: int = 5555,
        recording_dir: str | None = None,
    ):
        super().__init__(codec=codec, host=host, port=port, recording_dir=recording_dir)
        self.checkpoints_dir = checkpoints_dir.rstrip('/')
        self.checkpoint = checkpoint
        self.modality_config = modality_config
        self.modality_config_path = MODALITY_CONFIGS.get(modality_config, modality_config)
        self.groot_venv_path = groot_venv_path
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

    def _resolve_checkpoint_id(self, checkpoint_id: str | None) -> str:
        if checkpoint_id:
            return f'checkpoint-{checkpoint_id}'

        if self.checkpoint:
            return 'checkpoint-' + str(self.checkpoint).strip('/')

        return get_latest_checkpoint(self.checkpoints_dir, 'checkpoint-')

    async def resolve_model(self, model_id: str | None, websocket: WebSocket | None) -> tuple[Any, dict]:
        send_progress = self._progress_sender(websocket)

        resolved_path_id = self._resolve_checkpoint_id(model_id)

        available = list_checkpoints(self.checkpoints_dir, prefix='checkpoint-')
        if resolved_path_id not in available:
            raise ValueError(f'Checkpoint not found: {resolved_path_id}')

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
        await subprocess_obj.start_async(on_progress=send_progress)

        self.subprocess = subprocess_obj
        self.current_checkpoint_id = resolved_path_id
        self.current_checkpoint_dir = str(checkpoint_dir)
        return subprocess_obj, {'checkpoint_id': normalized_id, 'checkpoint_path': str(checkpoint_dir)}

    def create_policy(self, model_handle: Any) -> Policy:
        return Gr00tPolicy(model_handle.client)

    async def get_models(self) -> dict:
        checkpoints = list_checkpoints(self.checkpoints_dir, prefix='checkpoint-')
        normalized = sorted(
            int(cp.replace('checkpoint-', '')) for cp in checkpoints if cp.replace('checkpoint-', '').isdigit()
        )
        return {'models': [str(n) for n in normalized]}

    def shutdown_model(self):
        if self.subprocess is not None:
            self.subprocess.stop()


@cfn.config(
    codec=codecs.ee_quat,
    checkpoint=None,
    port=8000,
    groot_venv_path='/.venv/',
    modality_config='ee',
    recording_dir=None,
)
def server(
    codec: Codec,
    checkpoints_dir: str,
    checkpoint: str | None,
    port: int,
    groot_venv_path: str,
    modality_config: str,
    recording_dir: str | None,
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
            recording_dir=recording_dir,
        ).serve()


# Pre-configured server variants matching GR00T modality configs
ee = server.copy()  # Uses default codec=codecs.ee_quat, modality='ee'
ee_joints = server.override(codec=codecs.ee_quat_joints, modality_config='ee_q')
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
