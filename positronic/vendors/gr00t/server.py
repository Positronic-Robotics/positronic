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

from positronic.cfg.policy import action as act_cfg
from positronic.cfg.policy import observation as obs_cfg
from positronic.policy.action import ActionDecoder
from positronic.policy.observation import GrootObservationEncoder
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
from positronic.utils.serialization import deserialise, serialise
from positronic.vendors.gr00t import MODALITY_CONFIGS

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

    def get_client(self) -> PolicyClient:
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
        observation_encoder: GrootObservationEncoder,
        action_decoder: ActionDecoder,
        checkpoints_dir: str,
        checkpoint: str | None,
        modality_config: str,
        groot_venv_path: str,
        host: str = '0.0.0.0',
        port: int = 8000,
        zmq_port: int = 5555,
    ):
        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
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

        self.metadata = {'host': host, 'port': port, 'type': 'groot', 'modality_config': modality_config}
        if hasattr(observation_encoder, 'meta'):
            self.metadata.update({f'observation.{k}': v for k, v in observation_encoder.meta.items()})
        if hasattr(action_decoder, 'meta'):
            self.metadata.update({f'action.{k}': v for k, v in action_decoder.meta.items()})

        self.app = FastAPI()
        self.app.get('/api/v1/models')(self.get_models)
        self.app.websocket('/api/v1/session')(self.websocket_endpoint)
        self.app.websocket('/api/v1/session/{checkpoint_id}')(self.websocket_endpoint)

    def _resolve_checkpoint_id(self, checkpoint_id: str | None) -> str:
        if checkpoint_id:
            return checkpoint_id

        if self.checkpoint:
            return 'checkpoint-' + str(self.checkpoint).strip('/')

        return get_latest_checkpoint(self.checkpoints_dir, 'checkpoint-')

    def _ensure_subprocess(self, checkpoint_id: str):
        if self.current_checkpoint_id == checkpoint_id and self.subprocess is not None:
            return

        if self.subprocess is not None:
            logger.info(f'Stopping subprocess for checkpoint {self.current_checkpoint_id}')
            self.subprocess.stop()

        logger.info(f'Loading checkpoint {checkpoint_id}')
        checkpoint_path = f'{self.checkpoints_dir}/{checkpoint_id}'
        checkpoint_dir = pos3.download(checkpoint_path, exclude=['optimizer.pt'])

        logger.info(f'Starting subprocess for checkpoint {checkpoint_id}')
        self.subprocess = Gr00tSubprocess(
            checkpoint_dir=str(checkpoint_dir),
            modality_config_path=self.modality_config_path,
            groot_venv_path=self.groot_venv_path,
            zmq_port=self.zmq_port,
        )
        self.subprocess.start()
        self.current_checkpoint_id = checkpoint_id
        self.current_checkpoint_dir = str(checkpoint_dir)

    async def get_models(self):
        try:
            checkpoints = list_checkpoints(self.checkpoints_dir, prefix='checkpoint-')
            return {'models': checkpoints}
        except Exception:
            logger.exception('Failed to list checkpoints.')
            return {'models': []}

    async def websocket_endpoint(self, websocket: WebSocket, checkpoint_id: str | None = None):
        await websocket.accept()

        try:
            checkpoint_id = self._resolve_checkpoint_id(checkpoint_id)
            available = list_checkpoints(self.checkpoints_dir, prefix='checkpoint-')
            if checkpoint_id not in available:
                logger.error('Checkpoint not found: %s', checkpoint_id)
                await websocket.send_bytes(serialise({'error': 'Checkpoint not found'}))
                await websocket.close(code=1008, reason='Checkpoint not found')
                return
        except Exception as e:
            logger.exception('Failed to resolve checkpoint.')
            await websocket.send_bytes(serialise({'error': str(e)}))
            await websocket.close(code=1008, reason='Failed to resolve checkpoint')
            return

        logger.info(f'Connected from {websocket.client} requesting {checkpoint_id}')

        try:
            self._ensure_subprocess(checkpoint_id)
            client = self.subprocess.get_client()

            meta = {**self.metadata, 'checkpoint_id': checkpoint_id, 'checkpoint_path': self.current_checkpoint_dir}
            await websocket.send_bytes(serialise({'meta': meta}))

            client.reset()

            try:
                while True:
                    message = await websocket.receive_bytes()
                    try:
                        raw_obs = deserialise(message)
                        encoded_obs = self.observation_encoder.encode(raw_obs)
                        action_response, _info = client.get_action(encoded_obs)

                        action = {k: v[0] for k, v in action_response.items()}
                        lengths = {len(v) for v in action.values()}
                        time_horizon = lengths.pop()

                        decoded_actions = []
                        for i in range(time_horizon):
                            step_action = {k: v[i] for k, v in action.items()}
                            decoded = self.action_decoder.decode(step_action, raw_obs)
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

    def startup(self):
        """Start the subprocess on server startup."""
        try:
            checkpoint_id = self._resolve_checkpoint_id(None)
            self._ensure_subprocess(checkpoint_id)
        except Exception:
            logger.exception('Failed to start subprocess on startup')

    def shutdown(self):
        """Clean up subprocess on server shutdown."""
        if self.subprocess is not None:
            self.subprocess.stop()

    def serve(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level='info')
        server = uvicorn.Server(config)
        return server.serve()


@cfn.config(
    observation_encoder=obs_cfg.groot,
    action_decoder=act_cfg.groot,
    checkpoint=None,
    port=8000,
    groot_venv_path='/.venv/',
    modality_config='ee',
)
def main(
    observation_encoder: GrootObservationEncoder,
    action_decoder: ActionDecoder,
    checkpoints_dir: str,
    checkpoint: str | None,
    port: int,
    groot_venv_path: str,
    modality_config: str,
):
    """Starts the GR00T inference server with encoding/decoding."""
    with pos3.mirror():
        server = InferenceServer(
            observation_encoder=observation_encoder,
            action_decoder=action_decoder,
            checkpoints_dir=checkpoints_dir,
            checkpoint=checkpoint,
            modality_config=modality_config,
            groot_venv_path=groot_venv_path,
            port=port,
        )
        server.startup()
        try:
            asyncio.run(server.serve())
        except KeyboardInterrupt:
            logger.info('Server stopped by user')
        finally:
            server.shutdown()


if __name__ == '__main__':
    init_logging()
    cfn.cli(main)
