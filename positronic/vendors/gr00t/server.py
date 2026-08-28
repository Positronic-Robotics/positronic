import io
import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import msgpack
import numpy as np
import pos3
import zmq

from pimm.logging import init_logging
from positronic import keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT
from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress, wait_for_subprocess_ready, warmup
from positronic.policy import Policy, Session
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.policy.spec import ModelSource, remote
from positronic.utils.checkpoints import list_checkpoints
from positronic.vendors import gr00t
from positronic.vendors.gr00t import codecs

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

    def __init__(
        self,
        checkpoint_dir: str,
        modality_config_path: Path,
        groot_venv_path: str,
        zmq_port: int = 5555,
        ready_timeout: float = 120.0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.modality_config_path = modality_config_path
        self.groot_venv_path = groot_venv_path
        self.zmq_port = zmq_port
        self.ready_timeout = ready_timeout
        self.process: subprocess.Popen | None = None
        self._client: PolicyClient | None = None

    def start(self, on_progress: Callable[[str], None] | None = None):
        groot_root = Path(__file__).parents[4] / 'gr00t'
        python_bin = str(Path(self.groot_venv_path) / 'bin' / 'python')

        command = [python_bin, 'gr00t/eval/run_gr00t_server.py']
        command.extend(['--model_path', str(self.checkpoint_dir)])
        command.extend(['--embodiment_tag', 'NEW_EMBODIMENT'])
        command.extend(['--modality_config_path', str(self.modality_config_path)])
        command.extend(['--host', '127.0.0.1'])
        command.extend(['--port', str(self.zmq_port)])

        env = os.environ.copy()
        logger.info(f'Starting gr00t subprocess: {" ".join(command)}')
        self.process = subprocess.Popen(command, env=env, cwd=str(groot_root))
        self._wait_for_ready(on_progress)

    def _wait_for_ready(self, on_progress: Callable[[str], None] | None):
        client = PolicyClient(host='127.0.0.1', port=self.zmq_port, timeout_ms=2000)
        try:
            wait_for_subprocess_ready(
                client.ping,
                lambda: (self.process.poll() is not None, self.process.returncode),
                'gr00t subprocess',
                on_progress,
                max_wait=self.ready_timeout,
            )
        finally:
            client.close()

    @property
    def client(self) -> PolicyClient:
        if self._client is None:
            # The backend must not give up before the rig does, so this follows the rig's own per-call bound.
            # A warmup runs through here too, and pays the cold-start cost that bound exists to cover.
            timeout_ms = int(DEFAULT_INFER_TIMEOUT * 1000)
            self._client = PolicyClient(host='127.0.0.1', port=self.zmq_port, timeout_ms=timeout_ms)
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
# Policy and model source
###########################################################################################


class _Gr00tSession(Session):
    def __init__(self, client: PolicyClient, meta: dict[str, Any]):
        self._client = client
        self._meta = meta

    def __call__(self, obs, time_ns):
        action_response, _info = self._client.get_action(obs)
        action = {k: v[0] for k, v in action_response.items()}
        lengths = {len(v) for v in action.values()}
        assert len(lengths) == 1, f'All values in action must have the same length, got {lengths}'
        time_horizon = lengths.pop()
        return [{k: v[i] for k, v in action.items()} for i in range(time_horizon)]

    @property
    def meta(self):
        return self._meta


class Gr00tPolicy(Policy):
    """Talks to a GR00T ZMQ server subprocess, which it owns and stops on ``close()``."""

    def __init__(self, groot: Gr00tSubprocess, checkpoint_path: str):
        self._groot = groot
        self._meta = {keys.CHECKPOINT_PATH: checkpoint_path}

    def new_session(self, context=None, rt=None):
        self._groot.client.reset()
        return _Gr00tSession(self._groot.client, self._meta)

    def close(self):
        self._groot.stop()


def _step_id(raw: str) -> str:
    """The public id for a ``checkpoint-<raw>`` directory: its step number, free of any zero-padding."""
    return str(int(raw)) if raw.isdigit() else raw


def _warm_observation(modality: gr00t.ModalityConfig) -> dict[str, Any]:
    """Zero-filled inputs in GR00T's nested format, carrying the state block ``modality`` declares.

    Leading axes are ``(batch, time)``, the way a session hands one step over.
    """
    width, height = gr00t.IMAGE_SIZE
    frame = np.zeros((1, 1, height, width, 3), dtype=np.uint8)
    return {
        gr00t.VIDEO: dict.fromkeys(modality.cameras, frame),
        gr00t.STATE: {key: np.zeros((1, 1, dim), dtype=np.float32) for key, dim in modality.state.items()},
        gr00t.LANGUAGE: {modality.task_key: [['']]},
    }


class Gr00tSource(ModelSource):
    """GR00T checkpoints under ``checkpoints_dir``, each served through a dedicated ZMQ subprocess.

    Model ids are checkpoint step numbers (``'5000'`` for ``checkpoint-5000``). ``load`` downloads the
    checkpoint and boots the gr00t subprocess; the returned policy owns the subprocess.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        checkpoint: str | None = None,
        modality_config: str | gr00t.ModalityConfig = 'ee',
        groot_venv_path: str = '/.venv/',
        zmq_port: int = 5555,
        ready_timeout: float = 120.0,
    ):
        if isinstance(modality_config, str):
            if modality_config not in gr00t.MODALITY_CONFIGS:
                raise ValueError(
                    f'Unknown modality config: {modality_config}. Available: {sorted(gr00t.MODALITY_CONFIGS)}. '
                    'A config of your own is passed as a ModalityConfig, which states the state block to warm it with'
                )
            self._modality = gr00t.MODALITY_CONFIGS[modality_config]
        else:
            self._modality = modality_config
        self.checkpoints_dir = checkpoints_dir.rstrip('/')
        self.checkpoint = checkpoint
        # What to call the config being served: the alias where there is one, else the module it points at.
        self.modality_config = modality_config if isinstance(modality_config, str) else str(self._modality.path)
        self.groot_venv_path = groot_venv_path
        self.zmq_port = zmq_port
        self.ready_timeout = ready_timeout

    def _raw_ids(self) -> list[str]:
        return [cp.removeprefix('checkpoint-') for cp in list_checkpoints(self.checkpoints_dir, prefix='checkpoint-')]

    def _raw_for(self, model_id: str) -> str:
        """The directory's own suffix for ``model_id``, which may be zero-padded where the advertised id is not."""
        for r in self._raw_ids():
            if r == model_id or (r.isdigit() and model_id.isdigit() and int(r) == int(model_id)):
                return r
        raise ValueError(f'Checkpoint not found: {model_id}. Available: {self.get_models()}')

    def get_models(self) -> list[str]:
        return [_step_id(r) for r in self._raw_ids()]

    def resolve(self, model_id: str | None) -> str:
        """Explicit id > the configured ``checkpoint`` > latest, always as the id ``get_models`` advertises.

        The zero-padding a directory may carry stays out of the public id; ``load`` puts it back to reach
        the directory.
        """
        if model_id is None and self.checkpoint is not None:
            model_id = str(self.checkpoint).strip('/')
        if model_id is None:
            return _step_id(self._raw_ids()[-1])
        return _step_id(self._raw_for(model_id))

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        checkpoint_path = f'{self.checkpoints_dir}/checkpoint-{self._raw_for(model_id)}'
        logger.info(f'Downloading checkpoint {checkpoint_path}')
        checkpoint_dir = run_with_progress(
            lambda: pos3.download(checkpoint_path, exclude=['optimizer.pt']),
            f'Downloading checkpoint checkpoint-{model_id}',
            on_progress,
        )
        groot = Gr00tSubprocess(
            checkpoint_dir=str(checkpoint_dir),
            modality_config_path=self._modality.path,
            groot_venv_path=self.groot_venv_path,
            zmq_port=self.zmq_port,
            ready_timeout=self.ready_timeout,
        )
        try:
            groot.start(on_progress)
            policy = Gr00tPolicy(groot, str(checkpoint_dir))
            # The subprocess initializes CUDA on its first forward, which outlasts a rig's inference timeout.
            warmup(policy, _warm_observation(self._modality), on_progress)
        except Exception:
            groot.stop()
            raise
        return policy

    def meta(self, model_id: str) -> dict[str, Any]:
        return {
            keys.TYPE: 'groot',
            'modality_config': self.modality_config,
            keys.EXPERIMENT_NAME: self.checkpoints_dir.split('/')[-1] or '',
        }


###########################################################################################
# Serving configs
###########################################################################################


gr00t_source = cfn.Config(Gr00tSource)


# No ``ee_frame``: every checkpoint served here was trained on poses the rig reported in its ``default``,
# so none has a transform to declare.
@cfn.config(codec=codecs.ee_quat, source=gr00t_source)
def pipeline(codec, source):
    return StopOnFault() | ChunkedSchedule() | RestrictImageSize(*gr00t.IMAGE_SIZE) | remote | codec | source


# Each entry pairs the codec with the matching GR00T modality config; they must agree with training.
ee = pipeline
ee_joints = pipeline.override(codec=codecs.ee_quat_joints, **{'source.modality_config': 'ee_q'})
ee_rot6d = pipeline.override(codec=codecs.ee_rot6d, **{'source.modality_config': 'ee_rot6d'})
ee_rot6d_joints = pipeline.override(codec=codecs.ee_rot6d_joints, **{'source.modality_config': 'ee_rot6d_q'})
ee_rot6d_rel = pipeline.override(codec=codecs.ee_rot6d, **{'source.modality_config': 'ee_rot6d_rel'})
ee_rot6d_joints_rel = pipeline.override(codec=codecs.ee_rot6d_joints, **{'source.modality_config': 'ee_rot6d_q_rel'})
# The sim_stack checkpoint was trained on inverted-grip (1 = open) sim data, hence flip_grip.
sim_stack_pipe = pipeline.override(
    codec=codecs.ee_rot6d.override(flip_grip=True), **{'source.modality_config': 'ee_rot6d'}
)


# Every pipeline is a subcommand, and so is every deployment — a pipeline with its checkpoints bound.
COMMANDS = {
    'serve': serve.override(pipeline=ee),
    'ee': serve.override(pipeline=ee),
    'ee_joints': serve.override(pipeline=ee_joints),
    'ee_rot6d': serve.override(pipeline=ee_rot6d),
    'ee_rot6d_joints': serve.override(pipeline=ee_rot6d_joints),
    'ee_rot6d_rel': serve.override(pipeline=ee_rot6d_rel),
    'ee_rot6d_joints_rel': serve.override(pipeline=ee_rot6d_joints_rel),
    'phail': serve.override(
        pipeline=ee_rot6d_rel.override(
            codec=codecs.phail_v1,
            **{'source.checkpoints_dir': 's3://checkpoints/phail_unified/groot/270226-ee_rot6d_rel/'},
        ),
        recording_dir='s3://inference/phail_unified/server_recordings/groot/270226-ee_rot6d_rel/',
    ),
    'sim_stack': serve.override(
        pipeline=sim_stack_pipe.override(**{
            'source.checkpoints_dir': 's3://checkpoints/sim_stack/groot/ee_rot6d/230226/'
        }),
        recording_dir='s3://inference/sim_stack/server_recordings/groot/230226/',
    ),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
