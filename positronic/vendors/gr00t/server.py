import logging
import os
import subprocess
from enum import Enum, auto
from pathlib import Path
from typing import Any

import configuronic as cfn
import msgpack
import msgpack_numpy as mnp
import numpy as np
import pos3
import zmq

from pimm.logging import init_logging
from positronic.offboard import keys as offboard_keys
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT
from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress, wait_for_subprocess_ready, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codec import Codec, RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable
from positronic.utils.checkpoints import list_checkpoints
from positronic.vendors import gr00t
from positronic.vendors.gr00t import codecs

logger = logging.getLogger(__name__)

NUMPY_ARRAY = b'nd'
NUMPY_KIND = b'kind'
NUMPY_OBJECT_KIND = b'O'


class MsgSerializer:
    """N1.7's msgpack-numpy wire format, excluding pickle-bearing object arrays."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes, raw=False)

    @staticmethod
    def decode_custom_classes(obj):
        if isinstance(obj, dict):
            # TODO: Reject object-kind descriptors regardless of the array flag's value.
            if obj.get(NUMPY_ARRAY, obj.get(NUMPY_ARRAY.decode())) and obj.get(
                NUMPY_KIND, obj.get(NUMPY_KIND.decode())
            ) in (NUMPY_OBJECT_KIND, NUMPY_OBJECT_KIND.decode()):
                raise ValueError('Object arrays are not supported by the GR00T wire protocol')
            if obj.get(gr00t.MODALITY_CONFIG):
                return obj[gr00t.AS_JSON]
        return mnp.decode(obj)

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray) and obj.dtype.hasobject:
            raise TypeError('Object arrays are not supported by the GR00T wire protocol')
        return mnp.encode(obj)


class PingResult(Enum):
    SUCCESS = auto()
    FAILURE = auto()


class PolicyClient:
    """Client for communicating with GR00T N1.7 PolicyServer via ZMQ."""

    def __init__(self, host: str = 'localhost', port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.socket = self._make_socket()

    def _make_socket(self):
        socket = self.context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(f'tcp://{self.host}:{self.port}')
        return socket

    def ping(self) -> PingResult:
        try:
            self.call_endpoint(gr00t.PING)
            return PingResult.SUCCESS
        except (zmq.error.ZMQError, RuntimeError):
            return PingResult.FAILURE

    def call_endpoint(self, endpoint: str, data: dict | None = None) -> Any:
        request: dict = {gr00t.ENDPOINT: endpoint}
        if data is not None:
            request[gr00t.DATA] = data

        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            message = self.socket.recv()
        except zmq.error.ZMQError as err:
            self.socket.close(linger=0)
            self.socket = self._make_socket()
            raise RuntimeError(f'GR00T endpoint {endpoint} failed at {self.host}:{self.port}: {err}') from err

        if message == b'ERROR':
            raise RuntimeError('Server error. Make sure the correct policy server is running.')
        response = MsgSerializer.from_bytes(message)

        if isinstance(response, dict) and gr00t.ERROR in response:
            raise RuntimeError(f'Server error: {response[gr00t.ERROR]}')
        return response

    def get_action(self, observation: dict[str, Any]) -> tuple[dict, dict]:
        response = self.call_endpoint(gr00t.GET_ACTION, {gr00t.OBSERVATION: observation, gr00t.OPTIONS: None})
        return tuple(response)

    def reset(self) -> dict[str, Any]:
        return self.call_endpoint(gr00t.RESET, {gr00t.OPTIONS: None})

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


class Gr00tSubprocess:
    """Manages the gr00t ZMQ server subprocess."""

    def __init__(self, model_path: str, groot_venv_path: Path, zmq_port: int = 5555, ready_timeout: float = 120.0):
        self.model_path = model_path
        self.groot_venv_path = groot_venv_path
        self.zmq_port = zmq_port
        self.ready_timeout = ready_timeout
        self.process: subprocess.Popen | None = None
        self._client: PolicyClient | None = None

    def start(self):
        groot_root = Path(__file__).parents[4] / 'gr00t'
        python_bin = str(self.groot_venv_path / 'bin' / 'python')

        command = [python_bin, 'gr00t/eval/run_gr00t_server.py']
        command.extend(['--model_path', str(self.model_path)])
        command.extend(['--embodiment-tag', gr00t.EMBODIMENT])
        command.extend(['--host', '127.0.0.1'])
        command.extend(['--port', str(self.zmq_port)])

        env = os.environ.copy()
        logger.info(f'Starting gr00t subprocess: {" ".join(command)}')
        self.process = subprocess.Popen(command, env=env, cwd=str(groot_root))
        self._wait_for_ready()

    def _wait_for_ready(self):
        client = PolicyClient(host='127.0.0.1', port=self.zmq_port, timeout_ms=2000)
        try:
            wait_for_subprocess_ready(
                lambda: client.ping() is PingResult.SUCCESS,
                lambda: (self.process.poll() is not None, self.process.returncode),
                'gr00t subprocess',
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
                self.process.wait()
            self.process = None


class Gr00tModel(Model):
    """Talks to a GR00T ZMQ server subprocess, which it owns and stops on ``close()``."""

    def __init__(self, groot: Gr00tSubprocess, meta: dict[str, Any]):
        self._groot = groot
        self._meta = meta

    def __call__(self, obs: Obs, *, session_id: str):
        action_response, _info = self._groot.client.get_action(dict(obs))
        action = {k: v[0] for k, v in action_response.items()}
        lengths = {len(v) for v in action.values()}
        assert len(lengths) == 1, f'All values in action must have the same length, got {lengths}'
        time_horizon = lengths.pop()
        return [{k: v[i] for k, v in action.items()} for i in range(time_horizon)]

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self):
        self._groot.stop()


def _step_id(raw: str) -> str:
    """The public id for a ``checkpoint-<raw>`` directory: its step number, free of any zero-padding."""
    return str(int(raw)) if raw.isdigit() else raw


def _checkpoint_dir(model_source: str, checkpoint: str | None) -> str:
    """The ``checkpoint-<raw>`` directory of ``model_source`` that ``checkpoint`` names, else the latest one.

    A directory may zero-pad its step where ``checkpoint`` does not.
    """
    names = list_checkpoints(model_source, prefix=gr00t.CHECKPOINT_PREFIX)
    if checkpoint is None:
        return names[-1]
    wanted = checkpoint.strip('/')
    for name in names:
        raw = name.removeprefix(gr00t.CHECKPOINT_PREFIX)
        if raw == wanted or (raw.isdigit() and wanted.isdigit() and int(raw) == int(wanted)):
            return name
    raise ValueError(f'Checkpoint not found: {checkpoint}. Available: {names}')


def _warm_observation(modalities: dict) -> dict[str, Any]:
    """An observation in the checkpoint's own modalities, once they are ones the DROID adapter serves."""
    for name in (gr00t.VIDEO, gr00t.STATE):
        if modalities[name][gr00t.DELTA_INDICES] != [0]:
            raise ValueError(f'DROID adapter requires current-frame {name}, got {modalities[name]}')
    state_keys = modalities[gr00t.STATE][gr00t.MODALITY_KEYS]
    unknown = sorted(set(state_keys) - set(gr00t.STATE_DIMS))
    if unknown:
        raise ValueError(f'Checkpoint state keys {unknown} are not ones the DROID adapter serves')
    language_key = modalities[gr00t.LANGUAGE][gr00t.MODALITY_KEYS][0]
    if language_key != gr00t.TASK:
        raise ValueError(f'Checkpoint instruction key {language_key} does not match codec key {gr00t.TASK}')
    width, height = gr00t.IMAGE_SIZE
    state = {name: np.zeros((1, 1, gr00t.STATE_DIMS[name]), dtype=np.float32) for name in state_keys}
    if gr00t.EE_POSE in state:
        state[gr00t.EE_POSE][..., 3:] = [1, 0, 0, 0, 1, 0]
    return {
        gr00t.VIDEO: {
            name: np.zeros((1, 1, height, width, 3), dtype=np.uint8)
            for name in modalities[gr00t.VIDEO][gr00t.MODALITY_KEYS]
        },
        gr00t.STATE: state,
        gr00t.LANGUAGE: {gr00t.TASK: [['pick up the object']]},
    }


@cfn.config(
    model_source=gr00t.HF_MODEL_PREFIX + gr00t.BASE_MODEL,
    checkpoint=None,
    groot_venv_path=gr00t.VENV,
    zmq_port=5555,
    ready_timeout=600.0,
)
def gr00t_model(
    model_source: str, checkpoint: str | None, groot_venv_path: str, zmq_port: int, ready_timeout: float
) -> Model:
    """A Hugging Face model (``hf://owner/model``), or one checkpoint of a directory of fine-tuned ones:
    ``checkpoint``, else the latest one.

    Checkpoint ids are step numbers (``'5000'`` for ``checkpoint-5000``). The returned model owns the
    gr00t subprocess.
    """
    model_source = model_source.rstrip('/')
    if model_source.startswith(gr00t.HF_MODEL_PREFIX):
        if checkpoint is not None:
            raise ValueError('checkpoint step selection applies only to fine-tuned checkpoint directories')
        checkpoint_id = model_source.removeprefix(gr00t.HF_MODEL_PREFIX)
        model_path = model_source
    else:
        name = _checkpoint_dir(model_source, checkpoint)
        checkpoint_id = _step_id(name.removeprefix(gr00t.CHECKPOINT_PREFIX))
        model_path = run_with_progress(
            lambda: pos3.download(f'{model_source}/{name}', exclude=[gr00t.OPTIMIZER_FILENAME]),
            f'Downloading checkpoint {name}',
        )
    groot = Gr00tSubprocess(
        model_path=str(model_path),
        groot_venv_path=Path(groot_venv_path).expanduser(),
        zmq_port=zmq_port,
        ready_timeout=ready_timeout,
    )
    try:
        groot.start()
        modalities = groot.client.call_endpoint(gr00t.GET_MODALITY_CONFIG)
        policy = Gr00tModel(
            groot,
            {
                offboard_keys.CHECKPOINT_ID: checkpoint_id,
                policy_keys.TYPE: 'groot',
                policy_keys.CHECKPOINT_PATH: str(model_path),
                'embodiment': gr00t.EMBODIMENT,
                policy_keys.EXPERIMENT_NAME: model_source.split('/')[-1] or '',
            },
        )
        # The subprocess initializes CUDA on its first forward, which outlasts a rig's inference timeout.
        warmup(policy, _warm_observation(modalities))
    except Exception:
        groot.stop()
        raise
    return policy


@cfn.config(codec=codecs.droid)
def pipeline(codec: Codec, fps: float = 15.0, horizon_sec: float = 1.0):
    """Schedule DROID joint commands while the server codec performs checkpoint-specific conversion."""
    return PolicyDeployment(
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps, horizon_sec), RestrictImageSize(*gr00t.IMAGE_SIZE)), codec
    )


droid = pipeline
droid_three_cameras = pipeline.override(codec=codecs.droid_three_cameras)
COMMANDS = {
    'serve': serve.override(model=gr00t_model, pipeline=droid),
    'droid': serve.override(model=gr00t_model, pipeline=droid),
    'droid_three_cameras': serve.override(model=gr00t_model, pipeline=droid_three_cameras),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
