"""DreamZero inference server: the roboarena subprocess model source and its serving pipelines."""

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

from pimm.logging import init_logging
from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress, wait_for_subprocess_ready
from positronic.policy import Codec, Layer, Policy, Session
from positronic.policy import keys as policy_keys
from positronic.policy.codec import RestrictImageSize
from positronic.policy.spec import ModelSource, remote
from positronic.utils.checkpoints import list_checkpoints
from positronic.utils.serialization import deserialize, serialize
from positronic.vendors.dreamzero import codecs, roboarena

logger = logging.getLogger(__name__)


def _dreamzero_root():
    return Path(__file__).parents[4] / 'dreamzero'


def _download_checkpoint(model_path: str) -> Path:
    """Local checkpoint dir for ``model_path``: an ``s3://`` URL or local path via pos3, else a HuggingFace repo."""
    local = os.path.expanduser(model_path)
    if model_path.startswith('s3://') or os.path.exists(local):
        return pos3.download(local)
    return Path(snapshot_download(model_path))


def _is_run_directory(model_path: str) -> bool:
    """Whether ``model_path`` holds ``checkpoint-N`` children rather than being one checkpoint itself.

    Decided by shape, not by name: a run directory may be called anything, including the step number of
    the checkpoint inside it.
    """
    last = model_path.rstrip('/').split('/')[-1]
    return model_path.startswith('s3://') and not last.startswith('checkpoint-')


def _checkpoint_id(checkpoint_path: str) -> str:
    """The public id for a checkpoint: the step a ``checkpoint-N`` directory names, else the path itself.

    The step is kept as the directory writes it, zero-padding and all, so the id maps back to a directory
    that exists. Anything else — a HuggingFace repo, a local path — names no step and stays whole, since
    that whole string is the id a client addresses it by.
    """
    last = checkpoint_path.rstrip('/').split('/')[-1]
    return last.removeprefix('checkpoint-') if last.startswith('checkpoint-') else checkpoint_path


def _experiment_name(checkpoint_path: str) -> str:
    """The training run a resolved checkpoint belongs to."""
    parts = checkpoint_path.rstrip('/').split('/')
    return parts[-2] if len(parts) >= 2 and parts[-1].startswith('checkpoint-') else parts[-1]


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
        self._server_config: dict | None = None

    def connect(self):
        self._ws = websockets.sync.client.connect(
            f'ws://{self._host}:{self._port}', compression=None, max_size=None, ping_interval=60, ping_timeout=600
        )
        # First message from server is PolicyServerConfig metadata
        self._server_config = deserialize(self._ws.recv())
        logger.info(f'Connected to roboarena server, metadata: {self._server_config}')

    @property
    def server_config(self) -> dict:
        """The ``PolicyServerConfig`` this backend announced on connect: which cameras it wants, at what
        resolution, and whether it tracks sessions."""
        if self._server_config is None:
            raise RuntimeError('Not connected: the server announces its config on connect')
        return self._server_config

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


def _warm_observation(server_config: dict, session_id: str) -> dict[str, Any]:
    """Zero-filled inputs at the geometry and camera count ``server_config`` announced on connect."""
    if server_config[roboarena.NEEDS_STEREO_CAMERA]:
        raise ValueError('roboarena server asks for stereo cameras, which this source does not send')
    resolution = server_config[roboarena.RESOLUTION]
    if resolution is None:
        # Both backbone scripts announce one — wan2.1 pins 320x180, wan2.2 reports whatever it was configured
        # with — so an absent resolution is the protocol's optional field going unset, not a size to infer.
        # The codec's geometry is a rig-side setting this source cannot see, and the backbones disagree on it.
        raise ValueError('roboarena server announced no image resolution, so there is no geometry to warm it at')
    height, width = resolution
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    obs: dict[str, Any] = {
        roboarena.JOINT_POSITION: np.zeros(7, dtype=np.float32),
        roboarena.GRIPPER_POSITION: np.zeros(1, dtype=np.float32),
        roboarena.PROMPT: '',
        roboarena.SESSION_ID: session_id,
    }
    if server_config[roboarena.NEEDS_WRIST_CAMERA]:
        obs[roboarena.WRIST_IMAGE] = frame
    for i in range(server_config[roboarena.NUM_EXTERIOR_CAMERAS]):
        obs[roboarena.exterior_image(i)] = frame
    return obs


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

    def warmup(self, on_progress: Callable[[str], None] | None = None):
        """Run one inference so the backbone's first-call cost is paid before a rig connects.

        On its own connection, because the observation is built from what the server announces there. The
        backbone keeps per-session frame history, so this resets the session it opened rather than leaving it.
        """
        client = RoboarenaClient(port=self.roboarena_port)
        client.connect()
        session_id = str(uuid.uuid4())
        try:
            obs = _warm_observation(client.server_config, session_id)
            run_with_progress(lambda: client.infer(obs), 'Running warmup inference', on_progress)
            client.reset(session_id=session_id)
        finally:
            client.close()

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

    def __call__(self, obs, time_ns):
        obs = dict(obs)
        obs[roboarena.SESSION_ID] = self._session_id
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

    def new_session(self, context=None, rt=None):
        client = RoboarenaClient(port=self._subprocess.roboarena_port)
        client.connect()
        return _DreamZeroSession(client, str(uuid.uuid4()))

    def close(self):
        self._subprocess.stop()


class DreamZeroSource(ModelSource):
    """DreamZero checkpoints served through a torchrun subprocess speaking the roboarena protocol.

    ``model_path`` is an ``s3://`` run directory (served at its latest ``checkpoint-N``), a pinned
    checkpoint dir, a HuggingFace repo, or a local path. Model ids are checkpoint step numbers
    (``'100000'`` for ``checkpoint-100000``).
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

    def _checkpoint_path(self, model_id: str) -> str:
        """The checkpoint directory whose public id is ``model_id``.

        Composed rather than looked up: a session handshake reaches here, and must not need the
        checkpoint bucket to describe weights that are already loaded.
        """
        if not _is_run_directory(self._model_path):
            return self._model_path
        return f'{self._model_path.rstrip("/")}/checkpoint-{model_id}'

    def get_models(self) -> list[str]:
        if not _is_run_directory(self._model_path):
            return [_checkpoint_id(self._model_path)]
        return [_checkpoint_id(c) for c in list_checkpoints(self._model_path, prefix='checkpoint-')]

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        checkpoint_path = self._checkpoint_path(model_id)
        local_path = run_with_progress(
            lambda: _download_checkpoint(checkpoint_path), 'Downloading DreamZero checkpoint', on_progress
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
            sp.warmup(on_progress)
        except Exception:
            sp.stop()
            raise
        return DreamZeroPolicy(sp)

    def meta(self, model_id: str) -> dict[str, Any]:
        checkpoint_path = self._checkpoint_path(model_id)
        return {
            policy_keys.TYPE: 'dreamzero',
            'backbone': self._backbone,
            'num_gpus': self._num_gpus,
            policy_keys.CHECKPOINT_PATH: checkpoint_path,
            policy_keys.EXPERIMENT_NAME: _experiment_name(checkpoint_path),
        }


dreamzero_source = cfn.Config(DreamZeroSource)


@cfn.config(local=codecs.dreamzero_layers, codec=codecs.joints, source=dreamzero_source, width=320, height=176)
def pipeline(local: Layer, codec: Codec, source: ModelSource, width: int, height: int):
    """One DreamZero serving pipeline: the rig-side AR video context, the codec, the subprocess-backed source.

    ``width``/``height`` bound frames on the rig and follow the codec's own geometry.
    """
    return local | RestrictImageSize(width, height) | remote | codec | source


joints = pipeline
joints_traj = pipeline.override(codec=codecs.joints_traj)
joints_ik = pipeline.override(codec=codecs.joints_ik)
joints_ik_sim = pipeline.override(codec=codecs.joints_ik_sim)
# The pretrained DROID model (wan2.1) asserts exactly 320x180 frames.
droid = pipeline.override(codec=codecs.droid, height=180)


# Every pipeline is a subcommand, and so is every deployment — a pipeline with its checkpoint bound.
COMMANDS = {
    'serve': serve.override(pipeline=joints),
    'joints': serve.override(pipeline=joints),
    'joints_traj': serve.override(pipeline=joints_traj),
    'joints_ik': serve.override(pipeline=joints_ik),
    'joints_ik_sim': serve.override(pipeline=joints_ik_sim),
    # The PhAIL fine-tune. Trained with the joints_ik codec, whose inference decode is the shared joints
    # one, so the joints pipeline serves it; the backbone must be the one the run was trained on.
    # TODO: publish this checkpoint to positronic-public and point here, as the other PhAIL models are
    # (`utilities/release_phail.py`, `positronic.cfg.phail.v1_0.models`). Reading it needs credentials until then.
    'phail': serve.override(
        pipeline=joints.override(codec=codecs.phail_v1),
        recording_dir='s3://inference/phail_unified/server_recordings/dreamzero/w22f1_100k_200626/',
        **{
            'pipeline.source.model_path': 's3://checkpoints/phail/dreamzero/w22f1_100k_200626/',
            'pipeline.source.backbone': 'wan2.2',
        },
    ),
    # Public pretrained DROID checkpoint: wan2.1 backbone (the base default) paired with the DROID
    # pipeline whose codec feeds its required 320x180 frames.
    'droid': serve.override(pipeline=droid.override(**{'source.model_path': 'GEAR-Dreams/DreamZero-DROID'})),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
