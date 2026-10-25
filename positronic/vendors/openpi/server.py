import logging
import os
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
import pos3
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from websockets.sync.client import ClientConnection, connect

from pimm.logging import init_logging
from positronic.cfg.codecs import franka_hand_frame, panda_hand_frame
from positronic.drivers.roboarm.models import EEFrame
from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress, wait_for_subprocess_ready, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Codec, Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codecs import ACTION, ChangeEEFrame, ResizeMode, TransformImages
from positronic.policy.processors import ChunkedSchedule
from positronic.utils.checkpoints import get_latest_checkpoint
from positronic.vendors import openpi
from positronic.vendors.openpi import codecs, ensure_paligemma_tokenizer

logger = logging.getLogger(__name__)

PREALLOCATE_ENV = 'XLA_PYTHON_CLIENT_PREALLOCATE'


class PinglessWebsocketClientPolicy(WebsocketClientPolicy):
    """A websocket policy client that sends no keepalive pings. It talks to the openpi subprocess on the loopback.

    The subprocess runs ``infer`` on its event loop, so it answers no ping until the inference returns. A first
    inference compiles the model and can take longer than a keepalive window. A dead subprocess closes the socket.
    """

    def _wait_for_server(self) -> tuple[ClientConnection, dict]:
        connection = connect(self._uri, compression=None, max_size=None, ping_interval=None)
        return connection, msgpack_numpy.unpackb(connection.recv())


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
        self._client: PinglessWebsocketClientPolicy | None = None

    def _build_command(self) -> list[str]:
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
        """Start the subprocess and block until it accepts connections."""
        command = self._build_command()
        logger.info(f'Starting OpenPI subprocess: {" ".join(command)}')
        env = os.environ.copy()
        # JAX takes ~75% of the GPU at its first use, so a second server on that GPU finds none free.
        env.setdefault(PREALLOCATE_ENV, 'false')
        self.process = subprocess.Popen(command, env=env, cwd=str(self.openpi_root))
        self._wait_for_ready()

    def _is_ready(self) -> bool:
        """Check if OpenPI subprocess is ready by checking if port is accepting connections."""
        try:
            with socket.create_connection(('127.0.0.1', self.ws_port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            return False

    def _wait_for_ready(self):
        assert self.process is not None
        process = self.process
        wait_for_subprocess_ready(
            self._is_ready, lambda: (process.poll() is not None, process.returncode), 'OpenPI subprocess'
        )

    @property
    def client(self) -> PinglessWebsocketClientPolicy:
        """Get or create WebSocket client for inference."""
        if self._client is None:
            logger.info(f'Creating WebSocket client to OpenPI subprocess on port {self.ws_port}...')
            self._client = PinglessWebsocketClientPolicy(host='127.0.0.1', port=self.ws_port)
            logger.info('WebSocket client created successfully')
        return self._client

    def stop(self):
        self._client = None

        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning('OpenPI subprocess did not terminate, killing it')
                self.process.kill()
            self.process = None


class OpenpiModel(Model):
    """A running OpenPI subprocess; ``close()`` stops the subprocess."""

    def __init__(
        self,
        subproc: OpenpiSubprocess,
        meta: dict[str, Any],
        *,
        transform_actions: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self._subproc = subproc
        self._meta = meta
        self._transform_actions = transform_actions

    def __call__(self, obs: Obs, *, session_id: str):
        response = self._subproc.client.infer(obs)
        actions = response[openpi.ACTIONS]
        if self._transform_actions is not None:
            actions = self._transform_actions(actions)
        return [{ACTION: a} for a in actions]

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self):
        self._subproc.stop()


@cfn.config(config_name='pi05_positronic_lowmem', checkpoint=None, openpi_ws_port=8001)
def openpi_model(
    checkpoints_dir: str,
    config_name: str,
    checkpoint: str | None,
    openpi_ws_port: int,
    transform_actions: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Model:
    """One OpenPI checkpoint under ``checkpoints_dir``: ``checkpoint``, else the latest one, served by a
    serve_policy.py subprocess.

    A ``gs://`` checkpoints_dir is a published openpi checkpoint served as-is: openpi fetches it
    itself via fsspec[gcs] (pos3 handles only s3://), and there are no numeric-step subdirs to
    resolve — the dir is the single model.
    """
    checkpoints_dir = checkpoints_dir.rstrip('/')
    if checkpoints_dir.startswith('gs://'):
        checkpoint_id = checkpoints_dir.rsplit('/', 1)[-1]
        checkpoint_path = checkpoint_dir = checkpoints_dir
    else:
        checkpoint_id = checkpoint or get_latest_checkpoint(checkpoints_dir)
        checkpoint_path = f'{checkpoints_dir}/{checkpoint_id}'
        checkpoint_dir = run_with_progress(
            lambda: pos3.download(checkpoint_path), f'Downloading checkpoint {checkpoint_id}'
        )
    subproc = OpenpiSubprocess(checkpoint_dir=str(checkpoint_dir), config_name=config_name, ws_port=openpi_ws_port)
    try:
        subproc.start()
        policy = OpenpiModel(
            subproc,
            {
                policy_keys.CHECKPOINT_ID: checkpoint_id,
                policy_keys.TYPE: 'openpi',
                policy_keys.CONFIG_NAME: config_name,
                policy_keys.CHECKPOINT_PATH: checkpoint_path,
                policy_keys.EXPERIMENT_NAME: checkpoints_dir.rsplit('/', 1)[-1],
            },
            transform_actions=transform_actions,
        )
        # The subprocess compiles the model on its first inference, which outlasts a rig's inference timeout.
        warmup(policy, openpi.warm_observation())
    except Exception:
        subproc.stop()
        raise
    return policy


# ``ee_frame`` takes no default: a wrong frame does not error, it just puts the arm somewhere else, so a
# deployment states the frame of the checkpoint it binds.
@cfn.config(codec=codecs.ee)
def pipeline(codec: Codec, ee_frame: EEFrame | None, fps: float = 15.0, horizon_sec: float | None = None):
    """The OpenPI serving pipeline: rig-side chunk scheduling and the server-side codec.

    ``ee_frame`` is the frame this checkpoint's poses live in (``cfg.codecs.gripper_frame`` for a checkpoint
    trained on poses at ``GRIPPER_SITE``); ``None`` for one that speaks joints only.
    """
    local = Sequential(ChunkedSchedule(fps, horizon_sec), TransformImages(224, 224, mode=ResizeMode.BOUND))
    if ee_frame is not None:
        # Outermost, so everything downstream — the wire, the server's codec — sees poses already in ``ee_frame``.
        local = Sequential(ChangeEEFrame(ee_frame), local)
    return PolicyDeployment(local, codec)


# These bind no checkpoint, so they state no frame: whoever binds one passes ``--pipeline.ee_frame`` with it.
ee = pipeline
ee_joints = pipeline.override(codec=codecs.ee_joints)
ee_traj = pipeline.override(codec=codecs.ee_traj)
ee_joints_traj = pipeline.override(codec=codecs.ee_joints_traj)
# For checkpoints trained on inverted-grip (1 = open) data, e.g. the sim_stack recordings.
ee_flip_grip = pipeline.override(**{'codec.flip_grip': True})
# The joint-space codecs put no pose on the wire, so a checkpoint bound here speaks joints only. An EE-space
# DROID checkpoint would take ``cfg.codecs.droid_frame`` instead.
joints_traj = pipeline.override(codec=codecs.joints_traj, ee_frame=None)
droid_pipe = pipeline.override(codec=codecs.droid, ee_frame=None, horizon_sec=8 / 15)
droid_jointpos_pipe = pipeline.override(codec=codecs.droid_jointpos, ee_frame=None)
libero_pipe = pipeline.override(codec=codecs.libero, fps=20.0, horizon_sec=0.25)


# Every pipeline is a subcommand, and so is every deployment — a pipeline and the checkpoint it pairs with.
# ``droid``, ``droid_jointpos`` and ``libero`` also pair their codec with the openpi config that reads it.
COMMANDS = {
    'serve': serve.override(model=openpi_model, pipeline=ee),
    'ee': serve.override(model=openpi_model, pipeline=ee),
    'ee_joints': serve.override(model=openpi_model, pipeline=ee_joints),
    'ee_traj': serve.override(model=openpi_model, pipeline=ee_traj),
    'ee_joints_traj': serve.override(model=openpi_model, pipeline=ee_joints_traj),
    'joints_traj': serve.override(model=openpi_model, pipeline=joints_traj),
    'ee_flip_grip': serve.override(model=openpi_model, pipeline=ee_flip_grip),
    # Trained on phail recordings, whose poses are the real Franka's ``F_T_EE``.
    'phail': serve.override(
        model=openpi_model.override(
            checkpoints_dir='s3://checkpoints/phail_unified/openpi/pi05_positronic_lowmem/270226-ee/'
        ),
        pipeline=ee.override(codec=codecs.phail_v1, ee_frame=franka_hand_frame),
    ),
    # The sim_stack checkpoint was trained on inverted-grip (1 = open) sim data, hence the flip-grip pipeline.
    # Its poses are the sim panda's ``EE_LINK``.
    'sim_stack': serve.override(
        model=openpi_model.override(
            checkpoints_dir='s3://checkpoints/sim_stack/openpi/ee/pi05_positronic_lowmem/230226/'
        ),
        pipeline=ee_flip_grip.override(ee_frame=panda_hand_frame),
    ),
    'droid': serve.override(
        model=openpi_model.override(
            checkpoints_dir='s3://PUBLIC@positronic-public/checkpoints/openpi/pi05_droid/', config_name='pi05_droid'
        ),
        pipeline=droid_pipe,
    ),
    # The RoboLab leaderboard policy: openpi's DROID jointpos model, served from the checkpoint their
    # ``policies/pi0_family/README.md`` recipe pins (pass-through mode — openpi fetches gs:// itself).
    'droid_jointpos': serve.override(
        model=openpi_model.override(
            checkpoints_dir='gs://openpi-assets-simeval/pi05_droid_jointpos', config_name='pi05_droid_jointpos'
        ),
        pipeline=droid_jointpos_pipe,
    ),
    # TODO(#557): LIBERO reports its eef 38 mm and 90° from the shipped panda model's ``EE_LINK``, and the codec
    # is calibrated against what the env actually reports. ``panda_hand_frame`` undoes the ``PANDA_HAND_FRAME`` that
    # ``LiberoAdapter`` applies, so the checkpoint sees the poses the env reports.
    'libero': serve.override(
        model=openpi_model.override(
            checkpoints_dir='gs://openpi-assets/checkpoints/pi05_libero',
            config_name='pi05_libero',
            transform_actions=openpi.libero_actions_to_physical,
        ),
        pipeline=libero_pipe.override(ee_frame=panda_hand_frame),
    ),
}


if __name__ == '__main__':
    init_logging()
    ensure_paligemma_tokenizer()
    with pos3.mirror():
        cfn.cli(COMMANDS)
