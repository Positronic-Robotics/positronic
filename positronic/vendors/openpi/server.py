import logging
import os
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from positronic.offboard.server import PolicyServer
from positronic.offboard.server_utils import run_with_progress, wait_for_subprocess_ready
from positronic.policy import Codec, Policy, Session
from positronic.policy.codec import RestrictImageSize
from positronic.policy.spec import ModelSource, remote
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils.checkpoints import get_latest_checkpoint, list_checkpoints
from positronic.utils.logging import init_logging
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

    def start(self, on_progress: Callable[[str], None] | None = None):
        """Start the subprocess and block until it accepts connections, reporting progress."""
        command = self._build_command()
        logger.info(f'Starting OpenPI subprocess: {" ".join(command)}')
        # Don't pipe stdout/stderr so we can see the output
        self.process = subprocess.Popen(command, env=os.environ.copy(), cwd=str(self.openpi_root))
        self._wait_for_ready(on_progress)

    def _check_ready(self) -> bool:
        """Check if OpenPI subprocess is ready by checking if port is accepting connections."""
        try:
            with socket.create_connection(('127.0.0.1', self.ws_port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            return False

    def _wait_for_ready(self, on_progress: Callable[[str], None] | None, timeout: float = 300.0):
        assert self.process is not None
        process = self.process
        wait_for_subprocess_ready(
            self._check_ready,
            lambda: (process.poll() is not None, process.returncode),
            'OpenPI subprocess',
            on_progress,
            max_wait=timeout,
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
# Policy
###########################################################################################


class _OpenpiSession(Session):
    def __init__(self, client: WebsocketClientPolicy):
        self._client = client

    def __call__(self, obs):
        response = self._client.infer(obs)
        actions = response['actions']
        return [{'action': a} for a in actions]


class OpenpiPolicy(Policy):
    """A running OpenPI subprocess as a Policy; ``close()`` stops the subprocess."""

    def __init__(self, subproc: OpenpiSubprocess):
        self._subproc = subproc

    def new_session(self, context=None, now=None):
        client = self._subproc.client
        client.reset()
        return _OpenpiSession(client)

    def close(self):
        self._subproc.stop()


###########################################################################################
# Model source
###########################################################################################


class OpenpiSource(ModelSource):
    """OpenPI checkpoints under ``checkpoints_dir``; each load boots a serve_policy.py subprocess.

    A ``gs://`` checkpoints_dir is a published openpi checkpoint served as-is: openpi fetches it
    itself via fsspec[gcs] (pos3 handles only s3://), and there are no numeric-step subdirs to
    resolve — the dir is the single model.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        config_name: str = 'pi05_positronic_lowmem',
        checkpoint: str | None = None,
        openpi_ws_port: int = 8001,
    ):
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/')
        self.config_name = config_name
        self.checkpoint = checkpoint
        self.openpi_ws_port = openpi_ws_port

    @property
    def _passthrough(self) -> bool:
        return self.checkpoints_dir.startswith('gs://')

    def get_models(self) -> list[str]:
        if self._passthrough:
            return [self.checkpoints_dir.rsplit('/', 1)[-1]]
        checkpoints = list_checkpoints(self.checkpoints_dir)
        return [str(n) for n in sorted(int(cp) for cp in checkpoints if cp.isdigit())]

    def resolve(self, model_id: str | None) -> str:
        """Digit ids match numerically ('5000' resolves to '005000'); ``None`` picks the configured
        checkpoint, else the latest."""
        if self._passthrough:
            served = self.checkpoints_dir.rsplit('/', 1)[-1]
            if model_id and model_id != served:
                raise ValueError(f'Checkpoint not found or invalid ID: {model_id}. This server serves only {served}.')
            return served
        if model_id:
            available = list_checkpoints(self.checkpoints_dir)
            if model_id.isdigit():
                target = int(model_id)
                for cp in available:
                    if cp.isdigit() and int(cp) == target:
                        return cp
            raise ValueError(f'Checkpoint not found or invalid ID: {model_id}.')
        if self.checkpoint:
            return self.checkpoint
        return get_latest_checkpoint(self.checkpoints_dir)

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        if self._passthrough:
            checkpoint_dir = self.checkpoints_dir  # openpi's subprocess downloads gs:// itself
        else:
            path = f'{self.checkpoints_dir}/{model_id}'
            checkpoint_dir = run_with_progress(
                lambda: pos3.download(path), f'Downloading checkpoint {model_id}', on_progress
            )
        subproc = OpenpiSubprocess(
            checkpoint_dir=str(checkpoint_dir), config_name=self.config_name, ws_port=self.openpi_ws_port
        )
        try:
            subproc.start(on_progress)
        except Exception:
            subproc.stop()
            raise
        return OpenpiPolicy(subproc)

    def meta(self, model_id: str) -> dict[str, Any]:
        return {
            'type': 'openpi',
            'config_name': self.config_name,
            'checkpoint_path': self.checkpoints_dir,
            'experiment_name': self.checkpoints_dir.rsplit('/', 1)[-1],
        }


###########################################################################################
# Server configs
###########################################################################################


openpi_source = cfn.Config(OpenpiSource)


@cfn.config(codec=codecs.ee, source=openpi_source)
def pipe(codec: Codec, source: ModelSource):
    """The OpenPI serving pipe: rig-side chunk scheduling, the server-side codec, the checkpoint source."""
    return ChunkedSchedule() | RestrictImageSize.from_codec(codec) | remote | codec | source


PIPES = {
    'ee': pipe,
    'ee_joints': pipe.override(codec=codecs.ee_joints),
    'ee_traj': pipe.override(codec=codecs.ee_traj),
    'ee_joints_traj': pipe.override(codec=codecs.ee_joints_traj),
    'joints_traj': pipe.override(codec=codecs.joints_traj),
    # For checkpoints trained on inverted-grip (1 = open) data, e.g. the sim_stack recordings.
    'ee_flip_grip': pipe.override(**{'codec.flip_grip': True}),
    'droid': pipe.override(codec=codecs.droid, **{'source.config_name': 'pi05_droid'}),
    'droid_jointpos': pipe.override(codec=codecs.droid_jointpos, **{'source.config_name': 'pi05_droid_jointpos'}),
    'libero': pipe.override(codec=codecs.libero, **{'source.config_name': 'pi05_libero'}),
}


@cfn.config(
    pipe='ee',
    checkpoints_dir='',
    config_name=None,
    checkpoint=None,
    host='0.0.0.0',
    port=8000,
    openpi_ws_port=8001,
    recording_dir=None,
    idle_timeout_min=None,
)
def main(
    pipe: str,
    checkpoints_dir: str,
    config_name: str | None,
    checkpoint: str | None,
    host: str,
    port: int,
    openpi_ws_port: int,
    recording_dir: str | None,
    idle_timeout_min: float | None,
):
    """OpenPI inference server.

    Args:
        pipe: Named policy pipe from ``PIPES`` — picks the codec and, for droid/droid_jointpos/libero,
            the paired OpenPI config.
        checkpoints_dir: Directory containing model checkpoints (``gs://`` serves a published openpi
            checkpoint as-is).
        config_name: OpenPI config name; overrides the pipe's pairing (base pipes use
            pi05_positronic_lowmem).
        checkpoint: Specific checkpoint to serve by default (defaults to latest).
        host: Server host address.
        port: Server port.
        openpi_ws_port: Internal WebSocket port for the OpenPI subprocess.
        recording_dir: Directory for recording .rrd files (optional, supports S3 paths).
        idle_timeout_min: Shut down after this many minutes without activity.
    """
    overrides: dict[str, Any] = {
        'source.checkpoints_dir': checkpoints_dir,
        'source.checkpoint': checkpoint,
        'source.openpi_ws_port': openpi_ws_port,
    }
    if config_name is not None:
        overrides['source.config_name'] = config_name
    cfg = PIPES[pipe].override(**overrides)
    PolicyServer(cfg, host=host, port=port, recording_dir=recording_dir, idle_timeout_min=idle_timeout_min).serve()


phail = main.override(
    checkpoints_dir='s3://checkpoints/phail_unified/openpi/pi05_positronic_lowmem/270226-ee/',
    recording_dir='s3://inference/phail_unified/server_recordings/openpi/270226-ee/',
)
# The sim_stack checkpoint was trained on inverted-grip (1 = open) sim data, hence the flip-grip pipe.
sim_stack = main.override(
    pipe='ee_flip_grip',
    checkpoints_dir='s3://checkpoints/sim_stack/openpi/ee/pi05_positronic_lowmem/230226/',
    recording_dir='s3://inference/sim_stack/server_recordings/openpi/230226/',
)
droid = main.override(pipe='droid', checkpoints_dir='s3://PUBLIC@positronic-public/checkpoints/openpi/pi05_droid/')
# The RoboLab leaderboard policy: openpi's DROID jointpos model, served from the checkpoint their
# ``policies/pi0_family/README.md`` recipe pins (pass-through mode — openpi fetches gs:// itself).
droid_jointpos = main.override(pipe='droid_jointpos', checkpoints_dir='gs://openpi-assets-simeval/pi05_droid_jointpos')
libero = main.override(pipe='libero', checkpoints_dir='gs://openpi-assets/checkpoints/pi05_libero')


if __name__ == '__main__':
    init_logging()
    ensure_paligemma_tokenizer()
    with pos3.mirror():
        cfn.cli({
            'serve': main,
            'phail': phail,
            'sim_stack': sim_stack,
            'droid': droid,
            'droid_jointpos': droid_jointpos,
            'libero': libero,
        })
