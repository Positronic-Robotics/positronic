"""Serve a pi0.5 checkpoint fine-tuned on ABC's real data over positronic's session protocol.

The policy runs in the openpi environment of the author's fork, as the author's own openpi server, on this machine.
A session picks the crop of its views in the query of its URL, for example `?crop=yambox`.

Commands
  bottles_16k  The bottles checkpoint: one task, its own norm stats.
  abc4_30k     A checkpoint fine-tuned on the real data of four ABC tasks, with the prompt of each task.

Usage
  python -m positronic.vendors.pi05_abc.server <command> --model.checkpoint=<checkpoint dir> --model.fork=<fork>
"""

import os
import socket
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
from platform_client.policy_container import WARMUP_DEADLINE_S

from pimm.logging import init_logging
from positronic.offboard.roboarena import RoboarenaClient
from positronic.offboard.server import serve
from positronic.offboard.server_utils import wait_for_subprocess_ready, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codecs import ACTION, ResizeMode, TransformImages
from positronic.policy.processors import ChunkedSchedule, TrajectoryPlayer
from positronic.vendors import pi05_abc
from positronic.vendors.pi05_abc import codecs

# The norm stats of each checkpoint, in its checkpoint directory.
BOTTLES_NORM_STATS = Path('assets/put_the_plastic_bottles_in_the_bin/norm_stats.json')
ABC4_NORM_STATS = Path('assets/abc4_yam/norm_stats.json')
# The fork's interpreter, the author's server, and the openpi sources where the author runs the server.
FORK_PYTHON = Path('.venv/bin/python')
FORK_SERVER = Path('yam_abc_reproduce/deploy/servers/openpi_server.py')
FORK_OPENPI = Path('third_party/policy/openpi')
# The fork's `YamInputs` reads `observation/image`, `observation/left_wrist`, `observation/right_wrist` and
# `observation/state`; the server snapshot renames the camera roles to these keys.
IMAGE_KEY_MAP = f'{pi05_abc.TOP_CAMERA}=image,{pi05_abc.LEFT_CAMERA}=left_wrist,{pi05_abc.RIGHT_CAMERA}=right_wrist'
FLATTEN_PREFIX = 'observation/'
# JAX takes most of the GPU at its first use unless this is false, so a second server on that GPU finds none free.
PREALLOCATE_ENV = 'XLA_PYTHON_CLIENT_PREALLOCATE'
# The client plays this many actions of each 50-action chunk, as the author's evaluation does.
EXECUTED_ACTIONS = 15


class Pi05AbcModel(Model):
    """The checkpoint behind the author's server in a child process, which this model owns."""

    def __init__(self, backend: subprocess.Popen, port: int, meta: dict[str, Any]):
        self._backend = backend
        self._client = RoboarenaClient(port=port)
        self._meta = meta

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        return [{ACTION: action} for action in np.asarray(self._client.infer(obs)[pi05_abc.ACTIONS])]

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self) -> None:
        self._client.close()
        self._backend.terminate()
        try:
            self._backend.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._backend.kill()
            self._backend.wait()


def _warm_observation() -> dict[str, Any]:
    """Black views and a zero state, in the fields of the author's server."""
    width, height = codecs.VIEW_SIZE
    view = np.zeros((3, height, width), dtype=np.uint8)
    cameras = (pi05_abc.TOP_CAMERA, pi05_abc.LEFT_CAMERA, pi05_abc.RIGHT_CAMERA)
    return {
        pi05_abc.STATE: np.zeros(codecs.BIMANUAL_SIZE, dtype=np.float32),
        pi05_abc.IMAGES: dict.fromkeys(cameras, view),
        pi05_abc.PROMPT: codecs.TASK,
    }


def _exit_status(process: subprocess.Popen) -> tuple[bool, int | None]:
    code = process.poll()
    return code is not None, code


@cfn.config(config_name='pi05_abc130k', port=8001, norm_stats=BOTTLES_NORM_STATS)
def pi05_abc_model(checkpoint: str, fork: str, config_name: str, port: int, norm_stats: Path) -> Model:
    """The checkpoint directory `checkpoint`, served by the author's server in the fork checkout `fork`, with the norm
    stats file `norm_stats` of the checkpoint directory.

    The author's server loads the checkpoint and compiles the model before it listens.
    """
    # The child runs in the fork's openpi directory, where a relative path names another file.
    checkpoint_dir, fork_dir = Path(checkpoint).resolve(), Path(fork).resolve()
    # rules-allow: hardcoded-keys — the flags of the author's server as it parses them.
    command = [
        fork_dir / FORK_PYTHON,
        fork_dir / FORK_SERVER,
        '--host',
        '127.0.0.1',
        '--port',
        str(port),
        '--config',
        config_name,
        '--checkpoint',
        checkpoint_dir,
        '--norm-stats-path',
        checkpoint_dir / norm_stats,
        '--prompt',
        codecs.TASK,
        '--image-key-map',
        IMAGE_KEY_MAP,
        '--flatten-prefix',
        FLATTEN_PREFIX,
        '--state-key',
        f'{FLATTEN_PREFIX}state',
    ]
    # Refuse a port another server holds: the readiness probe would reach that server, not this checkpoint.
    with socket.socket() as port_check:
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_check.bind(('127.0.0.1', port))
    env = os.environ.copy()
    env.setdefault(PREALLOCATE_ENV, 'false')
    backend = subprocess.Popen(command, env=env, cwd=fork_dir / FORK_OPENPI)
    meta = {
        policy_keys.CHECKPOINT_ID: checkpoint_dir.name,
        policy_keys.TYPE: 'openpi',
        policy_keys.CONFIG_NAME: config_name,
        policy_keys.CHECKPOINT_PATH: str(checkpoint_dir),
    }
    model = Pi05AbcModel(backend, port, meta)
    try:
        probe = RoboarenaClient(port=port)
        wait_for_subprocess_ready(
            lambda: probe.probe() is None,
            partial(_exit_status, backend),
            "the author's openpi server",
            max_wait=WARMUP_DEADLINE_S,
        )
        # The author's server swallows a failed warm-up of its own, so this one proves that the policy answers.
        warmup(model, _warm_observation())
    except Exception:
        model.close()
        raise
    return model


@cfn.config(crop=codecs.Crop.YAMBOX_WIDE_TOP.value, prompts=codecs.BOTTLES_PROMPTS)
def pipeline(crop: str, prompts: dict[str, str]) -> PolicyDeployment:
    """The views cropped by the set `crop` of `codecs.CROPS`, which a session sets in its URL query, and the prompt of
    each task instruction in `prompts`. The client plays the first actions of each chunk at 30 Hz, and shrinks each
    view to fit the view size, or twice that size for a cropped view."""
    try:
        crop_set = codecs.Crop(crop)
    except ValueError:
        raise ValueError(f'Unknown crop {crop!r}; the crops are {sorted(c.value for c in codecs.Crop)}') from None
    crops = codecs.CROPS[crop_set]
    width, height = codecs.VIEW_SIZE
    upload_size = (2 * width, 2 * height) if crops else codecs.VIEW_SIZE
    schedule = Sequential(
        TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(codecs.FPS, horizon_sec=EXECUTED_ACTIONS / codecs.FPS)
    )
    return PolicyDeployment(
        Sequential(schedule, TransformImages(*upload_size, mode=ResizeMode.BOUND)),
        codecs.abc.override(prompts=prompts, crops=crops).instantiate(),
    )


# The checkpoint fine-tuned on the real data of four ABC tasks, and the prompt of each task.
abc4_model = pi05_abc_model.override(norm_stats=ABC4_NORM_STATS)
abc4_pipeline = pipeline.override(prompts=codecs.ABC4_PROMPTS)

COMMANDS = {
    'bottles_16k': serve.override(model=pi05_abc_model, pipeline=pipeline),
    'abc4_30k': serve.override(model=abc4_model, pipeline=abc4_pipeline),
}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
