"""Serve an ABC-DiT checkpoint over positronic's session protocol.

The policy runs in ABC's own interpreter, as ABC's policy server started by `backend.py`, on this machine.
`multitask_200k` serves `abc_dit_xl_200k_model.pt`, ABC's multi-task sim policy, with the prompts it trained under.

Usage
  python -m positronic.vendors.abc_dit.server multitask_200k --model.abc_python=<ABC's python> \
      --model.checkpoint=<dir>/abc_dit_xl_200k_model.pt
"""

import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
from platform_client.policy_container import PROVISIONING_DEADLINE_S

from pimm.logging import init_logging
from positronic.offboard.roboarena import RoboarenaClient
from positronic.offboard.server import serve
from positronic.offboard.server_utils import wait_for_subprocess_ready, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codecs import ACTION, Codec, RestrictImageSize
from positronic.policy.processors import ChunkedSchedule
from positronic.vendors import abc_dit
from positronic.vendors.abc_dit import codecs

# The script that ABC's interpreter runs: ABC's own websocket policy server, on loopback.
BACKEND_SCRIPT = Path(__file__).with_name('backend.py')
# The DiT checkpoints predict chunks of this many actions.
CHUNK_LENGTH = 30


class AbcDitModel(Model):
    """An ABC-DiT checkpoint behind ABC's policy server in a child process, which this model owns.

    Each request carries standard normal flow noise from one generator, as ABC's sim eval draws it.
    """

    def __init__(self, backend: subprocess.Popen, port: int, noise_seed: int, meta: dict[str, Any]):
        self._backend = backend
        self._client = RoboarenaClient(port=port)
        self._noise = np.random.default_rng(noise_seed)
        self._meta = meta

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        noise = self._noise.standard_normal((CHUNK_LENGTH, codecs.BIMANUAL_SIZE), dtype=np.float32)
        return [{ACTION: action} for action in self._client.infer({**obs, abc_dit.NOISE: noise})[abc_dit.ACTIONS]]

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


def _exit_status(process: subprocess.Popen) -> tuple[bool, int | None]:
    code = process.poll()
    return code is not None, code


def _warm_observation() -> dict[str, Any]:
    """Black views and a zero state, in the fields of ABC's policy server."""
    width, height = codecs.VIEW_SIZE
    view = np.zeros((3, height, width), dtype=np.uint8)
    cameras = (abc_dit.TOP_CAMERA, abc_dit.LEFT_CAMERA, abc_dit.RIGHT_CAMERA)
    return {
        abc_dit.STATE: np.zeros(codecs.BIMANUAL_SIZE, dtype=np.float32),
        abc_dit.IMAGES: dict.fromkeys(cameras, view),
        abc_dit.PROMPT: '',
    }


@cfn.config(fast_inference=True, noise_seed=0, backend_port=9000)
def abc_dit_model(checkpoint: str, abc_python: str, fast_inference: bool, noise_seed: int, backend_port: int) -> Model:
    """The DiT checkpoint at `checkpoint`, served by ABC's policy server in ABC's interpreter `abc_python`.

    `fast_inference` runs the DiT in bfloat16 and compiles it, as ABC's sim eval does; it needs a CUDA device.
    """
    checkpoint_path, python = Path(checkpoint), Path(abc_python)
    command = [
        python,
        '-P',
        BACKEND_SCRIPT,
        f'--policy.checkpoint-path={checkpoint_path}',
        '--policy-type=dit',
        '--policy.fast-inference' if fast_inference else '--policy.no-fast-inference',
        f'--port={backend_port}',
    ]
    backend = subprocess.Popen(command)
    meta = {
        policy_keys.CHECKPOINT_ID: checkpoint_path.name,
        policy_keys.TYPE: 'abc_dit',
        'fast_inference': fast_inference,
    }
    model = AbcDitModel(backend, backend_port, noise_seed, meta)
    try:
        probe = RoboarenaClient(port=backend_port)
        wait_for_subprocess_ready(
            lambda: probe.probe() is None,
            partial(_exit_status, backend),
            'ABC policy server',
            max_wait=PROVISIONING_DEADLINE_S,
        )
        warmup(model, _warm_observation())
    except Exception:
        model.close()
        raise
    return model


# ABC's synchronous sim eval executes this many actions of each chunk, then takes a new observation.
EXECUTED_ACTIONS = 15


@cfn.config(codec=codecs.multitask_200k)
def pipeline(codec: Codec) -> PolicyDeployment:
    """The client plays the first half of each chunk, as ABC's synchronous sim eval does, and `codec` runs on this
    server."""
    schedule = ChunkedSchedule(codecs.FPS, horizon_sec=EXECUTED_ACTIONS / codecs.FPS)
    return PolicyDeployment(Sequential(schedule, RestrictImageSize(*codecs.VIEW_SIZE)), codec)


COMMANDS = {'multitask_200k': serve.override(model=abc_dit_model, pipeline=pipeline)}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
