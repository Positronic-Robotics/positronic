"""Serve Cosmos3-Nano's DROID policy over positronic's session protocol.

The policy runs in NVIDIA's own interpreter, behind NVIDIA's RoboLab action server on this machine.
`droid` sends one exterior view to the policy twice, and `droid_3cam` sends both exterior views.

Usage
  python -m positronic.vendors.cosmos3.server droid --model.nvidia_python=<NVIDIA's python> \
      --model.checkpoint=nvidia/Cosmos3-Nano-Policy-DROID --model.revision=<commit>
"""

import subprocess
from functools import partial
from typing import Any

import configuronic as cfn
import numpy as np
from platform_client.policy_container import PROVISIONING_DEADLINE_S
from positronic_model_server import keys as offboard_keys

from pimm.logging import init_logging
from positronic import keys
from positronic.offboard.roboarena import RoboarenaClient
from positronic.offboard.server import serve
from positronic.offboard.server_utils import wait_for_subprocess_ready, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codec import ACTION, Codec, RestrictImageSize
from positronic.policy.processors import ChunkedSchedule, PauseOnUnavailable
from positronic.vendors import cosmos3
from positronic.vendors.cosmos3 import codecs

NVIDIA_SERVER_MODULE = 'cosmos_framework.scripts.action_policy_server_robolab'
# NVIDIA's server checks no token, so it listens on loopback only.
BACKEND_HOST = '127.0.0.1'


class Cosmos3Model(Model):
    """Cosmos3-Nano behind NVIDIA's server in a child process, which this model owns."""

    def __init__(self, backend: subprocess.Popen, port: int, meta: dict[str, Any]):
        self._backend = backend
        self._client = RoboarenaClient(BACKEND_HOST, port)
        self._meta = meta

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        return [{ACTION: action} for action in self._client.infer(obs)[cosmos3.ACTIONS]]

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
    """Black views, the arm at zero and the gripper open, as `codecs.droid` encodes them."""
    width, height = codecs.VIEW_SIZE
    view = np.zeros((height, width, 3), dtype=np.uint8)
    return codecs.droid.instantiate().encode({
        keys.WRIST_IMAGE: view,
        keys.EXTERIOR_IMAGE: view,
        keys.JOINTS: np.zeros(7),
        keys.GRIP: 0.0,
        keys.TASK: '',
    })


def _exit_status(process: subprocess.Popen) -> tuple[bool, int | None]:
    code = process.poll()
    return code is not None, code


@cfn.config(revision=None, backend_port=9000)
def cosmos3_model(checkpoint: str, revision: str | None, nvidia_python: str, backend_port: int) -> Model:
    """The policy at `checkpoint`, served by NVIDIA's server in NVIDIA's interpreter `nvidia_python`.

    `checkpoint` is a Hugging Face repository id with a `revision`, or a local directory with none.
    """
    # `-P` keeps the working directory off the module path, so a module there cannot shadow one of NVIDIA's.
    command = [nvidia_python, '-P', '-m', NVIDIA_SERVER_MODULE, '--checkpoint-path', checkpoint]
    command += ['--host', BACKEND_HOST, '--port', str(backend_port)]
    if revision is not None:
        command += ['--hf-revision', revision]
    backend = subprocess.Popen(command)
    meta = {offboard_keys.CHECKPOINT_ID: checkpoint, policy_keys.TYPE: 'cosmos3', 'revision': revision}
    model = Cosmos3Model(backend, backend_port, meta)
    try:
        probe = RoboarenaClient(BACKEND_HOST, backend_port)
        wait_for_subprocess_ready(
            lambda: probe.probe() is None,
            partial(_exit_status, backend),
            'Cosmos3 backend',
            max_wait=PROVISIONING_DEADLINE_S,
        )
        warmup(model, _warm_observation())
    except Exception:
        model.close()
        raise
    return model


@cfn.config(codec=codecs.droid)
def pipeline(codec: Codec) -> PolicyDeployment:
    """The client plays each chunk whole, as NVIDIA's RoboLab client does, and `codec` runs on this server."""
    return PolicyDeployment(
        Sequential(PauseOnUnavailable(), ChunkedSchedule(codecs.FPS), RestrictImageSize(*codecs.VIEW_SIZE)), codec
    )


COMMANDS = {
    'droid': serve.override(model=cosmos3_model, pipeline=pipeline),
    'droid_3cam': serve.override(model=cosmos3_model, pipeline=pipeline.override(codec=codecs.droid_3cam)),
}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
