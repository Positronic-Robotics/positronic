"""Serve FLUX 3 Action's DROID policy over positronic's session protocol.

The policy runs in Black Forest Labs' own interpreter, as `backend.py`, behind BFL's server on this machine.
`droid` sends one exterior view to the policy twice, and `droid_3cam` sends both exterior views.

Usage
  python -m positronic.vendors.flux3_action.server droid --model.flux_python=<BFL's python> \
      --model.checkpoint=black-forest-labs/flux-3-action-droid --model.revision=<commit> \
      --model.subfolder=variants/gd
"""

import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
from platform_client.policy_container import PROVISIONING_DEADLINE_S

from pimm.logging import init_logging
from positronic import keys
from positronic.offboard import keys as offboard_keys
from positronic.offboard.roboarena import RoboarenaClient
from positronic.offboard.server import serve
from positronic.offboard.server_utils import wait_for_subprocess_ready, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codec import ACTION, Codec, RestrictImageSize
from positronic.policy.processors import ChunkedSchedule, PauseOnUnavailable
from positronic.vendors import flux3_action
from positronic.vendors.flux3_action import codecs

BACKEND_SCRIPT = Path(__file__).with_name('backend.py')


class Flux3ActionModel(Model):
    """FLUX 3 Action behind BFL's server in a child process, which this model owns."""

    def __init__(self, backend: subprocess.Popen, port: int, meta: dict[str, Any]):
        self._backend = backend
        self._client = RoboarenaClient(port=port)
        self._meta = meta

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        return [{ACTION: action} for action in self._client.infer(obs)[flux3_action.ACTIONS]]

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


@cfn.config(revision=None, subfolder=None, backend_port=9000)
def flux3_action_model(
    checkpoint: str, revision: str | None, subfolder: str | None, flux_python: str, backend_port: int
) -> Model:
    """The policy package at `checkpoint`, served by `backend.py` in BFL's interpreter `flux_python`.

    `checkpoint` is a Hugging Face repository id, with a `revision` and a `subfolder`, or a local directory with
    neither.
    """
    # `-P` keeps this directory off the module path, so a positronic module here cannot shadow one of BFL's.
    command: list[str | Path] = [
        Path(flux_python),
        '-P',
        BACKEND_SCRIPT,
        '--checkpoint',
        checkpoint,
        '--port',
        str(backend_port),
    ]
    for flag, value in (('--revision', revision), ('--subfolder', subfolder)):
        if value is not None:
            command += [flag, value]
    backend = subprocess.Popen(command)
    meta = {
        offboard_keys.CHECKPOINT_ID: checkpoint,
        policy_keys.TYPE: 'flux3_action',
        'revision': revision,
        'subfolder': subfolder,
    }
    model = Flux3ActionModel(backend, backend_port, meta)
    try:
        probe = RoboarenaClient(port=backend_port)
        wait_for_subprocess_ready(
            lambda: probe.probe() is None,
            partial(_exit_status, backend),
            'FLUX 3 Action backend',
            max_wait=PROVISIONING_DEADLINE_S,
        )
        warmup(model, _warm_observation())
    except Exception:
        model.close()
        raise
    return model


@cfn.config(codec=codecs.droid)
def pipeline(codec: Codec) -> PolicyDeployment:
    """The client plays each chunk whole, as BFL's DROID settings do, and `codec` runs on this server."""
    return PolicyDeployment(
        Sequential(PauseOnUnavailable(), ChunkedSchedule(codecs.FPS), RestrictImageSize(*codecs.VIEW_SIZE)), codec
    )


COMMANDS = {
    'droid': serve.override(model=flux3_action_model, pipeline=pipeline),
    'droid_3cam': serve.override(model=flux3_action_model, pipeline=pipeline.override(codec=codecs.droid_3cam)),
}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
