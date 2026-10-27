"""A MolmoAct2 DROID policy server to edit by hand.

Change a knob or a function below, then build `docker/Dockerfile.serve-molmoact2-starter`. With no edit, the
server serves what `positro/molmoact2-droid` serves. `docs/nebius-competition.md` gives the commands.
"""

from typing import Any

import configuronic as cfn
import numpy as np

from pimm.logging import init_logging
from positronic.offboard.server import serve
from positronic.offboard.server_utils import warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy.base import Obs
from positronic.policy.codecs import ACTION, BinarizeGripInference
from positronic.vendors import molmoact2
from positronic.vendors.molmoact2 import codecs as molmoact2_codecs
from positronic.vendors.molmoact2.policy import MolmoAct2Model, warm_observation
from positronic.vendors.molmoact2.server import DEFAULT_HF_REPO, pipeline

# Knobs. The values here make the server act as the published example image.

# The text the model reads for an eval task. A task that is not here goes to the model unchanged.
INSTRUCTIONS: dict[str, str] = {
    # 'pick up the tissue.': 'pick up the white tissue from the table',
}

# The model predicts 15 actions, 1/15 s apart. The robot executes the first ACTIONS_PER_CHUNK of them, and then
# the server predicts a new chunk from a new observation. A smaller value replans more often. None executes all 15.
ACTIONS_PER_CHUNK: int | None = None

# The model predicts a grip from 0 (open) to 1 (closed). With a threshold, the grip is 1 above it and 0 below it.
GRIP_THRESHOLD: float | None = None

# The denoising steps of one prediction. More steps take more time on the GPU.
NUM_STEPS = 10


def instruction(task: str) -> str:
    """The text the model reads for the eval task `task`."""
    return INSTRUCTIONS.get(task, task)


def adjust(action: np.ndarray) -> np.ndarray:
    """One action of a chunk, as the robot executes it: 7 absolute joint positions in radians, then the grip."""
    return action


class StarterModel(Model):
    """The MolmoAct2 model, with `instruction` applied to what it reads and `adjust` to each action it returns."""

    def __init__(self, model: Model):
        self._model = model

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        # `obs` holds the views [exterior, exterior_2, wrist], the state [7 joint positions, grip] and the task.
        obs = {**obs, molmoact2.TASK: instruction(obs[molmoact2.TASK])}
        return [{ACTION: adjust(step[ACTION])} for step in self._model(obs, session_id=session_id)]

    def end_session(self, session_id: str) -> None:
        self._model.end_session(session_id)

    def meta(self) -> dict[str, Any]:
        return self._model.meta()

    def close(self) -> None:
        self._model.close()


@cfn.config(hf_repo=DEFAULT_HF_REPO, device_map='auto')
def starter_model(hf_repo: str, device_map: str) -> Model:
    model = StarterModel(MolmoAct2Model(hf_repo, device_map=device_map, num_steps=NUM_STEPS))
    warmup(model, warm_observation())
    return model


ACTION_FPS = 15.0  # The rate of the actions in a chunk of the DROID checkpoint.


@cfn.config()
def starter_pipeline() -> PolicyDeployment:
    """The `droid_3cam` pipeline, with ACTIONS_PER_CHUNK and GRIP_THRESHOLD applied."""
    codec = molmoact2_codecs.droid_3cam.instantiate()
    if GRIP_THRESHOLD is not None:
        codec = BinarizeGripInference(GRIP_THRESHOLD) | codec
    horizon_sec = None if ACTIONS_PER_CHUNK is None else ACTIONS_PER_CHUNK / ACTION_FPS
    return pipeline(codec=codec, fps=ACTION_FPS, horizon_sec=horizon_sec)


if __name__ == '__main__':
    init_logging()
    cfn.cli(serve.override(model=starter_model, pipeline=starter_pipeline))
