"""A stub of the model that the DAgger rounds train: its converter, its trainer and its policy server.

The converter lists the round and the record of each training episode. The trainer writes a checkpoint whose ID
depends on that list. The server serves a policy that commands the arm ``RISE`` above the pose that it measures, with
the grip that it measures, and states the ID of its checkpoint. So the arm leaves the path of the scripted expert of
the lift-cube task.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np

from pimm.logging import init_logging
from positronic import keys
from positronic.dagger import META_RECORD_UID, META_ROUND
from positronic.dataset.local_dataset import LocalDataset
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame
from positronic.offboard.server import serve
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codecs import ACTION, AbsolutePositionAction, ChangeEEFrame
from positronic.policy.codecs.observation import StateObservationCodec
from positronic.policy.processors import ChunkedSchedule, TrajectoryPlayer
from positronic.policy.sequential import Sequential

EPISODES = 'episodes.json'
CHECKPOINT = 'checkpoint.json'
WIRE_NAME = 'dagger_stub'
RISE = 0.01  # m
_STATE = 'observation.state'
# The measured pose and grip of the arm, in the order of the action of ``AbsolutePositionAction``.
_FEATURES = {keys.EE_POSE: 7, keys.GRIP: 1}
# The index of the height of the pose in the state and in the action.
_HEIGHT = 2


def episodes(converted: Path) -> list[dict[str, Any]]:
    """The round and the record of each training episode that the converter listed in ``converted``."""
    return json.loads((converted / EPISODES).read_text())


def checkpoint_id(listed: list[dict[str, Any]]) -> str:
    """The ID of the checkpoint that the trainer makes from the episodes ``listed``."""
    return hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest()[:16]


@cfn.config()
def convert(training_data: str, output_dir: str) -> None:
    listed = [
        {'round': episode.meta[META_ROUND], 'record_uid': episode.meta[META_RECORD_UID]}
        for episode in LocalDataset(Path(training_data))
    ]
    Path(output_dir).mkdir(parents=True)
    (Path(output_dir) / EPISODES).write_text(json.dumps(listed))


@cfn.config()
def train(input_path: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True)
    (Path(output_dir) / CHECKPOINT).write_text(json.dumps({'id': checkpoint_id(episodes(Path(input_path)))}))


class _Rises(Model):
    def __init__(self, checkpoint: str, license: str):
        self._id = json.loads((Path(checkpoint) / CHECKPOINT).read_text())['id']
        self._license = license

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        action = np.asarray(obs[_STATE], dtype=np.float64)
        action[_HEIGHT] += RISE
        return [{ACTION: action}]

    def meta(self) -> dict[str, Any]:
        return {policy_keys.TYPE: WIRE_NAME, policy_keys.LICENSE: self._license, policy_keys.CHECKPOINT_ID: self._id}


@cfn.config()
def pipeline() -> PolicyDeployment:
    codec = ChangeEEFrame(EEFrame(site=GRIPPER_SITE)) | (
        StateObservationCodec({_STATE: _FEATURES}) & AbsolutePositionAction(keys.TARGET_EE_POSE, keys.TARGET_GRIP)
    )
    return PolicyDeployment(Sequential(TrajectoryPlayer(replan_on_error=True), ChunkedSchedule(fps=15)), codec)


if __name__ == '__main__':
    init_logging()
    cfn.cli({'convert': convert, 'train': train, 'serve': serve.override(model=cfn.Config(_Rises), pipeline=pipeline)})
