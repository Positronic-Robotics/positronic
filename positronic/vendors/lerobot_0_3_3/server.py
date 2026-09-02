import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
from lerobot.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy

from pimm.logging import init_logging
from positronic import keys
from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress, warmup
from positronic.policy import Codec, Policy
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, StopOnFault
from positronic.policy.spec import ModelSource, remote
from positronic.utils.checkpoints import list_checkpoints, resolve_checkpoint
from positronic.vendors.lerobot_0_3_3 import codecs as lerobot_codecs
from positronic.vendors.lerobot_0_3_3.backbone import register_all
from positronic.vendors.lerobot_0_3_3.policy import LerobotPolicy, _detect_device, warm_observation

register_all()

logger = logging.getLogger(__name__)


def act(checkpoint_path: str) -> PreTrainedPolicy:
    return ACTPolicy.from_pretrained(checkpoint_path, strict=True)


class LerobotSource(ModelSource):
    """In-process LeRobot checkpoints from one experiment directory (its ``checkpoints/`` subdirectory),
    which ``load`` downloads one at a time.

    ``policy_factory`` builds the backbone policy from a checkpoint path — that is its whole contract,
    so any callable returning a ``PreTrainedPolicy`` works. ``model_type`` names what it built, for the
    handshake.
    """

    def __init__(
        self,
        policy_factory: Callable[[str], PreTrainedPolicy],
        checkpoints_dir: str | Path,
        checkpoint: str | None = None,
        device: str | None = None,
        model_type: str = 'act',
    ):
        self._policy_factory = policy_factory
        self._checkpoints_dir = str(checkpoints_dir).rstrip('/') + f'/{CHECKPOINTS_DIR}'
        self._checkpoint = checkpoint
        self._device = device or _detect_device()
        self._model_type = model_type
        self._experiment_name = str(checkpoints_dir).rstrip('/').split('/')[-1] or ''

    def get_models(self) -> list[str]:
        return list_checkpoints(self._checkpoints_dir)

    def resolve(self, model_id: str | None) -> str:
        return resolve_checkpoint(self._checkpoints_dir, self._checkpoint, model_id)

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        checkpoint_path = f'{self._checkpoints_dir}/{model_id}/{PRETRAINED_MODEL_DIR}'
        logger.info(f'Loading checkpoint from {checkpoint_path}')
        local = run_with_progress(
            lambda: pos3.download(checkpoint_path), f'Downloading checkpoint {model_id}', on_progress
        )
        backbone = self._policy_factory(str(local))
        meta = {keys.TYPE: self._model_type, keys.CHECKPOINT_PATH: checkpoint_path}
        policy = LerobotPolicy(backbone, self._device, extra_meta=meta)
        warmup(policy, warm_observation(backbone.config), on_progress)
        return policy

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'device': self._device, keys.EXPERIMENT_NAME: self._experiment_name}


lerobot_source = cfn.Config(LerobotSource, policy_factory=act)


# No ``ee_frame``: every checkpoint served here was trained on poses the rig reported in its ``default``,
# so none has a transform to declare.
@cfn.config(codec=lerobot_codecs.ee, source=lerobot_source)
def pipeline(codec: Codec, source: ModelSource):
    return StopOnFault() | ChunkedSchedule() | RestrictImageSize(224, 224) | remote | codec | source


ee = pipeline
joints = pipeline.override(codec=lerobot_codecs.joints)
ee_traj = pipeline.override(codec=lerobot_codecs.ee_traj)
joints_traj = pipeline.override(codec=lerobot_codecs.joints_traj)
joints_ik = pipeline.override(codec=lerobot_codecs.joints_ik)
joints_ik_sim = pipeline.override(codec=lerobot_codecs.joints_ik_sim)
trossen_joints = pipeline.override(codec=lerobot_codecs.trossen_joints)
# For checkpoints trained on inverted-grip (1 = open) sim data, which speak the flipped convention.
ee_flip = pipeline.override(codec=lerobot_codecs.ee.override(flip_grip=True))


# Every pipeline is a subcommand, and so is every deployment — a pipeline with its checkpoints bound.
# The sim_stack and demo checkpoints were trained on inverted-grip (1 = open) sim data, hence the flipped pipeline.
COMMANDS = {
    'serve': serve.override(pipeline=ee),
    'ee': serve.override(pipeline=ee),
    'joints': serve.override(pipeline=joints),
    'ee_traj': serve.override(pipeline=ee_traj),
    'joints_traj': serve.override(pipeline=joints_traj),
    'joints_ik': serve.override(pipeline=joints_ik),
    'joints_ik_sim': serve.override(pipeline=joints_ik_sim),
    'trossen_joints': serve.override(pipeline=trossen_joints),
    'ee_flip': serve.override(pipeline=ee_flip),
    'phail': serve.override(
        pipeline=ee.override(
            codec=lerobot_codecs.phail_v1,
            **{'source.checkpoints_dir': 's3://checkpoints/phail_unified/lerobot/270226-ee/'},
        ),
        recording_dir='s3://inference/phail_unified/server_recordings/lerobot/270226-ee/',
    ),
    'sim_stack': serve.override(
        pipeline=ee_flip.override(**{'source.checkpoints_dir': 's3://checkpoints/sim_stack/lerobot/230226-ee/'}),
        recording_dir='s3://inference/sim_stack/server_recordings/lerobot/230226-ee/',
    ),
    'demo': serve.override(
        pipeline=ee_flip.override(**{
            'source.checkpoints_dir': 's3://PUBLIC@positronic-public/checkpoints/sim_stack_cubes/act/'
        })
    ),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
