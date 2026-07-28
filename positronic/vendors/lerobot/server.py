import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3

from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress
from positronic.policy import Codec, Policy
from positronic.policy.codec import RestrictImageSize
from positronic.policy.spec import ModelSource, Pipeline, remote
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils.checkpoints import list_checkpoints, resolve_checkpoint
from positronic.utils.logging import init_logging
from positronic.vendors.lerobot import codecs as lerobot_codecs
from positronic.vendors.lerobot.policy import LerobotPolicy, _detect_device

logger = logging.getLogger(__name__)


class LerobotSource(ModelSource):
    """LeRobot 0.4.x checkpoints of one experiment directory, which ``load`` downloads one at a time.

    The policy type is auto-detected from each checkpoint's config, so this serves SmolVLA, ACT,
    Diffusion, or any other lerobot 0.4.x policy.
    """

    def __init__(self, checkpoints_dir: str | Path, checkpoint: str | None = None, device: str | None = None):
        self.checkpoints_dir = str(checkpoints_dir).rstrip('/') + '/checkpoints'
        self.checkpoint = checkpoint
        self.device = device or _detect_device()
        self.experiment_name = str(checkpoints_dir).rstrip('/').split('/')[-1] or ''

    def get_models(self) -> list[str]:
        return list_checkpoints(self.checkpoints_dir)

    def resolve(self, model_id: str | None) -> str:
        return resolve_checkpoint(self.checkpoints_dir, self.checkpoint, model_id)

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        checkpoint_path = f'{self.checkpoints_dir}/{model_id}/pretrained_model'
        logger.info(f'Loading checkpoint from {checkpoint_path}')
        local = run_with_progress(
            lambda: pos3.download(checkpoint_path), f'Downloading checkpoint {model_id}', on_progress
        )
        return LerobotPolicy(str(local), self.device, extra_meta={'checkpoint_path': checkpoint_path})

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'device': self.device, 'experiment_name': self.experiment_name}


lerobot_source = cfn.Config(LerobotSource, checkpoint=None, device=None)


@cfn.config(codec=lerobot_codecs.ee, source=lerobot_source)
def pipeline(codec: Codec, source: ModelSource) -> Pipeline:
    return ChunkedSchedule() | RestrictImageSize(512, 512) | remote | codec | source


ee = pipeline
joints = pipeline.override(codec=lerobot_codecs.joints)
joints_ik = pipeline.override(codec=lerobot_codecs.joints_ik)
joints_ik_sim = pipeline.override(codec=lerobot_codecs.joints_ik_sim)


# Every pipeline is a subcommand, and so is every deployment — a pipeline with its checkpoints bound.
COMMANDS = {
    'serve': serve.override(pipeline=ee),
    'ee': serve.override(pipeline=ee),
    'joints': serve.override(pipeline=joints),
    'joints_ik': serve.override(pipeline=joints_ik),
    'joints_ik_sim': serve.override(pipeline=joints_ik_sim),
    'phail': serve.override(
        pipeline=ee.override(**{'source.checkpoints_dir': 's3://checkpoints/phail_unified/smolvla/170316_ee/'}),
        recording_dir='s3://inference/phail_unified/server_recordings/smolvla/170316_ee/',
    ),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
