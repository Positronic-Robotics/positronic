import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3

from positronic.offboard.server import PolicyServer
from positronic.policy import Codec, Policy
from positronic.policy.spec import ModelSource, Pipe, remote
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils.checkpoints import list_checkpoints, resolve_checkpoint
from positronic.utils.logging import init_logging
from positronic.vendors.lerobot import codecs as lerobot_codecs
from positronic.vendors.lerobot.policy import LerobotPolicy, _detect_device

logger = logging.getLogger(__name__)


class LerobotSource(ModelSource):
    """LeRobot 0.4.x checkpoints of one experiment directory.

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
        return LerobotPolicy(checkpoint_path, self.device, extra_meta={'checkpoint_path': checkpoint_path})

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'device': self.device, 'experiment_name': self.experiment_name}


lerobot_source = cfn.Config(LerobotSource, checkpoint=None, device=None)


@cfn.config(codec=lerobot_codecs.ee, source=lerobot_source)
def pipe(codec: Codec, source: ModelSource) -> Pipe:
    return ChunkedSchedule() | remote | codec | source


PIPES = {
    'ee': pipe,
    'joints': pipe.override(codec=lerobot_codecs.joints),
    'joints_ik': pipe.override(codec=lerobot_codecs.joints_ik),
    'joints_ik_sim': pipe.override(codec=lerobot_codecs.joints_ik_sim),
}


@cfn.config(
    pipe='ee', checkpoint=None, device=None, port=8000, host='0.0.0.0', recording_dir=None, idle_timeout_min=None
)
def main(
    pipe: str,
    checkpoints_dir: str,
    checkpoint: str | None,
    device: str | None,
    port: int,
    host: str,
    recording_dir: str | None,
    idle_timeout_min: float | None,
):
    checkpoints_dir = str(pos3.download(checkpoints_dir))
    cfg = PIPES[pipe].override(**{
        'source.checkpoints_dir': checkpoints_dir,
        'source.checkpoint': checkpoint,
        'source.device': device,
    })
    PolicyServer(cfg, host=host, port=port, recording_dir=recording_dir, idle_timeout_min=idle_timeout_min).serve()


phail = main.override(
    checkpoints_dir='s3://checkpoints/phail_unified/smolvla/170316_ee/',
    recording_dir='s3://inference/phail_unified/server_recordings/smolvla/170316_ee/',
)


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli({'serve': main, 'phail': phail})
