import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy

from positronic.offboard.server import PolicyServer
from positronic.policy import Codec, Policy
from positronic.policy.codec import RestrictImageSize
from positronic.policy.spec import ModelSource, remote
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils.checkpoints import list_checkpoints, resolve_checkpoint
from positronic.utils.logging import init_logging
from positronic.vendors.lerobot_0_3_3 import codecs as lerobot_codecs
from positronic.vendors.lerobot_0_3_3.backbone import register_all
from positronic.vendors.lerobot_0_3_3.policy import LerobotPolicy, _detect_device

register_all()

logger = logging.getLogger(__name__)


def act(checkpoint_path: str) -> PreTrainedPolicy:
    return ACTPolicy.from_pretrained(checkpoint_path, strict=True)


class LerobotSource(ModelSource):
    """In-process LeRobot checkpoints from one experiment directory (its ``checkpoints/`` subdirectory).

    ``policy_factory`` builds the backbone policy from a checkpoint path — that is its whole contract,
    so any callable returning a ``PreTrainedPolicy`` works. ``model_type`` names what it built, for the
    handshake. Loads are synchronous and fast (<20s), so ``on_progress`` is unused.
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
        self._checkpoints_dir = str(checkpoints_dir).rstrip('/') + '/checkpoints'
        self._checkpoint = checkpoint
        self._device = device or _detect_device()
        self._model_type = model_type
        self._experiment_name = str(checkpoints_dir).rstrip('/').split('/')[-1] or ''

    def get_models(self) -> list[str]:
        return list_checkpoints(self._checkpoints_dir)

    def resolve(self, model_id: str | None) -> str:
        return resolve_checkpoint(self._checkpoints_dir, self._checkpoint, model_id)

    def load(self, model_id: str, on_progress: Callable[[str], None] | None = None) -> Policy:
        checkpoint_path = f'{self._checkpoints_dir}/{model_id}/pretrained_model'
        logger.info(f'Loading checkpoint from {checkpoint_path}')
        policy = self._policy_factory(checkpoint_path)
        meta = {'type': self._model_type, 'checkpoint_path': checkpoint_path}
        return LerobotPolicy(policy, self._device, extra_meta=meta)

    def meta(self, model_id: str) -> dict[str, Any]:
        return {'device': self._device, 'experiment_name': self._experiment_name}


lerobot_source = cfn.Config(LerobotSource, policy_factory=act)


@cfn.config(codec=lerobot_codecs.ee, source=lerobot_source)
def pipeline(codec: Codec, source: ModelSource):
    return ChunkedSchedule() | RestrictImageSize(224, 224) | remote | codec | source


PIPELINES = {
    'ee': pipeline,
    'joints': pipeline.override(codec=lerobot_codecs.joints),
    'ee_traj': pipeline.override(codec=lerobot_codecs.ee_traj),
    'joints_traj': pipeline.override(codec=lerobot_codecs.joints_traj),
    'joints_ik': pipeline.override(codec=lerobot_codecs.joints_ik),
    'joints_ik_sim': pipeline.override(codec=lerobot_codecs.joints_ik_sim),
    # For checkpoints trained on inverted-grip (1 = open) sim data, which speak the flipped convention.
    'ee_flip': pipeline.override(codec=lerobot_codecs.ee.override(flip_grip=True)),
}


@cfn.config(
    pipeline='ee',
    policy_factory=act,
    model_type='act',
    checkpoint=None,
    port=8000,
    host='0.0.0.0',
    recording_dir=None,
    idle_timeout_min=None,
)
def main(
    pipeline: str,
    policy_factory: Callable[[str], PreTrainedPolicy],
    model_type: str,
    checkpoints_dir: str,
    checkpoint: str | None,
    port: int,
    host: str,
    recording_dir: str | None,
    idle_timeout_min: float | None,
):
    cfg = PIPELINES[pipeline].override(**{
        'source.policy_factory': policy_factory,
        'source.model_type': model_type,
        'source.checkpoints_dir': str(pos3.download(checkpoints_dir)),
        'source.checkpoint': checkpoint,
    })
    PolicyServer(cfg, host=host, port=port, recording_dir=recording_dir, idle_timeout_min=idle_timeout_min).serve()


# Every pipeline is a subcommand, and so is every deployment — a pipeline with its checkpoints bound.
# The sim_stack and demo checkpoints were trained on inverted-grip (1 = open) sim data, hence the flipped pipeline.
COMMANDS = {
    'serve': main,
    **{name: main.override(pipeline=name) for name in PIPELINES},
    'phail': main.override(
        checkpoints_dir='s3://checkpoints/phail_unified/lerobot/270226-ee/',
        recording_dir='s3://inference/phail_unified/server_recordings/lerobot/270226-ee/',
    ),
    'sim_stack': main.override(
        pipeline='ee_flip',
        checkpoints_dir='s3://checkpoints/sim_stack/lerobot/230226-ee/',
        recording_dir='s3://inference/sim_stack/server_recordings/lerobot/230226-ee/',
    ),
    'demo': main.override(
        pipeline='ee_flip', checkpoints_dir='s3://PUBLIC@positronic-public/checkpoints/sim_stack_cubes/act/'
    ),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
