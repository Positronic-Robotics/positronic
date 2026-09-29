import logging

import configuronic as cfn
import pos3

from pimm.logging import init_logging
from positronic.offboard import keys as offboard_keys
from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Codec, Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable
from positronic.utils.checkpoints import resolve_checkpoint
from positronic.vendors.lerobot import codecs as lerobot_codecs
from positronic.vendors.lerobot.policy import LerobotModel, _detect_device, warm_observation

logger = logging.getLogger(__name__)


@cfn.config(checkpoint=None, device=None)
def lerobot_model(checkpoints_dir: str, checkpoint: str | None, device: str | None) -> Model:
    """One LeRobot 0.4.x checkpoint of an experiment directory: ``checkpoint``, else the latest one.

    The policy type is auto-detected from the checkpoint's config, so this serves SmolVLA, ACT,
    Diffusion, or any other lerobot 0.4.x policy.
    """
    experiment_dir = checkpoints_dir.rstrip('/')
    checkpoint_id = resolve_checkpoint(f'{experiment_dir}/checkpoints', checkpoint)
    checkpoint_path = f'{experiment_dir}/checkpoints/{checkpoint_id}/pretrained_model'
    device = device or _detect_device()
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    local = run_with_progress(lambda: pos3.download(checkpoint_path), f'Downloading checkpoint {checkpoint_id}')
    policy = LerobotModel(
        str(local),
        device,
        extra_meta={
            offboard_keys.CHECKPOINT_ID: checkpoint_id,
            policy_keys.CHECKPOINT_PATH: checkpoint_path,
            policy_keys.EXPERIMENT_NAME: experiment_dir.split('/')[-1],
            'device': device,
        },
    )
    warmup(policy, warm_observation(policy.config))
    return policy


# No ``ee_frame``: every checkpoint served here was trained on poses the rig reported in its ``default``,
# so none has a transform to declare.
@cfn.config(codec=lerobot_codecs.ee)
def pipeline(codec: Codec, fps: float = 15.0, horizon_sec: float | None = 1.0) -> PolicyDeployment:
    return PolicyDeployment(
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps, horizon_sec), RestrictImageSize(512, 512)), codec
    )


ee = pipeline
joints = pipeline.override(codec=lerobot_codecs.joints)
joints_ik = pipeline.override(codec=lerobot_codecs.joints_ik)
joints_ik_sim = pipeline.override(codec=lerobot_codecs.joints_ik_sim)


# Every pipeline is a subcommand, and so is every deployment — a pipeline with its checkpoints bound.
COMMANDS = {
    'serve': serve.override(model=lerobot_model, pipeline=ee),
    'ee': serve.override(model=lerobot_model, pipeline=ee),
    'joints': serve.override(model=lerobot_model, pipeline=joints),
    'joints_ik': serve.override(model=lerobot_model, pipeline=joints_ik),
    'joints_ik_sim': serve.override(model=lerobot_model, pipeline=joints_ik_sim),
    'phail': serve.override(
        model=lerobot_model.override(checkpoints_dir='s3://checkpoints/phail_unified/smolvla/170316_ee/'),
        pipeline=ee.override(codec=lerobot_codecs.phail_v1),
    ),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
