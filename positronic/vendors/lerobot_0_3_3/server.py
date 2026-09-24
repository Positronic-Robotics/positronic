import logging
from collections.abc import Callable

import configuronic as cfn
import pos3
from lerobot.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy

from pimm.logging import init_logging
from positronic import geom, keys
from positronic.cfg import codecs
from positronic.offboard import keys as offboard_keys
from positronic.offboard.server import serve
from positronic.offboard.server_utils import run_with_progress, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Codec, Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.codec import RestrictImageSize
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable
from positronic.utils.checkpoints import resolve_checkpoint
from positronic.vendors.lerobot_0_3_3.backbone import register_all
from positronic.vendors.lerobot_0_3_3.policy import LerobotModel, _detect_device, warm_observation

register_all()

logger = logging.getLogger(__name__)


def act(checkpoint_path: str) -> PreTrainedPolicy:
    return ACTPolicy.from_pretrained(checkpoint_path, strict=True)


@cfn.config(policy_factory=act, checkpoint=None, device=None, model_type='act')
def lerobot_model(
    policy_factory: Callable[[str], PreTrainedPolicy],
    checkpoints_dir: str,
    checkpoint: str | None,
    device: str | None,
    model_type: str,
) -> Model:
    """One in-process LeRobot checkpoint from an experiment directory (its ``checkpoints/`` subdirectory):
    ``checkpoint``, else the latest one.

    ``policy_factory`` builds the backbone policy from a checkpoint path — that is its whole contract,
    so any callable returning a ``PreTrainedPolicy`` works. ``model_type`` names what it built, for the
    handshake.
    """
    experiment_dir = checkpoints_dir.rstrip('/')
    checkpoint_id = resolve_checkpoint(f'{experiment_dir}/{CHECKPOINTS_DIR}', checkpoint)
    checkpoint_path = f'{experiment_dir}/{CHECKPOINTS_DIR}/{checkpoint_id}/{PRETRAINED_MODEL_DIR}'
    device = device or _detect_device()
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    local = run_with_progress(lambda: pos3.download(checkpoint_path), f'Downloading checkpoint {checkpoint_id}')
    backbone = policy_factory(str(local))
    meta = {
        offboard_keys.CHECKPOINT_ID: checkpoint_id,
        policy_keys.TYPE: model_type,
        policy_keys.CHECKPOINT_PATH: checkpoint_path,
        policy_keys.EXPERIMENT_NAME: experiment_dir.split('/')[-1],
        'device': device,
    }
    model = LerobotModel(backbone, device, extra_meta=meta)
    warmup(model, warm_observation(backbone.config))
    return model


# No ``ee_frame``: every checkpoint served here was trained on poses the rig reported in its ``default``,
# so none has a transform to declare.
@cfn.config(**{
    'obs': codecs.general_obs,
    'obs.state_name': 'observation.state',
    'obs.state_features': {keys.EE_POSE: 7, keys.GRIP: 1},
    'obs.image_mappings': {'observation.images.left': keys.WRIST_IMAGE, 'observation.images.side': keys.EXTERIOR_IMAGE},
    'obs.image_size': (224, 224),
    'action': codecs.absolute_pos_action,
})
def pipeline(
    obs: Codec,
    action: Codec,
    fps: float = 15.0,
    horizon_sec: float | None = 1.0,
    binarize_grip: tuple[str, ...] | None = None,
    flip_grip: bool = False,
    ee_frame: geom.Transform3D | None = None,
) -> PolicyDeployment:
    return PolicyDeployment(
        local=Sequential(
            PauseOnUnavailable(), ChunkedSchedule(fps=fps, horizon_sec=horizon_sec), RestrictImageSize(224, 224)
        ),
        codec=codecs.compose_data(
            obs=obs, action=action, binarize_grip=binarize_grip, flip_grip=flip_grip, ee_frame=ee_frame
        ),
    )


ee = pipeline
joints = pipeline.override(**{'obs.state_features': {keys.JOINTS: 7, keys.GRIP: 1}})
ee_traj = pipeline.override(**{
    'action.tgt_ee_pose_key': keys.EE_POSE,
    'action.tgt_grip_key': keys.GRIP,
    'binarize_grip': (keys.GRIP,),
})
joints_traj = pipeline.override(**{
    'obs.state_features': {keys.JOINTS: 7, keys.GRIP: 1},
    'action': codecs.absolute_joints_action,
    'action.tgt_joints_key': keys.JOINTS,
    'action.tgt_grip_key': keys.GRIP,
    'binarize_grip': (keys.GRIP,),
})
joints_ik = pipeline.override(**{
    'obs.state_features': {keys.JOINTS: 7, keys.GRIP: 1},
    'action': codecs.ik_joints_action,
    'action.solver': 'dls_limits',
})
joints_ik_sim = pipeline.override(**{
    'obs.state_features': {keys.JOINTS: 7, keys.GRIP: 1},
    'action': codecs.ik_joints_action,
    'action.solver': 'lm',
})
# For checkpoints trained on inverted-grip (1 = open) sim data, which speak the flipped convention.
ee_flip = pipeline.override(flip_grip=True)


phail = pipeline.override(**{'action': codecs.phail_v1_execution, 'action.action': codecs.absolute_pos_action})
demo = pipeline.override(**{
    'obs': codecs.general_obs,
    'obs.state_name': 'observation.state',
    'obs.state_features': {keys.EE_POSE: 7, keys.GRIP: 1},
    'obs.image_mappings': {'observation.images.left': keys.WRIST_IMAGE, 'observation.images.side': keys.EXTERIOR_IMAGE},
    'obs.image_size': (224, 224),
    'action': codecs.absolute_pos_action,
    'flip_grip': True,
    'fps': 15.0,
    'horizon_sec': 1.0,
})


# Every pipeline is a subcommand, and so is every deployment — a pipeline and the checkpoints it pairs with.
# The sim_stack and demo checkpoints were trained on inverted-grip (1 = open) sim data, hence the flipped pipeline.
COMMANDS = {
    'serve': serve.override(model=lerobot_model, pipeline=ee),
    'ee': serve.override(model=lerobot_model, pipeline=ee),
    'joints': serve.override(model=lerobot_model, pipeline=joints),
    'ee_traj': serve.override(model=lerobot_model, pipeline=ee_traj),
    'joints_traj': serve.override(model=lerobot_model, pipeline=joints_traj),
    'joints_ik': serve.override(model=lerobot_model, pipeline=joints_ik),
    'joints_ik_sim': serve.override(model=lerobot_model, pipeline=joints_ik_sim),
    'ee_flip': serve.override(model=lerobot_model, pipeline=ee_flip),
    'phail': serve.override(
        model=lerobot_model.override(checkpoints_dir='s3://checkpoints/phail_unified/lerobot/270226-ee/'),
        pipeline=phail,
    ),
    'sim_stack': serve.override(
        model=lerobot_model.override(checkpoints_dir='s3://checkpoints/sim_stack/lerobot/230226-ee/'), pipeline=ee_flip
    ),
    'demo': serve.override(
        model=lerobot_model.override(checkpoints_dir='s3://PUBLIC@positronic-public/checkpoints/sim_stack_cubes/act/'),
        pipeline=demo,
    ),
}


if __name__ == '__main__':
    init_logging()
    with pos3.mirror():
        cfn.cli(COMMANDS)
