import configuronic as cfn
import pos3

from positronic.cfg import codecs
from positronic.policy import Policy
from positronic.policy.action import ActionDecoder
from positronic.policy.lerobot import LerobotPolicy
from positronic.policy.observation import ObservationEncoder
from positronic.utils import get_latest_checkpoint


@cfn.config()
def placeholder():
    raise RuntimeError(
        'This config is not supposed to be instantiated, '
        'and is used only to simplify relative imports of other policy configs.'
    )


@cfn.config(observation=None, action=None)
def wrapped(base: Policy, observation: ObservationEncoder | None, action: ActionDecoder | None):
    from positronic.policy.base import DecodedEncodedPolicy

    extra_meta: dict[str, object] = {}
    if action is not None:
        extra_meta |= {f'action.{k}': v for k, v in action.meta.items()}
    if observation is not None:
        extra_meta |= {f'observation.{k}': v for k, v in observation.meta.items()}

    return DecodedEncodedPolicy(
        base,
        encoder=None if observation is None else observation.encode,
        decoder=None if action is None else action.decode,
        extra_meta=extra_meta,
        action_horizon_sec=action.action_horizon_sec if action is not None else None,
        action_fps=action.action_fps if action is not None else None,
    )


@cfn.config(checkpoint=None)
def act(checkpoints_dir: str, checkpoint: str | None, n_action_steps: int | None = None, device=None):
    from lerobot.policies.act.modeling_act import ACTPolicy

    from positronic.vendors.lerobot.backbone import register_all

    register_all()

    checkpoints_dir = checkpoints_dir.rstrip('/') + '/checkpoints/'
    if checkpoint is None:
        checkpoint = get_latest_checkpoint(checkpoints_dir)
    else:
        checkpoint = str(checkpoint).strip('/')

    fully_specified_checkpoint_dir = checkpoints_dir.rstrip('/') + '/' + checkpoint + '/pretrained_model/'
    policy = ACTPolicy.from_pretrained(pos3.download(fully_specified_checkpoint_dir), strict=True)
    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps

    return LerobotPolicy(policy, device, extra_meta={'type': 'act', 'checkpoint_path': fully_specified_checkpoint_dir})


@cfn.config()
def diffusion(checkpoint_path: str, device: str | None = None):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(pos3.download(checkpoint_path), local_files_only=True, strict=True)
    return LerobotPolicy(policy, device, extra_meta={'type': 'diffusion', 'checkpoint_path': checkpoint_path})


act_absolute = wrapped.override(base=act, observation=codecs.eepose, action=codecs.absolute_position)


@cfn.config(weights=None)
def sample(origins: list[cfn.Config], weights: list[float] | None):
    """One could use the following CLI:
    --policy=.sample --policy.origins='[".act"]' --policy.origins.0.checkpoint_path=<yada-yada>
    """
    from positronic.policy import SampledPolicy

    return SampledPolicy(*origins, weights=weights)


@cfn.config(host='localhost', port=8000, resize=640, model_id=None)
def remote(host: str, port: int, resize: int | None = None, model_id: str | None = None):
    from positronic.policy.remote import RemotePolicy

    return RemotePolicy(host, port, resize, model_id=model_id)


# Pre-configured policy instances
act_latest = act_absolute.override(**{
    'base.checkpoints_dir': 's3://checkpoints/full_ft/act/021225/',
    'base.n_action_steps': 15,
})
act_q_latest = act_absolute.override(**{
    'base.checkpoints_dir': 's3://checkpoints/full_ft_q/act/031225/',
    'observation': codecs.eepose_q,
    'base.n_action_steps': 15,
})
