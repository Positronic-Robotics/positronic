import configuronic as cfn
import pos3

from positronic.cfg import codecs
from positronic.policy import Codec, Policy
from positronic.utils import get_latest_checkpoint


@cfn.config()
def placeholder():
    raise RuntimeError(
        'This config is not supposed to be instantiated, '
        'and is used only to simplify relative imports of other policy configs.'
    )


@cfn.config(codec=None)
def wrapped(base: Policy, codec: Codec | None):
    if codec is None:
        return base
    return codec.wrap(base)


@cfn.config(checkpoint=None)
def act(checkpoints_dir: str, checkpoint: str | None, n_action_steps: int | None = None, device=None):
    from lerobot.policies.act.modeling_act import ACTPolicy

    from positronic.vendors.lerobot_0_3_3.backbone import register_all
    from positronic.vendors.lerobot_0_3_3.policy import LerobotPolicy

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


act_absolute = wrapped.override(
    base=act, codec=codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action)
)


@cfn.config(weights=None)
def sample(origins: list[cfn.Config], weights: list[float] | None):
    """One could use the following CLI:
    --policy=.sample --policy.origins='[".act"]' --policy.origins.0.checkpoint_path=<yada-yada>
    """
    from positronic.policy import SampledPolicy

    return SampledPolicy(*origins, weights=weights)


@cfn.config(host='localhost', port=8000, resize=640, model_id=None, horizon_sec=None, codec=None)
def remote(
    host: str,
    port: int,
    resize: int | None = None,
    model_id: str | None = None,
    horizon_sec: float | None = None,
    codec: Codec | None = None,
):
    from positronic.policy.remote import RemotePolicy

    effective_resize = None if codec and codec.meta.get('image_sizes') else resize
    policy = RemotePolicy(host, port, effective_resize, model_id=model_id, horizon_sec=horizon_sec)
    return codec.wrap(policy) if codec else policy
