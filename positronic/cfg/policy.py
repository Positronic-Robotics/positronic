import configuronic as cfn
import pos3

from positronic.cfg import codecs
from positronic.policy import Codec, Policy, RemotePolicy, SampledPolicy
from positronic.policy.sampler import Sampler
from positronic.policy.spec import PolicySource, inline
from positronic.policy.wrappers import ChunkedSchedule
from positronic.utils import get_latest_checkpoint


@cfn.config()
def placeholder():
    raise RuntimeError(
        'This config is not supposed to be instantiated, '
        'and is used only to simplify relative imports of other policy configs.'
    )


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


@cfn.config(
    base=act, codec=codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action, horizon=1.0)
)
def act_absolute(base: Policy, codec: Codec):
    """ACT with the absolute-position codec, composed in-process."""
    return inline(ChunkedSchedule() | codec | PolicySource(base))


@cfn.config(weights=None)
def sample(origins: list[cfn.Config], weights: list[float] | None):
    """One could use the following CLI:
    --policy=.sample --policy.origins='[".act"]' --policy.origins.0.checkpoint_path=<yada-yada>
    """
    return SampledPolicy(*origins, weights=weights)


remote = cfn.Config(RemotePolicy, host='localhost', port=8000, resize=640, params={})

remote_url = cfn.Config(RemotePolicy.from_url, resize=640)


@cfn.config(balance=2)
def balanced(balance: int):
    from positronic.policy.sampler import BalancedSampler

    return BalancedSampler(balance=balance)


@cfn.config(endpoints={}, weights={}, resize=640, recording_dir=None, sampler=None, group_fields=None)
def production(
    endpoints: dict[str, str],
    weights: dict[str, float],
    resize: int | None,
    recording_dir: str | None,
    sampler: Sampler | None,
    group_fields: list[str] | None,
):
    """Routes each episode to one of several remote endpoints, each named for CLI overrides.

    An endpoint is one URL (`RemotePolicy.from_url`), so `--policy.endpoints.groot=desktop:8000` adds or
    repoints one without restating the others. `weights` are read by the default uniform sampler.
    """
    if not endpoints:
        raise ValueError('At least one endpoint must be given, e.g. --policy.endpoints.groot=desktop:8000')
    policies = [RemotePolicy.from_url(url, resize, recording_dir=recording_dir) for url in endpoints.values()]
    w = [weights.get(name, 1.0) for name in endpoints] if weights else None
    return SampledPolicy(*policies, weights=w, sampler=sampler, group_fields=group_fields)


@cfn.config()
def phail_single(hostname, w_openpi=1.0, w_groot=1.0, w_act=1.0):
    openpi = RemotePolicy(hostname, 8000, resize=640)
    groot = RemotePolicy(hostname, 8001, resize=640)
    act = RemotePolicy(hostname, 8002, resize=640)

    return SampledPolicy(openpi, groot, act, weights=[w_openpi, w_groot, w_act])


_EVAL_GROUP_FIELDS = ['task', 'eval.object', 'eval.tote_placement', 'eval.external_camera']

phail_multiple = production.override(
    endpoints={'smolvla': 'notebook:8000', 'act': 'notebook:8001', 'groot': 'desktop:8000', 'openpi': 'vm-openpi:8000'},
    sampler=balanced,
    group_fields=_EVAL_GROUP_FIELDS,
)


spoons_ablation = production.override(
    endpoints={'groot': 'vm-openpi:8000', 'smolvla': 'vm-openpi:8001', 'act': 'vm-openpi:8002'},
    sampler=balanced,
    group_fields=_EVAL_GROUP_FIELDS,
)
