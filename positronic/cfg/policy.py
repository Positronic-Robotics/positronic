import configuronic as cfn
import pos3

from positronic import keys
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


remote = cfn.Config(RemotePolicy, url='localhost:8000')


@cfn.config(balance=2)
def balanced(balance: int):
    from positronic.policy.sampler import BalancedSampler

    return BalancedSampler(balance=balance)


@cfn.config(endpoints={}, weights={}, recording_dir=None, sampler=None, group_fields=None, headers=None)
def production(
    endpoints: dict[str, str],
    weights: dict[str, float],
    recording_dir: str | None,
    sampler: Sampler | None,
    group_fields: list[str] | None,
    headers: dict[str, str] | None,
):
    """Routes each episode to one of several remote endpoints, each named for CLI overrides.

    An endpoint is one URL, so `--policy.endpoints.groot=desktop:8000` adds or repoints one without
    restating the others. `weights` name the same endpoints and set their sampling odds; endpoints left
    out of it weigh 1.0. `headers` reach every endpoint, since one set of credentials fronts them all.

    The endpoint's name is what identifies it — the sampling key, and the field recorded on each episode.
    Two deployments of one checkpoint report the same server metadata, so only the name tells them apart.
    """
    if not endpoints:
        raise ValueError('At least one endpoint must be given, e.g. --policy.endpoints.groot=desktop:8000')
    if unknown := weights.keys() - endpoints.keys():
        raise ValueError(f'weights name unknown endpoints: {sorted(unknown)}; known are {sorted(endpoints)}')
    # Every Sampler but the default uniform one picks by episode counts alone, so weights would be dropped.
    if weights and sampler is not None:
        raise ValueError(f'weights cannot be combined with {type(sampler).__name__}, which samples by count')
    policies = [
        RemotePolicy(url, label=name, recording_dir=recording_dir, headers=headers) for name, url in endpoints.items()
    ]
    w = [weights.get(name, 1.0) for name in endpoints] if weights else None
    return SampledPolicy(*policies, weights=w, sampler=sampler, group_fields=group_fields, key_field='label')


@cfn.config()
def phail_single(hostname, w_openpi=1.0, w_groot=1.0, w_act=1.0):
    openpi = RemotePolicy(f'{hostname}:8000')
    groot = RemotePolicy(f'{hostname}:8001')
    act = RemotePolicy(f'{hostname}:8002')

    return SampledPolicy(openpi, groot, act, weights=[w_openpi, w_groot, w_act])


EVAL_GROUP_FIELDS = [keys.TASK, 'eval.object', 'eval.tote_placement', 'eval.external_camera']

phail_multiple = production.override(
    endpoints={'smolvla': 'notebook:8000', 'act': 'notebook:8001', 'groot': 'desktop:8000', 'openpi': 'vm-openpi:8000'},
    sampler=balanced,
    group_fields=EVAL_GROUP_FIELDS,
)

# The gyros deployments sit behind the workspace's proxy, so pass --policy.headers with its token.
gyros_p497 = production.override(
    endpoints={
        'nm167k_us': 'wss://runway-pythagoras-dev--gyros-p497-nm167k-us-gyrosserver-web.modal.run',
        'fm167k_us': 'wss://runway-pythagoras-dev--gyros-p497-fm167k-us-gyrosserver-web.modal.run',
        'nm167k': 'wss://runway-pythagoras-dev--gyros-p497-nm167k-gyrosserver-web.modal.run',
    },
    sampler=balanced,
    group_fields=EVAL_GROUP_FIELDS,
)

# Naming no sampler leaves the uniform one, so each episode draws between the two independently: nothing
# about the episodes so far narrows what runs next, which balancing by count would.
gyros_fm_act = production.override(
    endpoints={
        'actuni': 'wss://runway-pythagoras-dev--gyros-fm-actuni-gyrosserver-web.modal.run',
        'tsu_actuni': 'wss://runway-pythagoras-dev--gyros-fm-tsu-actuni-gyrosserver-web.modal.run',
    }
)
