import configuronic as cfn
import pos3

from positronic.cfg import codecs
from positronic.offboard.client import DEFAULT_INFER_TIMEOUT
from positronic.policy import ActionHorizon, Codec, Policy, Recorder, RemotePolicy
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
    base=act, codec=codecs.compose.override(obs=codecs.eepose_obs, action=codecs.absolute_pos_action, horizon=1.0)
)


@cfn.config(weights=None)
def sample(origins: list[cfn.Config], weights: list[float] | None):
    """One could use the following CLI:
    --policy=.sample --policy.origins='[".act"]' --policy.origins.0.checkpoint_path=<yada-yada>
    """
    from positronic.policy import SampledPolicy

    return SampledPolicy(*origins, weights=weights)


@cfn.config(
    host='localhost',
    port=8000,
    resize=640,
    model_id=None,
    horizon_sec=None,
    codec=None,
    secure=False,
    recording_dir=None,
    infer_timeout=DEFAULT_INFER_TIMEOUT,
)
def remote(
    host: str,
    port: int,
    resize: int | None = None,
    model_id: str | None = None,
    horizon_sec: float | None = None,
    codec: Codec | None = None,
    headers: dict[str, str] | None = None,
    secure: bool = False,
    recording_dir: str | None = None,
    infer_timeout: float = DEFAULT_INFER_TIMEOUT,
):
    effective_resize = None if codec and codec.meta.get('image_sizes') else resize
    policy = RemotePolicy(
        host, port, effective_resize, model_id=model_id, headers=headers, secure=secure, infer_timeout=infer_timeout
    )
    if horizon_sec is not None:
        codec = ActionHorizon(horizon_sec) | codec if codec else ActionHorizon(horizon_sec)
    if recording_dir is not None:
        rec = Recorder(pos3.sync(recording_dir))
        if codec is None:
            # No codec: the raw and server boundaries coincide, so a single tap.
            return rec.tap('raw').wrap(policy)
        return (rec.tap('raw') | codec | rec.tap('server')).wrap(policy)
    return codec.wrap(policy) if codec else policy


@cfn.config(
    host=None,
    port=8000,
    weight=1.0,
    model_id=None,
    resize=640,
    horizon_sec=None,
    codec=None,
    secure=False,
    recording_dir=None,
    infer_timeout=DEFAULT_INFER_TIMEOUT,
)
def weighted_remote(
    host: str | None,
    port: int,
    weight: float,
    model_id: str | None,
    resize: int | None,
    horizon_sec: float | None,
    codec: Codec | None = None,
    headers: dict[str, str] | None = None,
    secure: bool = False,
    recording_dir: str | None = None,
    infer_timeout: float = DEFAULT_INFER_TIMEOUT,
):
    if not host:
        return None

    effective_resize = None if codec and codec.meta.get('image_sizes') else resize
    policy = RemotePolicy(
        host, port, effective_resize, model_id=model_id, headers=headers, secure=secure, infer_timeout=infer_timeout
    )
    if horizon_sec is not None:
        codec = ActionHorizon(horizon_sec) | codec if codec else ActionHorizon(horizon_sec)
    if recording_dir is not None:
        rec = Recorder(pos3.sync(recording_dir))
        if codec is None:
            # No codec: the raw and server boundaries coincide, so a single tap.
            return rec.tap('raw').wrap(policy), weight
        return (rec.tap('raw') | codec | rec.tap('server')).wrap(policy), weight
    return (codec.wrap(policy) if codec else policy), weight


@cfn.config(balance=2)
def balanced(balance: int):
    from positronic.policy.sampler import BalancedSampler

    return BalancedSampler(balance=balance)


@cfn.config(
    groot=weighted_remote.copy(),
    openpi=weighted_remote.copy(),
    act=weighted_remote.copy(),
    smolvla=weighted_remote.copy(),
    extra=None,
    sampler=None,
    group_fields=None,
)
def production(groot, openpi, act, smolvla, extra, sampler, group_fields):
    from positronic.policy import SampledPolicy

    entries = [e for e in [groot, openpi, act, smolvla] if e is not None]
    if extra:
        entries.extend(e for e in extra if e is not None)
    if not entries:
        raise ValueError('At least one vendor policy must be enabled')
    policies, weights = zip(*entries, strict=False)
    return SampledPolicy(*policies, weights=weights, sampler=sampler, group_fields=group_fields)


@cfn.config()
def phail_single(hostname, w_openpi=1.0, w_groot=1.0, w_act=1.0):
    from positronic.policy import SampledPolicy

    openpi = RemotePolicy(hostname, 8000, resize=640)
    groot = RemotePolicy(hostname, 8001, resize=640)
    act = RemotePolicy(hostname, 8002, resize=640)

    return SampledPolicy(openpi, groot, act, weights=[w_openpi, w_groot, w_act])


phail_multiple = production.override(**{
    'smolvla.host': 'notebook',
    'smolvla.port': 8000,
    'act.host': 'notebook',
    'act.port': 8001,
    'groot.host': 'desktop',
    'groot.port': 8000,
    'openpi.host': 'vm-openpi',
    'openpi.port': 8000,
    'sampler': balanced,
    'group_fields': ['task', 'eval.object', 'eval.tote_placement', 'eval.external_camera'],
})


spoons_ablation = production.override(**{
    'groot.host': 'vm-openpi',
    'groot.port': 8000,
    'smolvla.host': 'vm-openpi',
    'smolvla.port': 8001,
    'act.host': 'vm-openpi',
    'act.port': 8002,
    'sampler': balanced,
    'group_fields': ['task', 'eval.object', 'eval.tote_placement', 'eval.external_camera'],
})
