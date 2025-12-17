from functools import partial

import configuronic as cfn
import pos3

from positronic.cfg.policy import action as act_cfg
from positronic.cfg.policy import observation as obs_cfg
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
    from positronic.policy.base import DecodedPolicy, EncodedPolicy

    if action is not None:
        meta = {f'action.{k}': v for k, v in action.meta.items()}
        base = DecodedPolicy(base, action.decode, extra_meta=meta)
    if observation is not None:
        meta = {f'observation.{k}': v for k, v in observation.meta.items()}
        base = EncodedPolicy(base, observation.encode, extra_meta=meta)
    return base


@cfn.config(checkpoint=None, use_temporal_ensembler=False)
def act(
    checkpoints_dir: str,
    checkpoint: str | None,
    use_temporal_ensembler: bool,
    n_action_steps: int | None = None,
    device=None,
):
    def factory(fully_specified_checkpoint_dir: str):
        from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

        policy = ACTPolicy.from_pretrained(pos3.download(fully_specified_checkpoint_dir), strict=True)

        if use_temporal_ensembler:
            policy.config.n_action_steps = 1
            policy.config.temporal_ensemble_coeff = 0.01
            policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

        if n_action_steps is not None:
            policy.config.n_action_steps = n_action_steps
        return policy

    checkpoints_dir = checkpoints_dir.rstrip('/') + '/checkpoints/'
    if checkpoint is None:
        checkpoint = get_latest_checkpoint(checkpoints_dir)
    else:
        checkpoint = str(checkpoint).strip('/')

    fully_specified_checkpoint_dir = checkpoints_dir.rstrip('/') + '/' + checkpoint + '/pretrained_model/'
    return LerobotPolicy(
        partial(factory, fully_specified_checkpoint_dir),
        device,
        extra_meta={'type': 'act', 'checkpoint_path': fully_specified_checkpoint_dir},
    )


@cfn.config()
def diffusion(checkpoint_path: str, device: str | None = None):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    def factory():
        return DiffusionPolicy.from_pretrained(pos3.download(checkpoint_path), local_files_only=True, strict=True)

    return LerobotPolicy(factory, device, extra_meta={'type': 'diffusion', 'checkpoint_path': checkpoint_path})


act_absolute = wrapped.override(base=act, observation=obs_cfg.eepose, action=act_cfg.absolute_position)


@cfn.config(host='localhost', port=8000, n_action_steps=None)
def openpi_remote(host: str, port: int, n_action_steps: int | None):
    """PI0/PI0.5 policy with Cartesian control."""
    from positronic.policy.openpi import OpenPIRemotePolicy

    return OpenPIRemotePolicy(host, port, n_action_steps)


openpi_positronic = wrapped.override(
    base=openpi_remote, observation=obs_cfg.openpi_positronic, action=act_cfg.absolute_position
)
openpi_droid = wrapped.override(
    base=openpi_remote.override(n_action_steps=15), observation=obs_cfg.openpi_droid, action=act_cfg.joint_delta
)


@cfn.config(n_action_steps=None)
def groot_remote(host: str = 'localhost', port: int = 9000, timeout_ms: int = 15000, n_action_steps: int | None = None):
    from positronic.policy.gr00t import Gr00tPolicy

    return Gr00tPolicy(host, port, timeout_ms, n_action_steps)


groot_ee = wrapped.override(base=groot_remote, observation=obs_cfg.groot_infer, action=act_cfg.groot_infer)
groot_ee_q = groot_ee.override(observation=obs_cfg.groot_ee_q_infer)


@cfn.config(weights=None)
def sample(origins: list[cfn.Config], weights: list[float] | None):
    """One could use the following CLI:
    --policy=.sample --policy.origins='[".act"]' --policy.origins.0.checkpoint_path=<yada-yada>
    """
    from positronic.policy import SampledPolicy

    return SampledPolicy(*origins, weights=weights)


@cfn.config(host='localhost', port=8000, resize=640)
def remote(host: str, port: int, resize: int | None = None):
    from positronic.policy.remote import RemotePolicy

    return RemotePolicy(host, port)
