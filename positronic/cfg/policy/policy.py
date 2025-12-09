import configuronic as cfn
import pos3

from positronic.policy.lerobot import LerobotPolicy


@cfn.config(use_temporal_ensembler=False)
def act(checkpoint_path: str, use_temporal_ensembler: bool, n_action_steps: int | None = None, device=None):
    def factory():
        from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

        policy = ACTPolicy.from_pretrained(pos3.download(checkpoint_path), strict=True)

        if use_temporal_ensembler:
            policy.config.n_action_steps = 1
            policy.config.temporal_ensemble_coeff = 0.01
            policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

        if n_action_steps is not None:
            policy.config.n_action_steps = n_action_steps
        return policy

    return LerobotPolicy(factory, device, extra_meta={'type': 'act', 'checkpoint_path': checkpoint_path})


def _get_diffusion_policy(checkpoint_path: str, device: str | None = None):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    def factory():
        return DiffusionPolicy.from_pretrained(pos3.download(checkpoint_path), local_files_only=True, strict=True)

    return LerobotPolicy(factory, device, extra_meta={'type': 'diffusion', 'checkpoint_path': checkpoint_path})


@cfn.config(host='localhost', port=8000, n_action_steps=None)
def openpi(host: str, port: int, n_action_steps: int | None):
    """PI0/PI0.5 policy with Cartesian control."""
    from positronic.policy.pi0 import OpenPIRemotePolicy

    return OpenPIRemotePolicy(host, port, n_action_steps)


droid = openpi.override(n_action_steps=15)
diffusion = cfn.Config(_get_diffusion_policy)


@cfn.config(n_action_steps=None)
def groot(host: str = 'localhost', port: int = 9000, timeout_ms: int = 15000, n_action_steps: int | None = None):
    from positronic.policy.gr00t import Gr00tPolicy

    return Gr00tPolicy(host, port, timeout_ms, n_action_steps)


@cfn.config(weights=None)
def sample(origins: list[cfn.Config], weights: list[float] | None):
    """One could use the following CLI:
    --policy=.sample --policy.origins='[".act"]' --policy.origins.0.checkpoint_path=<yada-yada>
    """
    from positronic.policy import SampledPolicy

    return SampledPolicy(*origins, weights=weights)
