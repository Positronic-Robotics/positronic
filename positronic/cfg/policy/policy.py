import configuronic as cfn

import positronic.utils.s3 as pos3
from positronic.policy.lerobot import LerobotPolicy


def _get_act_policy(
    checkpoint_path: str,
    use_temporal_ensembler: bool = False,
    n_action_steps: int | None = None,
    device: str | None = None,
):
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

    policy = ACTPolicy.from_pretrained(pos3.download(checkpoint_path), strict=True)

    if use_temporal_ensembler:
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps

    return LerobotPolicy(policy, device)


def _get_diffusion_policy(checkpoint_path: str, device: str | None = None):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(pos3.download(checkpoint_path), local_files_only=True, strict=True)
    return LerobotPolicy(policy, device)


@cfn.config(host='localhost', port=8000, n_action_steps=None)
def openpi(host: str, port: int, n_action_steps: int | None):
    """PI0/PI0.5 policy with Cartesian control."""
    from positronic.policy.pi0 import OpenPIRemotePolicy

    return OpenPIRemotePolicy(host, port, n_action_steps)


droid = openpi.override(n_action_steps=15)
act = cfn.Config(_get_act_policy, use_temporal_ensembler=False)
diffusion = cfn.Config(_get_diffusion_policy)
