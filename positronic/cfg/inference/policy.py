import configuronic as cfn


def _get_act_policy(checkpoint_path: str, use_temporal_ensembler: bool = False, n_action_steps: int | None = None):
    from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    policy = ACTPolicy.from_pretrained(checkpoint_path, strict=True)

    if use_temporal_ensembler:
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps

    return policy


def _get_diffusion_policy(checkpoint_path: str):
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    policy = DiffusionPolicy.from_pretrained(checkpoint_path, local_files_only=True, strict=True)
    return policy


def _get_pi0_policy(checkpoint_path: str):
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    policy = PI0Policy.from_pretrained(checkpoint_path, strict=True)
    return policy


def _get_pi0_fast_policy(checkpoint_path: str):
    from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
    policy = PI0FASTPolicy.from_pretrained(checkpoint_path, strict=True)
    return policy


act = cfn.Config(_get_act_policy, use_temporal_ensembler=False)
diffusion = cfn.Config(_get_diffusion_policy)
pi0 = cfn.Config(_get_pi0_policy)
pi0_fast = cfn.Config(_get_pi0_fast_policy)
