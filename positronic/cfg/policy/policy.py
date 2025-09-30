import configuronic as cfn


def _get_act_policy(checkpoint_path: str, use_temporal_ensembler: bool = False, n_action_steps: int | None = None):
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler

    policy = ACTPolicy.from_pretrained(checkpoint_path, strict=True)

    if use_temporal_ensembler:
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps

    return policy


def _get_diffusion_policy(checkpoint_path: str):
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(checkpoint_path, local_files_only=True, strict=True)
    return policy


@cfn.config(n_action_steps=30)
def pi0(n_action_steps: int | None = None):
    from positronic.policy.pi0 import PI0RemotePolicy

    return PI0RemotePolicy('localhost', 8000, n_action_steps)


act = cfn.Config(_get_act_policy, use_temporal_ensembler=False)
diffusion = cfn.Config(_get_diffusion_policy)
