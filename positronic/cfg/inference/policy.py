import ironic as ir


def _get_act_policy(checkpoint_path: str, use_temporal_ensembler: bool = False, n_action_steps: int | None = None):
    from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    policy = ACTPolicy.from_pretrained(checkpoint_path)

    if use_temporal_ensembler:
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = 0.01
        policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

    if n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps

    return policy


def _get_diffusion_policy(checkpoint_path: str):
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    return policy


act = ir.Config(
    _get_act_policy,
    use_temporal_ensembler=False
)

diffusion = ir.Config(
    _get_diffusion_policy,
)

policy_options = ir.Option(
    act=act,
    diffusion=diffusion
)
