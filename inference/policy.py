import os
from typing import Dict, Optional

import yaml



def get_config(checkpoint_path: str):
    with open(os.path.join(checkpoint_path, 'config.yaml'), 'r') as f:
        return yaml.safe_load(f)


def _get_policy_config(checkpoint_path: str, policy_name: str, policy_args: Dict):
    if policy_name == 'act':
        from lerobot.common.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
        policy = ACTPolicy.from_pretrained(checkpoint_path)

        if policy_args.get('use_temporal_ensembler'):
            policy.config.n_action_steps = 1
            policy.config.temporal_ensemble_coeff = 0.01
            policy.temporal_ensembler = ACTTemporalEnsembler(0.01, policy.config.chunk_size)

        if policy_args.get('n_action_steps'):
            policy.config.n_action_steps = policy_args['n_action_steps']
            
    elif policy_name == 'diffusion':
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    else:
        raise ValueError(f"Unsupported policy name: {policy_name}")

    return policy


def get_policy(checkpoint_path: str, policy_args: Optional[Dict] = None):
    config = get_config(checkpoint_path)
    policy_name = config['policy']['name']
    policy_args = policy_args or {}

    policy = _get_policy_config(checkpoint_path, policy_name, policy_args)
    policy.eval()

    return policy
