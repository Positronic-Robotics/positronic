import os
import time
from typing import Dict, Optional

import yaml
import torch

from lerobot.common.datasets.relative_position import delta_position_to_relative

class PlaybackPolicy:
    def __init__(self, path: str, relative: bool = False):
        s = torch.load(path)
        self.action = s['action']
        if relative:
            print("Converting to relative")
            self.action = delta_position_to_relative(self.action)
        self.current_step = 0


    def select_action(self, state: Dict):
        if len(self.action) <= self.current_step:
            return torch.zeros(1, 7)
        action = self.action[self.current_step]
        self.current_step += 1

        print(f"Action sent: {self.current_step}")
        return action

    def to(self, device: torch.device):
        pass

    def reset(self):
        self.current_step = 0

    def chunk_start(self):
        return False


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
