"""
Example:
`positronic-train --dataset_root=~/datasets/lerobot/stack_cubes`
"""

from dataclasses import dataclass, field
from pathlib import Path

import configuronic as cfn
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.envs.configs import EnvConfig, FeatureType, PolicyFeature
from lerobot.scripts import train as lerobot_train

import positronic.utils.s3 as pos3


@EnvConfig.register_subclass('positronic')
@dataclass
class PositronicEnvConfig(EnvConfig):
    """Configuration for the Positronic environment."""

    fps: int = 15
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            'action': PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
            'observation.images.left': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            'observation.images.side': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            'action': ACTION,
            'observation.images.left': OBS_IMAGE,
            'observation.images.side': OBS_IMAGE,
            'observation.state': OBS_STATE,
        }
    )

    @property
    def gym_kwargs(self) -> dict:
        return {}


def _update_config(cfg: TrainPipelineConfig, **cfg_kwargs):
    for k, v in cfg_kwargs.items():
        try:
            parts = k.split('.')
            config = cfg
            for part in parts[:-1]:
                config = getattr(config, part)
            setattr(config, parts[-1], v)
        except AttributeError as e:
            raise AttributeError(f'Could not update config for {k}') from e


@cfn.config()
@pos3.with_mirror()
def train(
    dataset_root: str,
    run_name: str,
    output_dir,
    base_config: str = 'positronic/training/train_config.json',
    **cfg_kwargs,
):
    assert Path(base_config).is_file(), f'Base config file {base_config} does not exist.'
    cfg = TrainPipelineConfig.from_pretrained(base_config)
    cfg.env = PositronicEnvConfig()
    cfg.job_name = run_name
    cfg.dataset.root = pos3.download(dataset_root)
    cfg.dataset.repo_id = 'local'
    cfg.eval_freq = 0
    cfg.policy.push_to_hub = False
    cfg.output_dir = pos3.upload(output_dir) / run_name
    _update_config(cfg, **cfg_kwargs)

    print('Starting training...')
    lerobot_train.init_logging()
    lerobot_train.train(cfg)
    print('Training finished.')


def _internal_main():
    cfn.cli(train)


if __name__ == '__main__':
    _internal_main()
