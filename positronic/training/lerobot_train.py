"""
Example:
`positronic-train --dataset_root=~/datasets/lerobot/stack_cubes`
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import configuronic as cfn
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.envs.configs import EnvConfig, FeatureType, PolicyFeature
from lerobot.scripts import train as lerobot_train

import positronic.utils.s3 as pos3
from positronic import utils
from positronic.utils.logging import init_logging


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
def train(dataset_root: str, run_name: str, output_dir, **cfg_kwargs):
    base_config = str(Path(__file__).resolve().parent.joinpath('train_config.json'))
    assert Path(base_config).is_file(), f'Base config file {base_config} does not exist.'
    run_name = str(run_name)
    cfg = TrainPipelineConfig.from_pretrained(base_config)
    cfg.env = PositronicEnvConfig()

    if os.getenv('WANDB_API_KEY'):
        cfg.wandb.enable = True
        cfg.wandb.project = 'lerobot-train'
        cfg.wandb.run_id = run_name
        cfg.wandb.disable_artifact = True

    cfg.job_name = run_name
    cfg.dataset.root = str(pos3.download(dataset_root))
    cfg.dataset.repo_id = 'local'
    cfg.eval_freq = 0
    cfg.policy.push_to_hub = False
    cfg.output_dir = pos3.sync(output_dir, exclude=[f'{run_name}/wandb/*']) / run_name
    _update_config(cfg, **cfg_kwargs)

    # Start a background thread to save metadata once the directory is created by lerobot
    # This avoids FileExistsError since lerobot expects the directory to not exist
    def _save_metadata_delayed():
        output_path = Path(cfg.output_dir)
        # Wait for directory to be created (max 5 min)
        for _ in range(60):
            if output_path.exists():
                utils.save_run_metadata(output_path, patterns=['*.py', '*.toml'])
                return
            time.sleep(5)
        logging.warning(f'Timed out waiting for output directory {output_path} to be created')

    threading.Thread(target=_save_metadata_delayed, daemon=True).start()

    logging.info('Starting training...')
    lerobot_train.init_logging()
    lerobot_train.train(cfg)
    logging.info('Training finished.')


def _internal_main():
    init_logging()
    cfn.cli(train)


if __name__ == '__main__':
    _internal_main()
