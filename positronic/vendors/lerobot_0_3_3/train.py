"""
LeRobot training script using ACT policy.

Example:
    # Train with default codec (ee)
    cd docker && docker compose run --rm lerobot-train \\
      --input_path=s3://interim/sim_stack/lerobot/eepose_absolute/ \\
      --exp_name=my_experiment \\
      --output_dir=s3://checkpoints/lerobot/

    # Override codec
    cd docker && docker compose run --rm lerobot-train \\
      --input_path=s3://interim/sim_stack/lerobot/joints_absolute/ \\
      --exp_name=my_experiment \\
      --codec=@positronic.vendors.lerobot_0_3_3.codecs.joints \\
      --output_dir=s3://checkpoints/lerobot/
"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import configuronic as cfn
import pos3
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.envs.configs import EnvConfig, FeatureType, PolicyFeature

from positronic import utils
from positronic.policy import Codec
from positronic.utils.logging import init_logging
from positronic.vendors.lerobot_0_3_3 import codecs as lerobot_codecs
from positronic.vendors.lerobot_0_3_3.backbone import BACKBONES


@EnvConfig.register_subclass('positronic')
@dataclass
class PositronicEnvConfig(EnvConfig):
    """Configuration for the Positronic environment."""

    fps: int = 15
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def gym_kwargs(self) -> dict:
        return {}


def build_env_config_from_codec(codec: Codec) -> PositronicEnvConfig:
    """Build PositronicEnvConfig from codec metadata."""
    inference_meta = codec.meta
    training_meta = codec.training_encoder.meta

    fps = int(inference_meta.get('action_fps', 15))

    assert 'lerobot_features' in training_meta, (
        f"Codec training_encoder missing 'lerobot_features'. Keys: {list(training_meta.keys())}"
    )
    lerobot_features = training_meta['lerobot_features']

    features = {}
    features_map = {}

    for key, meta in lerobot_features.items():
        assert 'shape' in meta, f"Feature '{key}' missing 'shape' in metadata: {meta}"
        dtype = meta.get('dtype', 'float32')

        match key, dtype:
            case 'action', _:
                features[key] = PolicyFeature(type=FeatureType.ACTION, shape=meta['shape'])
                features_map[key] = ACTION
            case _, 'video':
                h, w, c = meta['shape']
                assert c == 3, f"Visual feature '{key}' expected 3 channels, got {c}"
                features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
                features_map[key] = OBS_IMAGE
            case _, ('float32' | 'float64' | 'int32' | 'int64'):
                features[key] = PolicyFeature(type=FeatureType.STATE, shape=meta['shape'])
                features_map[key] = OBS_STATE
            case _:
                raise ValueError(f"Feature '{key}' has unsupported dtype '{dtype}'")

    if not features:
        raise ValueError('No features extracted from codec metadata')

    return PositronicEnvConfig(fps=fps, features=features, features_map=features_map)


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


@cfn.config(codec=lerobot_codecs.ee, num_train_steps=None, backbone=None)
@pos3.with_mirror()
def train(
    input_path: str,
    exp_name: str,
    output_dir: str,
    codec: Codec,
    num_train_steps: int | None,
    backbone: str | None,
    **cfg_kwargs,
):
    # Handle codec passed as string (e.g., from CLI)
    if isinstance(codec, str):
        parts = codec.split('.')
        module = __import__('.'.join(parts[:-1]), fromlist=[parts[-1]])
        codec = getattr(module, parts[-1])
        logging.info(f'Resolved codec from string: {codec}')

    base_config = str(Path(__file__).resolve().parent.joinpath('train_config.json'))
    assert Path(base_config).is_file(), f'Base config file {base_config} does not exist.'
    exp_name = str(exp_name)
    cfg = TrainPipelineConfig.from_pretrained(base_config)

    if backbone is not None:
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Available: {list(BACKBONES.keys())}")
        bb_name, bb_weights, _ = BACKBONES[backbone]
        cfg.policy.vision_backbone = bb_name
        cfg.policy.pretrained_backbone_weights = bb_weights

    cfg.env = build_env_config_from_codec(codec)

    if os.getenv('WANDB_API_KEY'):
        cfg.wandb.enable = True
        cfg.wandb.project = 'lerobot-train'
        cfg.wandb.run_id = exp_name
        cfg.wandb.disable_artifact = True

    cfg.job_name = exp_name
    cfg.dataset.root = str(pos3.download(input_path))
    cfg.dataset.repo_id = 'local'
    cfg.eval_freq = 0
    cfg.policy.push_to_hub = False
    cfg.output_dir = pos3.sync(output_dir, exclude=[f'{exp_name}/wandb/*']) / exp_name

    # Set training steps if provided
    if num_train_steps is not None:
        cfg.steps = num_train_steps

    _update_config(cfg, **cfg_kwargs)

    if cfg.resume:
        checkpoints_dir = Path(cfg.output_dir) / 'checkpoints'
        if checkpoints_dir.exists():
            # Find the latest checkpoint directory (highest integer value)
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name))
                config_path = latest_checkpoint / 'pretrained_model' / 'train_config.json'
                logging.info(f'Resuming run. Automatically setting config_path to {config_path}')
                # Hack: lerobot requires config_path to be in sys.argv for validation
                sys.argv.append(f'--config_path={config_path}')
            else:
                logging.critical(f'No numeric checkpoint directories found in {checkpoints_dir}')
        else:
            logging.critical(f'Checkpoints directory {checkpoints_dir} does not exist')

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
    from lerobot.scripts import train as lerobot_train

    lerobot_train.init_logging()
    lerobot_train.train(cfg)
    logging.info('Training finished.')


def _internal_main():
    init_logging()
    cfn.cli(train)


if __name__ == '__main__':
    _internal_main()
