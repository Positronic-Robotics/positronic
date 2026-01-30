"""
LeRobot training script using ACT policy.

Example:
    # Train with default codec (eepose_absolute)
    cd docker && docker compose run --rm lerobot-train \\
      --input_path=s3://interim/sim_stack/lerobot/eepose_absolute/ \\
      --exp_name=my_experiment \\
      --output_dir=s3://checkpoints/lerobot/

    # Override codec
    cd docker && docker compose run --rm lerobot-train \\
      --input_path=s3://interim/sim_stack/lerobot/joints_absolute/ \\
      --exp_name=my_experiment \\
      --codec=@positronic.vendors.lerobot.codecs.joints_absolute \\
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
from positronic.vendors.lerobot import codecs as lerobot_codecs


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


def build_env_config_from_codec(codec: Codec, fps: int = 15) -> PositronicEnvConfig:
    """Build PositronicEnvConfig from codec metadata with validation.

    Args:
        codec: Codec instance with observation and action encoders
        fps: Frames per second for the environment

    Returns:
        PositronicEnvConfig with features derived from codec metadata

    Raises:
        ValueError: If codec metadata is missing or invalid
    """
    # Validate codec has required metadata
    if not hasattr(codec.observation, 'meta') or 'lerobot_features' not in codec.observation.meta:
        raise ValueError(
            f"Codec observation encoder missing 'lerobot_features' metadata. "
            f'Available meta keys: {getattr(codec.observation, "meta", {}).keys()}'
        )

    if not hasattr(codec.action, 'meta') or 'lerobot_features' not in codec.action.meta:
        raise ValueError(
            f"Codec action decoder missing 'lerobot_features' metadata. "
            f'Available meta keys: {getattr(codec.action, "meta", {}).keys()}'
        )

    obs_features = codec.observation.meta['lerobot_features']
    action_features = codec.action.meta['lerobot_features']

    features = {}
    features_map = {}

    # Convert action features
    for key, meta in action_features.items():
        if 'shape' not in meta:
            raise ValueError(f"Action feature '{key}' missing 'shape' in metadata: {meta}")
        features[key] = PolicyFeature(type=FeatureType.ACTION, shape=meta['shape'])
        features_map[key] = ACTION

    # Convert observation features
    for key, meta in obs_features.items():
        if 'shape' not in meta or 'dtype' not in meta:
            raise ValueError(f"Observation feature '{key}' missing 'shape' or 'dtype': {meta}")

        if meta['dtype'] == 'video':
            # Validate image shape is (H, W, C)
            if len(meta['shape']) != 3:
                raise ValueError(f"Visual feature '{key}' expected shape (H, W, C), got {meta['shape']}")
            h, w, c = meta['shape']
            if c != 3:
                raise ValueError(f"Visual feature '{key}' expected 3 channels (RGB), got {c} channels")
            # Convert HWC → CHW for LeRobot
            features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w))
            features_map[key] = OBS_IMAGE
        elif meta['dtype'] in ('float32', 'float64', 'int32', 'int64'):
            # State features (scalars/vectors)
            features[key] = PolicyFeature(type=FeatureType.STATE, shape=meta['shape'])
            features_map[key] = OBS_STATE
        else:
            raise ValueError(
                f"Observation feature '{key}' has unsupported dtype '{meta['dtype']}'. "
                f"Expected 'video', 'float32', 'float64', 'int32', or 'int64'"
            )

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


@cfn.config(codec=lerobot_codecs.eepose_absolute, num_train_steps=None)
@pos3.with_mirror()
def train(input_path: str, exp_name: str, output_dir: str, codec: Codec, num_train_steps: int | None, **cfg_kwargs):
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
