"""
LeRobot 0.4.x training script (SmolVLA default).

Example:
    cd docker && docker compose run --rm lerobot-train \\
      --input_path=s3://interim/sim_stack/lerobot/ee/ \\
      --exp_name=my_experiment \\
      --output_dir=s3://checkpoints/lerobot/
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import configuronic as cfn
import pos3
from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.envs.configs import EnvConfig, FeatureType, PolicyFeature
from lerobot.policies.xvla.configuration_xvla import XVLAConfig  # noqa: F401 — registers policy choices
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from positronic import utils
from positronic.policy import Codec
from positronic.utils.logging import init_logging
from positronic.vendors.lerobot import codecs as lerobot_codecs


@EnvConfig.register_subclass('positronic')
@dataclass
class PositronicEnvConfig(EnvConfig):
    fps: int = 15
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def gym_kwargs(self) -> dict:
        return {}


def build_env_config_from_codec(codec: Codec) -> PositronicEnvConfig:
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
                features_map[key] = OBS_IMAGES
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


@cfn.config(codec=lerobot_codecs.ee, base_model='lerobot/smolvla_base', num_train_steps=None)
@pos3.with_mirror()
def train(
    input_path: str,
    exp_name: str,
    output_dir: str,
    codec: Codec,
    base_model: str,
    num_train_steps: int | None,
    **cfg_kwargs,
):
    if isinstance(codec, str):
        parts = codec.split('.')
        module = __import__('.'.join(parts[:-1]), fromlist=[parts[-1]])
        codec = getattr(module, parts[-1])
        logging.info(f'Resolved codec from string: {codec}')

    exp_name = str(exp_name)

    policy = PreTrainedConfig.from_pretrained(base_model)
    policy.pretrained_path = Path(base_model)
    policy.push_to_hub = False

    env_config = build_env_config_from_codec(codec)
    num_dataset_cameras = sum(1 for f in env_config.features.values() if f.type is FeatureType.VISUAL)
    num_base_cameras = sum(1 for f in (policy.input_features or {}).values() if f.type is FeatureType.VISUAL)
    policy.input_features = {}
    if num_base_cameras > num_dataset_cameras:
        policy.empty_cameras = num_base_cameras - num_dataset_cameras

    dataset_root = str(pos3.download(input_path))
    output_path = pos3.sync(output_dir, exclude=[f'{exp_name}/wandb/*']) / exp_name

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id='local', root=dataset_root),
        policy=policy,
        env=env_config,
        output_dir=output_path,
        job_name=exp_name,
        eval_freq=0,
        steps=num_train_steps if num_train_steps is not None else 100_000,
    )

    if os.getenv('WANDB_API_KEY'):
        cfg.wandb.enable = True
        cfg.wandb.project = 'lerobot-train'
        cfg.wandb.run_id = exp_name
        cfg.wandb.disable_artifact = True

    _update_config(cfg, **cfg_kwargs)

    if cfg.resume:
        checkpoints_dir = Path(cfg.output_dir) / 'checkpoints'
        if checkpoints_dir.exists():
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=lambda d: int(d.name))
                config_path = latest_checkpoint / 'pretrained_model' / 'train_config.json'
                logging.info(f'Resuming run. Automatically setting config_path to {config_path}')
                sys.argv.append(f'--config_path={config_path}')
            else:
                logging.critical(f'No numeric checkpoint directories found in {checkpoints_dir}')
        else:
            logging.critical(f'Checkpoints directory {checkpoints_dir} does not exist')

    logging.info('Starting training...')
    from lerobot.scripts.lerobot_train import train as lerobot_train

    lerobot_train(cfg)

    utils.save_run_metadata(Path(cfg.output_dir), patterns=['*.py', '*.toml'])
    logging.info('Training finished.')


def _internal_main():
    init_logging()
    cfn.cli(train)


if __name__ == '__main__':
    _internal_main()
