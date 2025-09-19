# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lerobot>=0.3.3",
#     "torch",
#     "configuronic",
# ]
# ///
"""
PEP 723 standalone script
-------------------------

This script is intentionally decoupled from the core library’s dependencies so
that `lerobot` does not become a hard requirement of `positronic`.

How to run (recommended):
- Use uv to run the script in an isolated env that also installs the local
  project and its dependencies, while adding only the script-specific deps
  (like `lerobot`) as declared above.

Examples:
`uv run -s positronic/training/lerobot_train.py --dataset_root=/tmp/datasets/`
"""
from dataclasses import dataclass, field
from pathlib import Path

import configuronic as cfn
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.envs.configs import EnvConfig, FeatureType, PolicyFeature
from lerobot.scripts import train as lerobot_train


@EnvConfig.register_subclass("positronic")
@dataclass
class PositronicEnvConfig(EnvConfig):
    """Configuration for the Positronic environment."""

    fps: int = 15
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
            "observation.images.left": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "observation.images.left": OBS_IMAGE,
            "observation.images.side": OBS_IMAGE,
            "observation.state": OBS_STATE,
        }
    )

    @property
    def gym_kwargs(self) -> dict:
        return {}


@cfn.config()
def train(dataset_root: str, base_config: str = "positronic/training/train_config.json"):
    assert Path(base_config).is_file(), f"Base config file {base_config} does not exist."
    cfg = TrainPipelineConfig.from_pretrained(base_config)
    cfg.env = PositronicEnvConfig()
    cfg.dataset.root = dataset_root
    cfg.dataset.repo_id = "local"
    cfg.eval_freq = 0
    cfg.policy.push_to_hub = False

    print('Starting training...')
    lerobot_train.init_logging()
    lerobot_train.train(cfg)
    print('Training finished.')


if __name__ == "__main__":
    cfn.cli(train)
