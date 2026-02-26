"""
Utility configurations for constructing datasets.
"""

from typing import Any

import configuronic as cfn
import pos3
from pos3 import Profile

from positronic.dataset.dataset import ConcatDataset, Dataset
from positronic.dataset.local_dataset import LocalDataset, load_all_datasets
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import EpisodeTransform, Group

PUBLIC = Profile(
    local_name='positronic-public', endpoint='https://storage.eu-north1.nebius.cloud', public=True, region='eu-north1'
)


@cfn.config()
def local(path: str, profile: Profile | None = None):
    return LocalDataset(pos3.download(path, profile=profile))


@cfn.config()
def local_all(path: str, profile: Profile | None = None):
    return load_all_datasets(pos3.download(path, profile=profile))


@cfn.config()
def concat_ds(datasets: list[Dataset]):
    return ConcatDataset(*datasets)


@cfn.config(transforms=[])
def transform(base: Dataset, transforms: list[EpisodeTransform], extra_meta: dict[str, Any] = None):
    return TransformedDataset(base, *transforms, extra_meta=extra_meta)


@cfn.config()
def group(transforms: list[EpisodeTransform]):
    return Group(*transforms)


@cfn.config()
def apply_codec(dataset: Dataset, codec):
    """Apply vendor codec to a dataset for training.

    Args:
        dataset: Raw Positronic dataset
        codec: Codec instance with training_encoder

    Returns:
        TransformedDataset with codec's training encoder applied

    Example:
        positronic-to-lerobot convert \\
          --dataset=@positronic.cfg.ds.apply_codec \\
          --dataset.dataset=.internal.droid \\
          --dataset.codec=@positronic.vendors.gr00t.codecs.ee_quat \\
          --output_dir=/data/lerobot_dataset
    """
    return TransformedDataset(dataset, codec.training_encoder)
