from pathlib import Path

import configuronic as cfn

from positronic.cfg.policy import action, observation
from positronic.dataset.transforms.episode import KeyFuncEpisodeTransform


@cfn.config()
def local(path: str):
    from positronic.dataset.local_dataset import load_all_datasets

    return load_all_datasets(Path(path))


@cfn.config(observation=observation.eepose_mujoco, action=action.absolute_position, task=None, pass_through=False)
def transformed(path, observation, action, task, pass_through):
    """Load datasets with observation/action transforms and optional task label."""
    from positronic.dataset.local_dataset import load_all_datasets
    from positronic.dataset.transforms import TransformedDataset

    extra = []
    if task is not None:
        extra.append(KeyFuncEpisodeTransform(task=lambda _ep: task))

    return TransformedDataset(load_all_datasets(Path(path)), observation, action, *extra, pass_through=pass_through)
