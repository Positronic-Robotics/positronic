import configuronic as cfn

import positronic.utils.s3 as pos3
from positronic.cfg.policy import action, observation
from positronic.dataset.local_dataset import LocalDataset, load_all_datasets
from positronic.dataset.transforms import Concatenate, TransformedDataset
from positronic.dataset.transforms.episode import KeyFuncEpisodeTransform


@cfn.config()
def local(path: str):
    return LocalDataset(pos3.download(path))


@cfn.config()
def local_all(path: str):
    return load_all_datasets(pos3.download(path))


@cfn.config(base=local_all, observation=observation.eepose_mujoco, action=action.absolute_position, task=None)
def encoded(base, observation, action, task):
    """Load datasets with encoded observation/action and optional task label."""

    tfs = [observation, action]
    if task is not None:
        tfs.append(KeyFuncEpisodeTransform(add={'task': lambda _ep: task}, pass_through=False))

    return TransformedDataset(base, Concatenate(*tfs))
