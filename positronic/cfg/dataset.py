import configuronic as cfn

import positronic.utils.s3 as pos3
from positronic.cfg.policy import action, observation
from positronic.dataset.local_dataset import LocalDataset, load_all_datasets
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import KeyFuncEpisodeTransform


@cfn.config()
def local(path: str):
    return LocalDataset(pos3.download(path))


@cfn.config()
def local_all(path: str):
    return load_all_datasets(pos3.download(path))


@cfn.config(
    base=local_all,
    observation=observation.eepose_mujoco,
    action=action.absolute_position,
    task=None,
    pass_through=False,
)
def transformed(base, observation, action, task, pass_through):
    """Load datasets with observation/action transforms and optional task label."""

    extra = []
    if task is not None:
        extra.append(KeyFuncEpisodeTransform(task=lambda _ep: task))

    return TransformedDataset(base, observation, action, *extra, pass_through=pass_through)
