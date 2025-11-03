import configuronic as cfn

import positronic.utils.s3 as pos3
from positronic.cfg.policy import action, observation
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import KeyFuncEpisodeTransform


@cfn.config()
def local(path: str):
    from positronic.dataset.local_dataset import load_all_datasets

    return load_all_datasets(pos3.download(path))


@cfn.config(
    base=local, observation=observation.eepose_mujoco, action=action.absolute_position, task=None, pass_through=False
)
def transformed(base, observation, action, task, pass_through):
    """Load datasets with observation/action transforms and optional task label."""

    extra = []
    if task is not None:
        extra.append(KeyFuncEpisodeTransform(task=lambda _ep: task))

    return TransformedDataset(base, observation, action, *extra, pass_through=pass_through)
