import configuronic as cfn

import positronic.utils.s3 as pos3
from positronic.cfg.policy import action, observation
from positronic.dataset.dataset import ConcatDataset
from positronic.dataset.local_dataset import LocalDataset, load_all_datasets
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import Derive, Group, Identity


@cfn.config()
def local(path: str):
    return LocalDataset(pos3.download(path))


@cfn.config()
def local_all(path: str):
    return load_all_datasets(pos3.download(path))


@cfn.config()
def concat_ds(datasets):
    return ConcatDataset(*datasets)


@cfn.config(transforms=[])
def transform(base, transforms):
    return TransformedDataset(base, *transforms)


@cfn.config()
def group(transforms):
    return Group(*transforms)


@cfn.config(base=local_all, observation=observation.eepose_mujoco, action=action.absolute_position, task=None)
def encoded(base, observation, action, task):
    """Load datasets with encoded observation/action and optional task label."""

    tfs = [observation, action]
    if task is not None:
        tfs.append(Derive(task=lambda _ep: task))
    else:
        tfs.append(Identity('task'))

    return TransformedDataset(base, Group(*tfs))
