from pathlib import Path

import configuronic as cfn

from positronic.cfg.policy import action, observation


@cfn.config()
def local(path: str):
    from positronic.dataset.local_dataset import load_all_datasets

    return load_all_datasets(Path(path))


@cfn.config(observation=observation.eepose_mujoco, action=action.absolute_position, pass_through=False)
def transformed(path, observation, action, pass_through):
    from positronic.dataset.local_dataset import load_all_datasets
    from positronic.dataset.transforms import TransformedDataset

    return TransformedDataset(load_all_datasets(Path(path)), observation, action, pass_through=pass_through)
