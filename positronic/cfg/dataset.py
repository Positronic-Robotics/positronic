from pathlib import Path

import configuronic as cfn

from positronic.cfg.policy import action, observation


@cfn.config()
def local(path: str):
    from positronic.dataset.local_dataset import LocalDataset

    return LocalDataset(Path(path))


@cfn.config(observation=observation.franka_mujoco_stackcubes, action=action.absolute_position, pass_through=False)
def transformed(path, observation, action, pass_through):
    from positronic.dataset.local_dataset import LocalDataset
    from positronic.dataset.transforms import TransformedDataset

    return TransformedDataset(LocalDataset(Path(path)), observation, action, pass_through=pass_through)
