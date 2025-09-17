from pathlib import Path
import configuronic as cfn

from positronic.cfg.policy import action, observation


@cfn.config()
def local(path: str):
    from positronic.dataset.local_dataset import LocalDataset
    return LocalDataset(Path(path))


@cfn.config(transforms=[
                observation.franka_mujoco_stackcubes,
                action.absolute_position,
            ],
            pass_through=False)
def transformed(path, transforms, pass_through):
    from positronic.dataset.local_dataset import LocalDataset
    from positronic.dataset.transforms import TransformedDataset
    return TransformedDataset(LocalDataset(Path(path)), *transforms, pass_through=pass_through)
