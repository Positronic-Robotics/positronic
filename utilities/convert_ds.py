"""Utility for exporting a transformed dataset to disk.

The CLI defined here reads a configured source dataset—most often a
``TransformedDataset`` that reshapes signals lazily—and streams it into a new
``LocalDataset`` directory. The default configuration targets
``update_v0_1_0`` transformation around specified local dataset,
but you are welcome to use your own transformations with ``original_ds`` config entry.

Most signals are copied sample-by-sample; however, video signals require
special handling. Instead of materialising individual frames, the underlying
``VideoSignal`` files (the encoded video and its frame index) are copied
verbatim so the resulting dataset preserves the original video assets without
re-encoding.

Example:

    python -m utilities.convert_ds --original_ds.path /path/to/transformed_source \
        --output_path /path/to/export_root
"""

import shutil
from pathlib import Path

import configuronic as cfn
import tqdm

from positronic.dataset import Dataset, transforms
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.signal import Kind
from positronic.dataset.transforms import KeyFuncEpisodeTransform, TransformedDataset
from positronic.dataset.video import VideoSignal


def _discover_image_signals(dataset: Dataset) -> list[str]:
    """Inspect episodes to find signals carrying image data."""
    episode = dataset[0]
    image_keys = [name for name, signal in episode.signals.items() if signal.kind == Kind.IMAGE]
    return image_keys


@cfn.config()
def update_v0_1_0(path: str):
    dataset = LocalDataset(Path(path))
    image_features = _discover_image_signals(dataset)

    funcs = {
        'controller_positions.right': lambda ep: transforms.concat(
            ep['right_controller_translation'], ep['right_controller_quaternion']
        ),
        'robot_commands.pose': lambda ep: transforms.concat(
            ep['target_robot_position_translation'], ep['target_robot_position_quaternion']
        ),
        'robot_state.q': lambda ep: ep['robot_joints'],
        'robot_state.dq': lambda ep: ep['robot_joints_velocity'],
        'robot_state.ee_pose': lambda ep: transforms.concat(
            ep['robot_position_translation'], ep['robot_position_quaternion']
        ),
    }

    return TransformedDataset(
        dataset, KeyFuncEpisodeTransform(**funcs), pass_through=['grip', 'target_grip'] + image_features
    )


@cfn.config(original_ds=update_v0_1_0)
def main(output_path: str, original_ds: Dataset | None = None):
    root = Path(output_path)
    with LocalDatasetWriter(root) as writer:
        for episode in tqdm.tqdm(original_ds):
            with writer.new_episode() as ew:
                for key, value in episode.static.items():
                    ew.set_static(key, value)

                for key, signal in episode.signals.items():
                    if signal.kind == Kind.IMAGE:
                        assert isinstance(signal, VideoSignal)
                        shutil.copy(signal.video_path, ew.path / signal.video_path.name)
                        shutil.copy(signal.frames_index_path, ew.path / signal.frames_index_path.name)
                        continue

                    for value, ts in signal:
                        ew.append(key, value, ts)


if __name__ == '__main__':
    cfn.cli(main)
