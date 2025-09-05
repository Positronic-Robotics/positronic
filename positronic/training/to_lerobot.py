# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lerobot>=0.3",
#     "torch",
#     "tqdm",
#     "configuronic",
#     "numpy",
#     "scipy",
# ]
# ///
from pathlib import Path
from typing import Any, Sequence
from collections.abc import Sequence as AbcSequence


import torch
import tqdm
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import configuronic as cfn
from positronic import geom
from positronic.dataset import transforms
from positronic.dataset import Signal
from positronic.dataset.local_dataset import LocalDataset


def seconds_to_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


class ActionEncoder(transforms.EpisodeTransform):

    def __init__(self, rot_repr: geom.Rotation.Representation = geom.Rotation.Representation.QUAT):
        self._rot_repr = rot_repr

    @property
    def keys(self) -> Sequence[str]:
        return ['actions']

    def transform(self, name: str, episode: transforms.Episode) -> Signal[Any] | Any:
        if name != 'actions':
            raise ValueError(f"Unknown action key: {name}")

        rotations = transforms.recode_rotation(geom.Rotation.Representation.QUAT, self._rot_repr,
                                               episode['target_robot_position_quaternion'])

        return transforms.concat(rotations, episode['target_robot_position_translation'], episode['target_grip'])

    def get_features(self):
        return {
            'actions': {
                "dtype": "float64",
                "shape": (self._rot_repr.size + 3 + 1, ),
                "names": [
                    *[f"rot_{i}" for i in range(self._rot_repr.size)], "translation_x", "translation_y",
                    "translation_z", "grip"
                ]
            }
        }


@cfn.config()
def action_baseline(rot_repr: geom.Rotation.Representation = geom.Rotation.Representation.QUAT):
    return ActionEncoder(rot_repr=rot_repr)


class ObservationEncoder(transforms.EpisodeTransform):

    def __init__(self, state_features: list[str], **image_configs):
        self._state_features = state_features
        self._image_configs = image_configs

    @property
    def keys(self) -> Sequence[str]:
        return ['observation.state'] + [f'observation.images.{k}' for k in self._image_configs.keys()]

    def transform(self, name: str, episode: transforms.Episode) -> Signal[Any] | Any:
        if name == 'observation.state':
            return transforms.concat(*[episode[k] for k in self._state_features])
        elif name.startswith('observation.images.'):
            key = name[len('observation.images.'):]
            input_key, (widht, height) = self._image_configs[key]
            return transforms.Image.resize_with_pad(widht, height, episode[input_key])
        else:
            raise ValueError(f"Unknown observation key: {name}")

    def get_features(self):
        features = {}
        for key, (input_key, (width, height)) in self._image_configs.items():
            features['observation.images.' + key] = {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
        features['observation.state'] = {
            "dtype": "float64",
            "shape": (8,),  # TODO: Invent the way to compute it dynamically
            "names": ["state"],
        }
        return features


@cfn.config()
def state_baseline():
    return ObservationEncoder(
        state_features=[
            'target_robot_position_quaternion',
            'target_robot_position_translation',
            'grip',
        ],
        left=('image.handcam_left', (224, 224)),
        side=('image.back_view', (224, 224)),
    )


class EpisodeDictDataset(torch.utils.data.Dataset):
    """
    This dataset is used to load the episode data from the file and encode it into a dictionary.
    """

    def __init__(self, input_dir: Path, state_encoder: ObservationEncoder, action_encoder: ActionEncoder, fps: int):
        self.dataset = LocalDataset(input_dir)
        self.observation_encoder = state_encoder
        self.action_encoder = action_encoder
        self.fps = fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        episode = transforms.TransformEpisode(self.dataset[idx], self.observation_encoder, self.action_encoder)
        start, finish = episode.start_ts, episode.last_ts
        timestamps = np.arange(start, finish, 1e9 / self.fps, dtype=np.int64)
        return episode.time[timestamps]


def _collate_fn(x):
    return x[0]


def append_data_to_dataset(dataset: LeRobotDataset,
                           input_dir: Path,
                           state_encoder: ObservationEncoder,
                           action_encoder: ActionEncoder,
                           task: str | None = None,
                           num_workers: int = 16,
                           fps: int = 30):
    dataset.start_image_writer(num_processes=num_workers)
    # Process each episode file
    total_length_sec = 0

    episode_dataset = EpisodeDictDataset(input_dir, state_encoder, action_encoder, fps=fps)
    dataloader = torch.utils.data.DataLoader(
        episode_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_fn
    )

    for episode_idx, ep_dict in enumerate(tqdm.tqdm(dataloader, desc="Processing episodes")):
        num_frames = len(ep_dict['actions'])
        total_length_sec += num_frames * 1 / dataset.fps

        for i in range(num_frames):
            ep_task = task
            if task is None and 'task' in ep_dict:
                ep_task = ep_dict['task']

            frame = {}
            for key, value in ep_dict.items():
                frame[key] = ep_dict[key]
                if isinstance(value, AbcSequence | np.ndarray) and len(value) == num_frames:
                    frame[key] = frame[key][i]

            dataset.add_frame(frame, task=ep_task or '')

        dataset.save_episode()
        dataset.encode_episode_videos(episode_idx)
    print(f"Total length of the dataset: {seconds_to_str(total_length_sec)}")


@cfn.config(
    fps=30,
    video=True,
    state_encoder=state_baseline,
    action_encoder=action_baseline,
    task="pick plate from the table and place it into the dishwasher",
)
def convert_to_lerobot_dataset(input_dir: str, output_dir: str, fps: int, video: bool,
                               state_encoder: ObservationEncoder, action_encoder: ActionEncoder, task: str):
    features = {**state_encoder.get_features(), **action_encoder.get_features()}

    dataset = LeRobotDataset.create(repo_id='local',
                                    fps=fps,
                                    root=Path(output_dir),
                                    use_videos=video,
                                    features=features,
                                    image_writer_threads=32)

    append_data_to_dataset(dataset=dataset,
                           input_dir=Path(input_dir),
                           state_encoder=state_encoder,
                           action_encoder=action_encoder,
                           task=task)
    print(f"Dataset converted and saved to {output_dir}")


@cfn.config(
    state_encoder=state_baseline,
    action_encoder=action_baseline,
    task="pick plate from the table and place it into the dishwasher",
)
def append_data_to_lerobot_dataset(dataset_dir: str, input_dir: Path, state_encoder: ObservationEncoder,
                                   action_encoder: ActionEncoder, task: str):
    dataset = LeRobotDataset(repo_id='local', root=dataset_dir)

    append_data_to_dataset(dataset=dataset,
                           input_dir=Path(input_dir),
                           state_encoder=state_encoder,
                           action_encoder=action_encoder,
                           task=task)
    print(f"Dataset extended with {input_dir} and saved to {dataset_dir}")


if __name__ == "__main__":
    cfn.cli({'convert': convert_to_lerobot_dataset, 'append': append_data_to_lerobot_dataset})
