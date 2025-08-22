# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "lerobot>=0.3",
#     "torch",
#     "tqdm",
#     "configuronic",
#     "numpy",
#     "fire",
#     "scipy",
# ]
# ///
from pathlib import Path
import numpy as np
import torch
import tqdm
import configuronic as cfn

from lerobot.datasets.lerobot_dataset import LeRobotDataset


from positronic.dataset.local_dataset import LocalDataset
import positronic.cfg.inference.action
import positronic.cfg.inference.observation
from positronic.inference.action import ActionDecoder
from positronic.inference.observation import ObservationEncoder, ImageTransform, ToArrayTransform


def seconds_to_str(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def observation_features(observation_encoder: ObservationEncoder) -> dict:
    features = {}

    for transform in observation_encoder.transforms:
        if isinstance(transform, ImageTransform):
            features[transform.output_key] = {
                "dtype": "video",
                "shape": (*transform.resize[::-1], 3),
                "names": ["height", "width", "channel"],
            }

        elif isinstance(transform, ToArrayTransform):
            features[transform.output_key] = {
                "dtype": "float64",
                "shape": (transform.n_features,),
                "names": ["state"],
            }
    return features


class EpisodeDictDataset(torch.utils.data.Dataset):
    """
    This dataset is used to load the episode data from the file and encode it into a dictionary.
    """

    def __init__(self, input_dir, observation_encoder: ObservationEncoder, action_encoder: ActionDecoder, fps: int):
        self.dataset = LocalDataset(input_dir)
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder
        self.fps = fps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        episode = self.dataset[idx]
        timestamps = np.linspace(episode.start_ts, episode.last_ts, int(episode.time.duration / 1e9 * self.fps))
        ep_dict = self.observation_encoder.episode_to_dict(episode, timestamps=timestamps)
        ep_dict['action'] = self.action_encoder.encode_episode(episode, timestamps=timestamps)

        return ep_dict


def append_data_to_dataset(
    dataset: LeRobotDataset,
    input_dir: Path,
    observation_encoder: ObservationEncoder,
    action_encoder: ActionDecoder,
    task: str,
    num_workers: int = 16,
    fps: int = 30,
):
    dataset.start_image_writer(num_processes=num_workers)
    # Process each episode file
    total_length = 0

    episode_dataset = EpisodeDictDataset(input_dir, observation_encoder, action_encoder, fps)
    dataloader = torch.utils.data.DataLoader(
        episode_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )

    for episode_idx, ep_dict in enumerate(tqdm.tqdm(dataloader, desc="Processing episodes")):
        num_frames = len(ep_dict['action'])
        total_length += num_frames * 1 / dataset.fps

        for i in range(num_frames):
            frame = {}
            for key in ep_dict.keys():
                frame[key] = ep_dict[key][i]
            dataset.add_frame(frame, task=task)

        dataset.save_episode()
        dataset.encode_episode_videos(episode_idx)
    print(f"Total length of the dataset: {seconds_to_str(total_length)}")


@cfn.config(
    fps=30,
    video=True,
    observation_encoder=positronic.cfg.inference.observation.end_effector_handcam,
    action_encoder=positronic.cfg.inference.action.relative_robot_position,
    task="pick up the red cube and place it on the green cube",
)
def convert_to_lerobot_dataset(
    input_dir: str,
    output_dir: str,
    fps: int,
    video: bool,
    observation_encoder: ObservationEncoder,
    action_encoder: ActionDecoder,
    task: str,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    features = observation_features(observation_encoder)
    features.update(action_encoder.get_features())

    dataset = LeRobotDataset.create(
        repo_id='local',
        fps=fps,
        root=output_dir,
        use_videos=video,
        features=features,
        image_writer_threads=32,
    )

    append_data_to_dataset(
        dataset=dataset,
        input_dir=input_dir,
        observation_encoder=observation_encoder,
        action_encoder=action_encoder,
        task=task,
        fps=fps,
    )
    print(f"Dataset converted and saved to {output_dir}")


@cfn.config(
    observation_encoder=positronic.cfg.inference.observation.end_effector_handcam,
    action_encoder=positronic.cfg.inference.action.relative_robot_position,
    task="pick up the red cube and place it on the green cube",
)
def append_data_to_lerobot_dataset(
    dataset_dir: str,
    input_dir: Path,
    observation_encoder: ObservationEncoder,
    action_encoder: ActionDecoder,
    task: str,
):
    dataset = LeRobotDataset(
        repo_id='local',
        root=dataset_dir,
    )

    append_data_to_dataset(
        dataset=dataset,
        input_dir=Path(input_dir),
        observation_encoder=observation_encoder,
        action_encoder=action_encoder,
        task=task,
        fps=dataset.fps,
    )
    print(f"Dataset extended with {input_dir} and saved to {dataset_dir}")


if __name__ == "__main__":
    cfn.cli({
        'convert': convert_to_lerobot_dataset,
        'append': append_data_to_lerobot_dataset,
    })
