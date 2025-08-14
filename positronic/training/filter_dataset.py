from pathlib import Path
from typing import Dict, Any
from io import BytesIO
import os

import imageio
import numpy as np
import tqdm
import fire
import torch


def _decode_video_from_array(array: torch.Tensor) -> torch.Tensor:
    """
    Decodes array with encoded video bytes into a video.

    Args:
        array (torch.Tensor): Tensor containing encoded video bytes.

    Returns:
        torch.Tensor: Decoded video frames.

    Raises:
        ValueError: If the video data cannot be decoded.
    """
    with BytesIO() as buffer:
        buffer.write(array.numpy().tobytes())
        buffer.seek(0)
        try:
            with imageio.get_reader(buffer, format='mp4') as reader:
                return torch.from_numpy(np.stack([frame for frame in reader]))
        except Exception:
            try:
                print("Failed to decode video data. Trying to read from file.")
                with open('tmp.mp4', 'wb') as f:
                    f.write(array.numpy().tobytes())
                return torch.from_numpy(np.stack(imageio.mimread('tmp.mp4')))
            except Exception as e:
                raise ValueError(f"Failed to decode video data: {str(e)}")
            finally:
                os.remove('tmp.mp4')


def get_consistent_samples(
        ds: Dict[str, Any],
        min_interval_length: int = 30,
):

    distance_between_controllers = torch.linalg.norm(ds['umi_left_translation'] - ds['umi_right_translation'], axis=1)

    # since controllers dosen't move relative to each other, the distance should be constant
    median_distance = torch.median(distance_between_controllers)

    # find samples where the distance is more than 1cm different from the median
    consistent_samples = torch.abs(distance_between_controllers - median_distance) < 0.01

    # get consistent intervals
    consistent_intervals = []

    # Find continuous intervals of consistent samples
    if consistent_samples.any():
        # Get indices where state changes (from consistent to inconsistent or vice versa)
        change_points = torch.where(consistent_samples[1:] != consistent_samples[:-1])[0] + 1

        # Add start and end points
        if consistent_samples[0]:
            starts = torch.cat([torch.tensor([0]), change_points[1::2]])
            ends = change_points[::2]
        else:
            starts = change_points[::2]
            ends = change_points[1::2]

        # Handle case where dataset ends with consistent samples
        if len(starts) > len(ends):
            ends = torch.cat([ends, torch.tensor([len(consistent_samples)])])

        # Create intervals as (start, end) tuples
        consistent_intervals = [(starts[i].item(), ends[i].item()) for i in range(len(starts))]

    # Remove intervals that are too short
    consistent_intervals = [
        (start, end)
        for start, end in consistent_intervals
        if end - start >= min_interval_length
    ]

    return consistent_intervals


def slice_sample(sample, consistent_intervals):
    sample_len = len(sample['umi_left_translation'])
    new_samples = []

    if len(consistent_intervals) == 1 and consistent_intervals[0][0] == 0 and consistent_intervals[0][1] == sample_len:
        return [sample]

    videos = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and '.image' in k:
            video = _decode_video_from_array(v)
            videos[k] = video

    for start, end in consistent_intervals:
        s = {}
        for k, v in sample.items():
            if isinstance(v, torch.Tensor) and '.image' in k:
                # Encode the video slice to bytes using imageio
                video_slice = videos[k][start:end]
                encoded_video = imageio.mimsave("<bytes>", video_slice, format='mp4')
                # Store the encoded frames
                s[k] = torch.from_numpy(np.frombuffer(encoded_video, dtype=np.uint8))
            elif isinstance(v, torch.Tensor) and len(v) == sample_len:
                s[k] = v[start:end]
            else:
                s[k] = v
        s['episode_start'] = s['image_timestamp'][0]
        new_samples.append(s)

    return new_samples


def filter_samples(path: str, target_path: str, min_interval_length: int = 30):
    """
    Filter the dataset to remove samples with non-consistent controller positions.

    Args:
        path: (str) Path to the dataset.
        target_path: (str) Path to save the filtered dataset.
        min_interval_length: (int) Minimum length of the interval to be considered.
    """
    target_path = Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    for episode in tqdm.tqdm(sorted(Path(path).glob('*.pt'))):
        sample = torch.load(episode)
        consistent_intervals = get_consistent_samples(sample, min_interval_length)
        new_samples = slice_sample(sample, consistent_intervals)
        for i, s in enumerate(new_samples):
            torch.save(s, target_path / f'{episode.stem}_filtered_{i}.pt')


if __name__ == '__main__':
    fire.Fire(filter_samples)
