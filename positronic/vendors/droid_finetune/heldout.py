"""The windows of a held-out Cosmos3-DROID tree, and the action error of a policy on them.

A window is 33 frames from an allowed start: the policy sees the first frame, and its 32 actions are scored against
the recorded ones. The allowed starts and the choice among them are those of `flux-action evaluate`, so a FLUX 3
Action evaluation and a Cosmos3 evaluation of one tree score the same windows.
"""

import itertools
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from .report import ScoredWindow

DATASET_DIR = 'droid_plus_lerobot_640x360_20260412'
SUCCESS_DIR = 'success'
KEEP_RANGES_FILE = 'keep_ranges.json'
FPS = 15
CHUNK = 32
PROMPT_SEPARATOR = ' | '

JOINT_POSITIONS = 'observation.state.joint_positions'
GRIPPER_POSITION = 'observation.state.gripper_position'
ACTION_JOINT_POSITION = 'action.joint_position'
ACTION_GRIPPER_POSITION = 'action.gripper_position'
# The observation keys of the RoboLab action servers, and the dataset feature of each view.
VIEWS = {
    'observation/wrist_image_left': 'observation.image.wrist_image_left',
    'observation/exterior_image_1_left': 'observation.image.exterior_image_1_left',
    'observation/exterior_image_2_left': 'observation.image.exterior_image_2_left',
}
OBS_JOINT_POSITION = 'observation/joint_position'
OBS_GRIPPER_POSITION = 'observation/gripper_position'
OBS_PROMPT = 'prompt'

Observation = dict[str, np.ndarray | str]
Predictor = Callable[[Observation], np.ndarray]


@dataclass(frozen=True)
class Video:
    file: Path
    first_frame: int


@dataclass(frozen=True)
class Episode:
    index: int
    episode_id: str
    prompt: str
    n_frames: int
    data_file: Path
    videos: dict[str, Video]
    valid_ranges: list[tuple[int, int]]


@dataclass(frozen=True)
class Window:
    episode: Episode
    start: int


# A keep-ranges key is a DROID recording path. The episode id is the path segment between these two.
_KEY_ID_START = '/r2d2-data-full/'
_KEY_ID_END = '/recordings/'


def episode_id_of_key(key: str) -> str:
    return key.split(_KEY_ID_START, 1)[1].split(_KEY_ID_END, 1)[0]


def read_episodes(tree: Path) -> list[Episode]:
    """The episodes of `tree/<DATASET_DIR>/success`, in the order of `flux-action index-droid`.

    The prompt is the first alternative of the stored task, as `flux-action evaluate` takes it.
    """
    root = tree / DATASET_DIR
    success = root / SUCCESS_DIR
    info = json.loads((success / 'meta' / 'info.json').read_text())
    if info['codebase_version'] != 'v3.0' or info['fps'] != FPS:
        raise ValueError(f'{success}: expected LeRobot v3.0 at {FPS} Hz')
    keep_ranges = json.loads((root / KEEP_RANGES_FILE).read_text())
    ranges = {episode_id_of_key(key): [(int(a), int(b)) for a, b in value] for key, value in keep_ranges.items()}
    episodes = []
    for path in sorted((success / 'meta' / 'episodes').rglob('*.parquet')):
        for row in pq.read_table(path).to_pylist():
            videos = {
                view: Video(
                    success
                    / info['video_path'].format(
                        video_key=feature,
                        chunk_index=row[f'videos/{feature}/chunk_index'],
                        file_index=row[f'videos/{feature}/file_index'],
                    ),
                    round(row[f'videos/{feature}/from_timestamp'] * FPS),
                )
                for view, feature in VIEWS.items()
            }
            data_file = info['data_path'].format(chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])
            episodes.append(
                Episode(
                    index=row['episode_index'],
                    episode_id=row['episode_id'],
                    prompt=row['tasks'][0].split(PROMPT_SEPARATOR)[0],
                    n_frames=row['length'],
                    data_file=success / data_file,
                    videos=videos,
                    valid_ranges=ranges.get(row['episode_id'], []),
                )
            )
    return episodes


def valid_starts(n_frames: int, valid_ranges: Sequence[tuple[int, int]]) -> list[int]:
    """Each start whose 33 frames lie inside one half-open keep range and inside the episode."""
    return [s for first, end in valid_ranges for s in range(max(first, 0), min(end, n_frames) - CHUNK)]


def spread(n: int, k: int) -> list[int]:
    """`k` indices spread evenly over `range(n)`, the first and the last included, distinct and sorted."""
    if n <= 0 or k <= 0:
        return []
    if k == 1 or n == 1:
        return [0]
    return sorted({round(j * (n - 1) / (k - 1)) for j in range(k)})


def plan_windows(episodes: Sequence[Episode], per_episode: int) -> list[Window]:
    windows = []
    for episode in episodes:
        starts = valid_starts(episode.n_frames, episode.valid_ranges)
        windows += [Window(episode, starts[i]) for i in spread(len(starts), per_episode)]
    return windows


def read_rows(episode: Episode) -> tuple[np.ndarray, np.ndarray]:
    """The state and the action of each frame of `episode`, each [n_frames, 8]: seven joints, then the gripper."""
    columns = [JOINT_POSITIONS, GRIPPER_POSITION, ACTION_JOINT_POSITION, ACTION_GRIPPER_POSITION]
    table = pq.read_table(
        episode.data_file, columns=['frame_index', *columns], filters=[('episode_index', '=', episode.index)]
    ).sort_by('frame_index')
    if table.num_rows != episode.n_frames:
        raise ValueError(
            f'{episode.data_file}: episode {episode.index} has {table.num_rows} of {episode.n_frames} rows'
        )
    # A converter writes the gripper either as a scalar or as a one-element list per frame.
    joints, grip, action_joints, action_grip = (
        np.stack(table[c].to_numpy(zero_copy_only=False)).reshape(table.num_rows, -1) for c in columns
    )
    return np.concatenate([joints, grip], axis=1), np.concatenate([action_joints, action_grip], axis=1)


def decode_frames(video: Video, starts: Sequence[int]) -> dict[int, np.ndarray]:
    """The RGB frame of `video` at each window start of its episode."""
    wanted = {video.first_frame + start: start for start in starts}
    frames = {}
    with av.open(str(video.file)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        assert time_base is not None
        # A seek lands on the keyframe at or before the first wanted frame.
        container.seek(int(min(wanted) / (FPS * time_base)), stream=stream)
        for frame in container.decode(stream):
            assert frame.pts is not None
            index = round(frame.pts * time_base * FPS)
            if index in wanted:
                frames[wanted[index]] = frame.to_ndarray(format='rgb24')
            if index >= max(wanted):
                break
    missing = sorted(set(starts) - frames.keys())
    if missing:
        raise ValueError(f'{video.file}: no frame for window starts {missing[:5]}')
    return frames


def _by_episode(windows: Sequence[Window]) -> Iterator[tuple[Episode, list[int]]]:
    for _, group in itertools.groupby(windows, key=lambda w: w.episode.index):
        group = list(group)
        yield group[0].episode, [w.start for w in group]


def score_windows(windows: Sequence[Window], predict: Predictor) -> tuple[list[ScoredWindow], np.ndarray]:
    """Ask `predict` for the 32 actions of each window from its first frame, state and prompt, and score them.

    Returns each window's mean squared error, and the mean over windows of the error of each channel: seven joints
    in radians, then the gripper. The observation carries the raw gripper closed fraction, as the dataset stores it;
    the action servers flip it.
    """
    scored, per_channel = [], []
    for episode, starts in _by_episode(windows):
        state, action = read_rows(episode)
        views = {view: decode_frames(video, starts) for view, video in episode.videos.items()}
        for start in starts:
            observation: Observation = {
                **{view: frames[start] for view, frames in views.items()},
                OBS_JOINT_POSITION: state[start, :7],
                OBS_GRIPPER_POSITION: state[start, 7:],
                OBS_PROMPT: episode.prompt,
            }
            predicted = np.asarray(predict(observation), dtype=np.float64)
            recorded = action[start : start + CHUNK].astype(np.float64)
            if predicted.shape != recorded.shape:
                raise ValueError(f'the policy returned actions of shape {predicted.shape}, not {recorded.shape}')
            channels = ((predicted - recorded) ** 2).mean(axis=0)
            scored.append(ScoredWindow(episode.index, start, float(channels.mean())))
            per_channel.append(channels)
    if not scored:
        raise ValueError('the tree has no window to score')
    return scored, np.mean(per_channel, axis=0)
