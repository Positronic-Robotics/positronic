import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from positronic.vendors.droid_finetune import heldout

DATA_PATH = 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
VIDEO_PATH = 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
KEY = 'gs://xembodiment_data/r2d2/r2d2-data-full/{id}/recordings/MP4--x/trajectory.h5'
# Two episodes share one data file and one video file per view, as LeRobot v3 packs them.
LENGTHS = {'ep-a': 40, 'ep-b': 45}
KEEP_RANGES = {'ep-a': [[0, 40]], 'ep-b': [[2, 45]]}
GRAY_STEP = 3


def _gray(global_frame: int) -> int:
    return GRAY_STEP * global_frame


def _write_video(path: Path, n_frames: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=heldout.FPS)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for i in range(n_frames):
            image = np.full((48, 64, 3), _gray(i), dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')))
        container.mux(stream.encode())


def _state(frame: np.ndarray) -> np.ndarray:
    """Seven joints and a gripper that name the global frame they belong to."""
    return np.concatenate([frame[:, None] + np.arange(7) / 10, (frame[:, None] % 10) / 10], axis=1)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    success = tmp_path / heldout.DATASET_DIR / heldout.SUCCESS_DIR
    (success / 'meta' / 'episodes').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'fps': heldout.FPS, 'data_path': DATA_PATH, 'video_path': VIDEO_PATH}
    (success / 'meta' / 'info.json').write_text(json.dumps(info))
    keep = {KEY.format(id=episode_id): ranges for episode_id, ranges in KEEP_RANGES.items()}
    (tmp_path / heldout.DATASET_DIR / heldout.KEEP_RANGES_FILE).write_text(json.dumps(keep))

    episodes, rows, first = [], [], 0
    for index, (episode_id, length) in enumerate(LENGTHS.items()):
        frames = np.arange(first, first + length)
        state = _state(frames)
        action = _state(frames + 1)
        rows.append({
            'index': frames,
            'episode_index': np.full(length, index),
            'frame_index': np.arange(length),
            heldout.JOINT_POSITIONS: list(state[:, :7].astype(np.float32)),
            heldout.GRIPPER_POSITION: list(state[:, 7:].astype(np.float32)),
            heldout.ACTION_JOINT_POSITION: list(action[:, :7].astype(np.float32)),
            heldout.ACTION_GRIPPER_POSITION: list(action[:, 7:].astype(np.float32)),
        })
        episode = {
            'episode_index': index,
            'episode_id': episode_id,
            'tasks': [f'pick the {episode_id} | grab the {episode_id}'],
            'length': length,
            'data/chunk_index': 0,
            'data/file_index': 0,
            'dataset_from_index': first,
            'dataset_to_index': first + length,
        }
        for feature in heldout.VIEWS.values():
            episode |= {
                f'videos/{feature}/chunk_index': 0,
                f'videos/{feature}/file_index': 0,
                f'videos/{feature}/from_timestamp': first / heldout.FPS,
                f'videos/{feature}/to_timestamp': (first + length) / heldout.FPS,
            }
        episodes.append(episode)
        first += length
    data = success / DATA_PATH.format(chunk_index=0, file_index=0)
    data.parent.mkdir(parents=True)
    pq.write_table(pa.concat_tables([pa.table(r) for r in reversed(rows)]), data)
    pq.write_table(pa.Table.from_pylist(episodes), success / 'meta' / 'episodes' / 'file-000.parquet')
    for feature in heldout.VIEWS.values():
        _write_video(success / VIDEO_PATH.format(video_key=feature, chunk_index=0, file_index=0), first)
    return tmp_path


def test_valid_starts_keep_the_33_frames_of_a_window_inside_the_range_and_the_episode():
    assert heldout.valid_starts(40, [(0, 40)]) == list(range(8))
    assert heldout.valid_starts(40, [(0, 35)]) == [0, 1, 2]
    assert heldout.valid_starts(40, [(-3, 90)]) == list(range(8))
    assert heldout.valid_starts(40, [(10, 40)]) == []


def test_plan_windows_picks_the_starts_that_flux_action_evaluate_picks(tree):
    windows = heldout.plan_windows(heldout.read_episodes(tree), per_episode=4)

    assert [(w.episode.episode_id, w.start) for w in windows] == [
        ('ep-a', 0), ('ep-a', 2), ('ep-a', 5), ('ep-a', 7),
        ('ep-b', 2), ('ep-b', 5), ('ep-b', 9), ('ep-b', 12),
    ]  # fmt: skip


def test_an_episode_without_keep_ranges_has_no_window(tree):
    episodes = heldout.read_episodes(tree)
    stripped = [heldout.Episode(**{**vars(e), 'valid_ranges': []}) if e.episode_id == 'ep-a' else e for e in episodes]

    assert {w.episode.episode_id for w in heldout.plan_windows(stripped, per_episode=4)} == {'ep-b'}


def test_score_windows_shows_the_policy_the_first_frame_state_and_prompt_of_each_window(tree):
    windows = heldout.plan_windows(heldout.read_episodes(tree), per_episode=4)
    seen = []

    def predict(observation):
        seen.append(observation)
        state = np.concatenate([observation[heldout.OBS_JOINT_POSITION], observation[heldout.OBS_GRIPPER_POSITION]])
        frame = int(round(state[0]))
        return _state(np.arange(frame + 1, frame + 1 + heldout.CHUNK)) + 0.5

    scored, per_channel = heldout.score_windows(windows, predict)

    first_frames = {'ep-a': 0, 'ep-b': LENGTHS['ep-a']}
    for window, observation in zip(windows, seen, strict=True):
        global_frame = first_frames[window.episode.episode_id] + window.start
        assert observation[heldout.OBS_JOINT_POSITION][0] == pytest.approx(global_frame)
        assert observation[heldout.OBS_PROMPT] == f'pick the {window.episode.episode_id}'
        for view in heldout.VIEWS:
            assert observation[view].mean() == pytest.approx(_gray(global_frame), abs=4)
    assert [(s.episode_index, s.start) for s in scored] == [(w.episode.index, w.start) for w in windows]
    assert [s.action_mse for s in scored] == pytest.approx([0.25] * len(windows), rel=1e-5)
    assert per_channel == pytest.approx(np.full(8, 0.25), rel=1e-5)  # the tree stores float32


def test_score_windows_refuses_actions_of_the_wrong_length(tree):
    windows = heldout.plan_windows(heldout.read_episodes(tree), per_episode=1)

    with pytest.raises(ValueError, match='shape'):
        heldout.score_windows(windows, lambda observation: np.zeros((heldout.CHUNK - 1, 8)))
