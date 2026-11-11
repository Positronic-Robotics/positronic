"""
Write positronic recordings as Cosmos3-DROID training sets, which the Cosmos3 and FLUX 3 Action DROID recipes read.

Each tree `<output_dir>/train/` and `<output_dir>/heldout/` holds `droid_plus_lerobot_640x360_20260412/success/`
(LeRobot v3.0, 15 Hz, three views at 640x360) and `keep_ranges.json`. A kept episode starts at its first frame
15 cm below the first frame's end-effector height. `<output_dir>/start_trim.csv` gives the cut of each episode and
names the episodes that never get that low, which are left out. The Cosmos3 DROID loader opens `success/` and
`failure/` unless its dataset sets `use_success_only=True`.

Usage
  uv run --extra lerobot python -m positronic.vendors.lerobot.to_cosmos3_droid split \\
    --dataset=@positronic.cfg.ds.local_all --dataset.path=<recordings> --output=<heldout.json>
  uv run --extra lerobot python -m positronic.vendors.lerobot.to_cosmos3_droid convert \\
    --dataset=@positronic.cfg.ds.local_all --dataset.path=<recordings> --output_dir=<dir> --heldout=<heldout.json>
"""

import csv
import dataclasses
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import configuronic as cfn
import numpy as np
import pos3
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import tqdm
from eval_vocabulary.episode import INSTRUCTION
from eval_vocabulary.timeline import recording_timeline
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pimm.logging import init_logging
from positronic import keys
from positronic.dataset import Dataset
from positronic.dataset.episode import META_CREATED_TS_NS, META_UID, Episode, EpisodeContainer
from positronic.dataset.time import Time
from positronic.dataset.transforms.image import ResizeMode, transform_frame
from positronic.vendors.cosmos3 import droid_dataset as droid
from positronic.vendors.lerobot.to_lerobot import _collate_fn, _raise_fd_limit

TREES = ('train', 'heldout')
START_TRIM_FILE = 'start_trim.csv'
# Each DROID view and the recorded camera that fills it. A recording with one exterior camera fills both slots.
_VIEWS = {
    droid.EXTERIOR_IMAGE_1: (keys.EXTERIOR_IMAGE,),
    droid.EXTERIOR_IMAGE_2: (keys.EXTERIOR_IMAGE_2, keys.EXTERIOR_IMAGE),
    droid.WRIST_IMAGE: (keys.WRIST_IMAGE,),
}
_WIDTH, _HEIGHT = droid.VIEW_SIZE
FEATURES = {
    **{
        view: {'dtype': 'video', 'shape': (_HEIGHT, _WIDTH, 3), 'names': ['height', 'width', 'channels']}
        for view in _VIEWS
    },
    droid.JOINT_POSITIONS: {'dtype': 'float32', 'shape': (7,), 'names': None},
    droid.GRIPPER_POSITION: {'dtype': 'float32', 'shape': (1,), 'names': None},
    droid.CARTESIAN_POSITION: {'dtype': 'float32', 'shape': (6,), 'names': None},
    droid.ACTION_JOINT_POSITION: {'dtype': 'float32', 'shape': (7,), 'names': None},
    droid.ACTION_GRIPPER_POSITION: {'dtype': 'float32', 'shape': (1,), 'names': None},
}


def _views(episode: Episode) -> dict[str, str]:
    return {view: next(name for name in sources if name in episode.signals) for view, sources in _VIEWS.items()}


def _prompt(episode: Episode) -> str:
    if INSTRUCTION in episode.signals:
        return droid.prompt(list(episode.signals[INSTRUCTION].values()))
    return episode.static[keys.TASK]


_STATE_SIGNALS = (keys.JOINTS, keys.GRIP, keys.TARGET_GRIP, keys.EE_POSE)


@dataclass(frozen=True)
class KeptFrames:
    """The rows, the resized views and the prompt of the frames that the trim keeps."""

    rows: dict[str, np.ndarray]
    images: dict[str, list[np.ndarray]]
    prompt: str


@dataclass(frozen=True)
class TrimmedEpisode:
    uid: str
    frames: int
    cut: int | None
    kept: KeptFrames | None


@dataclass(frozen=True)
class TrimRow:
    """A line of `start_trim.csv`. `tree` is empty for an episode that the trim leaves out."""

    uid: str
    tree: str
    frames: int
    cut_frame: int | None
    cut_s: float | None


class DroidEpisodes(torch.utils.data.Dataset):
    """Each episode, sampled at 15 Hz and trimmed at its start."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> TrimmedEpisode:
        episode = self.dataset[idx]
        views = _views(episode)
        used = EpisodeContainer({name: episode.signals[name] for name in {*_STATE_SIGNALS, *views.values()}})
        timeline = recording_timeline(used.timelines)
        start, finish = used.bounds(timeline)
        grid = [Time(**{timeline: t}) for t in np.arange(start, finish, 1e9 / droid.FPS, dtype=np.int64)]
        state = {name: np.asarray(used.signals[name].time[grid].values(), dtype=np.float64) for name in _STATE_SIGNALS}
        ee_pose = state[keys.EE_POSE]
        cut = droid.start_cut_frame(ee_pose[:, 2])
        if cut is None or len(grid) - cut < 2:
            return TrimmedEpisode(episode.meta[META_UID], len(grid), None, None)
        rows = droid.droid_rows(state[keys.JOINTS], state[keys.GRIP], state[keys.TARGET_GRIP], ee_pose, cut)
        kept = grid[cut : len(grid) - 1]
        decoded = {name: used.signals[name].time[kept].values() for name in set(views.values())}
        images = {
            view: [transform_frame(frame, _WIDTH, _HEIGHT, mode=ResizeMode.PAD) for frame in decoded[name]]
            for view, name in views.items()
        }
        return TrimmedEpisode(episode.meta[META_UID], len(grid), cut, KeptFrames(rows, images, _prompt(episode)))


def _add_episode(lr_dataset: LeRobotDataset, kept: KeptFrames) -> None:
    # LeRobot stamps frame i at float32(i / 15), which equals the float32 grid bit for bit (a test pins it).
    for i in range(len(kept.rows[droid.JOINT_POSITIONS])):
        frame = {name: values[i] for name, values in kept.rows.items()}
        frame.update({view: frames[i] for view, frames in kept.images.items()})
        lr_dataset.add_frame({**frame, 'task': kept.prompt})
    lr_dataset.save_episode(parallel_encoding=False)


def _add_episode_ids(root: Path, episode_ids: list[str]) -> None:
    """Both recipes read an episode's id from the `episode_id` column of `meta/episodes`."""
    for path in sorted((root / 'meta' / 'episodes').rglob('*.parquet')):
        table = pq.read_table(path)
        ids = [episode_ids[i] for i in table['episode_index'].to_pylist()]
        pq.write_table(table.append_column(droid.EPISODE_ID, pa.array(ids, pa.string())), path)


@cfn.config(heldout=None, num_workers=8)
def convert(dataset: Dataset, output_dir: str, heldout: str | None, num_workers: int):
    """Write the train and held-out trees. `heldout` is a JSON list of the uids of the held-out episodes."""
    _raise_fd_limit()
    held = set(json.loads(Path(heldout).read_text())) if heldout else set()
    output = Path(pos3.sync(output_dir, interval=None, sync_on_error=False))
    targets: dict[str, LeRobotDataset] = {}
    lengths: dict[str, dict[str, int]] = {tree: {} for tree in TREES}
    report: list[TrimRow] = []
    loader = torch.utils.data.DataLoader(
        DroidEpisodes(dataset), batch_size=1, num_workers=num_workers, collate_fn=_collate_fn,
        prefetch_factor=1 if num_workers else None,
    )  # fmt: skip
    for item in tqdm.tqdm(loader, desc='Episodes'):
        tree = 'heldout' if item.uid in held else 'train'
        cut_s = None if item.cut is None else round(item.cut / droid.FPS, 3)
        report.append(TrimRow(item.uid, tree if item.kept else '', item.frames, item.cut, cut_s))
        if item.kept is None:
            logging.info(f'Left out {item.uid}: the end effector never gets {droid.MIN_START_DROP_M} m below its start')
            continue
        if tree not in targets:
            root = output / tree / droid.DATASET_DIR / droid.SUCCESS_DIR
            targets[tree] = LeRobotDataset.create(repo_id='local', fps=droid.FPS, root=root, features=FEATURES)
        _add_episode(targets[tree], item.kept)
        lengths[tree][item.uid] = len(item.kept.rows[droid.JOINT_POSITIONS])
    for tree, target in targets.items():
        target.finalize()
        root = output / tree / droid.DATASET_DIR
        _add_episode_ids(root / droid.SUCCESS_DIR, list(lengths[tree]))
        ranges = {droid.keep_ranges_key(uid): [[0, n]] for uid, n in lengths[tree].items()}
        (root / droid.KEEP_RANGES_FILE).write_text(json.dumps(ranges, indent=1) + '\n')
    with (output / START_TRIM_FILE).open('w', newline='') as f:
        writer = csv.DictWriter(f, [field.name for field in dataclasses.fields(TrimRow)])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in report)


@cfn.config(group='eval.object', share=0.1, seed=0)
def split(dataset: Dataset, output: str, group: str, share: float, seed: int):
    """Hold out whole recording sessions (one UTC day of one `group` value), about `share` of each group."""
    sessions: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for episode in dataset:
        day = datetime.fromtimestamp(episode.meta[META_CREATED_TS_NS] / 1e9, UTC).date().isoformat()
        sessions[episode.static[group]][day].append(episode.meta[META_UID])
    rng = np.random.default_rng(seed)
    held = []
    for value in sorted(sessions):
        days = sorted(sessions[value])
        target = share * sum(len(uids) for uids in sessions[value].values())
        taken = 0
        for day in rng.permutation(days):
            if taken >= target:
                break
            held += sessions[value][day]
            taken += len(sessions[value][day])
    Path(output).write_text(json.dumps(sorted(held), indent=1) + '\n')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({'convert': convert, 'split': split})


if __name__ == '__main__':
    _internal_main()
