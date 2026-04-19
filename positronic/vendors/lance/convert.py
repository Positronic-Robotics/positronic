"""Convert Positronic datasets to Lance format.

Example:
    docker compose run --rm lance-convert convert \\
      --dataset.dataset=@positronic.cfg.ds.phail.sim_stack_cubes \\
      --dataset.codec=@positronic.vendors.lance.codecs.ee \\
      --output_dir=/data/lance/sim_stack_cubes
"""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import av
import configuronic as cfn
import lance
import numpy as np
import pos3
import pyarrow as pa
import tqdm

from positronic import utils
from positronic.cfg.ds import apply_codec
from positronic.dataset import Dataset
from positronic.dataset.episode import Episode
from positronic.dataset.signal import Kind
from positronic.utils.logging import init_logging


def _write_mp4(path: Path, frames: Iterable[np.ndarray], fps: int) -> dict:
    """Encode frames sequentially to an mp4. Returns {duration, num_frames, width, height}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode='w')
    stream = None
    width = height = None
    count = 0
    try:
        for frame in frames:
            if stream is None:
                height, width = frame.shape[:2]
                stream = container.add_stream('h264', rate=fps)
                stream.width = width
                stream.height = height
                stream.pix_fmt = 'yuv420p'
            video_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(video_frame):
                container.mux(packet)
            count += 1
        if stream is not None:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()
    return {'duration': count / fps if fps else 0.0, 'num_frames': count, 'width': width or 0, 'height': height or 0}


def _column(name: str) -> str:
    """Sanitize episode keys for Lance: dots are interpreted as struct paths."""
    return name.replace('.', '_')


def _video_columns(name: str, uri: str, meta: dict) -> dict:
    base = _column(name)
    return {
        f'{base}_mp4_uri': uri,
        f'{base}_video_uri_duration': float(meta['duration']),
        f'{base}_number_of_frames': int(meta['num_frames']),
        f'{base}_width': int(meta['width']),
        f'{base}_height': int(meta['height']),
    }


def _episode_row(episode: Episode, fps: int, output_dir: Path) -> dict:
    step_ns = int(round(1e9 / fps))
    ts_grid = slice(episode.start_ts, episode.last_ts + 1, step_ns)

    row: dict[str, Any] = {_column(k): v for k, v in episode.static.items()}
    ep_uuid = row.get('uuid')
    if not ep_uuid:
        raise ValueError("Episode is missing 'uuid' — codec must derive it as a static scalar")

    for key, sig in episode.signals.items():
        view = sig.time[ts_grid]
        values = view._values_at(slice(None))
        if sig.kind is Kind.IMAGE:
            rel = Path('videos') / ep_uuid / f'{_column(key)}.mp4'
            video_path = output_dir / rel
            row.update(_video_columns(key, str(rel), _write_mp4(video_path, values, fps)))
        else:
            arr = np.asarray(values)
            row[_column(key)] = arr.tolist() if arr.ndim == 1 else arr.reshape(arr.shape[0], -1).tolist()
    return row


def _table_from_rows(rows: list[dict]) -> pa.Table:
    keys = list({k for row in rows for k in row.keys()})
    return pa.table({k: [row.get(k) for row in rows] for k in keys})


@cfn.config(dataset=apply_codec, fps=None)
def convert(output_dir: str, fps: int | None, dataset: Dataset):
    if fps is None:
        assert 'action_fps' in dataset.meta, "--fps not provided and dataset has no 'action_fps' metadata"
        fps = int(dataset.meta['action_fps'])

    output_dir = pos3.sync(output_dir, interval=None, sync_on_error=False)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    rows = []
    for episode in tqdm.tqdm(dataset, desc='Converting episodes'):
        rows.append(_episode_row(episode, fps=fps, output_dir=out_path))

    table = _table_from_rows(rows)
    lance.write_dataset(table, str(out_path / 'data.lance'), mode='overwrite')
    logging.info(f'Wrote {len(rows)} episodes to {out_path / "data.lance"}')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli({'convert': convert})


if __name__ == '__main__':
    _internal_main()
