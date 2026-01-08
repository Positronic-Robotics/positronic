"""Migrate dataset from remote server to local storage.

NOTE: This script writes directly to the episode directory, bypassing the normal EpisodeWriter.
It is tied to the current storage format of the library. If the storage format changes, this
script must be updated accordingly.

Example:
    python -m positronic.dataset.utilities.migrate_remote \\
        --source_url=http://localhost:8080 --dest_path=/path/to/output
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.dataset.remote import RemoteDataset


def migrate_remote_dataset(source_url: str, dest_path: Path) -> None:
    """Download a remote dataset to local storage without quality loss."""
    with RemoteDataset(source_url) as remote_ds, LocalDatasetWriter(dest_path) as writer:
        for episode in tqdm.tqdm(remote_ds, total=len(remote_ds), desc='Migrating'):
            with writer.new_episode() as ew:
                for key, value in episode.static.items():
                    ew.set_static(key, value)

                for key, signal in episode.signals.items():
                    if signal.encoding_format is not None:
                        _write_encoded_signal(signal, ew.path, key)
                    else:
                        for value, ts in signal:
                            ew.append(key, value, ts)


def _write_encoded_signal(signal, episode_path: Path, signal_name: str) -> None:
    fmt = signal.encoding_format
    if fmt == 'positronic.video.v1':
        _write_video_v1(signal.iter_encoded_chunks(), episode_path, signal_name)
    else:
        raise ValueError(f'Unknown encoding format: {fmt}')


def _write_video_v1(chunks: Iterator[bytes], episode_path: Path, signal_name: str) -> None:
    """Parse positronic.video.v1 format and write files."""
    buffer = b''
    chunks_iter = iter(chunks)

    def read_bytes(n: int) -> bytes:
        nonlocal buffer
        while len(buffer) < n:
            buffer += next(chunks_iter)
        result, buffer = buffer[:n], buffer[n:]
        return result

    # Read and write video
    video_size = struct.unpack('<Q', read_bytes(8))[0]
    with open(episode_path / f'{signal_name}.mp4', 'wb') as f:
        remaining = video_size
        while remaining > 0:
            chunk = read_bytes(min(remaining, 64 * 1024))
            f.write(chunk)
            remaining -= len(chunk)

    # Read Arrow IPC, write as parquet
    arrow_size = struct.unpack('<Q', read_bytes(8))[0]
    frames_table = pa.ipc.open_stream(read_bytes(arrow_size)).read_all()
    pq.write_table(frames_table, episode_path / f'{signal_name}.frames.parquet')


if __name__ == '__main__':
    import configuronic as cfn

    @cfn.config()
    def main(source_url: str, dest_path: str):
        migrate_remote_dataset(source_url, Path(dest_path))

    cfn.cli(main)
