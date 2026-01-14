"""Migrate dataset from remote server to local or S3 storage.

NOTE: This script writes directly to the episode directory, bypassing the normal EpisodeWriter.
It is tied to the current storage format of the library. If the storage format changes, this
script must be updated accordingly.

Example:
    python -m positronic.dataset.utilities.migrate_remote \\
        --source_url=http://localhost:8080 --dest_path=/path/to/output

    python -m positronic.dataset.utilities.migrate_remote \\
        --source_url=http://localhost:8080 --dest_path=s3://bucket/path

TODO: Fix episode creation timestamp preservation (for future PR)
======================================================================

PROBLEM:
    The `created_ts_ns` field in episode metadata gets overwritten during migration with the
    migration timestamp instead of preserving the original data collection timestamp. This
    causes the server visualization (cfg/server.py) to show migration time instead of the
    actual episode creation time.

    Example:
    - Original (s3://raw/droid/towels/...): created_ts_ns = 1760091240100852093 (Oct 10, 2025)
    - Migrated (s3://positronic-public/...): created_ts_ns = 1768326957281971891 (Jan 13, 2026)

ROOT CAUSE:
    1. This migration script only copies static data (line 43-44), not metadata
    2. LocalDatasetWriter.new_episode() creates DiskEpisodeWriter which ALWAYS sets
       created_ts_ns to time.time_ns() (local_dataset.py:99)
    3. There's no API to preserve original metadata during episode creation

IMPACT:
    - Server UI shows incorrect "started" time for all migrated episodes
    - Historical analysis of data collection sessions is inaccurate
    - Episode provenance is lost

SOLUTION (Multi-step fix):

    Step 1: Fix data collection at the source (data_collection.py)
    ----------------------------------------------------------------
    - Modify data_collection.py:216 (static_getter function) to include episode start time
    - Change from: lambda: {'task': task}
    - Change to: lambda: {'task': task, 'started_ts_ns': time.time_ns()}
    - This ensures all NEW episodes have creation time in static data

    Step 2: Update migration script (this file)
    --------------------------------------------
    - Add preservation of original created_ts_ns from episode.meta to static data:

        for key, value in episode.static.items():
            ew.set_static(key, value)

        # Preserve original creation timestamp in static data
        if 'started_ts_ns' not in episode.static and 'created_ts_ns' in episode.meta:
            ew.set_static('started_ts_ns', episode.meta['created_ts_ns'])

    Step 3: Update internal dataset configs (cfg/ds/internal.py)
    ------------------------------------------------------------
    - Add a derive transform to ensure 'started_ts_ns' exists for all episodes
    - For episodes from old migrations (only have meta.created_ts_ns), derive from meta
    - For episodes from new data collection (have static.started_ts_ns), use that
    - Example:

        def _ensure_started_ts(ep: Episode) -> int:
            # Prefer static (new format) over meta (migration artifact)
            return ep.static.get('started_ts_ns', ep.meta.get('created_ts_ns'))

        droid_ds = TransformedDataset(
            base_dataset,
            Group(Derive(started_ts_ns=_ensure_started_ts), Identity())
        )

    Step 4: Update server config (cfg/server.py)
    ---------------------------------------------
    - Change from: datetime.fromtimestamp(ep.meta['created_ts_ns'] / 1e9)
    - Change to: datetime.fromtimestamp(ep['started_ts_ns'] / 1e9)
    - This uses the derived field which has correct timestamp regardless of source

    Step 5: Re-run all migrations
    ------------------------------
    - Re-migrate all 3 public datasets (phail, sim-stack-cubes, sim-pick-place)
    - Upload to s3://positronic-public/datasets/ to replace current data
    - Verify that migrated episodes have correct started_ts_ns in static.json

    Step 6: Update migration documentation
    ---------------------------------------
    - Update cfg/ds/phail.py docstrings to note that started_ts_ns is preserved
    - Document that new datasets should include started_ts_ns in static at collection time

VERIFICATION:
    After implementing above steps, verify:
    1. New data collection includes started_ts_ns in static.json
    2. Migration preserves started_ts_ns in static.json
    3. Internal configs derive started_ts_ns correctly (static > meta fallback)
    4. Server displays correct episode creation times
    5. Old episodes (pre-fix) still work via meta fallback

FILES TO MODIFY:
    - positronic/data_collection.py (Step 1)
    - positronic/dataset/utilities/migrate_remote.py (Step 2)
    - positronic/cfg/ds/internal.py (Step 3)
    - positronic/cfg/server.py (Step 4)
    - Run migration scripts for Step 5
    - Update cfg/ds/phail.py docstrings (Step 6)
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import pos3
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.dataset.remote import RemoteDataset


def migrate_remote_dataset(source_url: str, dest_path: str) -> None:
    """Download a remote dataset to local or S3 storage without quality loss.

    Args:
        source_url: URL of the remote dataset server
        dest_path: Local path or S3 path (s3://bucket/path) for output
    """
    # pos3 handles both local and S3 paths
    resolved_path = pos3.upload(dest_path, sync_on_error=True, interval=None)

    with RemoteDataset(source_url) as remote_ds, LocalDatasetWriter(resolved_path) as writer:
        for episode in tqdm.tqdm(remote_ds, total=len(remote_ds), desc='Migrating'):
            with writer.new_episode() as ew:
                for key, value in episode.static.items():
                    ew.set_static(key, value)

                for key, signal in episode.signals.items():
                    if signal.encoding_format is not None:
                        _write_encoded_signal(signal, ew.path, key)
                    else:
                        # Batch fetch in chunks to avoid memory pressure
                        chunk_size = 10_000
                        for i in range(0, len(signal), chunk_size):
                            end = min(i + chunk_size, len(signal))
                            indices = list(range(i, end))
                            values = signal._values_at(indices)
                            timestamps = signal._ts_at(indices)
                            for value, ts in zip(values, timestamps, strict=True):
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
        with pos3.mirror():
            migrate_remote_dataset(source_url, dest_path)

    cfn.cli(main)
