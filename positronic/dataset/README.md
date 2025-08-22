# Positronic Dataset library

This is a library for recording, storing, sharing and using robotic datasets. We differentiate between recording and storing the data and using it from with PyTorch when training. The first part is represented by data model while the second one is the view of that model.

## Core concepts
__Signal__ – strictly typed stepwise function of time, represented as a sequence of `(data, ts)` elements, strictily ordered by `ts`.
The value at time `t` is defined as: $f(t) = \text{data}_i$ where $i = \max\{j : \text{ts}_j \leq t\}$. For $t < \text{ts}_0$ the function is not defined.

Three types are currently supported.
  * __scalar__ of any supported type
  * __vector__ – of any length
  * __image__ – 3-channel images (of uint8 dtype)

__Episode__ – collection of Signals recorded together plus static, episode-level metadata. All dynamic signals in an Episode share a common time axis.

__Dataset__ – ordered collection of Episodes with sequence-style access (indexing, slicing, and index arrays by position). Implementations decide storage and discovery; for example, `LocalDataset` stores episodes in a directory on disk.

### We optimize for:
* Fast append during recording (low latency).
* Random access at query time by the timestamp.
* Window slices like "5 seconds before time X".

## Public API
Signal implements `Sequence[(T, int)]` (iterable, indexable, reversible).
We support three kinds of `Signal`s: scalar, vector, and image (video). Also, we support one and only one timestamp type per `Signal`.
```python
class Signal[T]:
    # Returns the number of records in the signal
    def __len__(self) -> int:
        pass

    # Access data by index or slice
    # signal[idx] returns (value, timestamp_ns) tuple
    # signal[start:end] returns a `Signal` view of the slice
    # signal[[i1, i2, ...]] returns a `Signal` view with the selected indices
    # (boolean masks are not supported)
    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> Tuple[T, int] | Signal[T]:
        pass

    # Timestamp-based indexer property for accessing data by time:
    # * signal.time[ts_ns] returns (value, timestamp_ns) for closest record at or before ts_ns.
    # * signal.time[start_ts:end_ts] returns `Signal` view for time window [start_ts, end_ts).
    # * signal.time[start:end:step] returns a `Signal` sampled at requested timestamps:
    #     t_i = start + i * step (for i >= 0 while t_i < end).
    #     Each returned element is (value_at_or_before_t_i, t_i). If any requested timestamp
    #     precedes the first record, a KeyError is raised. step must be positive.
    # * signal.time[[t1, t2, ...]] returns a `Signal` sampled at the provided timestamps.
    #     Each element is (value_at_or_before_t_i, t_i). Raises KeyError if any t_i precedes
    #     the first record.
    @property
    def time(self):
        pass

class SignalWriter[T]:
    # Appends data with timestamp. Fails if ts_ns is not increasing or data shape/dtype doesn't match
    def append(self, data: T, ts_ns: int) -> None:
        pass

    # Writers are context managers. Exiting the context finalizes the file.
    def __enter__(self) -> "SignalWriter[T]":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    # Abort writing and remove any partial outputs; subsequent calls raise
    def abort(self) -> None:
        pass

class Episode:
    # Names of all items (dynamic signals + static items)
    @property
    def keys(self):
        pass

    # Access by name: returns a `Signal` for dynamic items or the value for static items
    def __getitem__(self, name: str) -> Signal[Any] | Any:
        pass

    # Read-only system metadata (not included in keys, not accessible via __getitem__)
    @property
    def meta(self) -> dict:
        pass

    # Latest start and end timestamps across all dynamic signals
    @property
    def start_ts(self) -> int:
        pass

    @property
    def last_ts(self) -> int:
        pass

    # Episode-wide time accessor:
    # * ep.time[ts] -> dict merging static items with sampled values from each `Signal` at-or-before ts
    # * ep.time[start:end] -> EpisodeView windowed to [start, end)
    # * ep.time[start:end:step] -> EpisodeView sampled at t_i = start + i*step (end-exclusive)
    # * ep.time[[t1, t2, ...]] -> EpisodeView sampled at provided timestamps
    @property
    def time(self):
        pass

class EpisodeWriter:
    # Append dynamic `Signal` data; timestamps must be strictly increasing per signal
    # Raises if the `Signal` name conflicts with existing static items
    def append(self, signal_name: str, data: T, ts_ns: int) -> None:
        pass

    # Set static (non-time-varying) item; raises on name conflicts
    def set_static(self, name: str, data: Any) -> None:
        pass

    # Writers are context managers. Exiting the context finalizes the episode.
    def __enter__(self) -> "EpisodeWriter[T]":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    # Abort the episode; underlying writers are aborted and the `Episode` directory is removed
    def abort(self) -> None:
        pass

class Dataset:
    # Ordered collection of Episodes with sequence-style access
    def __len__(self) -> int:
        pass

    # Indexing returns an Episode; slices and index arrays return lists of Episodes
    def __getitem__(self, index_or_slice: int | slice | Sequence[int] | np.ndarray) -> `Episode` | list[Episode]:
        pass

class DatasetWriter:
    # Allocate a new `Episode` and return an EpisodeWriter (context-managed)
    def new_episode(self) -> EpisodeWriter:
        pass
```

## Signal implementations

`Signal` and `SignalWriter` are abstract interfaces.

We provide implementations for scalar and vector `Signal`s (`SimpleSignal`/`SimpleSignalWriter`) and for image `Signal`s (`VideoSignal`/`VideoSignalWriter`).

### Scalar / Vector

Every sequence is stored in one parquet file, with `timestamp` and `value` columns. The timestamp-based access via the `time` property relies on binary search and hence has `O(log N)` compute time.

All the classes are lazy, in the sense that they don't perform any IO or computations until requested. The `SimpleSignal` keeps all the data in numpy arrays in memory after loading from the parquet file. Once the data is loaded into memory, we provide efficient access through views.

#### Access semantics

When accessing data via slices (either index-based like `signal[0:100]` or time-based like `signal.time[start_ts:end_ts]`), the library returns `Signal` views that share the underlying data with the original `Signal`. These views have the same API as the original `Signal` and provide zero-copy access to the data.

For stepped time slicing `signal.time[start:end:step]`, the returned `Signal` contains samples located at the requested timestamps t_i = start + i * step (end-exclusive), as described above in the public API section. NOTE: stepped time slicing creates a copy of data, so it is not a "free" operation.

Both index and time indexing supports arrays access:
- Index-based arrays: `signal[[i1, i2, ...]]` or `signal[np.array([i1, i2, ...])]` returns a `Signal` view containing the records at the specified indices. Boolean masks are not supported.
- Time-based arrays: `signal.time[[t1, t2, ...]]` returns a `Signal` sampled at the provided timestamps. Each element is `(value_at_or_before_t_i, t_i)`. A KeyError is raised if any requested timestamp precedes the first record. Implementation will allocate new arrays internally and therefore is not a zero-copy operation.

### Video
When implementing `VideoSignal` we are balancing the following trade-offs:

* The disk size – the less the better,
* The performance of random access to any given frame – must be a constant time,
* Memory footprint – must also be constant for video data (timestamps are loaded into the memory)
* User should have control over the size / performance trade off.

We store image streams as a **separate video file** (e.g., MP4/MKV with H.264/H.265) and keep a **single Parquet index** (`frames.parquet`) for timestamp mapping. All timestamps are `timestamp('ns')`. Files are append-only.

Currently we use only one video file per `VideoSignal`, though in the future we might support multiple files.

#### Schema

```text
frames.parquet
  ts_ns    : int64   # presentation timestamp of the frame in nanoseconds
```

* Frames are strictly sorted by `ts_ns` and must be strictly increasing.
* Frame numbers are implicit - they are simply the row indices (0, 1, 2, ...).
* We rely on the modern video container's internal frame index for seeking.

#### Access semantics

Access interface is the same as for `SimpleSignals`, i.e. both index and time interfaces support plain integers, slices and arrays.
All access patterns return views over the underlying index data (zero-copy for indices/timestamps; note that timestamps for sampled views are materialized). Frames are decoded on demand when accessed and are not duplicated across views unless decoded again.

Returned frame type is **decoded uint8 image (H×W×3)**. Decoding is on-demand; memory usage stays O(1) with respect to the number of frames (timestamps are kept in memory). Grayscale (HxWx1) images are not supported yet.

#### Recording
`VideoSignalWriter` takes the path to the video file, frame index file, and encoding settings (codec, GOP size, fps).

* Frame dimensions (width, height) are automatically inferred from the first frame.
* Writer encodes frames to video file using the specified codec (default: H.264).
* For every input frame, the timestamp is appended to the `frames.parquet` index.
* The frame number in the video corresponds to the index position in the timestamp array.

## Episodes

An `Episode` is a collection of `Signal`s recorded together plus static, episode-level metadata. All dynamic signals in an `Episode` share a common time axis.

### Recording

Episodes are recorded via an `EpisodeWriter` implementations. You add time-varying data by calling `append(signal_name, data, ts_ns)` where timestamps are strictly increasing per `Signal` name; you add episode-level metadata via `set_static(name, data)`. All static items are stored together in a single `static.json`, while each dynamic `Signal` is stored in its own format, defined by the particular `SignalWriter` implementation. (e.g., Parquet for scalar/vector; video file plus frame index for image signals).

Name collisions are disallowed: attempting to `append` to a name that already exists as a static item raises an error, and vice versa.

Use as a context manager: exiting the `with` block finalizes all underlying `Signal` writers and persists metadata.
Aborting: `abort()` stops recording, asks each underlying `Signal` writer to abort, and removes the `Episode` directory. After abort, all writer operations (`append`, `set_static`) raise.

### System Metadata (meta)

- Purpose: store system-generated, immutable information separate from user static items.
- Storage: sidecar JSON file `meta.json` inside the `Episode` directory.
- Accessor: `Episode.meta` (read-only dict). Not included in `Episode.keys` and not accessible via `__getitem__`.
- Written: immediately on `EpisodeWriter` creation (side-effect of constructing the writer).
- Contents (concise schema):
  - `schema_version: int` – manifest version (starts at 1).
  - `created_ts_ns: int` – `Episode` creation time in nanoseconds.
  - `writer: object` – environment and provenance:
    - `name: str` – fully-qualified writer class (e.g., `positronic.dataset.episode.DiskEpisodeWriter`).
    - `version: str|null` – package version if available.
    - `python: str` – interpreter version.
    - `platform: str` – platform string.
    - `git: {commit, branch, dirty}` – present if a Git repo is detected.

Signal schemas (dtype, shape, etc.) are not duplicated here; they reside in the `Signal` files themselves (e.g., Parquet/Arrow metadata or frame index files).

### Time accessor

Episode supports time-based access across all signals while preserving static items:

- `ep.time[ts] -> dict`
  - Snapshot: merges all static items with sampled values from each dynamic `Signal` at-or-before `ts`.

- `ep.time[start:end] -> EpisodeView`
  - Window: each dynamic `Signal` is restricted to `[start, end)`. Static items are preserved.

- `ep.time[start:end:step] -> EpisodeView`
  - Sampling: each dynamic `Signal` is sampled at `t_i = start + i*step` (end-exclusive). Static items are preserved.

- `ep.time[[t1, t2, ...]] -> EpisodeView`
  - Arbitrary timestamps: each dynamic `Signal` is sampled at the provided times. Static items are preserved.

Access semantics mirror what is already defined for `Signal.time`: sampling returns values at-or-before requested timestamps; windowing returns views that share underlying storage; stepped sampling materializes requested timestamps. There is no index-based access for episodes — access is time-based only via `ep.time`.

## Datasets

`Dataset` organizes many `Episode`s and provide simple sequence-style access. Implementations decide how episodes are stored and discovered (e.g., filesystem), but must expose a consistent order and length.
- Access: `ds[i] -> Episode`; `ds[start:stop:step] -> list[Episode]`; `ds[[i1, i2, ...]] -> list[Episode]`. Boolean masks are not supported.

### Local dataset

`LocalDataset` is a filesystem-backed implementation that stores episodes in a directory.

### Writing datasets

`DatasetWriter` is a factory for `EpisodeWriter` instances. Implementations allocate a new `Episode` slot and return an `EpisodeWriter` for recording:

```python
with dataset_writer.new_episode() as ew:
    ew.set_static("task", "pick_place")
    ew.set_static("id", 123)
    ew.append("state", np.array([...]), ts_ns)
```
