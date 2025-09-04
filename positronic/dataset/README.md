# Positronic Dataset library

This is a library for recording, storing, sharing and using robotic datasets. We differentiate between recording and storing the data and using it from with PyTorch when training. The first part is represented by data model while the second one is the view of that model.

## Core concepts
__Signal__ – strictly typed stepwise function of time, represented as a sequence of `(data, ts)` elements with strictly increasing `ts`.
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
Signal implements `Sequence[(T, int)]` (iterable, indexable). We support three kinds of `Signal`s: scalar, vector, and image (video). Timestamps are int nanoseconds.
```python
T = TypeVar('T')  # The type of the data we manage

IndicesLike = Sequence[int] | np.ndarray
RealNumericArrayLike = Sequence[int] | Sequence[float] | np.ndarray

class Signal[T]:
    # Minimal abstract interface (implementations must provide):
    def __len__(self) -> int: ...                # number of records
    def _ts_at(self, indices: IndicesLike) -> IndicesLike: ...         # list-like only
    def _values_at(self, indices: IndicesLike) -> Sequence[T]: ...     # list-like only
    def _search_ts(self, ts_array: RealNumericArrayLike) -> IndicesLike: ...
        # list-like only; floor indices, -1 if before first

    # Index-based access (provided by the library):
    # * signal[i] -> (value, ts) at i (negative indices supported)
    # * signal[a:b:s] -> Signal view over [a:b:s], step>0 required
    # * signal[[i1, i2, ...]] -> Signal view at those integer positions (no boolean masks)

    # Time-based access (provided by the library):
    # * signal.time[ts] -> (value_at_or_before_ts, ts_at_or_before) or KeyError if ts < first
    # * signal.time[start:stop] -> Signal view for [start, stop). Empty signal -> empty view.
    #     If start < first: the window intersects to [first, stop). If start is between two
    #     records and >= first, a carried-back sample is injected at exactly `start`.
    # * signal.time[start:stop:step] -> sampled at t_i = start + i*step (end-exclusive). step>0
    #     and start required; start < first -> KeyError. Timestamps in the result are the
    #     requested ones.
    # * signal.time[[t1, t2, ...]] -> sampled at provided timestamps. Empty arrays are supported
    #     and return an empty Signal; non-numeric dtype raises TypeError (floats accepted);
    #     any t < first -> KeyError.

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
    # * ep.time[start:end] -> NOT SUPPORTED (to ensure equal-length sequences across signals)
    # * ep.time[start:end:step] -> dict with static items and, for each signal, a sequence of signal values sampled at t_i = start + i*step (end‑exclusive). If `end` is omitted, it defaults to the episode's `last_ts` (common stop for all signals) to ensure equal-length sequences. Note that information about the actual timestamps where values originate from is not provided.
    # * ep.time[[t1, t2, ...]] -> dict with static items and, for each signal, a sequence of signal values sampled at provided timestamps.
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
All `Signal` implementations (scalar/vector/video) only implement the minimal interface shown above.
The library provides the full indexing/time behavior so every Signal behaves identically regardless of
the backing store. This keeps implementations small and focused (e.g., Parquet arrays for vectors,
video decoding for images) while ensuring consistent semantics.

We provide implementations for scalar and vector `Signal`s (`SimpleSignal`/`SimpleSignalWriter`) and for image `Signal`s (`VideoSignal`/`VideoSignalWriter`).

### Scalar / Vector

Every sequence is stored in one parquet file, with `timestamp` and `value` columns. The timestamp-based access via the `time` property relies on binary search and hence has `O(log N)` compute time.

All the classes are lazy, in the sense that they don't perform any IO or computations until requested. The `SimpleSignal` keeps all the data in numpy arrays in memory after loading from the parquet file. Once the data is loaded into memory, we provide efficient access through views.

#### Access semantics

- Indexing by position: integers, positive-step slices, and integer arrays. Boolean masks are not supported.
- Time scalar: at-or-before with KeyError if before first.
- Time windows (non-stepped): [start, stop), inject start sample when start is between records (>= first).
  If start < first, the window intersects to [first, stop). Empty signal -> empty view.
- Time stepped windows: [start, stop: step], step > 0 and start required; start < first -> KeyError.
  Samples at requested timestamps; values are carried back.
- Time arrays: empty arrays are allowed (empty result); non-numeric dtype raises TypeError (floats accepted);
  any t < first -> KeyError.

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

Episode supports time-based access across all signals while preserving static items. All episode time queries return a plain dict:

- `ep.time[ts] -> dict`
  - Snapshot: merges all static items with sampled values (no timestamps) from each dynamic `Signal` at-or-before `ts`.

- `ep.time[start:end]`
  - Not supported: use stepped slicing (`start:end:step`) or explicit timestamp arrays to guarantee equal-length sequences across signals.

- `ep.time[start:end:step] -> dict`
  - Sampling: for each dynamic `Signal`, returns a sequence of signal values sampled at `t_i = start + i*step` (end-exclusive). Static items are preserved as-is. `step > 0`, `start` are required. If `end` is omitted, it defaults to the episode's `last_ts` so that all per-signal sequences have the same length regardless of when each signal stops. If `start` is before the episode's `start_ts` (max of signal starts), a `KeyError` is raised.

- `ep.time[[t1, t2, ...]] -> dict`
  - Arbitrary timestamps: for each dynamic `Signal`, returns a sequence of signal values sampled at the provided times. Static items are preserved as-is.

Access semantics mirror those of `Signal.time` for selecting timestamps; the episode-level result aggregates per-signal sequences but omits timestamps (values only). There is no index-based access for episodes — access is time-based only via `ep.time`.

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

## DsWriterAgent (streaming recorder)

`DsWriterAgent` is a control-loop component (based on our `pimm` library) that turns live inputs into episode recordings using a flexible serializer pipeline. It listens for episode lifecycle commands (start/stop/abort) and, while an episode is open, appends any updated inputs with timestamps from the provided clock.

Key ideas
- Inputs are declared up front via `signals_spec: dict[name -> Serializer|None]`.
- The agent polls inputs at a configurable rate and appends only on updates.
- A separate `command` channel controls episode lifecycle.

`Serializer` is a pure function that know how to translate the incoming data into a format that `SignalWriter` can accept:
- A serializer receives the latest value for the input and can return:
  - Transformed value: recorded under the same input name.
  - Dict of suffix -> value: expanded and recorded as `name + suffix` for each
    item (use empty suffix `""` to keep the base name as-is).
  - `None`: the sample is dropped (not recorded).
- If the serializer is `None` in `signals_spec`, the value is passed through.

Built‑in serializers (`positronic.dataset.ds_writer_agent.Serializers`)
- `transform_3d(pose: Transform3D) -> np.ndarray`
  - Returns `[tx, ty, tz, qx, qy, qz, qw]` (shape `(7,)`).
- `robot_state(state: roboarm.State) -> dict | None`
  - Drops samples when `status == RobotStatus.RESETTING`.
  - Otherwise expands to `{'.q': q, '.dq': dq, '.ee_pose': transform_3d(ee)}`.
- `robot_command(command) -> dict`
  - `CartesianMove(pose)` -> `{'.pose': transform_3d(pose)}`
  - `JointMove(positions)` -> `{'.joints': positions}`
  - `Reset()` -> `{'.reset': 1}`

Lifecycle
- `START_EPISODE`: allocates a new episode writer and applies provided static
  metadata (`DsWriterCommand(static_data=...)`).
- `STOP_EPISODE`: finalizes the episode (applies static data then closes).
- `ABORT_EPISODE`: aborts and discards the episode directory.

Notes
- Timestamps come from the agent’s clock; they are strictly increasing per signal.
