# Positronic Dataset library

This is a library for recording, storing, sharing and using robotic datasets. We differentiate between recording and storing the data and using it from with PyTorch when training. The first part is represented by data model while the second one is the view of that model.

## Table of Contents
- [Core concepts](#core-concepts)
  - [We optimize for](#we-optimize-for)
  - [Layers](#layers)
- [Time](#time)
- [Public API](#public-api)
  - [Signal metadata](#signal-metadata)
  - [Signal implementations](#signal-implementations)
    - [Scalar / Vector](#scalar--vector)
      - [Access semantics](#access-semantics)
    - [Video](#video)
      - [Schema](#schema)
      - [Access semantics](#access-semantics-1)
      - [Recording](#recording)
- [Episodes](#episodes)
  - [Recording](#recording-1)
  - [System Metadata (meta)](#system-metadata-meta)
  - [Time accessor](#time-accessor)
- [Datasets](#datasets)
  - [Local dataset](#local-dataset)
  - [Writing datasets](#writing-datasets)
  - [Editing datasets](#editing-datasets)
- [`DsWriterAgent` (streaming recorder)](#dswriteragent-streaming-recorder)
- [`DsPlayerAgent` (dataset player)](#dsplayeragent-dataset-player)
- [Transforms](#transforms)
  - [Building blocks](#building-blocks)
  - [Derived helpers](#derived-helpers)
  - [Episode and dataset transforms](#episode-and-dataset-transforms)
  - [Example](#example)
- [resized view; original imagery untouched](#resized-view-original-imagery-untouched)

## Core concepts

__Time__ – a moment described by timestamps on one or more named timelines, such as wall time, simulation time, or tick number. A time series can live on several timelines at once.

__Signal__ – strictly typed stepwise function of time, represented as a sequence of `(data, ts)` elements with strictly increasing `ts`.
Think of it as a time series where each value stays in effect until the next timestamp updates it. Formally, $f(t) = \text{data}_i$ where $i = \max\{j : \text{ts}_j \leq t\}$, and nothing is defined before the very first timestamp. Because values propagate forward like this, every `Signal` can answer "what did we know at time _t_?" with a single lookup.

Three types are currently supported.
  * __scalar__ of any supported type
  * __vector__ – of any length
  * __image__ – 3-channel images (of uint8 dtype)

__Episode__ – collection of Signals recorded together plus static, episode-level metadata. Signals may have overlapping or disjoint timelines. A time query selects the signals that contain every requested timeline.

__Dataset__ – ordered collection of Episodes with sequence-style access (indexing, slicing, and index arrays by position). Implementations decide storage and discovery; for example, `LocalDataset` stores episodes in a directory on disk.

### We optimize for:
* High write throughput during recording.
* Random access at query time by the timestamp.
* Window slices like "5 seconds before time X".

### Layers

A dataset read composes up to four layers, all behind the same `Dataset`/`Episode` interface:

1. **Storage backends** read immutable recordings — `LocalDataset` (the native on-disk format), `RemoteDataset` (HTTP). The API is deliberately format-agnostic so further formats can plug in as read adapters.
2. **Edits** persist post-hoc facts (operator verdicts, analysis scores) as a declarative log applied as a view — see [Editing datasets](#editing-datasets).
3. **Transforms** are lazy compute views (derived signals, renames, model codecs) — see [Transforms](#transforms).
4. **Consumers** — training pipelines, viewers, converters — work against the interface and don't know which layers sit underneath.

Recordings are never modified: edits persist but never compute, transforms compute but never persist.

## Time

`Time(world=1000, wall=2000, tick=4)` represents one moment on several named timelines.
Like a simple timestamp, it is immutable and supports comparison, addition, and subtraction.
Operations match coordinates by name, regardless of their order.

Ordering and arithmetic require the same set of timeline names. `a <= b` means every coordinate
in `a` is at most its counterpart in `b`; `<` also requires at least one to be smaller.
If timelines disagree on order, the two values are incomparable.

### Storage and compatibility

Scalar/vector signals use one Parquet file with a `value` column and one non-null `int64`
column per timeline: `ts.world`, `ts.wall`, `ts.tick`, etc. Video signals use the same timestamp
columns in their Parquet frame index. Column order gives no timeline priority. Names are literal:
`ts.server.wall` represents the name `server.wall`.

Native files carry `positronic.signal_version = 2` in Parquet schema metadata. Empty files written
without any appends have no timeline columns. Readers reject unknown format versions.

Files without a version marker are decoded automatically using the legacy layout. Their `timestamp`
(scalar/vector) or `ts_ns` (video) column is exposed on the timeline named by `positronic.timeline`,
or `recorded` when that metadata is absent. Legacy auxiliary columns remain in the file but are not
exposed as timelines. Reading requires no conversion and never rewrites the recording.
Migration and `convert_ds` preserve every timeline exposed by the source API. HTTP access uses
`/api/v2`; client and server must both support this named-timestamp protocol.
Encoded video streams advertise `positronic.video.v2` for named timeline indexes and
`positronic.video.v1` for legacy indexes. Migration accepts both formats without re-encoding video.

## Public API

`Signal` implements `Sequence[tuple[T, Time]]`. Index access returns a value and its full original
coordinates. Index slices and strictly increasing index lists return signal views; reverse slices,
repeated indices, and boolean masks are rejected.

```python
signal.timelines                              # tuple[str, ...]
episode.timelines                             # union of its signals' timeline names
value, original_time = signal[0]
values = signal.values()                     # Sequence[T]
world_times = signal.timestamps("world")     # Sequence[int]
times = signal.timestamps(("world", "tick"))  # Sequence[Time]
bounds = signal.bounds("world")              # TimeBounds[int]
span = bounds.finish - bounds.start          # int
first, last = signal.bounds(("world", "tick")) # TimeBounds[Time], unpackable
named_span = last - first                    # Time
```

`bounds` and `timestamps` accept a single name or a nonempty tuple of unique names.
As with `Time` indexing, a string selects integer coordinates; a tuple selects `Time` values,
even with just one name. Bounds return an immutable `TimeBounds` named tuple with inclusive
`start` and `finish` endpoints. Episode bounds use the same return types.
Unknown signal timelines raise `KeyError`.
Empty signals have no bounds and raise `ValueError`.
Tuple order controls presentation, never query results. Timestamp sequences may be lazy; their
numeric storage and materialization are backend details.

### Signal time access

```python
value, original_time = signal.time[Time(world=150)]
sampled = signal.time[[Time(world=100), Time(world=150), Time(world=200)]]
window = signal.time[Time(world=150):Time(world=250)]
sampled = signal.time[Time(world=100):Time(world=250):Time(world=50)]
```

- A point query returns the last record whose coordinates satisfy **all** requested `<=` bounds.
  Unqueried coordinates impose no bound. Repeated coordinates on a selected subset resolve to the
  last record. Unknown names or no qualifying record raise `KeyError`; an empty query is invalid.
- Batch queries contain `Time` values with one name set, strictly ordered componentwise. Empty
  batches produce empty views. Each output uses the requested coordinates on queried timelines
  and the selected source record's coordinates on every other timeline.
- Non-stepped windows carry a sample to `start` when a qualifying record exists and exclude `stop`.
  With both endpoints supplied, the window is empty unless `start < stop` componentwise.
  Without a qualifying carry record, the window begins at the first record satisfying every lower bound.
  An omitted start keeps the first source record. Incompatible ordering after injecting a carried
  timestamp raises `ValueError`.
- Stepped windows require a named start. Start, stop, and step have the same name set. Each step
  coordinate is nonnegative and at least one is positive. Sampling continues while the next `Time`
  is `< stop`; an omitted stop uses the final selected coordinates as an inclusive bound.
- Point queries, slice endpoints, steps, and batch elements require `Time` values.
  Construct them with `Time(**coordinates)` when timeline names are dynamic.

For source coordinates `(world, wall) = (100, 900), (140, 950), (220, 1100)`, sampling at world
`100, 150, 200, 250` yields `(100, 900), (150, 950), (200, 950), (250, 1100)`.
A point query at world `150` returns the original coordinates `(140, 950)`.

### Backend interface

Backends implement these operations without requiring the public API to materialize timestamps:

```python
class Signal[T]:
    @property
    def timelines(self) -> tuple[str, ...]: ...
    def __len__(self) -> int: ...
    def _ts_at(self, indices: IndicesLike, timelines: tuple[str, ...]) -> Sequence[Time]: ...
    def _values_at(self, indices: IndicesLike) -> Sequence[T]: ...

    # Optional optimized search; the base class supplies binary search.
    def _search_ts(self, queries: Sequence[Time]) -> Sequence[int]: ...

    # Optional optimized bounds; the base class reads the first and last timestamps.
    def _bounds(self, timelines: tuple[str, ...]) -> TimeBounds[Time]: ...
```

`SignalWriter.append(value, timestamps: Time)` and
`EpisodeWriter.append(signal_name, value, timestamps: Time)` enforce the
[timestamp ordering rules](#writing-datasets).
Writers are context managers; exiting finalizes output, and `abort()` removes partial output.
`DatasetWriter.new_episode()` allocates an episode writer.

### Signal metadata

Every `Signal` exposes a `SignalMeta` object that captures element dtype, shape, and a semantic kind (numeric or image). The base implementation infers these fields lazily from the first stored value, keeping implementations lightweight.

## Signal implementations

`Signal` and `SignalWriter` are abstract interfaces.
All `Signal` implementations (scalar/vector/video) only implement the minimal interface shown above.
The library provides the full indexing/time behavior so every Signal behaves identically regardless of
the backing store. This keeps implementations small and focused (e.g., Parquet arrays for vectors,
video decoding for images) while ensuring consistent semantics for the `time` accessor.

We provide implementations for scalar and vector `Signal`s (`SimpleSignal`/`SimpleSignalWriter`) and for image `Signal`s (`VideoSignal`/`VideoSignalWriter`).

### Scalar / Vector

Each signal uses a Parquet file with `value` and named timestamp columns. Queries binary-search each requested coordinate, taking the earliest of the resulting record indices.

All the classes are lazy, in the sense that they don't perform any IO or computations until requested. The `SimpleSignal` keeps all the data in numpy arrays in memory after loading from the parquet file. Once the data is loaded into memory, we provide efficient access through views.

#### Access semantics

All scalar/vector backends implement the named access rules above. Bounds use Parquet footer
statistics where available; querying timestamps does not load values.

### Video
When implementing `VideoSignal` we are balancing the following trade-offs:

* The disk size – the less the better,
* The performance of random access to any given frame – must be a constant time,
* Memory footprint – must also be constant for video data (timestamps are loaded into the memory)
* User should have control over the size / performance trade off.

We store image streams as a **separate video file** (e.g., MP4/MKV with H.264/H.265) and keep a **single Parquet index** (`frames.parquet`) for timestamp mapping. Timestamp coordinates are signed `int64`. Files are append-only.

Each `VideoSignal` has one video file.

#### Schema

```text
frames.parquet
  ts.world : int64 not null
  ts.wall  : int64 not null
```

* Every coordinate is non-decreasing; each frame advances at least one timeline.
* Frame numbers are implicit - they are simply the row indices (0, 1, 2, ...).
* We rely on the modern video container's internal frame index for seeking.

#### Access semantics

Index and named time access follow the same rules as `SimpleSignal`. Frame decoding is deferred
until values are requested; timeline discovery and bounds use the Parquet index.

Returned frame type is **decoded uint8 image (H×W×3)**. Decoding is on-demand; memory usage stays O(1) with respect to the number of frames (timestamps are kept in memory). Grayscale (HxWx1) images are not supported yet.

#### Recording
`VideoSignalWriter` takes the path to the video file, frame index file, and encoding settings (codec, GOP size, fps).

* Frame dimensions (width, height) are automatically inferred from the first frame.
* Writer encodes frames to video file using the specified codec (default: H.264).
* For every input frame, the timestamp is appended to the `frames.parquet` index.
* The frame number in the video corresponds to the index position in the timestamp array.

## Episodes

An `Episode` is a collection of `Signal`s recorded together plus static, episode-level metadata. Each signal fixes its own timeline set; signals may have no timelines in common.

### Recording

Episodes are recorded via `EpisodeWriter` implementations. You add time-varying data by calling `append(signal_name, data, timestamps)` where coordinates never decrease and at least one increases per `Signal` name; you add episode-level metadata via `set_static(name, data)`. All static items are stored together in a single `static.json`, while each dynamic `Signal` is stored in its own format, defined by the particular `SignalWriter` implementation (e.g., Parquet for scalar/vector; video file plus frame index for image signals).

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
  - `uid: str` – episode identity (uuid4 hex), stamped at recording time; episodes lacking one derive a stable `ts-<created_ts_ns>` uid from their recording timestamp. Stable across views, copies, and exports; position in a dataset is access, the uid is reference.
  - `created_ts_ns: int` – `Episode` creation time in nanoseconds.
  - `writer: object` – environment and provenance:
    - `name: str` – fully-qualified writer class (e.g., `positronic.dataset.local_dataset.DiskEpisodeWriter`).
    - `version: str|null` – package version if available.
    - `python: str` – interpreter version.
    - `platform: str` – platform string.
    - `git: {commit, branch, dirty}` – present if a Git repo is detected.

Signal schemas (dtype, shape, etc.) are not duplicated here; they reside in the `Signal` files themselves (e.g., Parquet/Arrow metadata or frame index files).

### Time accessor

Episode queries include only signals containing **all** requested timelines, plus every static item.
An eligible signal without a qualifying sample raises `KeyError`. With no eligible signals, point
and explicit-batch queries return only static items.

```python
first, last = episode.bounds(("world", "tick"))
snapshot = episode.time[Time(world=1000, tick=4)]
sampled = episode.time[[Time(world=1000), Time(world=2000)]]
sampled = episode.time[Time(world=1000):Time(world=3000):Time(world=500)]
```

`bounds(names)` takes the coordinatewise maximum of eligible signals' starts and ends. No eligible
signals, or an eligible signal without bounds, raises `ValueError`. Bounds need not identify a
stored record. Subtract the endpoints to obtain spans; a tick span is a tick count, not nanoseconds.

All queries return a dictionary. Points contain signal values; batches and stepped slices contain
per-signal value sequences, with statics unchanged. An omitted stepped stop uses one episode-wide
inclusive bound to give signals equal-length results. Slices without a step are unsupported.
An empty explicit batch returns empty per-signal sequences plus static items.

## Datasets

`Dataset` organizes many `Episode`s and provide simple sequence-style access. Implementations decide how episodes are stored and discovered (e.g., filesystem), but must expose a consistent order and length.
- Access: `ds[i] -> Episode`; `ds[start:stop:step] -> list[Episode]`; `ds[[i1, i2, ...]] -> list[Episode]`. Boolean masks are not supported.
- Schema discovery: inspect a representative episode (for example `ds[0]['signal'].meta`) or maintain an external manifest if you need to reason about schemas without materializing episodes.

### Local dataset

`LocalDataset` is a filesystem-backed implementation that stores episodes in a directory.

### Writing datasets

`DatasetWriter` is a factory for `EpisodeWriter` instances. Implementations allocate a new `Episode` slot and return an `EpisodeWriter` for recording.

Writers require one `Time` value with all coordinates for each append. The first successful
append fixes the signal's timeline names. Every subsequent record supplies exactly those names,
never decreases a coordinate, and strictly increases at least one. There is no main or default
timeline. Stored coordinates must fit a signed 64-bit integer.

```python
from positronic.dataset import Time

with dataset_writer.new_episode() as episode:
    episode.set_static("task", "pick_place")
    episode.set_static("id", 123)
    episode.append("state", state, Time(world=1000, wall=2000, tick=4))
    episode.append("camera", image, Time(world=1000, wall=2010))
```

### Editing datasets

Recorded episodes are immutable. Post-hoc facts — an operator's verdict, analysis scores — are *edits*: declarative records appended to `edits.jsonl` in the dataset directory and applied as a view on load. The edit layer lives in `positronic.dataset.edits`.

```python
from positronic.dataset.local_dataset import load_dataset

ds = load_dataset(root)  # an EditedDataset: LocalDataset(root) with the edit log applied
ds = ds.set_static(episode.meta['uid'], {'eval.outcome': 'success', 'notes': 'clean run'})
ds = ds.drop(bad_episode.meta['uid'])  # remove from the view; the recording stays on disk
```

`set_static`/`drop`/`undrop` append a record and return a *new* `EditedDataset` over the same recordings, so a held reference keeps its shape while the returned one reflects the edit.

Each line of `edits.jsonl` is one JSON record carrying its op. `{"op": "set_static", "v": 1, "ep": "<uid>", "data": {...}}` merges static items over the episode's recorded ones, in log order with last-write-wins per key; values follow the same restrictions as `EpisodeWriter.set_static`, and a key colliding with a signal name raises when that key is read (identity stays readable, so a colliding episode can still be filtered or dropped). `{"op": "drop", "v": 1, "ep": "<uid>"}` removes the episode from the loaded view; `{"op": "undrop", "v": 1, "ep": "<uid>"}` restores it (the last drop/undrop wins). Records target episodes by `meta['uid']`; corrupt or unrecognized records fail loudly when the log is loaded.

`EditedDataset(base, edits_dir)` reads the log at `edits_dir`, hides dropped episodes, and overlays each remaining episode's static edits; `load_dataset`/`load_all_datasets` compose it over a `LocalDataset`, while `LocalDataset` itself always reads the raw recordings. The log is plain appendable JSON so external tools can write it; the dataset directory has a single writer.

## `DsWriterAgent` (streaming recorder)

`DsWriterAgent` records data collection and replay runs. Inference recording belongs to
[`Harness`](../policy/harness.py), which writes through the same dataset interfaces.
Harness adds `harness.world` and `harness.wall`: first receipt for inputs, emission for commands.
These coordinates align inputs and commands while preserving the original message timestamps.
The agent is a control-loop component (based on our `pimm` library) that turns live inputs into episode recordings using a flexible serializer pipeline. It listens for episode lifecycle commands (start/stop/abort) and, while an episode is open, appends updated inputs with their `pimm.Message.time` timestamps.

Key ideas
- Inputs are registered explicitly through `DsWriterAgent.add_signal(name, serializer=None)`.
- Each `START_EPISODE` names the path its episode records into, and `None` names nowhere. The agent holds no dataset of its own: it opens each named one through the factory it was built with (`DsWriterAgent(LocalDatasetWriter)`), once per name, and closes them all when it stops.
- The agent polls inputs at a configurable rate and appends only on updates.
- Recording is best effort, and this is a deliberate trade rather than an oversight. Each input arrives over a one-slot `pimm` signal where a new value overwrites one still unread, so a recorder that stalls for longer than the gap between two samples loses the older one — commands exactly as much as camera frames or arm state. An episode is what the recorder managed to observe, not a guaranteed-complete log of what happened; treat a missing sample as possible in any analysis that counts them.
- A separate `command` channel controls episode lifecycle.
- Every message coordinate is saved: emission, first delivery, and optional producer timelines.
  Policy joins, viewing, and playback prefer `harness.world`, then `received.world`: wall time
  on hardware and simulation time in simulation. Legacy datasets use `recorded` without conversion or renaming.
  Viewing and playback also accept an explicit timeline.

`Serializer` is a pure function that know how to translate the incoming data into a format that `SignalWriter` can accept:
- A serializer receives the latest value for the input and can return:
  - Transformed value: recorded under the same input name.
  - Dict of suffix -> value: expanded and recorded as `name + suffix` for each
    item (use empty suffix `""` to keep the base name as-is).
  - `None`: the sample is dropped (not recorded).
- Omitting the serializer (or passing `None`) records the value unchanged.

Built‑in serializers (`positronic.dataset.serializers.Serializers`)
- `transform_3d(pose: Transform3D) -> np.ndarray`
  - Returns `[tx, ty, tz, qw, qx, qy, qz]` (shape `(7,)`).
- `robot_state(state: roboarm.State) -> dict | None`
  - Expands to `{'.status': status, '.q': q, '.dq': dq, '.ee_pose': transform_3d(ee)}`; every sample is
    recorded whatever the status.
- `robot_command(command) -> dict`
  - `CartesianPosition(pose)` -> `{'.pose': transform_3d(pose)}`
  - `JointPosition(positions)` -> `{'.joints': positions}`

Lifecycle
- `START_EPISODE`: opens a new episode in the dataset the command names and applies provided static
  metadata (`DsWriterCommand.START(output_path, static_data)`). A command that names no path opens an episode
  that records nowhere, so the `STOP_EPISODE` that ends it is as ordinary as any other.
- `STOP_EPISODE`: finalizes the episode (applies static data then closes).
- `ABORT_EPISODE`: aborts and discards the episode directory.

## `DsPlayerAgent` (dataset player)

`DsPlayerAgent` replays recorded `Episode` objects back into a live `pimm` world by streaming signal values on demand. It mirrors the lifecycle style of `DsWriterAgent`, making it easy to pipe existing datasets through simulators, robots, or other consumers.

Component layout
- Outputs are dynamically declared via `player.outputs[name]` before playback begins; every declared name must map to a dynamic signal in the episode. Static-only items raise `ValueError`, and missing signals raise `KeyError` so wiring mistakes surface immediately.
- `command` receives control messages. `DsPlayerStartCommand(episode, start_ts=None, end_ts=None, timeline=None)`
  starts playback on the selected timeline, optionally restricting the time window.
  `DsPlayerAbortCommand()` stops immediately without emitting `finished`.
- `finished` emits the originating `DsPlayerStartCommand` once all scheduled samples have been streamed.
- `poll_hz` (default `100 Hz`) governs how frequently the agent checks for new work.

Playback semantics
- `timeline` selects the nanosecond clock used for scheduling and command bounds. When omitted,
  playback prefers `harness.world`, then `received.world`, then `recorded` for legacy data.
  Other timelines require an explicit name. Every requested output must expose the selected
  timeline. Window selection uses the dataset's carry-back semantics.
- Playback anchors the first sample to the world clock when `START` is handled and preserves
  inter-sample spacing. Messages carry this scheduled time as `playback.scheduled`; pimm stamps
  their actual emission and first delivery times.

Typical use cases
- Driving robots or simulators from a stored episode while optionally recording the run again. See [`positronic/replay_record.py`](../../positronic/replay_record.py) where `DsPlayerAgent` feeds a Mujoco simulation and simultaneously streams into a `DsWriterAgent` to capture the replay.
- Scrubbing subsets of an episode (via `start_ts`/`end_ts`) for debugging or visualization tools without rewriting the dataset.
- Streaming into analysis pipelines that expect live `pimm` signals; consumers just connect to `player.outputs[...]` as if they were real hardware feeds.

## Transforms

`positronic.dataset.transforms` provides lazy views for deriving new signals and datasets without duplicating storage. Each transform wraps existing `Signal`/`Episode`/`Dataset` objects and only computes when you access the data, so recordings remain immutable.

### Building blocks
- `Elementwise(signal, fn)`: wraps a single signal and maps batches of values through `fn` while keeping the timestamp index untouched. Most other helpers eventually call into this class.
- `Join(*signals, timelines, include_ref_ts=False)`: aligns multiple signals on the required tuple of
  timeline names with carry-back semantics. Each input coordinate is carried forward to at least the
  latest input start on that timeline. The result yields tuples of values (and, optionally, reference
  timestamps) at every combined timestamp. Only selected timelines survive; duplicate projected
  coordinates collapse and incompatible ordering raises `ValueError`. Reference timestamps retain
  every source coordinate.
- `IndexOffsets(signal, *relative_indices, include_ref_ts=False)`: samples neighbouring indices around each position (e.g., `i-1`, `i`, `i+1`) to build finite-difference style windows. Length shrinks when offsets fall out of bounds.
- `TimeOffsets(signal, *offsets, include_ref_ts=False)`: samples values at named `Time` offsets and preserves all base timestamps. Can be used to lookup into "past" or "future".

Transforms operate purely on values; if you need semantic labels, maintain them alongside your data at a higher layer.

### Derived helpers
Common utilities stack the building blocks to cover frequent needs:
- `image.resize(...)` and `image.resize_with_pad(...)`: resize RGB frames per sample using OpenCV or PIL. Import from `positronic.dataset.transforms.image`.
- `concat(*signals, timelines, dtype=None)`: align signals with `Join` on the required tuple of timeline names and concatenate their vector values into one array view.
- `astype(signal, dtype)`: cast vector signals on the fly via `Elementwise`.
- `pairwise(a, b, op, *, timelines)`: join two signals on the required tuple of timeline names and apply a custom binary operator to every aligned pair.
- `recode_rotation(rep_from, rep_to, signal)`: convert rotation representations using `positronic.geom` utilities.
- `view(signal, slice_obj)`: create a zero-copy view that slices each frame (e.g., select quaternion components from a pose vector) while preserving timestamps.

Typical use cases include building model-ready tensors, normalizing values, resizing video streams, or deriving velocities. Because every helper is a view, you can stack them freely and continue to use standard access patterns (`signal.time[...]`, indexing, slicing) without materializing intermediate results.

### Episode and dataset transforms

`EpisodeTransform` transforms one episode into another episode. They follow a simple protocol:

```python
class EpisodeTransform(ABC):
    @abstractmethod
    def __call__(self, episode: Episode) -> Episode:
        """Transform an episode into a new episode."""
        ...

    @property
    def meta(self) -> dict[str, Any]:
        """Optional metadata for this transform."""
        return {}
```

Each transform is responsible for defining which keys are available in the output `Episode`. The library provides several built-in transforms:

- **`Derive(**functions)`**: Create new keys by applying functions to the input episode. Each keyword argument maps an output key to a function `(Episode) -> Signal | Any`. Values are computed lazily on first access per key.
- **`Group(*transforms)`**: Apply multiple transforms in parallel to the same input episode and merge their results lazily. If transforms produce overlapping keys, the first transform takes precedence.
- **`Rename(**mapping)`**: Rename episode keys according to `new_key='old_key'` (output -> input). Only renamed keys are included. For non-identifier keys, pass via dict expansion: `Rename(**{'a.b': 'c.d'})`.
- **`Identity(*keys)`**: Select specific keys to keep, or pass through the entire episode unchanged if no keys are specified.
- **`Eager(transform)`**: Force eager evaluation of a wrapped transform. Use when you want all values computed upfront (e.g., for debugging or when you know all values will be accessed).

Helper callables (used within `Derive`):
- **`Concat(*keys, timelines)`**: Concatenate multiple signals on the required tuple of timeline names into a single array signal.
- **`FromValue(value)`**: Return a constant value (useful for adding static labels).

`TransformedEpisode` applies a sequence of transforms lazily—transforms are chained sequentially where each receives the output of the previous one. Transformation happens on first access and results are cached. `TransformedDataset` lifts the same pattern to the dataset level so every retrieved episode is automatically transformed.

Each `EpisodeTransform` can expose metadata via its `meta` property. `TransformedDataset.meta` merges these dictionaries (along with the wrapped dataset's `meta`) so downstream consumers can inspect accumulated metadata without materializing episodes.

### Example

```python
from positronic.dataset import transforms
from positronic.dataset.signal import RECORDED_TIME
from positronic.dataset.transforms import image
from positronic.dataset.transforms.episode import Derive, Group, Identity
import numpy as np


# Define a custom encoder
class Features(transforms.EpisodeTransform):
    def __call__(self, episode: Episode) -> Episode:
        joint_q = episode["robot.q"]
        ee_pose = episode["robot.ee_pose"]
        features = transforms.concat(joint_q, ee_pose, timelines=(RECORDED_TIME,), dtype=np.float32)
        resized_image = image.resize(width=224, height=224, signal=episode["rgb_camera"])
        return EpisodeContainer(
            {"features": features, "resized_image": resized_image},
            episode.meta
        )


# Or use built-in Derive for simple cases
features_transform = Derive(
    features=lambda ep: transforms.concat(ep["robot.q"], ep["robot.ee_pose"], timelines=(RECORDED_TIME,), dtype=np.float32),
    resized_image=lambda ep: image.resize(width=224, height=224, signal=ep["rgb_camera"])
)

# Apply transform to dataset and select keys to keep
dataset = transforms.TransformedDataset(
    raw_dataset,
    Group(features_transform, Identity('robot.ee_pose'))
)

episode = dataset[0]
# Resized view; original imagery untouched
frame0, _ts = episode['resized_image'].time[episode.bounds((RECORDED_TIME,)).start]
```
