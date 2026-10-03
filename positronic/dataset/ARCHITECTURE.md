# Dataset Library — Design Principles

Repository-wide goals and principles are in [ARCHITECTURE.md](../../ARCHITECTURE.md); the user-facing guide is [README.md](README.md). Only the recording is irreplaceable, and everything else stays re-derivable from it.

## One API, many backends

Raw recordings and the views over them, through a single interface: `Signal`/`Episode`/`Dataset` and the layers composed over them. Storage formats are backends behind that interface (`LocalDataset` is the native one, `RemoteDataset` serves it over HTTP, and foreign formats plug in as read adapters). A foreign format is wrapped into the interface, never the reverse. Never push a capability into a storage format when it can live in a layer above it.

## Layering: backend → edits → transforms → consumer

Every dataset read composes in this order:

- **Backend** reads immutable recordings (`LocalDataset`, `RemoteDataset`).
- **Edits** (`edits.py`) persist post-hoc facts as a declarative log applied as a view. Edits bind to recorded keys and never compute.
- **Transforms** compute lazy views over the curated episode. Transforms never persist.
- **Consumers** (codecs, viewers, converters) see one `Dataset` interface and don't know which layers are present.

The shape is that of Lightroom catalogs over raw photos and Delta Lake logs over parquet: identity-keyed (uid, never path or position), time-addressed (absolute ns timestamps, never indices), append-only, dumb plain data with versioned records so a log replays forever.

## Episode data model

An Episode has three kinds of data with distinct roles:

- **Signals** and **static** are episode *content*. They appear in `episode.keys()`, are accessed via `episode[name]`, and transforms can add, remove, or modify them. Signals are time-series; static values are constants.

- **Meta** (`episode.meta`) is *about* the episode — recording facts like `created_ts_ns`, `schema_version`, `writer`. Meta is not part of episode content, not in `keys()`, and transforms pass it through unchanged. Meta keys are optional and may vary by implementation (e.g. `size_mb` exists for disk episodes, may not for others).

`Time` holds immutable integer coordinates on a nonempty set of named timelines.
Ordering and arithmetic match coordinates by name and require identical name sets.
Writers require this value on each append. Each signal fixes its timeline set on its first record:
all records contain every coordinate, never decrease any, and strictly increase at least one.
There is no main or default timeline. Signals in an episode may have overlapping or disjoint sets.

Queries name their timelines explicitly. Point lookup selects the last record satisfying all named
upper bounds and returns its complete original coordinates. Batch sampling replaces queried
coordinates with the requested values and retains other coordinates from the same selected record.
Episode queries include only signals containing every requested timeline. Joins retain an explicitly
selected common subset and reject incompatible ordering.

Native Parquet signal files declare `positronic.signal_version = 2` and store each coordinate in a
non-null `int64` column named `ts.<literal name>`. Video frame indexes use the same layout.
Files without a marker decode their legacy timestamp column on its stored timeline name, or
`recorded` when unnamed. Legacy auxiliary columns remain unexposed; reads require no conversion.
Migration preserves every coordinate exposed by the source API.

## Identity

Every episode is stamped with `meta['uid']` (a uuid4 hex) at recording time — the identity contract. Episodes lacking a stamped uid derive a stable `ts-<created_ts_ns>` one from their recording timestamp, which is equally immutable and travels with the episode. Position in a `Dataset` is *access*, not identity: `FilterDataset`/`ConcatDataset` renumber episodes freely. The uid is *reference* — stable across views, processes, copies, and exports. Because transforms pass meta through unchanged, a transformed episode keeps its recording's uid: it is a view of the same recording event.

## Edits

Recordings are immutable. All post-hoc modification goes through one mechanism: an append-only edit log (`edits.jsonl` in the dataset directory) of uid-keyed declarative records, applied as a view on read. `EditedDataset(base, edits_dir)` is both that view and the handle that amends it: curated reads (drops hidden, static edits overlaid) plus `set_static`/`drop`/`undrop` methods that append a record and return a fresh view over the same recordings — so a held reference never changes shape underneath a consumer. `load_dataset`/`load_all_datasets` compose it over a `LocalDataset`, while `LocalDataset` itself reads raw recordings. The static overlay primitive (`EditedEpisode`) is backend-agnostic; the edit layer reads a local `edits_dir`, the seam to reopen when a second edit-storage format appears.

- One JSON record per line, each carrying its op and version so a log stays replayable forever. `{"op": "set_static", "v": 1, "ep": "<uid>", "data": {...}}` merges static items over the recorded ones (log order, last write per key wins); `{"op": "drop", "v": 1, "ep": "<uid>"}` removes the episode from the loaded view while the recording stays on disk, and `{"op": "undrop", ...}` restores it — the last drop/undrop per episode wins.
- The format stays dumb plain data — smarts live in the library — so external editors can write it. The dataset directory assumes a single writer; readers fail loudly on corrupt or unrecognized records.

## Episode bounds

`episode.bounds(names)` derives named endpoints from signals containing all selected timelines.
It takes the coordinatewise maximum of signal starts and ends. Empty eligible signals or no
eligible signals raise `ValueError`. Subtraction yields a named span with units owned by consumers.
Bounds and spans are not episode metadata; transformed views derive them from their signals.

## Laziness

Nothing expensive happens until needed:
- Listing episodes should not touch signal data
- Accessing named bounds should not load signal values
- Accessing one signal should not load other signals

`SimpleSignal` reads Parquet row-group statistics (file footer) for named bounds and length without
loading values. Timestamp columns load independently when indexed or searched. HTTP discovery
and bounds also defer value metadata and payload reads. Public timestamp collections use
`Sequence[Time]`; backends may share immutable numeric storage and timeline names between rows.

Laziness is what keeps the layering honest: if reading through the abstraction were expensive, a consumer would reach around it for the backend.

## Transforms

`TransformedEpisode` wraps any `Episode` — it must not assume the underlying type or bypass the standard Episode interface. Correctness comes from the abstraction; performance comes from caching at the signal and dataset levels.
