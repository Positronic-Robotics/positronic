# Dataset Library — Design Principles

## One API, many backends

The library covers the data path of the robot-learning loop — record → curate/annotate → convert → train → eval → re-score — through a single interface: `Signal`/`Episode`/`Dataset` and the layers composed over them. Storage formats are backends behind that interface (`LocalDataset` is the native one, `RemoteDataset` serves it over HTTP, and foreign formats plug in as read adapters). Never push a capability into a storage format when it can live in a layer above it.

## Layering: backend → edits → transforms → consumer

Every dataset read composes in this order:

- **Backend** reads immutable recordings (`LocalDataset`, `RemoteDataset`).
- **Edits** (`edits.py`) persist post-hoc facts as a declarative log applied as a view. Edits bind to recorded keys and never compute.
- **Transforms** compute lazy views over the curated episode. Transforms never persist.
- **Consumers** (codecs, viewers, converters) see one `Dataset` interface and don't know which layers are present.

The shape mirrors the systems that got this right — Lightroom catalogs over raw photos, video EDLs, git, Delta Lake logs over parquet: identity-keyed (uid, never path or position), time-addressed (absolute ns timestamps, never indices), append-only, dumb plain data with versioned records so a log replays forever.

## Episode data model

An Episode has three kinds of data with distinct roles:

- **Signals** and **static** are episode *content*. They appear in `episode.keys()`, are accessed via `episode[name]`, and transforms can add, remove, or modify them. Signals are time-series; static values are constants.

- **Meta** (`episode.meta`) is *about* the episode — recording facts like `created_ts_ns`, `schema_version`, `writer`. Meta is not part of episode content, not in `keys()`, and transforms pass it through unchanged. Meta keys are optional and may vary by implementation (e.g. `size_mb` exists for disk episodes, may not for others).

## Identity

Every episode is stamped with `meta['uid']` (a uuid4 hex) at recording time — the identity contract. Episodes lacking a stamped uid derive a stable `ts-<created_ts_ns>` one from their recording timestamp, which is equally immutable and travels with the episode. Position in a `Dataset` is *access*, not identity: `FilterDataset`/`ConcatDataset` renumber episodes freely. The uid is *reference* — stable across views, processes, copies, and exports. Because transforms pass meta through unchanged, a transformed episode keeps its recording's uid: it is a view of the same recording event.

## Edits

Recordings are immutable. All post-hoc modification goes through one mechanism: an append-only edit log (`edits.jsonl` in the dataset directory) of uid-keyed declarative records, applied as a view on read. `EditedDataset(base, edits_dir)` is both that view and the handle that amends it: curated reads (drops hidden, static edits overlaid) plus `set_static`/`drop`/`undrop` methods that append a record and return a fresh view over the same recordings — so a held reference never changes shape underneath a consumer. `load_dataset`/`load_all_datasets` compose it over a `LocalDataset`, while `LocalDataset` itself reads raw recordings. The static overlay primitive (`EditedEpisode`) is backend-agnostic; the edit layer reads a local `edits_dir` for now — when a second edit-storage format appears, `edits_dir` is where the seam reopens.

- One JSON record per line, each carrying its op and version so a log stays replayable forever. `{"op": "set_static", "v": 1, "ep": "<uid>", "data": {...}}` merges static items over the recorded ones (log order, last write per key wins); `{"op": "drop", "v": 1, "ep": "<uid>"}` removes the episode from the loaded view while the recording stays on disk, and `{"op": "undrop", ...}` restores it — the last drop/undrop per episode wins.
- The format stays dumb plain data — smarts live in the library — so external editors can write it. The dataset directory assumes a single writer; readers fail loudly on corrupt or unrecognized records.

## Episode properties

`duration_ns`, `start_ts`, `last_ts` are **first-class properties on Episode**, always derived from signals. They are never stored in meta. If a transform changes signals, these properties reflect the change.

Implementations may cache these values internally (e.g. `DiskEpisode` reads a cached `duration_ns` from `meta.json`), but this is a private optimization — `episode.meta` must not expose `duration_ns`.

## Laziness

Nothing expensive should happen until needed. The library is designed around lazy evaluation:
- Listing episodes should not touch signal data
- Accessing `duration_ns` should not load signal values
- Accessing one signal should not load other signals

`SimpleSignal` reads parquet row-group statistics (file footer) for `start_ts`/`last_ts`/`len` without touching actual data. Full timestamps and values are loaded only when indexed or searched.

## Transforms

`TransformedEpisode` wraps any `Episode` — it must not assume the underlying type or bypass the standard Episode interface. Correctness comes from the abstraction; performance comes from caching at the signal and dataset levels.
