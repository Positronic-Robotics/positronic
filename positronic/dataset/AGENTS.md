# Dataset Library — Design Principles

## Episode data model

An Episode has three kinds of data with distinct roles:

- **Signals** and **static** are episode *content*. They appear in `episode.keys()`, are accessed via `episode[name]`, and transforms can add, remove, or modify them. Signals are time-series; static values are constants.

- **Meta** (`episode.meta`) is *about* the episode — recording facts like `created_ts_ns`, `schema_version`, `writer`. Meta is not part of episode content, not in `keys()`, and transforms pass it through unchanged. Meta keys are optional and may vary by implementation (e.g. `size_mb` exists for disk episodes, may not for others).

## Identity

Every episode is stamped with `meta['uid']` (a uuid4 hex) at recording time — the identity contract. Position in a `Dataset` is *access*, not identity: `FilterDataset`/`ConcatDataset` renumber episodes freely. The uid is *reference* — stable across views, processes, copies, and exports. Because transforms pass meta through unchanged, a transformed episode keeps its recording's uid: it is a view of the same recording event.

## Edits

Recordings are immutable. All post-hoc modification goes through one mechanism: an append-only edit log (`edits.jsonl` in the dataset directory) of uid-keyed declarative records, applied as a view on read. `LocalDataset` applies a discovered log automatically, so every consumer sees the edited view.

- One JSON record per line: `{"op": "set_static", "v": 1, "ep": "<uid>", "data": {...}}`. Records apply in log order; the last write per key wins. Each record carries its op and version, so a log stays replayable forever.
- The layering order is **backend → edits → transforms → consumer**: edits bind to recorded keys and persist; transforms compute on the edited episode and never persist.
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
