# Dataset Library — Design Principles

## Episode data model

An Episode has three kinds of data with distinct roles:

- **Signals** and **static** are episode *content*. They appear in `episode.keys()`, are accessed via `episode[name]`, and transforms can add, remove, or modify them. Signals are time-series; static values are constants.

- **Meta** (`episode.meta`) is *about* the episode — recording facts like `created_ts_ns`, `schema_version`, `writer`. Meta is not part of episode content, not in `keys()`, and transforms pass it through unchanged. Meta keys are optional and may vary by implementation (e.g. `size_mb` exists for disk episodes, may not for others).

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
