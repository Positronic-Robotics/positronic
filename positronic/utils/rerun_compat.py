from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import rerun as rr


def flatten_numeric(value: Any) -> np.ndarray | None:
    """Convert arbitrary numeric-like inputs into a 1D float64 array."""
    try:
        arr = np.array(value, dtype=np.float64)
    except (TypeError, ValueError):
        try:
            arr = np.array([float(value)], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def log_numeric_series(base_path: str, value: Any) -> None:
    """Log numeric samples at *base_path* while handling rerun API differences."""
    arr = flatten_numeric(value)
    if arr is None or arr.size == 0:
        return

    if hasattr(rr, 'Scalars'):
        rr.log(base_path, rr.Scalars(arr))
        return

    if arr.size == 1:
        rr.log(base_path, rr.Scalar(float(arr[0])))
        return

    for idx, sample in enumerate(arr):
        rr.log(f'{base_path}/{idx}', rr.Scalar(float(sample)))


def log_series_styles(base_path: str, names: Sequence[str], *, static: bool = True) -> None:
    """Log per-channel series styling information at *base_path*."""
    if not names:
        return

    if hasattr(rr, 'SeriesLines'):
        rr.log(base_path, rr.SeriesLines(names=list(names)), static=static)
        return

    if len(names) == 1:
        rr.log(base_path, rr.SeriesLine(name=names[0]), static=static)
        return

    for idx, name in enumerate(names):
        rr.log(f'{base_path}/{idx}', rr.SeriesLine(name=name), static=static)


def _to_nanos(timestamp: Any) -> int | None:
    """Best-effort conversion of *timestamp* to integer nanoseconds."""
    if isinstance(timestamp, np.datetime64):
        return int(timestamp.astype('datetime64[ns]').astype('int64'))

    if isinstance(timestamp, (np.integer, int, np.floating, float)):
        return int(timestamp)

    try:
        return int(np.datetime64(timestamp, 'ns').astype('int64'))
    except Exception:
        return None


def set_timeline_time(timeline: str, timestamp: Any) -> None:
    """Set *timeline* to *timestamp* while handling API differences."""
    ts_ns = _to_nanos(timestamp)
    if ts_ns is None:
        return

    if hasattr(rr.time, 'set_time'):
        rr.time.set_time(timeline, timestamp=np.datetime64(ts_ns, 'ns'))
        return

    if hasattr(rr.time, 'set_time_nanos'):
        rr.time.set_time_nanos(timeline, ts_ns)
        return

    if hasattr(rr, 'set_time_sequence'):
        rr.set_time_sequence(timeline, ts_ns)
        return
