from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    """Log numeric samples at *base_path*."""
    arr = flatten_numeric(value)
    if arr is None or arr.size == 0:
        return
    rr.log(base_path, rr.Scalars(arr))


def log_series_styles(base_path: str, names: Sequence[str], *, static: bool = True) -> None:
    """Log per-channel series styling information at *base_path*."""
    if not names:
        return
    rr.log(base_path, rr.SeriesLines(names=list(names)), static=static)


def _to_nanos(timestamp: Any) -> int | None:
    """Best-effort conversion of *timestamp* to integer nanoseconds."""
    if isinstance(timestamp, np.datetime64):
        return int(timestamp.astype('datetime64[ns]').astype('int64'))

    if isinstance(timestamp, np.integer | int | np.floating | float):
        return int(timestamp)

    try:
        return int(np.datetime64(timestamp, 'ns').astype('int64'))
    except Exception:
        return None


def set_timeline_time(timeline: str, timestamp: Any) -> None:
    """Set *timeline* to *timestamp*."""
    ts_ns = _to_nanos(timestamp)
    if ts_ns is None:
        return
    rr.time.set_time(timeline, timestamp=np.datetime64(ts_ns, 'ns'))


def set_timeline_sequence(timeline: str, value: int) -> None:
    """Set *timeline* to an integer sequence value."""
    rr.time.set_time(timeline, sequence=value)
