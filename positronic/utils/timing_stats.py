"""HACK: scratch per-episode wall-time collectors for the pi-timing sizing pass (internal#55). Never for merge.

Collectors accumulate named wall-time samples from anywhere in the eval process; the harness drains them into
each episode's static meta at finalize, so the stats ride the normal dataset sync to S3.
"""

from collections import defaultdict

_SERIES: dict[str, list[float]] = defaultdict(list)


def record(key: str, seconds: float) -> None:
    _SERIES[key].append(seconds)


def drain() -> dict[str, float | int]:
    """Summarize and clear all series, as flat dotted keys for the episode static meta."""
    out: dict[str, float | int] = {}
    for key, vals in _SERIES.items():
        s = sorted(vals)
        n = len(s)
        out[f'timing.{key}.count'] = n
        out[f'timing.{key}.sum_s'] = sum(s)
        out[f'timing.{key}.p50_ms'] = s[n // 2] * 1e3
        out[f'timing.{key}.p95_ms'] = s[min(n - 1, int(n * 0.95))] * 1e3
        out[f'timing.{key}.max_ms'] = s[-1] * 1e3
    _SERIES.clear()
    return out
