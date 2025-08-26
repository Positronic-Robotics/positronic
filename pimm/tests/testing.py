"""Testing helpers for pimm users.

Exposes lightweight fakes/mocks to simplify deterministic testing of
control loops and components that depend on `pimm.Clock`.
"""

from ..core import Clock


class MockClock(Clock):
    """Deterministic, manual-advance clock for tests.

    - `now()`/`now_ns()` return the current simulated time.
    - Use `advance(delta_sec)` or `set(time_sec)` to control it.
    """

    def __init__(self, start_time: float = 0.0):
        self._time = float(start_time)

    def now(self) -> float:
        return self._time

    def now_ns(self) -> int:
        return int(self._time * 1e9)

    def advance(self, delta_sec: float) -> float:
        self._time += float(delta_sec)
        return self._time

    def set(self, time_sec: float) -> float:
        self._time = float(time_sec)
        return self._time
