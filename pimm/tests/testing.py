"""Testing helpers for pimm users.

Exposes lightweight fakes/mocks to simplify deterministic testing of
control loops and components that depend on `pimm.Clock`.
"""

from ..calls import ControlSystemCaller, ControlSystemHandler
from ..core import Clock
from ..world import World


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


def wire_call(world: World, caller: ControlSystemCaller, handler: ControlSystemHandler) -> None:
    """Bind a caller to a handler in-process, without scheduling either owner.

    ``World.start`` is what otherwise binds a connected pair's transports, so a test that drives a control
    system's generator itself reaches for this instead.
    """
    for emitter, receiver in ((caller.requests, handler.requests), (handler.replies, caller.replies)):
        physical_emitter, physical_receiver = world.local_pipe(maxsize=0)
        emitter._bind(physical_emitter)
        receiver._bind(physical_receiver)
