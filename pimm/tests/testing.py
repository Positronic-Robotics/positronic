"""Testing helpers for pimm users.

Exposes lightweight fakes/mocks to simplify deterministic testing of
control loops and components that depend on `pimm.Clock`.
"""

from typing import Generic, TypeVar

from ..calls import Call, ControlSystemCaller, ControlSystemHandler
from ..core import Clock, ControlSystem, Sleep
from ..world import World

Req = TypeVar('Req')
Res = TypeVar('Res')


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


class Passive(ControlSystem):
    """An owner for a caller or handler that no test schedules."""

    def run(self, should_stop, clock):
        while not should_stop.value:
            yield Sleep(0.001)


class FakeCall(Call[Req, Res], Generic[Req, Res]):
    """A call answered by hand, recording the one answer it is allowed."""

    def __init__(self, request: Req):
        self._request = request
        self.answered = False
        self.result: Res | None = None
        self.exception: BaseException | None = None

    @property
    def request(self) -> Req:
        return self._request

    def set_result(self, value: Res) -> None:
        self.answered, self.result = True, value

    def set_exception(self, exc: BaseException) -> None:
        self.answered, self.exception = True, exc


def wire_call(world: World, caller: ControlSystemCaller, handler: ControlSystemHandler) -> None:
    """Bind a caller to a handler in-process, without scheduling either owner.

    ``World.start`` is what otherwise binds a connected pair's transports, so a test that drives a control
    system's generator itself reaches for this instead.
    """
    for emitter, receiver in ((caller.requests, handler.requests), (handler.replies, caller.replies)):
        physical_emitter, physical_receiver = world.local_pipe(maxsize=0)
        emitter._bind(physical_emitter)
        receiver._bind(physical_receiver)
