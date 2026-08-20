"""Shared by the drivers: the handles a run owns, and the synchronous move a driver carries."""

from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np

import pimm

Req = TypeVar('Req')


class DriverRun:
    """What a driver has only while it runs: the clock it reads, the rate it ticks at, the stop it watches.

    ``World.start`` pickles a background control system, so none of it survives being built in ``__init__``.
    """

    def __init__(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock, hz: float):
        self.should_stop = should_stop
        self.clock = clock
        self.limiter = pimm.RateLimiter(clock, hz=hz)
        self.errored = False


class MoveStatus(Enum):
    """Where a move stands."""

    MOVING = auto()
    ARRIVED = auto()
    GAVE_UP = auto()


class MoveAbandoned(RuntimeError):
    """A move the world came down under."""

    def __init__(self):
        super().__init__('the world stopped before the move arrived')


def abandon_queued(calls: pimm.calls.ControlSystemHandler[Req, None]) -> None:
    """Answer every call still queued on ``calls``, because nothing is left to serve them."""
    for call in calls.incoming():
        call.set_exception(MoveAbandoned())


class PendingMove(Generic[Req]):
    """The synchronous move a driver has in flight, for a device whose loop cannot be held for its duration.

    The driver carries one across ticks and settles it against what the device reads back. A move owns the
    device until it settles: a driver leaves its command stream unread while ``active``.
    """

    def __init__(self, tol: float, calls: pimm.calls.ControlSystemHandler[Req, None]):
        self._tol = tol
        self._calls = calls
        self._call: pimm.calls.Call[Req, None] | None = None
        self._target: np.ndarray | float = 0.0
        self._deadline = 0.0
        # A move settled but not yet answered, with what to answer it with: None for an arrival
        self._settled: tuple[pimm.calls.Call[Req, None], TimeoutError | None] | None = None
        # Set by a move that does not arrive, cleared by the next that does: the device is not where it was put
        self.errored = False

    def __enter__(self) -> 'PendingMove[Req]':
        return self

    def __exit__(self, exc_type, exc: BaseException | None, tb) -> None:
        self.abandon(exc)

    @property
    def active(self) -> bool:
        return self._call is not None

    @property
    def settled(self) -> bool:
        """The move is over and its asker not yet told."""
        return self._settled is not None

    @property
    def target(self) -> np.ndarray | float:
        """What the move in flight asked for."""
        assert self._call is not None, 'no move is in flight'
        return self._target

    def take(self) -> pimm.calls.Call[Req, None] | None:
        """The next move asked for, if the device is free to take one."""
        return None if self.active or self.settled else next(self._calls.incoming(), None)

    def accept(
        self, call: pimm.calls.Call[Req, None], target: np.ndarray | float, now: float, timeout_s: float
    ) -> None:
        """Take `call` as the move in flight, aiming at `target`, with `timeout_s` to get there."""
        self._call, self._target, self._deadline = call, target, now + timeout_s

    def fail(self, exc: BaseException) -> None:
        """Hand a settled move its own outcome, and `exc` to one still in flight. Both, if there are both."""
        self.answer()
        if self._call is None:
            return
        self._call.set_exception(exc)
        self._call, self.errored = None, True

    def abandon(self, exc: BaseException | None) -> None:
        """Answer everything outstanding — in flight, settled, still queued — because nothing will serve it."""
        self.fail(exc or MoveAbandoned())
        for call in self._calls.incoming():
            call.set_exception(MoveAbandoned())

    def settle(self, position: np.ndarray | float, now: float) -> MoveStatus:
        """Where the move in flight stands, once the device reads back at ``position``.

        ARRIVED and GAVE_UP end the move without answering it; ``answer`` hands the outcome over.
        """
        assert self._call is not None, 'no move is in flight'
        if bool(np.all(np.abs(np.asarray(position) - np.asarray(self._target)) < self._tol)):
            self._settled, self._call, self.errored = (self._call, None), None, False
            return MoveStatus.ARRIVED
        if now >= self._deadline:
            short = TimeoutError(f'stopped at {np.round(position, 3)}, short of {np.round(self._target, 3)}')
            self._settled, self._call, self.errored = (self._call, short), None, True
            return MoveStatus.GAVE_UP
        return MoveStatus.MOVING

    def answer(self) -> None:
        """Hand a settled move its outcome, once the state that goes with it is published."""
        if self._settled is None:
            return
        call, short = self._settled
        self._settled = None
        if short is None:
            call.set_result(None)
        else:
            call.set_exception(short)


# Fingers stopped by what they are holding never reach their target
_GRIP_TIMEOUT_S = 3.0


def _clamped(grip: float) -> float:
    """``grip`` saturated to the range the fingers report back."""
    return max(0.0, min(1.0, float(grip)))


def grip_setpoint(
    move: PendingMove[float], stream: pimm.SignalReceiver[float], grip: float, now: float
) -> float | None:
    """The width to command the fingers this tick, or ``None`` to leave the last one standing.

    A move in flight owns the fingers; one that gives up hands back the width they stopped at, which the
    driver writes before calling ``PendingMove.answer``.
    """
    if move.active:
        return grip if move.settle(grip, now) is MoveStatus.GAVE_UP else None
    if (call := move.take()) is not None:
        target = _clamped(call.request)
        move.accept(call, target, now, _GRIP_TIMEOUT_S)
        return target
    if (streamed := pimm.value_updated(stream)) is not None:
        return _clamped(streamed)
    return None
