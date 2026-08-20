"""Helpers shared by the drivers: what a driver has only while it runs, and when a device counts as
having arrived where it was sent."""

from enum import Enum, auto
from typing import Any

import numpy as np

import pimm


class DriverRun:
    """What a driver has only while it runs: the clock it reads, the rate it ticks at, the stop it watches.

    ``World.start`` pickles a background control system before it runs, so none of this can be built in
    ``__init__``, and a helper that suspends inside a move cannot hold it in a local either.
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


class PendingMove:
    """The synchronous move a driver has in flight, if any.

    For a device whose control loop cannot be held for the duration of a move: the driver carries one of
    these across ticks and settles it against what the device reads back. A move owns the device until it
    settles, so a driver leaves its command stream unread while ``active`` — a superseding setpoint would
    otherwise fail a move for something its asker never did.
    """

    def __init__(self, tol: float):
        self._tol = tol
        self._call: pimm.calls.Call[Any, None] | None = None
        self._target: np.ndarray | float = 0.0
        self._deadline = 0.0
        # A move settled but not yet answered, with what to answer it with: None for an arrival
        self._settled: tuple[pimm.calls.Call[Any, None], TimeoutError | None] | None = None
        # Set by a move that does not arrive, cleared by the next that does: the device is not where it was put
        self.errored = False

    @property
    def active(self) -> bool:
        return self._call is not None

    @property
    def target(self) -> np.ndarray | float:
        """What the move in flight asked for, for a device that must put its reading in the same terms."""
        assert self._call is not None, 'no move is in flight'
        return self._target

    def accept(
        self, call: pimm.calls.Call[Any, None], target: np.ndarray | float, now: float, timeout_s: float
    ) -> None:
        """Take `call` as the move in flight, aiming at `target`, with `timeout_s` to get there.

        How long a move may take is the driver's to say: a device that ramps to a capped speed needs longer
        for a longer trip, and one stopped by what it is holding needs a fixed grace instead.
        """
        self._call, self._target, self._deadline = call, target, now + timeout_s

    def fail(self, exc: BaseException) -> None:
        """Hand `exc` to the move in flight, or its own outcome to one already settled."""
        if self._settled is not None:
            self.answer()
            return
        if self._call is None:
            return
        self._call.set_exception(exc)
        self._call, self.errored = None, True

    def settle(self, position: np.ndarray | float, now: float) -> MoveStatus:
        """Where the move in flight stands, once the device reads back at ``position``.

        ARRIVED and GAVE_UP end the move but do not answer it: the device is free from here, and the state
        saying so has to reach the asker before the answer does. ``answer`` is what hands it over.
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
        """Hand a settled move its outcome. Call once the state that goes with it has been published."""
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
    move: PendingMove,
    calls: pimm.calls.ControlSystemHandler[float, None],
    stream: pimm.SignalReceiver[float],
    grip: float,
    now: float,
) -> float | None:
    """The width to command the fingers this tick, or ``None`` to leave the last one standing.

    A move in flight owns the fingers, so neither the calls nor the stream is read until it settles. One
    that gives up hands back the width the fingers stopped at, so they stop pushing at a width they could
    not reach.

    ``grip`` is the width the driver has already emitted this tick, so a settled move is answered here: the
    reading that says the fingers arrived is out before its asker learns they did.
    """
    if move.active:
        settled = move.settle(grip, now)
        move.answer()
        return grip if settled is MoveStatus.GAVE_UP else None
    if (call := next(calls.incoming(), None)) is not None:
        target = _clamped(call.request)
        move.accept(call, target, now, _GRIP_TIMEOUT_S)
        return target
    if (streamed := pimm.value_updated(stream)) is not None:
        return _clamped(streamed)
    return None
