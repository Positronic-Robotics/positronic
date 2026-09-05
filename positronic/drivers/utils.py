"""Shared by the drivers: the handles a run owns, and the moves a device is asked to make."""

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np

import pimm

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MoveStatus(Enum):
    """Where a move stands."""

    MOVING = auto()
    ARRIVED = auto()
    GAVE_UP = auto()


class MoveAbandoned(RuntimeError):
    """A move the world came down under."""

    # Defaulted rather than fixed: an exception crossing a process boundary is rebuilt from its ``args``
    def __init__(self, message: str = 'the world stopped before the move arrived'):
        super().__init__(message)


class Moves(Generic[T]):
    """Both ways a device is asked to move, and the move it has in flight.

    A synchronous move is a call: its asker waits to hear the device arrive, and owns the device until it
    does. An asynchronous move is a streamed setpoint nobody waits on. A driver whose loop can be held for
    the whole travel answers a call within the tick it takes it, and so never has one in flight.
    """

    def __init__(self, sync_move: pimm.calls.ControlSystemHandler[T, None], async_move: pimm.SignalReceiver[T]):
        self._sync_move = sync_move
        self._async_move = async_move
        self._call: pimm.calls.Call[T, None] | None = None
        self._target: np.ndarray | float = 0.0
        self._tol = 0.0
        self._deadline = 0.0
        # A move settled but not yet answered, with what to answer it with: None for an arrival
        self._settled: tuple[pimm.calls.Call[T, None], TimeoutError | None] | None = None
        # Set by a move that does not arrive, cleared by the next that does: the device is not where it was put
        self.errored = False

    def __enter__(self) -> 'Moves[T]':
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
    def busy(self) -> bool:
        """A move owns the device: one is in flight, or one has settled and its asker is still owed the news."""
        return self.active or self.settled

    @property
    def target(self) -> np.ndarray | float:
        """What the move in flight asked for."""
        assert self._call is not None, 'no move is in flight'
        return self._target

    def take_newest_setpoint(self) -> T | None:
        """The newest setpoint streamed at the device, letting go of every setpoint older than it.

        A transport that queues setpoints hands the oldest over first, and a setpoint says where the device
        is wanted now, not where it was wanted when the setpoint was written.
        """
        latest = None
        while (message := self._async_move.read()) is not None and message.updated:
            latest = message.data
        return latest

    def next_request(self) -> pimm.calls.Call[T, None] | T | None:
        """What the device is asked for now: a call whose asker waits to hear it arrive, a streamed setpoint
        nobody waits on, or nothing.

        A call comes first: a setpoint says where the device is wanted now, and the move that follows it
        puts the device somewhere else. A device a move already owns is asked for nothing, and the setpoints
        streamed at it while it travels are let go for the same reason.
        """
        newest = self.take_newest_setpoint()
        if self.busy:
            return None
        if (call := next(self._sync_move.incoming(), None)) is not None:
            return call
        return newest

    def accept(
        self, call: pimm.calls.Call[T, None], target: np.ndarray | float, tol: float, now: float, timeout_s: float
    ) -> None:
        """Take `call` as the move in flight, aiming at `target` within `tol`, with `timeout_s` to get there."""
        self._call, self._target, self._tol = call, target, tol
        self._deadline = now + timeout_s

    def fail(self, exc: BaseException) -> None:
        """Hand a settled move its own outcome, and `exc` to one still in flight. Both, if there are both."""
        self.answer()
        if self._call is None:
            return
        self._call.set_exception(exc)
        self._call, self.errored = None, True

    def abandon(self, exc: BaseException | None) -> None:
        """Answer the move this device took; the world answers what it never reached."""
        self.fail(exc or MoveAbandoned())

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


class DriverRun(Generic[T]):
    """What a driver has only while it runs: the clock it reads, the rate it ticks at, the stop it watches,
    and the moves it is asked to make.

    ``World.start`` pickles a background control system, so none of it survives being built in ``__init__``.
    """

    def __init__(
        self,
        sync_move: pimm.calls.ControlSystemHandler[T, None],
        async_move: pimm.SignalReceiver[T],
        should_stop: pimm.SignalReceiver,
        clock: pimm.Clock,
        hz: float,
    ):
        self.should_stop = should_stop
        self.clock = clock
        self.limiter = pimm.RateLimiter(clock, hz=hz)
        self.moves = Moves[T](sync_move, async_move)


@contextmanager
def log_failure(request: object) -> Iterator[None]:
    """Log whatever the block raises against ``request``, the counterpart of ``pimm.calls.raise_to`` for a
    setpoint nobody is waiting on."""
    try:
        yield
    # rules-allow: swallowed-error — a command stream cannot end the run; the next setpoint supersedes
    except Exception as exc:
        logger.warning(f'{request} not applied: {exc}')


_GRIP_TIMEOUT_S = 3.0  # fingers stopped by what they are holding never reach their target
_GRIP_TOL = 0.05  # the fingers report width, so arrival is judged from the reading


def _clamped(grip: float) -> float:
    """``grip`` saturated to the range the fingers report back; raises what is no width at all.

    ``min``/``max`` order NaN by whichever side it lands on, so saturating it silently yields a width.
    """
    grip = float(grip)
    if not math.isfinite(grip):
        raise ValueError(f'{grip} is not a grip width')
    return max(0.0, min(1.0, grip))


def grip_setpoint(moves: Moves[float], grip: float, now: float) -> float | None:
    """The width to command the fingers this tick, or ``None`` to leave the last one standing.

    A move in flight owns the fingers; one that gives up hands back the width they stopped at, which the
    driver writes before calling ``Moves.answer``.
    """
    if moves.active:
        # A width streamed at fingers a move owns is older than where the move puts them.
        moves.take_newest_setpoint()
        return grip if moves.settle(grip, now) is MoveStatus.GAVE_UP else None
    asked = moves.next_request()
    if isinstance(asked, pimm.calls.Call):
        with pimm.calls.raise_to(asked):  # a width the fingers cannot be put at is the asker's to hear about
            target = _clamped(asked.request)
            moves.accept(asked, target, _GRIP_TOL, now, _GRIP_TIMEOUT_S)
            return target
    elif asked is not None:
        with log_failure(asked):
            return _clamped(asked)
    return None
