"""The run-log records of a YAM settle pass and of a YAM device error, one JSON object per line.

Read them with a grep for ``YAM settle`` and ``YAM device error``. Joints carry the names of ``YAM_JOINT_NAMES``.
"""

import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..models import YAM_JOINT_NAMES
from .settle import SettleTuning

logger = logging.getLogger(__name__)

SETTLE_SAMPLES_PER_S = 25  # per arm: a start settle of 4 s writes about 100 sample lines
_DIGITS = 5  # 0.01 mrad, against the 2 mrad stillness rule
_MAX_CAUSES = 10

SettleOutcome = Literal['arrived', 'gave_up', 'timeout', 'stopped']


class ChainReading(BaseModel):
    """One reading of the chain. A torque is the effort that the chain reports, None when it reports none."""

    q: list[float]
    dq: list[float]
    torque: list[float] | None
    grip: float
    grip_dq: float | None
    grip_torque: float | None


class SettlePass(BaseModel):
    """The start of one settle pass: the goal, the reference that the ramp ends on, and the rules it settles by."""

    arm: str
    move: str
    attempt: int
    goal: list[float]
    reference: list[float]
    start: list[float]
    travel_s: float
    timeout_s: float
    tolerance_rad: float
    still_position_rad: float
    still_time_s: float


class SettleSample(BaseModel):
    """One reading in a settle pass. ``spread`` is the position span per joint over the last ``still_time_s``, None
    until the pass has rested that long; ``unstill`` names the joints whose span breaks the stillness rule."""

    arm: str
    move: str
    attempt: int
    t: float
    ramping: bool
    reading: ChainReading
    error: list[float]
    spread: list[float] | None
    unstill: list[str]
    off_goal: list[str]


class SettleEnd(BaseModel):
    """How one settle pass ended. ``last_unstill_s`` gives, per joint, the last time it broke the stillness rule;
    ``max_spread`` the largest span per joint over the pass."""

    arm: str
    move: str
    attempt: int
    outcome: SettleOutcome
    t: float
    reading: ChainReading
    error: list[float]
    spread: list[float] | None
    unstill: list[str]
    off_goal: list[str]
    last_unstill_s: dict[str, float]
    max_spread: list[float] | None


class ErrorLink(BaseModel):
    type: str
    message: str


class DeviceErrorRecord(BaseModel):
    """A device error of one arm: the exception and its causes, the last reading before it, and the last settle pass
    that ended on the arm."""

    arm: str
    during: str
    errors: list[ErrorLink]
    reading_age_s: float | None
    reading: ChainReading | None
    settle: SettleEnd | None


@contextlib.contextmanager
def _never_raises(record: str) -> Iterator[None]:
    # rules-allow: swallowed-error — a run-log record must never stop a move, a park or a fault report
    try:
        yield
    except Exception:
        logger.exception(f'Could not write the {record} record')


def _values(a: np.ndarray | Sequence[float]) -> list[float]:
    return [round(float(x), _DIGITS) for x in np.asarray(a, dtype=np.float64).ravel()]


def _names(mask: np.ndarray) -> list[str]:
    return [YAM_JOINT_NAMES[i] for i in np.flatnonzero(mask)]


def _optional(a: np.ndarray | Sequence[float] | None) -> list[float] | None:
    return None if a is None else _values(a)


def _scalar(a: np.ndarray | Sequence[float] | None) -> float | None:
    return None if a is None else _values(a)[0]


def chain_reading(
    q: np.ndarray,
    dq: np.ndarray,
    torque: np.ndarray | None,
    grip: float,
    grip_dq: np.ndarray | None,
    grip_torque: np.ndarray | None,
) -> ChainReading:
    """The reading of the joints and of the gripper motor, as the chain reports them."""
    return ChainReading(
        q=_values(q),
        dq=_values(dq),
        torque=_optional(torque),
        grip=round(float(grip), _DIGITS),
        grip_dq=_scalar(grip_dq),
        grip_torque=_scalar(grip_torque),
    )


def _error_chain(exc: BaseException) -> list[ErrorLink]:
    links: list[ErrorLink] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(links) < _MAX_CAUSES:
        seen.add(id(current))
        links.append(ErrorLink(type=f'{type(current).__module__}.{type(current).__qualname__}', message=str(current)))
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return links


def log_device_error(
    arm: str,
    during: str,
    exc: BaseException,
    reading: ChainReading | None = None,
    reading_age_s: float | None = None,
    settle: SettleEnd | None = None,
) -> DeviceErrorRecord | None:
    """Log the device error ``exc`` of ``arm``, with its traceback, and return its record. It never raises."""
    record = None
    with _never_raises('YAM device error'):
        record = DeviceErrorRecord(
            arm=arm,
            during=during,
            errors=_error_chain(exc),
            reading_age_s=None if reading_age_s is None else round(reading_age_s, 3),
            reading=reading,
            settle=settle,
        )
        logger.error('YAM device error %s', record.model_dump_json(), exc_info=exc)
    return record


class SettleTrace:
    """Logs one settle pass of one arm: its start, its readings at up to ``SETTLE_SAMPLES_PER_S``, and its end. No
    method raises: after a record fails to write, the trace writes no more records of the pass."""

    def __init__(
        self,
        arm: str,
        move: str,
        attempt: int,
        goal: np.ndarray,
        reference: np.ndarray,
        start: np.ndarray,
        travel_s: float,
        timeout_s: float,
        tuning: SettleTuning,
    ):
        self._arm, self._move, self._attempt = arm, move, attempt
        self._goal = np.asarray(goal, dtype=np.float64)
        self._tuning = tuning
        self._next_sample_s = 0.0
        self._last_unstill_s: dict[str, float] = {}
        self._max_spread: np.ndarray | None = None
        self._failed = False
        with self._guarded('YAM settle pass'):
            record = SettlePass(
                arm=arm,
                move=move,
                attempt=attempt,
                goal=_values(goal),
                reference=_values(reference),
                start=_values(start),
                travel_s=round(travel_s, 3),
                timeout_s=round(timeout_s, 3),
                tolerance_rad=tuning.tolerance_rad,
                still_position_rad=tuning.still_position_rad,
                still_time_s=tuning.still_time_s,
            )
            logger.info('YAM settle pass %s', record.model_dump_json())

    @contextlib.contextmanager
    def _guarded(self, record: str) -> Iterator[None]:
        with _never_raises(record):
            try:
                yield
            except Exception:
                self._failed = True
                raise

    def _judge(self, t: float, q: np.ndarray, spread: np.ndarray | None) -> tuple[np.ndarray, list[str], list[str]]:
        error = np.asarray(q, dtype=np.float64) - self._goal
        unstill: list[str] = []
        if spread is not None:
            unstill = _names(spread > self._tuning.still_position_rad)
            for name in unstill:
                self._last_unstill_s[name] = round(t, 3)
            self._max_spread = spread if self._max_spread is None else np.maximum(self._max_spread, spread)
        return error, unstill, _names(np.abs(error) >= self._tuning.tolerance_rad)

    def observe(
        self, t: float, ramping: bool, q: np.ndarray, spread: np.ndarray | None, reading: Callable[[], ChainReading]
    ) -> None:
        """Take the joints ``q`` at ``t`` s into the pass, and log ``reading()`` when a sample is due."""
        if self._failed:
            return
        with self._guarded('YAM settle sample'):
            error, unstill, off_goal = self._judge(t, q, spread)
            if t < self._next_sample_s or not logger.isEnabledFor(logging.INFO):
                return
            self._next_sample_s = t + 1 / SETTLE_SAMPLES_PER_S
            record = SettleSample(
                arm=self._arm,
                move=self._move,
                attempt=self._attempt,
                t=round(t, 3),
                ramping=ramping,
                reading=reading(),
                error=_values(error),
                spread=_optional(spread),
                unstill=unstill,
                off_goal=off_goal,
            )
            logger.info('YAM settle sample %s', record.model_dump_json())

    def end(
        self,
        outcome: SettleOutcome,
        t: float,
        q: np.ndarray,
        spread: np.ndarray | None,
        reading: Callable[[], ChainReading],
    ) -> SettleEnd | None:
        """Log how the pass ended, and return the record."""
        record = None
        if self._failed:
            return record
        with self._guarded('YAM settle end'):
            error, unstill, off_goal = self._judge(t, q, spread)
            record = SettleEnd(
                arm=self._arm,
                move=self._move,
                attempt=self._attempt,
                outcome=outcome,
                t=round(t, 3),
                reading=reading(),
                error=_values(error),
                spread=_optional(spread),
                unstill=unstill,
                off_goal=off_goal,
                last_unstill_s=dict(self._last_unstill_s),
                max_spread=_optional(self._max_spread),
            )
            level = logging.INFO if outcome == 'arrived' else logging.WARNING
            logger.log(level, 'YAM settle end %s', record.model_dump_json())
        return record
