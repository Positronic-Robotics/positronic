"""Shared helpers for scripted control-system tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import pimm

ScriptStep = tuple[Callable[[], None] | None, float]

T = TypeVar('T')


def drive_scheduler(iterator: Iterable[pimm.Sleep], *, clock=None, steps: int = 200) -> None:
    """Advance a scheduler iterator while optionally syncing a test clock.

    Args:
        iterator: Iterator returned by ``World.start`` or ``World.interleave``.
        clock: Optional mock clock exposing ``advance``; used to keep deterministic time.
        steps: Maximum number of iterations to execute before stopping.
    """
    for _ in range(steps):
        try:
            sleep = next(iterator)
        except StopIteration:
            break
        if clock is not None:
            clock.advance(sleep.seconds)


@dataclass(eq=False)
class ManualDriver(pimm.ControlSystem):
    """Deterministic control system that replays a scripted sequence of actions."""

    script: Sequence[ScriptStep]

    def __post_init__(self) -> None:
        self.script = tuple(self.script)

    def run(self, should_stop: pimm.SignalReceiver, _clock: pimm.Clock):
        for action, sleep_time in self.script:
            if should_stop.value:
                return
            if action is not None:
                action()
            yield pimm.Sleep(sleep_time)


def scripted_driver(*steps: ScriptStep) -> ManualDriver:
    """Convenience factory mirroring ``ManualDriver`` construction."""
    return ManualDriver(script=steps)


class RecordingEmitter(pimm.SignalEmitter[T]):
    """Emitter that records all emissions for later assertions."""

    def __init__(self) -> None:
        self.emitted: list[tuple[int, T]] = []

    def emit(self, data: T, ts: int = -1) -> bool:
        self.emitted.append((ts, data))
        return True


class ManualCommandReceiver(pimm.SignalReceiver[T]):
    """Receiver stub with push/read semantics convenient for tests."""

    def __init__(self) -> None:
        self._pending: list[pimm.Message[T]] = []
        self._last: pimm.Message[T] | None = None

    def push(self, data: T, ts: int | None = None) -> None:
        if ts is None:
            base = self._pending[-1].ts if self._pending else (self._last.ts if self._last else -1)
            ts = base + 1
        self._pending.append(pimm.Message(data, ts))

    def read(self) -> pimm.Message[T] | None:
        if self._pending:
            self._last = self._pending.pop(0)
            return self._last
        return self._last


class MutableShouldStop:
    """Mutable flag to coordinate manual shutdown in tests."""

    def __init__(self, initial: bool = False) -> None:
        self._value = initial

    @property
    def value(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = value


def drive_until(loop: Iterator[pimm.Sleep], clock, condition, max_steps: int = 100) -> None:
    for _ in range(max_steps):
        clock.advance(next(loop).seconds)
        if condition():
            return
    raise AssertionError('Condition not reached within step limit')


def run_scripted_agent(
    agent: pimm.ControlSystem, script: Sequence[ScriptStep], *, world: pimm.World, clock: Any, steps: int = 200
) -> None:
    """Run ``agent`` alongside a scripted driver within ``world``."""
    driver = ManualDriver(script=script)
    scheduler = world.start([agent, driver])
    drive_scheduler(scheduler, clock=clock, steps=steps)
