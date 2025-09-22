"""Shared helpers for scripted control-system tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import pimm

ScriptStep = tuple[Callable[[], None], float]


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
            action()
            yield pimm.Sleep(sleep_time)


def scripted_driver(*steps: ScriptStep) -> ManualDriver:
    """Convenience factory mirroring ``ManualDriver`` construction."""
    return ManualDriver(script=steps)
