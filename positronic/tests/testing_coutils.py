"""Shared helpers for scripted control-system tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import pimm
from positronic.eval import Task
from positronic.policy import Policy, Session
from positronic.policy.harness import Harness, Rollout

# The driver runs a step for its effect, so a step that hands something back — a call's answer — is one too.
ScriptStep = tuple[Callable[[], object] | None, float]

T = TypeVar('T')


def drive_scheduler(iterator: Iterable[pimm.Command], *, steps: int = 200) -> None:
    """Pump a world scheduler up to ``steps`` times.

    A virtual-time world advances its own clock as it runs, so there is nothing to
    drive here but the iterator itself.

    Args:
        iterator: Iterator returned by ``World.start`` or ``World.interleave``.
        steps: Maximum number of iterations to execute before stopping.
    """
    for _ in range(steps):
        try:
            next(iterator)
        except StopIteration:
            break


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
            yield pimm.Sleep(sleep_time) if sleep_time > 0 else pimm.Yield()


def scripted_driver(*steps: ScriptStep) -> ManualDriver:
    """Convenience factory mirroring ``ManualDriver`` construction."""
    return ManualDriver(script=steps)


def episode_caller(
    world: pimm.World, harness: Harness, policy: Policy, output_path: Path | None = Path('dataset')
) -> Callable[[Task], pimm.calls.Answer[dict[str, Any]]]:
    """A caller that asks ``harness`` for a task the way a driver does: it opens the session that runs it and
    names the path it records into. Nothing writes there unless the test runs a recorder of its own."""
    perform_task = world.pair(harness.perform_task)

    def ask(task: Task):
        return perform_task(Rollout(task, policy, output_path))

    return ask


class RecordingEmitter(pimm.SignalEmitter[T]):
    """Emitter that records all emissions for later assertions."""

    def __init__(self) -> None:
        self.emitted: list[tuple[int, T]] = []

    def emit(self, data: T, ts: int = -1):
        self.emitted.append((ts, data))


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
        elif self._last is not None:
            self._last.updated = False
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


def drive_until(loop: Iterator[pimm.Command], condition, max_steps: int = 100) -> None:
    for _ in range(max_steps):
        next(loop)
        if condition():
            return
    raise AssertionError('Condition not reached within step limit')


def run_scripted_agent(
    agent: pimm.ControlSystem, script: Sequence[ScriptStep], *, world: pimm.World, steps: int = 200
) -> None:
    """Run ``agent`` alongside a scripted driver within ``world``."""
    driver = ManualDriver(script=script)
    scheduler = world.start([agent, driver])
    drive_scheduler(scheduler, steps=steps)


class IdleSession(Session):
    """A policy session that records what it is shown and commands nothing.

    The recording lands on its policy's ``observations`` list.
    """

    def __init__(self, policy):
        self._policy = policy

    def __call__(self, obs, time_ns):
        self._policy.observations.append(obs)
        return []

    @property
    def meta(self):
        return {}

    def close(self):
        pass
