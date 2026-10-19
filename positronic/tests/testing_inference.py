"""Shared devices and browser-world setup for attended inference tests."""

import os
import random
import socket
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any

import httpx

import pimm
from positronic import wire
from positronic.eval import Embodiment, Task
from positronic.eval import keys as eval_keys
from positronic.gui.web import EndBody, StationConsole, Status
from positronic.policy import Policy
from positronic.policy.harness import Harness

ARM_STOPPED_SHORT = 'the arm stopped short of its target'


class ReadyDevices(pimm.ControlSystem):
    """The arm and fingers of a rig that reaches wherever it is asked to go ``move_s`` after it is asked. The arm's
    first ``arm_failures`` moves stop short. Its teardown creates ``parked``, when one is given."""

    def __init__(self, move_s: float = 0.0, arm_failures: int = 0, parked: Path | None = None):
        self._move_s = move_s
        self._arm_failures = arm_failures
        self._parked = parked
        self.arm = pimm.calls.ControlSystemHandler[Any, None](self)
        self.gripper = pimm.calls.ControlSystemHandler[Any, None](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        moving: list[tuple[pimm.calls.Call[Any, None], float]] = []
        while not should_stop.value:
            now = clock.now()
            for call in self.arm.incoming():
                if self._arm_failures > 0:
                    self._arm_failures -= 1
                    call.set_exception(RuntimeError(ARM_STOPPED_SHORT))
                else:
                    moving.append((call, now + self._move_s))
            moving.extend((call, now + self._move_s) for call in self.gripper.incoming())
            for call, _ in [(call, at) for call, at in moving if at <= now]:
                call.set_result(None)
            moving = [(call, at) for call, at in moving if at > now]
            yield pimm.Sleep(0.01)
        if self._parked is not None:
            self._parked.touch()


def make_embodiment(simulated: bool = False, move_s: float = 0.0, arm_failures: int = 0) -> Embodiment:
    devices = ReadyDevices(move_s, arm_failures)
    return Embodiment(
        descriptor='stub',
        observations={},
        commands={},
        prepare_handlers={eval_keys.ARM: devices.arm, eval_keys.GRIPPER: devices.gripper},
        static_meta={},
        meta_source=None,
        control_systems=(devices,),
        simulated=simulated,
    )


def make_trial(instruction: str = 'stub') -> Callable[[], Task]:
    prepare_args = {eval_keys.ARM: 'start-pose', eval_keys.GRIPPER: 0.0}
    return partial(Task, instruction_source=instruction, timeout_sec=None, prepare_args=prepare_args)


# Below the range Linux and macOS hand out for port 0, so that a server that binds port 0 in another test cannot take
# the port between this probe and the console's own bind. Each xdist worker draws from its own block of ports.
_FIRST_PORT = 20000
_PORTS_PER_WORKER = 400
_WORKER_BLOCKS = 30


def free_port() -> int:
    worker = os.environ.get('PYTEST_XDIST_WORKER', '').removeprefix('gw')
    first = _FIRST_PORT + (int(worker) % _WORKER_BLOCKS if worker.isdigit() else 0) * _PORTS_PER_WORKER
    for port in random.sample(range(first, first + _PORTS_PER_WORKER), _PORTS_PER_WORKER):
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                continue
            return port
    raise RuntimeError(f'no free port in {first}-{first + _PORTS_PER_WORKER - 1}')


def read_status(base: str) -> Status | None:
    try:
        return Status.model_validate(httpx.get(f'{base}/status', timeout=2.0).json())
    except httpx.TransportError:
        return None


class PageOperator(pimm.ControlSystem):
    """Presses the station page's buttons over HTTP from inside the world, as ``script`` says. Its return ends
    the world."""

    def __init__(self, script: Callable[[pimm.Clock], Iterator[pimm.Command]]):
        self._script = script

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        yield from self._script(clock)


def wait(condition: Callable[[], Any], what: str, clock: pimm.Clock, seconds: float = 10.0) -> Iterator[pimm.Command]:
    """Yield until ``condition`` holds. Raise when ``seconds`` pass on ``clock`` first."""
    deadline = clock.now() + seconds
    while not condition():
        if clock.now() > deadline:
            raise AssertionError(f'timed out waiting for {what}')
        yield pimm.Sleep(0.02)


def run_attended(
    policy: Policy,
    script: Callable[[str, pimm.Clock], Iterator[pimm.Command]],
    move_s: float = 0.0,
    arm_failures: int = 0,
) -> None:
    """The world ``web`` composes, with the station console in a process of its own and ``script`` pressing the
    page's buttons from beside the harness. The rig takes ``move_s`` for each move, and its arm's first
    ``arm_failures`` moves stop short."""
    port = free_port()
    console = StationConsole(make_trial('pick up the cube'), policy, None, host='127.0.0.1', port=port)
    embodiment = make_embodiment(move_s=move_s, arm_failures=arm_failures)
    harness = Harness(embodiment)
    operator = PageOperator(partial(script, f'http://127.0.0.1:{port}'))
    with pimm.World() as world:
        wire.wire_embodiment(world, harness, embodiment, done=console.done)
        world.connect(console.perform_task, harness.perform_task)
        world.run([operator, harness, *embodiment.control_systems], [console])


def post(base: str, path: str, body: EndBody | None = None) -> int:
    """The HTTP status of the press. A press the console refuses must not raise inside the world."""
    return httpx.post(f'{base}{path}', json=None if body is None else body.model_dump(mode='json')).status_code
