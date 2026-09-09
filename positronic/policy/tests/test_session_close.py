"""Whether an episode gives its inference session back before the next episode asks for one.

A server that serves one session at a time frees the slot on the WebSocket close handshake, so an
episode that drops its connection holds that slot for the whole process. These drive the harness the way
a run does and watch the wire.
"""

import threading
from typing import Any

import numpy as np
import pytest

import pimm
from positronic import keys
from positronic.drivers.roboarm.tests.fakes import make_robot_state
from positronic.eval import Task
from positronic.offboard.client import InferenceSession
from positronic.policy.base import Policy
from positronic.policy.harness import Harness, Rollout
from positronic.policy.layers import ChunkedSchedule
from positronic.policy.remote import INFER, RemoteSession, round_trip
from positronic.policy.tests.test_harness import CAM, make_embodiment
from positronic.tests.testing_coutils import drive_scheduler


class Wire(InferenceSession):
    """One server-side session slot, taken on open and given back when ``close`` RETURNS.

    ``hold`` keeps a round trip in flight until the test releases it, so a test can end an episode with a
    call still outstanding — the shape a close has to survive.
    """

    def __init__(self, name: str, log: list[tuple[str, str]], action: list[dict[str, Any]], hold=None):
        self.name = name
        self._log = log
        self._action = action
        self._hold = hold
        log.append(('open', name))

    def infer(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        self._log.append(('infer.start', self.name))
        if self._hold is not None:
            assert self._hold.wait(timeout=10.0), 'the test never released the held round trip'
        self._log.append(('infer.end', self.name))
        return self._action

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        self._log.append(('close.enter', self.name))
        self._log.append(('close.return', self.name))


class WiredPolicy(Policy):
    """A real ``RemoteSession`` per episode over a fresh ``Wire``, so each episode takes and gives back a
    slot through the same close chain a run does."""

    def __init__(self, log: list[tuple[str, str]], action: list[dict[str, Any]], holds: dict[int, Any] | None = None):
        self._log = log
        self._action = action
        self._holds = holds or {}
        self.opened = 0

    def new_session(self, context=None, rt=None) -> RemoteSession:
        assert rt is not None, 'the harness supplies the runtime'
        index = self.opened
        self.opened += 1
        return RemoteSession(Wire(f'wire{index}', self._log, self._action, self._holds.get(index)), rt)

    @property
    def functions(self):
        return {INFER: round_trip}


ACTION = [{keys.TARGET_GRIP: 0.3, keys.ACTION_TIMESTAMP: 0.0}]


class Episodes(pimm.ControlSystem):
    """Runs ``count`` episodes back to back, one at a time, the way a run's driver does.

    ``closes_each_rollout`` is ``Rollout``'s contract that whoever opens one closes it, honoured or not.
    """

    def __init__(self, policy: Policy, log: list, *, count: int = 2, closes_each_rollout: bool = True):
        self._policy = policy
        self._log = log
        self._count = count
        self._closes_each_rollout = closes_each_rollout
        self.opened: list[Rollout] = []
        self.perform_task = pimm.calls.ControlSystemCaller[Rollout, dict[str, Any]](self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        for index in range(self._count):
            rollout = Rollout(Task(instruction_source=f't{index}', timeout_sec=0.05), self._policy, None)
            self.opened.append(rollout)
            try:
                answer = self.perform_task(rollout)
                while not answer.done():
                    if should_stop.value:
                        return
                    yield pimm.Sleep(0.01)
                answer.result()
            finally:
                if self._closes_each_rollout:
                    self._log.append(('rollout.close.enter', f'ep{index}'))
                    rollout.close()
                    self._log.append(('rollout.close.return', f'ep{index}'))
        yield pimm.Sleep(0.1)


class Rig(pimm.ControlSystem):
    """The devices, publishing every round, so the harness always has an observation to infer on."""

    def __init__(self, camera, robot_state, grip):
        self._camera = camera
        self._robot_state = robot_state
        self._grip = grip

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock):
        frame = pimm.shared_memory.NumpySMAdapter((2, 2, 3), np.dtype(np.uint8))
        frame.array[:] = np.zeros((2, 2, 3), dtype=np.uint8)
        while not should_stop.value:
            self._camera.emit(frame)
            self._robot_state.emit(make_robot_state([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
            self._grip.emit(0.25)
            yield pimm.Sleep(0.005)


def _run(policy: Policy, log: list, *, closes_each_rollout: bool = True, count: int = 2) -> Episodes:
    with pimm.World(virtual_time=True) as world:
        harness = Harness(make_embodiment())
        rig = Rig(
            world.pair(harness.observations[CAM]),
            world.pair(harness.observations[keys.ROBOT_STATE]),
            world.pair(harness.observations[keys.GRIP]),
        )
        world.pair(harness.commands[keys.ROBOT_COMMAND])
        world.pair(harness.commands[keys.TARGET_GRIP])
        world.pair(harness.ds_command)
        world.pair(harness.deadline_ns)
        driver = Episodes(policy, log, count=count, closes_each_rollout=closes_each_rollout)
        world.connect(driver.perform_task, harness.perform_task)
        scheduler = world.start([harness, rig, driver])
        drive_scheduler(scheduler, steps=4000)
    return driver


def _order(log: list, event: str, name: str) -> int:
    for index, entry in enumerate(log):
        if entry == (event, name):
            return index
    return -1


@pytest.mark.timeout(60.0)
def test_the_first_episode_gives_its_session_back_before_the_second_asks_for_one():
    """The slot a server serves one of. The second episode's wire must open after the first one's close
    has RETURNED — not merely been called, and not after the process ends."""
    log: list[tuple[str, str]] = []
    policy = ChunkedSchedule().wrap(WiredPolicy(log, ACTION))
    _run(policy, log)

    assert _order(log, 'open', 'wire1') >= 0, f'the second episode never opened a session: {log}'
    closed = _order(log, 'close.return', 'wire0')
    assert closed >= 0, f'the first episode never finished closing its session: {log}'
    assert closed < _order(log, 'open', 'wire1'), f'the second session opened before the first was closed: {log}'


@pytest.mark.timeout(60.0)
def test_a_session_is_closed_even_when_a_round_trip_was_in_flight_at_the_end():
    """An episode can end with a wire round trip still outstanding. The close must still happen, must wait
    the call out first, and must return."""
    log: list[tuple[str, str]] = []
    hold = threading.Event()
    threading.Timer(1.0, hold.set).start()  # the server answers, late, after the episode has ended
    policy = ChunkedSchedule().wrap(WiredPolicy(log, ACTION, holds={0: hold}))
    try:
        _run(policy, log, count=1)
    finally:
        hold.set()

    assert _order(log, 'infer.end', 'wire0') >= 0, f'the held round trip never finished: {log}'
    assert _order(log, 'close.return', 'wire0') >= 0, f'the session was never closed: {log}'
    assert _order(log, 'infer.end', 'wire0') < _order(log, 'close.enter', 'wire0'), (
        f'the session was closed under a live round trip: {log}'
    )


@pytest.mark.timeout(60.0)
def test_nothing_closes_a_rollout_its_caller_drops():
    """``Rollout`` says whoever opens it closes it, and nothing underneath makes up for a caller that does
    not: no finalizer, and neither the harness nor the world reaches the session. A driver that drops each
    episode's rollout leaves every wire of the run open.

    The dropped rollouts are closed here once the assertions have read the log. Their runtimes serve on
    threads that outlive the world, and a thread still answering round trips records its spans into
    whatever telemetry the NEXT test binds.
    """
    log: list[tuple[str, str]] = []
    policy = ChunkedSchedule().wrap(WiredPolicy(log, ACTION))
    driver = _run(policy, log, closes_each_rollout=False)
    try:
        assert _order(log, 'infer.end', 'wire0') >= 0, f'the first episode never ran a round trip: {log}'
        assert _order(log, 'close.enter', 'wire0') < 0, f'something closed the dropped rollout: {log}'
    finally:
        for rollout in driver.opened:
            rollout.close()
