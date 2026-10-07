"""An env server that serves several slots, each driven by a client of its own.

``_ClonedScene`` holds one step count per clone and FREEZES a clone that ends: it stops advancing and reports
the verdict it ended on, the way RoboLab's env does instead of re-rolling a terminated clone. Each clone also
reports the grip of the last action it received, so a test can see which client's action reached it.
``positronic/simulator/robolab/validate.py`` runs the same shape against the real benchmark on a RoboLab box.
"""

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from typing import Any

import numpy as np
import pytest

import pimm
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.env_server.server import EnvProtocol
from positronic.simulator.env_server.tests.conftest import serve_env
from positronic.tests.testing_coutils import drive_scheduler

_SCENE = 'cloned_scene'
_STEPS = 'steps'
_GRIP = 'grip'
# How long a test waits to see that a request is still held back before it lets the batch run.
_HELD = 0.3


def _action(grip: float) -> dict[str, Any]:
    return protocol.single_arm_action({protocol.COMMAND_TYPE: protocol.HOLD}, grip)


class _ClonedScene(EnvProtocol):
    """``len(ends_at)`` clones of one scene: clone ``i`` ends on step ``ends_at[i]`` with verdict ``succeeds[i]``."""

    def __init__(self, ends_at: list[int], succeeds: list[bool] | None = None):
        self._ends_at = ends_at
        self._succeeds = succeeds if succeeds is not None else [False] * len(ends_at)
        self._steps = [0] * len(ends_at)
        self._grips = [0.0] * len(ends_at)
        self.resets: list[Any] = []
        self.stepped: list[set[int]] = []  # the slots each env step received an action for

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        return [{'name': _SCENE}]

    def _observed(self, slot: int) -> dict[str, Any]:
        return {_STEPS: np.full(2, self._steps[slot], dtype=np.float64), _GRIP: self._grips[slot]}

    def reset(self, token: Any) -> dict[str, Any]:
        self.resets.append(token)
        self._steps = [0] * len(self._ends_at)
        self._grips = [0.0] * len(self._ends_at)
        return {
            protocol.SLOTS: [{protocol.FRAME_OBS: self._observed(slot)} for slot in range(len(self._ends_at))],
            protocol.FRAME_META: {},
            protocol.FRAME_ROBOT_META: {},
            protocol.FRAME_CONTROL_DT: 0.1,
        }

    def step(self, actions: dict[int, dict[str, Any]]) -> dict[str, Any]:
        self.stepped.append(set(actions))
        for slot, action in actions.items():
            self._grips[slot] = float(action[protocol.TARGET_GRIP])
        slots = []
        for slot, ends_at in enumerate(self._ends_at):
            frozen = self._steps[slot] >= ends_at
            if not frozen:
                self._steps[slot] += 1
            done = self._steps[slot] >= ends_at
            slots.append({
                protocol.FRAME_OBS: self._observed(slot),
                protocol.FRAME_DONE: done,
                protocol.FRAME_SUCCESS: done and self._succeeds[slot],
            })
        return {protocol.SLOTS: slots, protocol.FRAME_CONTROL_DT: 0.1}

    def close(self) -> None:
        pass


@contextmanager
def _clients(env: _ClonedScene) -> Iterator[list[EnvConnection]]:
    """One connection per clone of ``env``, opened in slot order."""
    with serve_env(env, slots=len(env._ends_at)) as (host, port):
        conns = []
        try:
            for _ in env._ends_at:
                conns.append(EnvConnection(host, port))
                conns[-1].tasks({})  # a request the server answers at once, so this client holds its slot
            yield conns
        finally:
            for conn in conns:
                conn.close()


def _together(*calls: Callable[[], Any]) -> list[Any]:
    """Run ``calls`` at once, one thread each, and return their results in order."""
    with ThreadPoolExecutor(len(calls)) as pool:
        return [future.result(timeout=10.0) for future in [pool.submit(call) for call in calls]]


def _held_back(future: Future) -> bool:
    try:
        future.result(timeout=_HELD)
    except TimeoutError:
        return True
    return False


def _steps(frame: dict[str, Any]) -> float:
    return float(frame[protocol.FRAME_OBS][_STEPS][0])


@pytest.mark.timeout(60.0)
def test_each_client_drives_its_own_slot():
    """Each client's action reaches its own clone, and each client reads its own clone back."""
    env = _ClonedScene(ends_at=[9, 9, 9])
    with _clients(env) as conns:
        _together(*[lambda conn=conn: conn.reset(None) for conn in conns])
        frames = _together(*[
            lambda conn=conn, slot=slot: conn.step(_action(0.1 * slot)) for slot, conn in enumerate(conns)
        ])

    assert [frame[protocol.FRAME_OBS][_GRIP] for frame in frames] == pytest.approx([0.0, 0.1, 0.2])
    assert env.stepped == [{0, 1, 2}], 'the three actions did not reach one env step'


@pytest.mark.timeout(60.0)
def test_a_step_waits_for_every_slot_in_an_episode():
    with _clients(_ClonedScene(ends_at=[9, 9])) as (first, second), ThreadPoolExecutor(1) as pool:
        _together(lambda: first.reset(None), lambda: second.reset(None))
        pending = pool.submit(first.step, _action(0.0))

        assert _held_back(pending), 'the step ran before the other slot sent its action'
        second.step(_action(0.0))
        assert _steps(pending.result(timeout=10.0)) == 1.0


@pytest.mark.timeout(60.0)
def test_a_reset_waits_for_every_slot():
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env) as (first, second), ThreadPoolExecutor(1) as pool:
        pending = pool.submit(first.reset, 'scene')

        assert _held_back(pending), 'the reset ran before the other slot asked for one'
        second.reset('scene')
        pending.result(timeout=10.0)
        assert env.resets == ['scene']


@pytest.mark.timeout(60.0)
def test_a_slot_that_ends_early_stops_holding_the_others_back():
    """The ended clone keeps its final state and verdict; its neighbour steps on alone."""
    env = _ClonedScene(ends_at=[1, 3], succeeds=[True, False])
    with _clients(env) as (first, second):
        _together(lambda: first.reset(None), lambda: second.reset(None))
        ended, _ = _together(lambda: first.step(_action(0.0)), lambda: second.step(_action(0.0)))
        last = [second.step(_action(0.0)) for _ in range(2)][-1]

    assert ended[protocol.FRAME_DONE] and ended[protocol.FRAME_SUCCESS]
    assert last[protocol.FRAME_DONE] and not last[protocol.FRAME_SUCCESS] and _steps(last) == 3.0
    assert env.stepped == [{0, 1}, {1}, {1}]


@pytest.mark.timeout(60.0)
def test_a_slot_whose_client_resets_early_waits_without_an_action():
    """A client that ends its trial before the env does asks for the next one, and its clone gets no action."""
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env) as (first, second), ThreadPoolExecutor(1) as pool:
        _together(lambda: first.reset(None), lambda: second.reset(None))
        next_trial = pool.submit(first.reset, None)
        second.step(_action(0.0))

        assert _held_back(next_trial), 'the batch reset ran while a slot was still in its episode'
        second.reset(None)
        assert _steps(next_trial.result(timeout=10.0)) == 0.0
    assert env.stepped == [{1}] and len(env.resets) == 2


@pytest.mark.timeout(60.0)
def test_a_client_that_leaves_holds_nobody_back():
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env) as (first, second):
        first.close()
        second.reset(None)
        second.step(_action(0.0))
    assert env.stepped == [{1}]


@pytest.mark.timeout(60.0)
def test_slots_that_ask_for_different_resets_are_refused():
    """The clones share one scene, so one batch reset cannot serve two tokens."""
    with _clients(_ClonedScene(ends_at=[9, 9])) as (first, second):
        with pytest.raises(RuntimeError, match='2 different resets'):
            _together(lambda: first.reset('one scene'), lambda: second.reset('another scene'))


@pytest.mark.timeout(60.0)
def test_a_client_past_the_slot_count_is_refused():
    env = _ClonedScene(ends_at=[9])
    with serve_env(env) as (host, port):
        first = EnvConnection(host, port)
        first.tasks({})
        extra = EnvConnection(host, port)
        try:
            with pytest.raises(RuntimeError, match='all 1 slots are taken'):
                extra.tasks({})
        finally:
            extra.close()
            first.close()


@pytest.mark.timeout(60.0)
def test_a_step_before_the_first_reset_is_refused():
    with _clients(_ClonedScene(ends_at=[9])) as (conn,):
        with pytest.raises(RuntimeError, match='slot 0 stepped while idle'):
            conn.step(_action(0.0))


class _GripAdapter(EnvAdapter):
    """Commands the grip its policy names and reports the grip its clone last received."""

    def __init__(self, grip: float):
        self._grip = grip

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{eval_keys.TASK: record['name']} for record in records]

    def reset_token(self, params: dict[str, Any]) -> Any:
        return None

    def action(self, commands: dict[str, pimm.Message | None]) -> dict[str, Any]:
        return _action(self._grip)

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {_GRIP: raw_obs[_GRIP]}

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        return {eval_keys.SUCCESS: result[protocol.FRAME_SUCCESS]} if result[protocol.FRAME_DONE] else None


def _drive_one_proxy(host: str, port: int, grip: float, started: threading.Barrier) -> float:
    """Run one proxy in a World of its own through a reset and a few steps; return the grip it observed."""
    with pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_GripAdapter(grip), nullcontext((host, port)))
        grip_rx = world.pair(proxy.observations[_GRIP])
        scheduler = world.start([proxy])
        started.wait(timeout=10.0)
        proxy.reset({})
        drive_scheduler(scheduler, steps=4)
        return grip_rx.value


@pytest.mark.timeout(60.0)
def test_two_proxies_drive_two_slots_of_one_env():
    """Two Worlds, as two evals would run them, each command and observe their own clone of one env."""
    with serve_env(_ClonedScene(ends_at=[99, 99]), slots=2) as (host, port):
        started = threading.Barrier(2)
        grips = _together(
            lambda: _drive_one_proxy(host, port, 0.25, started), lambda: _drive_one_proxy(host, port, 0.75, started)
        )
    assert grips == [0.25, 0.75]
