"""An env server that serves several slots, each driven by a client of its own.

``_ClonedScene`` holds one step count per clone and FREEZES a clone that ends: it stops advancing and reports
the verdict it ended on, the way RoboLab's env does instead of re-rolling a terminated clone. Each clone also
reports the grip of the last action it received, so a test can see which client's action reached it. Each reset
builds as many clones as the batch has slots, as RoboLab's env does.
"""

import ast
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from websockets.frames import OP_PING
from websockets.sync.client import connect
from websockets.sync.server import serve as websocket_serve

import pimm
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server import protocol
from positronic.simulator.env_server import server as server_module
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.env_server.client import EnvConnection
from positronic.simulator.env_server.proxy import RemoteEnvControlSystem
from positronic.simulator.env_server.server import DEFAULT_JOIN_DEADLINE, EnvProtocol
from positronic.simulator.env_server.tests.conftest import serve_env
from positronic.simulator.env_server.tests.mujoco_env import make_mujoco_env
from positronic.tests.testing_coutils import drive_scheduler

_SCENE = 'cloned_scene'
_STEPS = 'steps'
_GRIP = 'grip'
# How long a test waits to see that a request is still held back before it lets the batch run.
_HELD = 0.3
# A reset token that ``_ClonedScene`` refuses.
_FAILING_TOKEN = 'unknown scene'


def _action(grip: float) -> dict[str, Any]:
    return protocol.single_arm_action({protocol.COMMAND_TYPE: protocol.HOLD}, grip)


class _ClonedScene(EnvProtocol):
    """Up to ``len(ends_at)`` clones of one scene: clone ``i`` ends on step ``ends_at[i]`` with verdict
    ``succeeds[i]``."""

    def __init__(self, ends_at: list[int], succeeds: list[bool] | None = None):
        self._max_ends_at = ends_at
        self._ends_at = ends_at
        self._succeeds = succeeds if succeeds is not None else [False] * len(ends_at)
        self._steps = [0] * len(ends_at)
        self._grips = [0.0] * len(ends_at)
        self.resets: list[Any] = []
        self.built: list[int] = []  # the clone count of each reset
        self.slot_resets: list[tuple[Any, int]] = []  # the token and the env slot of each reset of one slot
        self.stepped: list[set[int]] = []  # the slots each env step received an action for

    @property
    def num_slots(self) -> int:
        return len(self._max_ends_at)

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        return [{'name': _SCENE, protocol.TASK_INSTRUCTION: f'run {_SCENE}'}]

    def _observed(self, slot: int) -> dict[str, Any]:
        return {_STEPS: np.full(2, self._steps[slot], dtype=np.float64), _GRIP: self._grips[slot]}

    def reset(self, token: Any, slots: int | None = None) -> dict[str, Any]:
        self.resets.append(token)
        if token == _FAILING_TOKEN:
            raise ValueError(f'no scene for {token!r}')
        self._ends_at = self._max_ends_at[: self.num_slots if slots is None else slots]
        self.built.append(len(self._ends_at))
        self._steps = [0] * len(self._ends_at)
        self._grips = [0.0] * len(self._ends_at)
        return {
            protocol.SLOTS: [{protocol.FRAME_OBS: self._observed(slot)} for slot in range(len(self._ends_at))],
            protocol.FRAME_META: {},
            protocol.FRAME_ROBOT_META: {},
            protocol.FRAME_CONTROL_DT: 0.1,
        }

    def reset_slot(self, token: Any, slot: int) -> dict[str, Any]:
        self.slot_resets.append((token, slot))
        self._steps[slot], self._grips[slot] = 0, 0.0
        return {
            protocol.SLOTS: {slot: {protocol.FRAME_OBS: self._observed(slot)}},
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


def _connect(host: str, port: int) -> EnvConnection:
    conn = EnvConnection(host, port, connect_deadline=2.0)
    conn.tasks({})  # a request the server answers at once, so this client holds its slot
    return conn


@contextmanager
def _clients(
    env: _ClonedScene, count: int | None = None, join_deadline: float = DEFAULT_JOIN_DEADLINE
) -> Iterator[list[EnvConnection]]:
    """``count`` connections, one per clone of ``env`` by default, opened in slot order."""
    with serve_env(env, join_deadline) as (host, port):
        conns = []
        try:
            for _ in range(env.num_slots if count is None else count):
                conns.append(_connect(host, port))
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
def test_a_client_that_leaves_its_episode_holds_nobody_back():
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env) as (first, second):
        _together(lambda: first.reset(None), lambda: second.reset(None))
        first.close()
        second.step(_action(0.0))
        second.reset(None)
    assert env.stepped == [{1}] and len(env.resets) == 2


@pytest.mark.timeout(60.0)
def test_a_slot_no_client_claims_sits_out_the_reset_after_the_join_deadline():
    """1 of 4 clients never connects; the other 3 reset when the deadline passes and run to the end."""
    env = _ClonedScene(ends_at=[2, 2, 2, 2])
    with _clients(env, count=3, join_deadline=0.5) as conns:
        _together(*[lambda conn=conn: conn.reset(None) for conn in conns])
        _together(*[lambda conn=conn: conn.step(_action(0.0)) for conn in conns])
        last = _together(*[lambda conn=conn: conn.step(_action(0.0)) for conn in conns])

    assert all(frame[protocol.FRAME_DONE] for frame in last)
    assert env.stepped == [{0, 1, 2}, {0, 1, 2}]


@pytest.mark.timeout(60.0)
def test_a_client_that_never_asks_for_a_reset_sits_out_after_the_join_deadline():
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env, join_deadline=0.5) as (first, _silent):
        first.reset(None)
        first.step(_action(0.0))
    assert env.stepped == [{0}]


@pytest.mark.timeout(60.0)
def test_a_refill_resets_an_ended_slot_while_its_neighbour_runs():
    env = _ClonedScene(ends_at=[1, 5])
    with _clients(env) as (first, second):
        _together(lambda: first.reset('scene'), lambda: second.reset('scene'))
        _together(lambda: first.step(_action(0.0)), lambda: second.step(_action(0.0)))
        refilled = first.reset('scene', 2, refill=True)
        stepped, neighbour = _together(lambda: first.step(_action(0.5)), lambda: second.step(_action(0.0)))

    assert _steps(refilled) == 0.0 and _steps(stepped) == 1.0 and _steps(neighbour) == 2.0
    assert stepped[protocol.FRAME_OBS][_GRIP] == pytest.approx(0.5)
    assert env.resets == ['scene'] and env.slot_resets == [('scene', 0)]
    assert env.stepped == [{0, 1}, {0, 1}]


def test_a_refill_of_a_slot_whose_episode_still_runs_is_refused():
    env = _ClonedScene(ends_at=[5, 5])
    with _clients(env) as (first, second):
        _together(lambda: first.reset('scene'), lambda: second.reset('scene'))
        with pytest.raises(RuntimeError, match='still runs'):
            first.reset('scene', 2, refill=True)

    assert env.slot_resets == []


@pytest.mark.timeout(60.0)
def test_a_new_client_refills_the_env_slot_that_a_client_left():
    env = _ClonedScene(ends_at=[1, 5])
    with serve_env(env) as (host, port):
        first, second = _connect(host, port), _connect(host, port)
        _together(lambda: first.reset('scene'), lambda: second.reset('scene'))
        _together(lambda: first.step(_action(0.0)), lambda: second.step(_action(0.0)))
        first.close()
        third = _connect(host, port)
        third.reset('scene', refill=True)
        stepped, _ = _together(lambda: third.step(_action(0.7)), lambda: second.step(_action(0.0)))
        third.close()
        second.close()

    assert stepped[protocol.FRAME_OBS][_GRIP] == pytest.approx(0.7)
    assert env.slot_resets == [('scene', 0)]


@pytest.mark.timeout(60.0)
def test_a_refill_finds_no_env_slot_while_every_slot_of_the_batch_is_held():
    env = _ClonedScene(ends_at=[5, 5, 5])
    with _clients(env) as (first, second, third):
        _together(lambda: first.reset('scene', 2), lambda: second.reset('scene', 2))
        with pytest.raises(RuntimeError, match='every env slot of the batch of 2 is held'):
            third.reset('scene', refill=True)
    assert env.slot_resets == []


@pytest.mark.timeout(60.0)
@pytest.mark.parametrize(
    ('token', 'batch_size', 'error'),
    [('another scene', None, 'the token of that batch'), ('scene', 1, 'the batch that runs has 2 slots')],
)
def test_a_refill_into_another_batch_is_refused(token, batch_size, error):
    env = _ClonedScene(ends_at=[1, 5])
    with _clients(env) as (first, second):
        _together(lambda: first.reset('scene'), lambda: second.reset('scene'))
        _together(lambda: first.step(_action(0.0)), lambda: second.step(_action(0.0)))
        with pytest.raises(RuntimeError, match=error):
            first.reset(token, batch_size, refill=True)
    assert env.slot_resets == []


@pytest.mark.timeout(60.0)
def test_a_refill_is_refused_while_a_slot_waits_to_join_the_next_batch():
    env = _ClonedScene(ends_at=[1, 5, 5])
    with serve_env(env) as (host, port), ThreadPoolExecutor(1) as pool:
        first, second, third = (_connect(host, port) for _ in range(3))
        _together(lambda: first.reset('scene', 2), lambda: second.reset('scene', 2))
        _together(lambda: first.step(_action(0.0)), lambda: second.step(_action(0.0)))
        joining = pool.submit(third.reset, 'scene', 1)
        assert _held_back(joining)
        with pytest.raises(RuntimeError, match='waits to join the next batch'):
            first.reset('scene', 2, refill=True)
        first.close()
        second.close()
        joining.result(timeout=10.0)
        third.close()


def _watch_resets(monkeypatch) -> Callable[..., None]:
    """A wait that returns once the server has taken ``count`` resets for ``token``: parked them, or run them."""
    taken: list[Any] = []
    changed = threading.Condition()
    take = server_module.EnvServer._take

    def recording_take(server: server_module.EnvServer, request: Any) -> None:
        take(server, request)
        if getattr(request, 'msg', {}).get(protocol.CMD) == protocol.Command.RESET.value:
            with changed:
                taken.append(request.msg[protocol.TOKEN])
                changed.notify_all()

    monkeypatch.setattr(server_module.EnvServer, '_take', recording_take)

    def wait_for(token: str, count: int = 1) -> None:
        with changed:
            took = changed.wait_for(lambda: taken.count(token) >= count, timeout=10.0)
            assert took, f'the server never took {count} resets {token!r}'

    return wait_for


@pytest.mark.timeout(60.0)
def test_a_slot_that_asks_inside_the_join_deadline_joins_the_reset(monkeypatch):
    taken = _watch_resets(monkeypatch)
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env, join_deadline=10.0) as (first, second), ThreadPoolExecutor(1) as pool:
        pending = pool.submit(first.reset, 'scene')
        taken('scene')

        assert _held_back(pending), 'the reset ran before the deadline, without the slow slot'
        second.reset('scene')
        pending.result(timeout=10.0)
        _together(lambda: first.step(_action(0.0)), lambda: second.step(_action(0.0)))
    assert env.stepped == [{0, 1}] and len(env.resets) == 1


def _test_clock(monkeypatch) -> list[float]:
    """Give the server a clock that only the test moves: the one value in the returned list."""
    now = [0.0]
    monkeypatch.setattr(server_module, 'time', SimpleNamespace(monotonic=lambda: now[0]))
    return now


def _move_clock_until_done(now: list[float], future: Future) -> None:
    while not future.done():
        now[0] += 2.0
        time.sleep(0.05)


@pytest.mark.timeout(60.0)
def test_a_batch_waits_the_join_deadline_only_for_its_first_reset(monkeypatch):
    """The slots that reset together are the batch; a later reset of the batch waits only for them.

    The test moves the clock only for the first reset, so a later reset that waited the deadline would never answer.
    """
    now = _test_clock(monkeypatch)
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env, count=1, join_deadline=1.0) as (conn,), ThreadPoolExecutor(1) as pool:
        first = pool.submit(conn.reset, None)
        _move_clock_until_done(now, first)
        first.result()
        pool.submit(conn.reset, None).result(timeout=10.0)
    assert len(env.resets) == 2


@pytest.mark.timeout(60.0)
def test_a_batch_whose_reset_failed_retries_without_the_join_deadline(monkeypatch):
    now = _test_clock(monkeypatch)
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env, count=1, join_deadline=1.0) as (conn,), ThreadPoolExecutor(1) as pool:
        failed = pool.submit(conn.reset, _FAILING_TOKEN)
        _move_clock_until_done(now, failed)
        with pytest.raises(RuntimeError, match='no scene for'):
            failed.result()
        pool.submit(conn.reset, 'known scene').result(timeout=10.0)
    assert env.resets == [_FAILING_TOKEN, 'known scene']


@pytest.mark.timeout(60.0)
def test_a_slot_whose_reset_failed_cannot_step():
    with _clients(_ClonedScene(ends_at=[9])) as (conn,):
        with pytest.raises(RuntimeError, match='no scene for'):
            conn.reset(_FAILING_TOKEN)
        with pytest.raises(RuntimeError, match='slot 0 stepped while reset failed'):
            conn.step(_action(0.0))


@pytest.mark.timeout(60.0)
def test_a_batch_resets_at_its_join_deadline_when_no_request_comes(monkeypatch):
    """The server wakes at the deadline, not at its next idle check, which this test moves far away."""
    monkeypatch.setattr(server_module, '_IDLE_CHECK_INTERVAL', 10.0)
    env = _ClonedScene(ends_at=[9, 9])
    with _clients(env, count=1, join_deadline=0.05) as (conn,), ThreadPoolExecutor(1) as pool:
        pool.submit(conn.reset, None).result(timeout=5.0)
    assert len(env.resets) == 1


@pytest.mark.timeout(60.0)
def test_the_join_deadline_counts_from_when_the_previous_batch_left(monkeypatch):
    """A slot that waited through a running batch does not start the next batch alone."""
    now = _test_clock(monkeypatch)
    taken = _watch_resets(monkeypatch)
    env = _ClonedScene(ends_at=[9, 9])
    with serve_env(env, join_deadline=1.0) as (host, port), ThreadPoolExecutor(1) as pool:
        first = _connect(host, port)
        _move_clock_until_done(now, pool.submit(first.reset, 'first batch'))
        late = _connect(host, port)
        joining = pool.submit(late.reset, 'second batch')
        taken('second batch')
        assert _held_back(joining), 'the late slot joined a batch that was already running'
        now[0] += 10.0  # the late slot waits past the deadline while the batch runs
        first.close()
        newcomer = _connect(host, port)
        newcomer.reset('second batch')
        joining.result(timeout=10.0)
        _together(lambda: late.step(_action(0.0)), lambda: newcomer.step(_action(0.0)))
        late.close()
        newcomer.close()
    assert env.resets == ['first batch', 'second batch']
    assert env.stepped == [{0, 1}]


def _connect_when_a_slot_frees(host: str, port: int) -> EnvConnection:
    """A connection that holds a slot, made as soon as the server frees one."""
    deadline = time.monotonic() + 10.0
    while True:
        conn = EnvConnection(host, port, connect_deadline=2.0)
        try:
            conn.tasks({})
            return conn
        except RuntimeError:
            conn.close()
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


@pytest.mark.timeout(60.0)
def test_a_reset_parked_by_a_client_that_disconnects_never_joins_a_batch(monkeypatch):
    """A client asks to join while a batch runs, then drops its socket; the next clients get its slot."""
    now = _test_clock(monkeypatch)
    env = _ClonedScene(ends_at=[9, 9])
    with serve_env(env, join_deadline=1.0) as (host, port), ThreadPoolExecutor(1) as pool:
        first = _connect(host, port)
        _move_clock_until_done(now, pool.submit(first.reset, 'first batch'))
        dropped = connect(f'ws://{host}:{port}/', compression=None)
        dropped.send(protocol.encode({protocol.CMD: protocol.Command.RESET.value, protocol.TOKEN: 'stale batch'}))
        dropped.close()
        newcomers = [_connect_when_a_slot_frees(host, port)]
        first.close()
        newcomers.append(_connect(host, port))
        _together(*[lambda conn=conn: conn.reset('second batch') for conn in newcomers])
        _together(*[lambda conn=conn: conn.step(_action(0.0)) for conn in newcomers])
        for conn in newcomers:
            conn.close()
    assert env.resets == ['first batch', 'second batch']
    assert env.stepped == [{0, 1}]


@pytest.mark.timeout(60.0)
def test_a_slot_that_sat_out_a_batch_joins_the_next_one(monkeypatch):
    """A late client waits while the batch runs, also through its later resets, and resets once the batch leaves."""
    taken = _watch_resets(monkeypatch)
    env = _ClonedScene(ends_at=[9, 9])
    with serve_env(env, join_deadline=0.5) as (host, port), ThreadPoolExecutor(1) as pool:
        first = _connect(host, port)
        first.reset('first batch')
        late = _connect(host, port)
        joining = pool.submit(late.reset, 'second batch')
        taken('second batch')
        first.step(_action(0.0))
        first.reset('first batch')

        assert _held_back(joining), 'the late slot joined a batch that was already running'
        first.close()
        frame = joining.result(timeout=10.0)
        late.step(_action(0.0))
        late.close()
    assert _steps(frame) == 0.0
    assert env.resets == ['first batch', 'first batch', 'second batch']
    assert env.stepped == [{0}, {0}]


@pytest.mark.timeout(60.0)
def test_a_batch_smaller_than_the_env_starts_at_once_on_as_many_clones():
    """The join deadline is the default 600 s, so a reset that waited for the other two clones would time out."""
    env = _ClonedScene(ends_at=[9, 9, 9, 9])
    with _clients(env, count=2) as conns:
        _together(*[lambda conn=conn: conn.reset('scene', batch_size=2) for conn in conns])
        _together(*[lambda conn=conn: conn.step(_action(0.0)) for conn in conns])
    assert env.built == [2]
    assert env.stepped == [{0, 1}]


@pytest.mark.timeout(60.0)
def test_the_slots_of_a_batch_drive_the_first_clones_in_slot_order():
    """The client on slot 0 never resets; the batch of the clients on slots 1 and 2 drives clones 0 and 1."""
    env = _ClonedScene(ends_at=[9, 9, 9])
    with _clients(env) as (_idle, first, second):
        _together(lambda: first.reset('scene', batch_size=2), lambda: second.reset('scene', batch_size=2))
        frames = _together(lambda: first.step(_action(0.25)), lambda: second.step(_action(0.75)))
    assert [frame[protocol.FRAME_OBS][_GRIP] for frame in frames] == [0.25, 0.75]
    assert env.built == [2] and env.stepped == [{0, 1}]


@pytest.mark.timeout(60.0)
def test_each_batch_builds_the_clones_its_own_size_asks_for():
    env = _ClonedScene(ends_at=[9, 9, 9])
    with serve_env(env) as (host, port):
        for size in (2, 3, 1):
            conns = [_connect(host, port) for _ in range(size)]
            _together(*[lambda conn=conn, size=size: conn.reset('scene', batch_size=size) for conn in conns])
            for conn in conns:
                conn.close()
    assert env.built == [2, 3, 1]


@pytest.mark.timeout(60.0)
def test_a_slot_past_the_batch_size_waits_for_the_next_batch(monkeypatch):
    taken = _watch_resets(monkeypatch)
    env = _ClonedScene(ends_at=[9, 9, 9])
    with _clients(env) as (first, second, third), ThreadPoolExecutor(2) as pool:
        first.reset('first batch', batch_size=1)
        joining = [pool.submit(conn.reset, 'second batch', batch_size=1) for conn in (second, third)]
        taken('second batch', count=2)
        first.close()
        joining[0].result(timeout=10.0)

        assert _held_back(joining[1]), 'a slot past the batch size of 1 joined the batch'
        second.close()
        joining[1].result(timeout=10.0)
    assert env.resets == ['first batch', 'second batch', 'second batch'] and env.built == [1, 1, 1]


@pytest.mark.timeout(60.0)
def test_a_reset_that_asks_for_another_batch_size_than_the_waiting_slots_is_refused(monkeypatch):
    taken = _watch_resets(monkeypatch)
    with _clients(_ClonedScene(ends_at=[9, 9, 9])) as (first, second, _), ThreadPoolExecutor(1) as pool:
        pending = pool.submit(first.reset, 'scene', batch_size=2)
        taken('scene')
        with pytest.raises(RuntimeError, match='the next batch has 2 slots, but this reset asks for a batch of 3'):
            second.reset('scene', batch_size=3)
        second.reset('scene', batch_size=2)
        pending.result(timeout=10.0)


@pytest.mark.timeout(60.0)
@pytest.mark.parametrize('size', [0, 3])
def test_a_batch_size_the_env_cannot_serve_is_refused(size):
    with _clients(_ClonedScene(ends_at=[9, 9]), count=1) as (conn,):
        with pytest.raises(RuntimeError, match=f'a batch of {size} slots is asked for, but this env serves 1 to 2'):
            conn.reset('scene', batch_size=size)


@pytest.mark.timeout(60.0)
@pytest.mark.parametrize('size', [1.0, True])
def test_a_batch_size_that_is_not_an_integer_is_refused_and_the_server_keeps_serving(size):
    with _clients(_ClonedScene(ends_at=[9, 9]), count=1) as (conn,):
        with pytest.raises(RuntimeError, match='a batch size is a whole number'):
            conn.reset('scene', batch_size=size)
        conn.reset('scene', batch_size=1)


@pytest.mark.timeout(60.0)
def test_a_slot_whose_client_left_is_taken_by_a_new_client():
    env = _ClonedScene(ends_at=[1])
    with serve_env(env) as (host, port):
        for batch in ('first batch', 'second batch'):
            conn = _connect(host, port)
            conn.reset(batch)
            assert conn.step(_action(0.0))[protocol.FRAME_DONE]
            conn.close()
    assert env.resets == ['first batch', 'second batch']


@pytest.mark.timeout(60.0)
def test_the_server_keeps_serving_after_every_slot_leaves():
    env = _ClonedScene(ends_at=[9, 9])
    with serve_env(env) as (host, port):
        for conn in [_connect(host, port), _connect(host, port)]:
            conn.close()
        again = EnvConnection(host, port, connect_deadline=2.0)
        try:
            assert again.tasks({}) == env.tasks({})
        finally:
            again.close()


@pytest.mark.timeout(60.0)
def test_slots_that_ask_for_different_resets_are_refused():
    """The clones share one scene, so one batch reset cannot serve two tokens."""
    with _clients(_ClonedScene(ends_at=[9, 9])) as (first, second):
        with pytest.raises(RuntimeError, match='all slots of one env server share one scene and one task'):
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


def _drive_one_proxy(
    host: str, port: int, grip: float, started: threading.Barrier, batch_size: int | None = None, refill: bool = False
) -> float:
    """Run one proxy in a World of its own through a reset and a few steps; return the grip it observed."""
    with pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_GripAdapter(grip), nullcontext((host, port)), batch_size, refill=refill)
        grip_rx = world.pair(proxy.observations[_GRIP])
        scheduler = world.start([proxy])
        started.wait(timeout=10.0)
        proxy.reset({})
        drive_scheduler(scheduler, steps=4)
        return grip_rx.value


@pytest.mark.timeout(60.0)
def test_two_proxies_drive_two_slots_of_one_env():
    """Two Worlds, as two evals would run them, each command and observe their own clone of one env."""
    with serve_env(_ClonedScene(ends_at=[99, 99])) as (host, port):
        started = threading.Barrier(2)
        grips = _together(
            lambda: _drive_one_proxy(host, port, 0.25, started), lambda: _drive_one_proxy(host, port, 0.75, started)
        )
    assert grips == [0.25, 0.75]


@pytest.mark.timeout(60.0)
def test_two_proxies_start_a_batch_of_two_on_an_env_of_four_clones():
    env = _ClonedScene(ends_at=[99, 99, 99, 99])
    with serve_env(env) as (host, port):
        started = threading.Barrier(2)
        grips = _together(
            lambda: _drive_one_proxy(host, port, 0.25, started, batch_size=2),
            lambda: _drive_one_proxy(host, port, 0.75, started, batch_size=2),
        )
    assert grips == [0.25, 0.75] and env.built == [2]


@pytest.mark.timeout(60.0)
def test_a_refill_proxy_resets_its_clone_alone_into_the_batch():
    env = _ClonedScene(ends_at=[99, 99])
    with serve_env(env, join_deadline=0.2) as (host, port):
        first = _connect(host, port)
        first.reset(None, 2)  # the deadline starts the batch with one of its two clones
        first.close()
        grip = _drive_one_proxy(host, port, 0.5, threading.Barrier(1), batch_size=2, refill=True)
    assert grip == 0.5 and env.resets == [None] and env.slot_resets == [(None, 0)]


def test_an_env_serves_one_slot_unless_it_says_otherwise():
    assert make_mujoco_env([]).num_slots == 1


@pytest.mark.parametrize('env_source', ['libero/env.py', 'molmo_spaces/env.py'])
def test_a_one_scene_benchmark_keeps_the_default_of_one_slot(env_source):
    """These envs import only in their own interpreters, so the check reads the class from the source."""
    tree = ast.parse((Path(__file__).parents[2] / env_source).read_text())
    (env_class,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and any(isinstance(b, ast.Name) and b.id == 'EnvProtocol' for b in node.bases)
    ]
    members = {node.name for node in env_class.body if isinstance(node, ast.FunctionDef)}
    assert 'num_slots' not in members, f'{env_class.name} serves more than the one slot its single scene holds'


class _SlowScene(_ClonedScene):
    """A scene whose reset takes `load_s` seconds, as a native scene load does."""

    def __init__(self, load_s: float):
        super().__init__([1])
        self._load_s = load_s

    def reset(self, token: Any, slots: int | None = None) -> dict[str, Any]:
        time.sleep(self._load_s)
        return super().reset(token, slots)


@pytest.mark.timeout(10.0)
def test_a_scene_load_past_the_websockets_ping_timeout_keeps_its_client(monkeypatch):
    """A native scene load blocks the server's heartbeat, so the server states its own ping timeout. Here the
    websockets default is 0.05 s, and the client answers no ping."""
    monkeypatch.setattr(server_module, 'serve', partial(websocket_serve, ping_interval=0.01, ping_timeout=0.05))
    with serve_env(_SlowScene(load_s=0.5)) as (host, port):
        conn = EnvConnection(host, port)
        receive_frame = conn._ws.protocol.recv_frame
        monkeypatch.setattr(
            conn._ws.protocol, 'recv_frame', lambda frame: None if frame.opcode == OP_PING else receive_frame(frame)
        )
        try:
            assert protocol.FRAME_OBS in conn.reset(_SCENE)
        finally:
            conn.close()
