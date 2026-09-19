"""An env server that steps several slots at once, and what a frozen slot does to the wire.

``_ClonedScene`` holds each clone's own step count and FREEZES a clone that ends: it stops advancing, keeps
its place in every answer, and reports the verdict it ended on, the way RoboLab's own env does instead of
re-rolling a terminated clone. ``positronic/simulator/robolab/validate.py`` runs the same shape against the
real benchmark on a RoboLab box.
"""

from contextlib import nullcontext
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

_SCENE = 'cloned_scene'
_STEPS = 'steps'  # the one observation this fixture's env publishes and its adapter reads
_NO_COMMAND = {protocol.ACTION_COMMAND: {protocol.COMMAND_TYPE: protocol.HOLD}, protocol.ACTION_GRIP: 0.0}


class _ClonedScene(EnvProtocol):
    """``ends_at`` clones of one scene: clone ``i`` ends on step ``ends_at[i]`` with verdict ``succeeds[i]``.

    Each clone observes its own step count, so a clone that froze is the one whose count stopped moving.
    """

    def __init__(self, ends_at: list[int], succeeds: list[bool] | None = None, control_dt: float = 0.1):
        self._ends_at = ends_at
        self._succeeds = succeeds if succeeds is not None else [False] * len(ends_at)
        self._control_dt = control_dt
        self._steps = [0] * len(ends_at)
        self._frozen = [False] * len(ends_at)

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        return [{'name': _SCENE}]

    def reset(self, token: Any) -> dict[str, Any]:
        self._steps = [0] * len(self._ends_at)
        self._frozen = [False] * len(self._ends_at)
        slots = [{protocol.FRAME_OBS: obs} for obs in self._observed()]
        return {
            protocol.SLOTS: slots,
            protocol.FRAME_META: {},
            protocol.FRAME_ROBOT_META: {},
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def step(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        if len(actions) != len(self._ends_at):
            raise ValueError(f'this env serves {len(self._ends_at)} slots; {len(actions)} actions arrived')
        for slot, ends_at in enumerate(self._ends_at):
            if self._frozen[slot]:
                continue  # a frozen clone is stepped with a zeroed action and holds its final state
            self._steps[slot] += 1
            self._frozen[slot] = self._steps[slot] >= ends_at
        return {
            protocol.SLOTS: [
                {
                    protocol.FRAME_OBS: obs,
                    protocol.FRAME_DONE: self._frozen[slot],
                    protocol.FRAME_SUCCESS: self._frozen[slot] and self._succeeds[slot],
                }
                for slot, obs in enumerate(self._observed())
            ],
            protocol.FRAME_CONTROL_DT: self._control_dt,
        }

    def _observed(self) -> list[dict[str, Any]]:
        return [{_STEPS: np.full(2, steps, dtype=np.float64)} for steps in self._steps]

    def close(self) -> None:
        pass


def _steps_of(frame: dict[str, Any]) -> list[float]:
    """Each slot's observed step count, in slot order."""
    return [float(slot[protocol.FRAME_OBS][_STEPS][0]) for slot in frame[protocol.SLOTS]]


def _field_of(frame: dict[str, Any], field: str) -> list[Any]:
    """One field of every slot, in slot order."""
    return [slot[field] for slot in frame[protocol.SLOTS]]


@pytest.mark.timeout(60.0)
def test_a_single_slot_env_answers_one_entry_per_per_slot_field():
    """The default is one clone, and ``slots`` then holds exactly one entry."""
    with serve_env(_ClonedScene(ends_at=[3])) as (host, port):
        conn = EnvConnection(host, port)
        try:
            reset = conn.reset(None)
            assert len(reset[protocol.SLOTS]) == 1
            step = conn.step([_NO_COMMAND])
            assert _steps_of(step) == [1.0]
            assert _field_of(step, protocol.FRAME_DONE) == [False]
            assert _field_of(step, protocol.FRAME_SUCCESS) == [False]
        finally:
            conn.close()


@pytest.mark.timeout(60.0)
def test_a_batch_answers_one_entry_per_slot():
    """Four clones step inside one request and answer four observations, in slot order."""
    with serve_env(_ClonedScene(ends_at=[9, 9, 9, 9])) as (host, port):
        conn = EnvConnection(host, port)
        try:
            assert len(conn.reset(None)[protocol.SLOTS]) == 4
            frame = conn.step([_NO_COMMAND] * 4)
            assert _steps_of(frame) == [1.0] * 4
            assert _field_of(frame, protocol.FRAME_DONE) == [False] * 4
        finally:
            conn.close()


@pytest.mark.timeout(60.0)
def test_a_slot_that_ends_early_leaves_the_others_running():
    """The freeze path: the ended clone holds its final state and its verdict while its neighbours advance.

    The reads never narrow — an answer carries every slot for as long as the batch runs — so a client reading
    slot 3 after slot 1 ended still reads slot 3.
    """
    with serve_env(_ClonedScene(ends_at=[5, 2, 5, 5], succeeds=[False, True, False, False])) as (host, port):
        conn = EnvConnection(host, port)
        try:
            conn.reset(None)
            frame = conn.step([_NO_COMMAND] * 4)
            for _ in range(3):
                frame = conn.step([_NO_COMMAND] * 4)

            assert _steps_of(frame) == [4.0, 2.0, 4.0, 4.0], 'the frozen slot kept advancing, or froze a neighbour'
            assert _field_of(frame, protocol.FRAME_DONE) == [False, True, False, False]
            assert _field_of(frame, protocol.FRAME_SUCCESS) == [False, True, False, False]
        finally:
            conn.close()


@pytest.mark.timeout(60.0)
def test_a_batch_runs_until_its_slowest_slot_ends():
    """Every clone reports its own verdict, and the batch is over only once all of them have."""
    with serve_env(_ClonedScene(ends_at=[1, 3], succeeds=[True, False])) as (host, port):
        conn = EnvConnection(host, port)
        try:
            conn.reset(None)
            dones = [_field_of(conn.step([_NO_COMMAND] * 2), protocol.FRAME_DONE) for _ in range(3)]

            assert dones == [[True, False], [True, False], [True, True]]
        finally:
            conn.close()


@pytest.mark.timeout(60.0)
def test_an_env_refuses_a_step_that_is_not_as_wide_as_its_batch():
    """A short action list would leave clones un-commanded, so it is refused rather than padded."""
    with serve_env(_ClonedScene(ends_at=[9, 9, 9])) as (host, port):
        conn = EnvConnection(host, port)
        try:
            conn.reset(None)
            with pytest.raises(RuntimeError, match='3 slots; 2 actions arrived'):
                conn.step([_NO_COMMAND] * 2)
        finally:
            conn.close()


class _StepCountAdapter(EnvAdapter):
    """The smallest adapter that lets the proxy run: it commands nothing and reports the step count."""

    def task_params(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{eval_keys.TASK: record['name']} for record in records]

    def reset_token(self, params: dict[str, Any]) -> Any:
        return None

    def action(self, commands: dict[str, pimm.Message]) -> dict[str, Any]:
        return _NO_COMMAND

    def observations(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {_STEPS: raw_obs[_STEPS]}

    def privileged(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        return {}

    def terminal(self, result: dict[str, Any]) -> dict[str, Any] | None:
        return {eval_keys.SUCCESS: result[protocol.FRAME_SUCCESS]} if result[protocol.FRAME_DONE] else None


@pytest.mark.timeout(60.0)
def test_the_proxy_drives_one_slot_and_reads_it():
    """One clone is what a World runs, and the proxy emits that clone's observation unwrapped."""
    with serve_env(_ClonedScene(ends_at=[9])) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_StepCountAdapter(), nullcontext((host, port)))
        steps_rx = world.pair(proxy.observations[_STEPS])
        world.start([proxy])

        proxy.reset({})

        emitted = steps_rx.read()
        assert emitted is not None and list(emitted.data) == [0.0, 0.0]


@pytest.mark.timeout(60.0)
def test_the_proxy_refuses_a_server_wider_than_one_slot():
    """A World records one episode, so driving a batch from here would drop every clone but one."""
    with serve_env(_ClonedScene(ends_at=[9, 9])) as (host, port), pimm.World(virtual_time=True) as world:
        proxy = RemoteEnvControlSystem(_StepCountAdapter(), nullcontext((host, port)))
        world.pair(proxy.observations[_STEPS])
        world.start([proxy])

        with pytest.raises(ValueError, match='serves 2 slots'):
            proxy.reset({})
