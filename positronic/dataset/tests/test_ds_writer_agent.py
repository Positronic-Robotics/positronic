from typing import Any, List, Tuple

import pytest

import pimm
from pimm.testing import MockClock
from positronic.dataset.ds_writer_agent import (
    DsWriterAgent,
    DsWriterCommand,
    DsWriterCommandType,
)
from positronic.dataset.core import DatasetWriter, EpisodeWriter
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter


@pytest.fixture
def clock():
    return MockClock()


@pytest.fixture
def world(clock):
    with pimm.World(clock=clock) as w:
        yield w


@pytest.fixture
def run_interleaved(clock, world):
    def _run(*loops, steps: int = 50):
        it = world.interleave(*loops)
        for _ in range(steps):
            try:
                sleep = next(it)
            except StopIteration:
                break
            clock.advance(sleep.seconds)
    return _run


@pytest.fixture
def run_agent(run_interleaved):
    """Helper to run an agent alongside a provided driver control loop.

    Usage: run_agent(agent, driver, steps=...)
    """
    def _run_agent(agent, driver, *, steps: int = 50):
        return run_interleaved(lambda s, c: agent.run(s, c), driver, steps=steps)

    return _run_agent


class FakeEpisodeWriter(EpisodeWriter[Any]):
    def __init__(self) -> None:
        self.statics: dict[str, Any] = {}
        self.appends: List[Tuple[str, Any, int]] = []
        self.exited = False
        self.aborted = False

    def append(self, signal_name: str, data: Any, ts_ns: int) -> None:
        self.appends.append((signal_name, data, int(ts_ns)))

    def set_static(self, name: str, data: Any) -> None:
        self.statics[name] = data

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def abort(self) -> None:
        self.aborted = True


class FakeDatasetWriter(DatasetWriter):
    def __init__(self) -> None:
        self.created: List[FakeEpisodeWriter] = []

    def new_episode(self) -> FakeEpisodeWriter:
        w = FakeEpisodeWriter()
        self.created.append(w)
        return w


def build_agent_with_pipes(signal_names: list[str], ds_writer: DatasetWriter, world: pimm.World):
    agent = DsWriterAgent(ds_writer, signal_names)
    # command channel
    cmd_em, cmd_rd = world.local_pipe()
    agent.command = cmd_rd  # type: ignore

    # inputs channels, keep emitters for tests
    emitters = {}
    readers = []
    for name in signal_names:
        em, rd = world.local_pipe(maxsize=8)
        emitters[name] = em
        readers.append(rd)
    agent.inputs = agent.inputs.__class__(*readers)  # replace namedtuple instance

    return agent, cmd_em, emitters


def test_start_stop_happy_path(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes(["a", "b"], ds, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE, {"user": "alice"}))
        yield pimm.Sleep(0.001)
        emitters["a"].emit(1)
        yield pimm.Sleep(0.001)
        emitters["b"].emit(2)
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE, {"done": True}))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.statics.get("user") == "alice"
    assert [(s, v) for (s, v, _) in w.appends] == [("a", 1), ("b", 2)]
    assert w.exited is True
    assert w.statics.get("done") is True


def test_ignore_duplicate_commands_and_empty_stop(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes(["x"], ds, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))  # ignored
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))  # ignored
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.exited is True


def test_abort_flow_then_restart(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes(["s"], ds, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        emitters["s"].emit(10)
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.ABORT_EPISODE))
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        emitters["s"].emit(11)
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    assert len(ds.created) == 2
    w1, w2 = ds.created[0], ds.created[1]
    assert w1.aborted is True and w1.exited is True
    assert [(s, v) for (s, v, _) in w2.appends] == [("s", 11)]


def test_appends_only_on_updates_and_timestamps_from_clock(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes(["a"], ds, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        emitters["a"].emit(1)
        # Allow agent to process first update
        yield pimm.Sleep(0.001)
        # No new update -> nothing appended
        yield pimm.Sleep(0.001)
        emitters["a"].emit(2)
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    w = ds.created[-1]
    assert len(w.appends) == 2
    assert w.appends[1][2] > w.appends[0][2]


def test_integration_with_local_dataset_writer(tmp_path, world, clock, run_agent):
    writer = LocalDatasetWriter(tmp_path)
    agent, cmd_em, emitters = build_agent_with_pipes(["a", "b"], writer, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE, {"task": "unit"}))
        yield pimm.Sleep(0.001)
        emitters["a"].emit(10)
        yield pimm.Sleep(0.001)
        emitters["b"].emit(20)
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE, {"ok": True}))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]
    assert set(ep.keys) == {"a", "b", "task", "ok"}
    a = ep["a"]
    b = ep["b"]
    assert len(a) == 1 and len(b) == 1
    assert a[0][0] == 10 and b[0][0] == 20
