from typing import Any, List, Tuple

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic import geom
from positronic.dataset.core import DatasetWriter, EpisodeWriter
from positronic.dataset.ds_writer_agent import (
    DsWriterAgent,
    DsWriterCommand,
    DsWriterCommandType,
    Serializers,
)
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.drivers.roboarm import command as rcmd


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


def build_agent_with_pipes(signals_spec: dict[str, Any], ds_writer: DatasetWriter, world: pimm.World):
    """Build agent with given signals spec and wire local pipes.

    - signals_spec maps input name -> serializer (or None for pass-through).
    - A serializer can:
        * return a transformed value (recorded under the same name),
        * return a dict mapping suffixes to values (recorded as name+suffix),
        * return None to drop the sample (not recorded at all).
    Returns (agent, cmd_emitter, emitters_by_name).
    """
    agent = DsWriterAgent(ds_writer, signals_spec)
    emitters: dict[str, pimm.SignalEmitter[Any]] = {}
    for name in signals_spec.keys():
        emitters[name], agent.inputs[name] = world.local_pipe(maxsize=8)

    cmd_em, agent.command = world.local_pipe()

    return agent, cmd_em, emitters


def test_start_stop_happy_path(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({"a": None, "b": None}, ds, world)

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
    agent, cmd_em, emitters = build_agent_with_pipes({"x": None}, ds, world)

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
    agent, cmd_em, emitters = build_agent_with_pipes({"s": None}, ds, world)

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
    agent, cmd_em, emitters = build_agent_with_pipes({"a": None}, ds, world)

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
    agent, cmd_em, emitters = build_agent_with_pipes({"a": None, "b": None}, writer, world)

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


def test_inputs_mapping_is_immutable(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({"a": None}, ds, world)

    # Cannot add new key
    with pytest.raises(TypeError):
        agent.inputs["b"] = pimm.NoOpReader()
    # Can modify existing key's value
    new_em, new_rd = world.local_pipe(maxsize=8)
    agent.inputs["a"] = new_rd
    assert agent.inputs["a"] is new_rd
    # Deleting keys is not allowed
    with pytest.raises(TypeError):
        del agent.inputs["a"]


def test_serializer_scalar_transform(world, clock, run_agent):
    ds = FakeDatasetWriter()

    # Serializer doubles the value
    def double(x):
        return x * 2

    agent, cmd_em, emitters = build_agent_with_pipes({"x": double}, ds, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        emitters["x"].emit(3)
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _) in w.appends] == [("x", 6)]


def test_serializer_dict_expansion(world, clock, run_agent):
    ds = FakeDatasetWriter()

    # Serializer splits into two signals:
    # - empty key keeps base name ("img")
    # - non-empty keys are treated as suffixes appended to base name (e.g., ".extra")
    def expand(v):
        return {"": v, ".extra": v + 1}

    agent, cmd_em, emitters = build_agent_with_pipes({"img": expand}, ds, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        emitters["img"].emit(10)
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    w = ds.created[-1]
    names_and_vals = [(s, v) for (s, v, _) in w.appends]
    assert ("img", 10) in names_and_vals
    assert ("img.extra", 11) in names_and_vals


def test_serializer_none_drops_sample(world, clock, run_agent):
    ds = FakeDatasetWriter()

    # Serializer drops negative values by returning None (sample is not recorded)
    def drop_negative(v):
        return None if v < 0 else v

    agent, cmd_em, emitters = build_agent_with_pipes({"x": drop_negative}, ds, world)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        emitters["x"].emit(3)    # kept
        yield pimm.Sleep(0.001)
        emitters["x"].emit(-1)   # dropped
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    w = ds.created[-1]
    # Only the positive value should be recorded
    assert [(s, v) for (s, v, _) in w.appends] == [("x", 3)]


def test_transform_3d_serializer(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({"pose": Serializers.transform_3d}, ds, world)

    t = np.array([0.1, -0.2, 0.3])
    q = geom.Rotation.identity
    pose = geom.Transform3D(translation=t, rotation=q)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        emitters["pose"].emit(pose)
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    w = ds.created[-1]
    names_vals = [(s, v) for (s, v, _) in w.appends]
    assert len(names_vals) == 1 and names_vals[0][0] == "pose"
    np.testing.assert_allclose(names_vals[0][1][:3], t)
    np.testing.assert_allclose(names_vals[0][1][3:], q.as_quat)


class _FakeState(roboarm.State):
    def __init__(self, q, dq, ee_pose, status):
        self._q = q
        self._dq = dq
        self._ee = ee_pose
        self._status = status

    @property
    def q(self):
        return self._q

    @property
    def dq(self):
        return self._dq

    @property
    def ee_pose(self):
        return self._ee

    @property
    def status(self):
        return self._status


def test_robot_state_serializer_drops_reset_and_emits_components(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({"robot_state": Serializers.robot_state}, ds, world)

    q = np.arange(7, dtype=np.float32)
    dq = np.arange(7, dtype=np.float32) + 10
    t = np.array([0.0, 0.1, 0.2])
    pose = geom.Transform3D(translation=t, rotation=geom.Rotation.identity)

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        # First, RESETTING -> should be dropped
        emitters["robot_state"].emit(_FakeState(q, dq, pose, roboarm.RobotStatus.RESETTING))
        yield pimm.Sleep(0.001)
        # Then, AVAILABLE -> should emit .q, .dq, .ee_pose
        emitters["robot_state"].emit(_FakeState(q, dq, pose, roboarm.RobotStatus.AVAILABLE))
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    w = ds.created[-1]
    items = {name: val for (name, val, _) in w.appends}
    # Should not contain any data from RESETTING
    assert set(items.keys()) == {"robot_state.q", "robot_state.dq", "robot_state.ee_pose"}
    np.testing.assert_allclose(items["robot_state.q"], q)
    np.testing.assert_allclose(items["robot_state.dq"], dq)
    np.testing.assert_allclose(items["robot_state.ee_pose"], np.concatenate([t, geom.Rotation.identity.as_quat]))


def test_robot_command_serializer_variants(world, clock, run_agent):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({"cmd": Serializers.robot_command}, ds, world)

    pose = geom.Transform3D(translation=np.array([0.2, 0.0, -0.1]), rotation=geom.Rotation.identity)
    joints = np.arange(7, dtype=np.float32) * 0.1

    def driver(stop_reader, clk):
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE))
        yield pimm.Sleep(0.001)
        # Cartesian move
        emitters["cmd"].emit(rcmd.CartesianMove(pose))
        yield pimm.Sleep(0.001)
        # Joint move
        emitters["cmd"].emit(rcmd.JointMove(joints))
        yield pimm.Sleep(0.001)
        # Reset
        emitters["cmd"].emit(rcmd.Reset())
        yield pimm.Sleep(0.001)
        cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
        yield pimm.Sleep(0.001)

    run_agent(agent, driver)

    w = ds.created[-1]
    items = {name: val for (name, val, _) in w.appends}
    np.testing.assert_allclose(items["cmd.pose"], np.concatenate([pose.translation, pose.rotation.as_quat]))
    np.testing.assert_allclose(items["cmd.joints"], joints)
    assert items["cmd.reset"] == 1
