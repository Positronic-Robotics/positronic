from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic import geom
from positronic.dataset import DatasetWriter, EpisodeWriter
from positronic.dataset.ds_writer_agent import (
    DsWriterAgent,
    DsWriterCommand,
    DsWriterCommandType,
    Serializers,
    TimeMode,
)
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.drivers import roboarm
from positronic.drivers.roboarm import command as rcmd
from positronic.tests.testing_coutils import run_scripted_agent


@pytest.fixture
def clock():
    return MockClock()


@pytest.fixture
def world(clock):
    with pimm.World(clock=clock) as w:
        yield w


class FakeEpisodeWriter(EpisodeWriter[Any]):
    def __init__(self) -> None:
        self.statics: dict[str, Any] = {}
        self.appends: list[tuple[str, Any, int]] = []
        self.exited = False
        self.aborted = False
        self.meta_calls: dict[str, Sequence[str] | None] = {}

    def append(self, signal_name: str, data: Any, ts_ns: int) -> None:
        self.appends.append((signal_name, data, int(ts_ns)))

    def set_signal_meta(self, signal_name: str, *, names: Sequence[str] | None = None) -> None:
        if signal_name in self.meta_calls and self.meta_calls[signal_name] != names:
            raise AssertionError('set_signal_meta called with different names')
        self.meta_calls[signal_name] = names

    def set_static(self, name: str, data: Any) -> None:
        self.statics[name] = data

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def abort(self) -> None:
        self.aborted = True


class FakeDatasetWriter(DatasetWriter):
    def __init__(self) -> None:
        self.created: list[FakeEpisodeWriter] = []

    def new_episode(self) -> FakeEpisodeWriter:
        w = FakeEpisodeWriter()
        self.created.append(w)
        return w

    def __exit__(self, exc_type, exc, tb) -> None:
        return False


def build_agent_with_pipes(
    signals_spec: dict[str, Any], ds_writer: DatasetWriter, world: pimm.World, *, time_mode: TimeMode = TimeMode.CLOCK
):
    """Build agent with given signals spec and wire it using ``world.pair``.

    - signals_spec maps input name -> serializer (or None for pass-through).
    - A serializer can:
        * return a transformed value (recorded under the same name),
        * return a dict mapping suffixes to values (recorded as name+suffix),
        * return None to drop the sample (not recorded at all).
    Returns (agent, cmd_emitter, emitters_by_name).
    """
    agent = DsWriterAgent(ds_writer, signals_spec, time_mode=time_mode)
    emitters: dict[str, pimm.SignalEmitter[Any]] = {name: world.pair(agent.inputs[name]) for name in signals_spec}

    cmd_em = world.pair(agent.command)

    return agent, cmd_em, emitters


def test_start_stop_happy_path(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None, 'b': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE, {'user': 'alice'})), 0.001),
        (partial(emitters['a'].emit, 1), 0.001),
        (partial(emitters['b'].emit, 2), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE, {'done': True})), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.statics.get('user') == 'alice'
    assert [(s, v) for (s, v, _) in w.appends] == [('a', 1), ('b', 2)]
    assert w.exited is True
    assert w.statics.get('done') is True


def test_episode_finalizes_when_run_stops(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 42), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert [(s, v) for (s, v, _) in w.appends] == [('a', 42)]
    assert w.exited is True


def test_ignore_duplicate_commands_and_empty_stop(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'x': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.exited is True


def test_abort_flow_then_restart(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'s': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['s'].emit, 10), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.ABORT_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['s'].emit, 11), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    assert len(ds.created) == 2
    w1, w2 = ds.created[0], ds.created[1]
    assert w1.aborted is True and w1.exited is True
    assert [(s, v) for (s, v, _) in w2.appends] == [('s', 11)]


def test_appends_only_on_updates_and_timestamps_from_clock(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 1), 0.001),
        (None, 0.001),
        (partial(emitters['a'].emit, 2), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    assert len(w.appends) == 2
    assert w.appends[1][2] > w.appends[0][2]


def test_time_mode_message_uses_signal_timestamp(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world, time_mode=TimeMode.MESSAGE)

    ts_first = 123_000_000
    ts_second = 456_000_000

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 1, ts=ts_first), 0.001),
        (partial(emitters['a'].emit, 2, ts=ts_second), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _) in w.appends] == [('a', 1), ('a', 2)]
    assert [ts for (_, _, ts) in w.appends] == [ts_first, ts_second]


def test_serializer_names_declared_on_start(world, clock):
    ds = FakeDatasetWriter()

    def ser(x):
        return x

    ser.names = 'feat'

    agent, cmd_em, _ = build_agent_with_pipes({'a': ser}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    writer = ds.created[-1]
    assert writer.meta_calls.get('a') == ['feat']
    assert writer.appends == []


def test_dict_serializer_value_with_names(world, clock):
    ds = FakeDatasetWriter()

    def ser(x):
        return {'.x': np.array([1.0, 2.0])}

    ser.names = {'.x': ['a', 'b']}

    agent, cmd_em, emitters = build_agent_with_pipes({'sig': ser}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['sig'].emit, np.array([1.0, 2.0])), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    writer = ds.created[-1]
    assert writer.meta_calls.get('sig.x') == ['a', 'b']
    assert len(writer.appends) == 1
    name, arr, _ = writer.appends[0]
    assert name == 'sig.x'
    np.testing.assert_allclose(arr, np.array([1.0, 2.0]))


def test_integration_with_local_dataset_writer(tmp_path, world, clock):
    with LocalDatasetWriter(tmp_path) as writer:
        agent, cmd_em, emitters = build_agent_with_pipes({'a': None, 'b': None}, writer, world)

        script = [
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE, {'task': 'unit'})), 0.001),
            (partial(emitters['a'].emit, 10), 0.001),
            (partial(emitters['b'].emit, 20), 0.001),
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE, {'ok': True})), 0.001),
        ]

        run_scripted_agent(agent, script, world=world, clock=clock)

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]
    assert set(ep.keys) == {'a', 'b', 'task', 'ok'}
    a = ep['a']
    b = ep['b']
    assert len(a) == 1 and len(b) == 1
    assert a[0][0] == 10 and b[0][0] == 20


def test_inputs_mapping_is_immutable(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    # Cannot add new key
    with pytest.raises(TypeError):
        agent.inputs['b'] = pimm.NoOpReceiver()
    # Can modify existing key's value
    new_em, new_rd = world.local_pipe(maxsize=8)
    agent.inputs['a'] = new_rd
    assert agent.inputs['a'] is new_rd
    # Deleting keys is not allowed
    with pytest.raises(TypeError):
        del agent.inputs['a']


def test_serializer_scalar_transform(world, clock):
    ds = FakeDatasetWriter()

    # Serializer doubles the value
    def double(x):
        return x * 2

    agent, cmd_em, emitters = build_agent_with_pipes({'x': double}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['x'].emit, 3), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _) in w.appends] == [('x', 6)]


def test_serializer_dict_expansion(world, clock):
    ds = FakeDatasetWriter()

    # Serializer splits into two signals:
    # - empty key keeps base name ("img")
    # - non-empty keys are treated as suffixes appended to base name (e.g., ".extra")
    def expand(v):
        return {'': v, '.extra': v + 1}

    agent, cmd_em, emitters = build_agent_with_pipes({'img': expand}, ds, world)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['img'].emit(10), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    names_and_vals = [(s, v) for (s, v, _) in w.appends]
    assert ('img', 10) in names_and_vals
    assert ('img.extra', 11) in names_and_vals


def test_serializer_none_drops_sample(world, clock):
    ds = FakeDatasetWriter()

    # Serializer drops negative values by returning None (sample is not recorded)
    def drop_negative(v):
        return None if v < 0 else v

    agent, cmd_em, emitters = build_agent_with_pipes({'x': drop_negative}, ds, world)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['x'].emit(3), 0.001),
        (lambda: emitters['x'].emit(-1), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    # Only the positive value should be recorded
    assert [(s, v) for (s, v, _) in w.appends] == [('x', 3)]


def test_transform_3d_serializer(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'pose': Serializers.transform_3d}, ds, world)

    t = np.array([0.1, -0.2, 0.3])
    q = geom.Rotation.identity
    pose = geom.Transform3D(translation=t, rotation=q)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['pose'].emit(pose), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    names_vals = [(s, v) for (s, v, _) in w.appends]
    assert len(names_vals) == 1 and names_vals[0][0] == 'pose'
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


def test_robot_state_serializer_drops_reset_and_emits_components(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'robot_state': Serializers.robot_state}, ds, world)

    q = np.arange(7, dtype=np.float32)
    dq = np.arange(7, dtype=np.float32) + 10
    t = np.array([0.0, 0.1, 0.2])
    pose = geom.Transform3D(translation=t, rotation=geom.Rotation.identity)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['robot_state'].emit(_FakeState(q, dq, pose, roboarm.RobotStatus.RESETTING)), 0.001),
        (lambda: emitters['robot_state'].emit(_FakeState(q, dq, pose, roboarm.RobotStatus.AVAILABLE)), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    items = {name: val for (name, val, _) in w.appends}
    # Should not contain any data from RESETTING
    assert set(items.keys()) == {'robot_state.q', 'robot_state.dq', 'robot_state.ee_pose'}
    np.testing.assert_allclose(items['robot_state.q'], q)
    np.testing.assert_allclose(items['robot_state.dq'], dq)
    np.testing.assert_allclose(items['robot_state.ee_pose'], np.concatenate([t, geom.Rotation.identity.as_quat]))


def test_robot_command_serializer_variants(world, clock):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'cmd': Serializers.robot_command}, ds, world)

    pose = geom.Transform3D(translation=np.array([0.2, 0.0, -0.1]), rotation=geom.Rotation.identity)
    joints = np.arange(7, dtype=np.float32) * 0.1

    script = [
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.CartesianMove(pose)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.JointMove(joints)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.Reset()), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world, clock=clock)

    w = ds.created[-1]
    items = {name: val for (name, val, _) in w.appends}
    np.testing.assert_allclose(items['cmd.pose'], np.concatenate([pose.translation, pose.rotation.as_quat]))
    np.testing.assert_allclose(items['cmd.joints'], joints)
    assert items['cmd.reset'] == 1
