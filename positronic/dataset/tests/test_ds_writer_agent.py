import pickle
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest

import pimm
from positronic import geom, keys, telemetry, telemetry_keys
from positronic.dataset import DatasetWriter, EpisodeWriter
from positronic.dataset.ds_writer_agent import DatasetFactory, DsWriterAgent, DsWriterCommand, TimeMode
from positronic.dataset.local_dataset import LocalDataset, LocalDatasetWriter
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm import command as rcmd
from positronic.drivers.roboarm.tests.fakes import FakeRobotState
from positronic.tests.testing_coutils import run_scripted_agent

# Where a scripted episode records; a fake dataset opens the same one whatever the path.
OUTPUT_PATH = Path('dataset')


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


class FakeEpisodeWriter(EpisodeWriter[Any]):
    def __init__(self) -> None:
        self.statics: dict[str, Any] = {}
        self.appends: list[tuple[str, Any, int, dict[str, int] | None]] = []
        self.exited = False
        self.aborted = False

    def append(self, signal_name: str, data: Any, ts_ns: int, extra_ts: dict[str, int] | None = None) -> None:
        self.appends.append((signal_name, data, int(ts_ns), extra_ts))

    def set_static(self, name: str, data: Any) -> None:
        self.statics[name] = data

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def abort(self) -> None:
        self.aborted = True


class FakeDatasetWriter(DatasetWriter):
    """One in-memory dataset, and the factory that opens it: it answers every path with itself."""

    def __init__(self) -> None:
        self.created: list[FakeEpisodeWriter] = []

    def __call__(self, output_path: Path) -> 'FakeDatasetWriter':
        return self

    def new_episode(self) -> FakeEpisodeWriter:
        w = FakeEpisodeWriter()
        self.created.append(w)
        return w

    def __exit__(self, exc_type, exc, tb) -> None:
        return False


def build_agent_with_pipes(
    signals_spec: dict[str, Any],
    dataset_factory: DatasetFactory,
    world: pimm.World,
    *,
    time_mode: TimeMode = TimeMode.CLOCK,
):
    """Build agent with given signals spec and wire it using ``world.pair``.

    - signals_spec maps input name -> serializer (or None for pass-through).
    - A serializer can:
        * return a transformed value (recorded under the same name),
        * return a dict mapping suffixes to values (recorded as name+suffix),
        * return None to drop the sample (not recorded at all).
    Returns (agent, cmd_emitter, emitters_by_name).
    """
    agent = DsWriterAgent(dataset_factory, time_mode=time_mode)
    for name, serializer in signals_spec.items():
        agent.add_signal(name, serializer)
    emitters: dict[str, pimm.SignalEmitter[Any]] = {name: world.pair(agent.inputs[name]) for name in signals_spec}

    cmd_em = world.pair(agent.command)

    return agent, cmd_em, emitters


def test_start_stop_happy_path(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None, 'b': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH, {'user': 'alice'})), 0.001),
        (partial(emitters['a'].emit, 1), 0.001),
        (partial(emitters['b'].emit, 2), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP({'done': True})), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.statics.get('user') == 'alice'
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 1), ('b', 2)]
    assert w.exited is True
    assert w.statics.get('done') is True


def test_episode_finalizes_when_run_stops(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['a'].emit, 42), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 42)]
    assert w.exited is True


def test_ignore_duplicate_commands_and_empty_stop(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'x': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.exited is True


def test_abort_flow_then_restart(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'s': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['s'].emit, 10), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.ABORT()), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['s'].emit, 11), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 2
    w1, w2 = ds.created[0], ds.created[1]
    assert w1.aborted is True and w1.exited is True
    assert [(s, v) for (s, v, _, _) in w2.appends] == [('s', 10), ('s', 11)]  # 10 is what the channel held


def test_appends_only_on_updates_and_timestamps_from_clock(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['a'].emit, 1), 0.001),
        (None, 0.001),
        (partial(emitters['a'].emit, 2), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert len(w.appends) == 2
    assert w.appends[1][2] > w.appends[0][2]


def test_records_what_the_inputs_hold_when_the_episode_opens(world):
    """The opening turn records what each channel holds, whenever that value was produced: a producer that
    published as it readied the scene has its frame in the episode, not dropped for predating START."""
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(emitters['a'].emit, 99), 0.001),  # latched on the channel before the episode opens
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['a'].emit, 7), 0.001),  # the first value that arrives in-episode
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 99), ('a', 7)]


def test_time_mode_message_uses_signal_timestamp(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world, time_mode=TimeMode.MESSAGE)

    ts_first = 123_000_000
    ts_second = 456_000_000

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['a'].emit, 1, ts=ts_first), 0.001),
        (partial(emitters['a'].emit, 2, ts=ts_second), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 1), ('a', 2)]
    assert [ts for (_, _, ts, _) in w.appends] == [ts_first, ts_second]


def test_integration_with_local_dataset_writer(tmp_path, world):
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None, 'b': None}, LocalDatasetWriter, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(tmp_path, {'task': 'unit'})), 0.001),
        (partial(emitters['a'].emit, 10), 0.001),
        (partial(emitters['b'].emit, 20), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP({'ok': True})), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    ds = LocalDataset(tmp_path)
    assert len(ds) == 1
    ep = ds[0]
    assert set(ep.keys()) == {'a', 'b', 'task', 'ok'}
    a = ep['a']
    b = ep['b']
    assert len(a) == 1 and len(b) == 1
    assert a[0][0] == 10 and b[0][0] == 20

    # Verify extra timelines are in the parquet files
    table_a = pq.read_table(ep._dir / 'a.parquet')
    assert 'ts_ns.message' in table_a.column_names
    assert 'ts_ns.system' in table_a.column_names
    assert 'ts_ns.world' in table_a.column_names


def test_each_episode_records_into_the_dataset_its_start_names(tmp_path, world):
    """Each START names where its episode records, so one run writes into as many datasets as it names."""
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, LocalDatasetWriter, world)
    first, second = tmp_path / 'first', tmp_path / 'second'

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(first)), 0.001),
        (partial(emitters['a'].emit, 10), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.START(second)), 0.001),
        (partial(emitters['a'].emit, 20), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    (first_ep,) = LocalDataset(first)
    (second_ep,) = LocalDataset(second)
    assert first_ep['a'][-1][0] == 10
    assert second_ep['a'][-1][0] == 20


def test_a_start_that_names_no_dataset_opens_no_episode(world):
    """There is nowhere to record, and the STOP that follows finds nothing open."""
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(None)), 0.001),
        (partial(emitters['a'].emit, 10), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert ds.created == []


def test_inputs_mapping_is_immutable(world):
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


def test_serializer_scalar_transform(world):
    ds = FakeDatasetWriter()

    # Serializer doubles the value
    def double(x):
        return x * 2

    agent, cmd_em, emitters = build_agent_with_pipes({'x': double}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['x'].emit, 3), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('x', 6)]


def test_serializer_dict_expansion(world):
    ds = FakeDatasetWriter()

    # Serializer splits into two signals:
    # - empty key keeps base name ("img")
    # - non-empty keys are treated as suffixes appended to base name (e.g., ".extra")
    def expand(v):
        return {'': v, '.extra': v + 1}

    agent, cmd_em, emitters = build_agent_with_pipes({'img': expand}, ds, world)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (lambda: emitters['img'].emit(10), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    names_and_vals = [(s, v) for (s, v, _, _) in w.appends]
    assert ('img', 10) in names_and_vals
    assert ('img.extra', 11) in names_and_vals


def test_serializer_none_drops_sample(world):
    ds = FakeDatasetWriter()

    # Serializer drops negative values by returning None (sample is not recorded)
    def drop_negative(v):
        return None if v < 0 else v

    agent, cmd_em, emitters = build_agent_with_pipes({'x': drop_negative}, ds, world)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (lambda: emitters['x'].emit(3), 0.001),
        (lambda: emitters['x'].emit(-1), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    # Only the positive value should be recorded
    assert [(s, v) for (s, v, _, _) in w.appends] == [('x', 3)]


def test_transform_3d_serializer(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'pose': Serializers.transform_3d}, ds, world)

    t = np.array([0.1, -0.2, 0.3])
    q = geom.Rotation.identity
    pose = geom.Transform3D(translation=t, rotation=q)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (lambda: emitters['pose'].emit(pose), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    names_vals = [(s, v) for (s, v, _, _) in w.appends]
    assert len(names_vals) == 1 and names_vals[0][0] == 'pose'
    np.testing.assert_allclose(names_vals[0][1][:3], t)
    np.testing.assert_allclose(names_vals[0][1][3:], q.as_quat)


def test_robot_state_serializer_records_a_busy_arm_beside_its_pose(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({keys.ROBOT_STATE: Serializers.robot_state}, ds, world)

    q = np.arange(7, dtype=np.float32)
    dq = np.arange(7, dtype=np.float32) + 10
    t = np.array([0.0, 0.1, 0.2])
    pose = geom.Transform3D(translation=t, rotation=geom.Rotation.identity)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (lambda: emitters[keys.ROBOT_STATE].emit(FakeRobotState(q, dq, pose, RobotStatus.BUSY)), 0.001),
        (lambda: emitters[keys.ROBOT_STATE].emit(FakeRobotState(q, dq, pose, RobotStatus.AVAILABLE)), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    by_name = {}
    for name, val, _, _ in w.appends:
        by_name.setdefault(name, []).append(val)
    expected = {keys.JOINTS: q, keys.JOINT_VEL: dq, keys.EE_POSE: np.concatenate([t, geom.Rotation.identity.as_quat])}
    assert set(by_name) == {keys.ROBOT_STATUS, *expected}
    assert by_name[keys.ROBOT_STATUS] == [RobotStatus.BUSY, RobotStatus.AVAILABLE]
    for name, value in expected.items():
        assert len(by_name[name]) == 2, name
        np.testing.assert_allclose(by_name[name][0], value)


def test_robot_command_serializer_variants(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'cmd': Serializers.robot_command}, ds, world)

    pose = geom.Transform3D(translation=np.array([0.2, 0.0, -0.1]), rotation=geom.Rotation.identity)
    delta = geom.Transform3D(translation=np.array([0.01, -0.02, 0.03]), rotation=geom.Rotation.identity)
    delta_frame = geom.Transform3D(translation=np.array([0.0, 0.0, 0.05]), rotation=geom.Rotation.identity)
    joints = np.arange(7, dtype=np.float32) * 0.1
    impedance = rcmd.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.CartesianPosition(pose)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.CartesianDelta(delta, delta_frame)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.JointPosition(joints)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.JointPosition(joints, mode=impedance)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.JointPosition(joints, mode=rcmd.PositionControl())), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.JointPosition(joints, mode=rcmd.PositionControl((100.0,) * 7))), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    items = {name: val for (name, val, _, _) in w.appends}
    np.testing.assert_allclose(items['cmd.pose'], np.concatenate([pose.translation, pose.rotation.as_quat]))
    np.testing.assert_allclose(items['cmd.pose_delta'], np.concatenate([delta.translation, delta.rotation.as_quat]))
    np.testing.assert_allclose(
        items['cmd.pose_delta_frame'], np.concatenate([delta_frame.translation, delta_frame.rotation.as_quat])
    )
    np.testing.assert_allclose(items['cmd.joints'], joints)
    np.testing.assert_allclose(items['cmd.mode.impedance.kq'], impedance.kq)
    # A pin naming no stiffness is a marker: a vector would fix the signal's shape, and what a mode does not
    # name has no shape to fix it at.
    assert items['cmd.mode.position_control'] == 1
    np.testing.assert_allclose(items['cmd.mode.position_control.stiffness'], [100.0] * 7)
    mode_appends = [name for (name, _, _, _) in w.appends if '.mode' in name]
    assert len(mode_appends) == 6, 'a command pinning nothing records no mode'


def test_multiple_timelines_recorded(world):
    """Test that DsWriterAgent records message, system, and world timelines."""
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['a'].emit, 42), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert len(w.appends) == 1
    name, value, _primary_ts, extra_ts = w.appends[0]

    assert name == 'a'
    assert value == 42
    assert extra_ts is not None

    # Should have message and system timelines
    assert 'message' in extra_ts
    assert 'system' in extra_ts

    # The virtual-time world drives a simulated clock, so 'world' is present too
    assert 'world' in extra_ts

    # All timestamps should be positive integers
    assert isinstance(extra_ts['message'], int) and extra_ts['message'] > 0
    assert isinstance(extra_ts['system'], int) and extra_ts['system'] > 0
    assert isinstance(extra_ts['world'], int) and extra_ts['world'] > 0


def test_pickles_with_every_constructor_argument_filled():
    """``World.start_in_subprocess`` spawns a background control system through plain pickle, so an agent
    carrying every argument the wiring fills — telemetry span factory included — has to survive a round-trip."""
    agent = DsWriterAgent(
        FakeDatasetWriter(),
        poll_hz=500.0,
        time_mode=TimeMode.MESSAGE,
        virtual_time=True,
        telemetry_span=partial(telemetry.span, telemetry_keys.SPAN_RECORD_IO),
    )
    agent.add_signal(keys.ROBOT_COMMAND, Serializers.robot_command)
    agent.add_signal(keys.ROBOT_STATE, Serializers.robot_state)

    loaded = pickle.loads(pickle.dumps(agent))

    assert set(loaded.inputs) == {keys.ROBOT_COMMAND, keys.ROBOT_STATE}
    with loaded._telemetry_span():
        pass


def test_serializer_plain_list_value(world):
    """A serializer returning a plain list is appended as one sample, values and all."""
    ds = FakeDatasetWriter()

    def to_list(_):
        return [1, 2, 3]

    agent, cmd_em, emitters = build_agent_with_pipes({'v': to_list}, ds, world)
    script = [
        (partial(cmd_em.emit, DsWriterCommand.START(OUTPUT_PATH)), 0.001),
        (partial(emitters['v'].emit, 0), 0.001),
        (partial(cmd_em.emit, DsWriterCommand.STOP()), 0.001),
    ]
    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('v', [1, 2, 3])]
