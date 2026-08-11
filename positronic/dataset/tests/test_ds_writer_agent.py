import json
import logging
import pickle
from functools import partial
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import pytest

import pimm
from positronic import geom, keys, telemetry, telemetry_keys
from positronic.dataset import DatasetWriter, DiscardReason, EpisodeWriter
from positronic.dataset.ds_writer_agent import (
    DsWriterAgent,
    DsWriterCommand,
    DsWriterCommandType,
    TimeMode,
    TrajectoryOverrideSerializer,
)
from positronic.dataset.local_dataset import DISCARD_MARKER, DISCARD_REASON, LocalDataset, LocalDatasetWriter
from positronic.dataset.serializers import Serializers
from positronic.dataset.vector import SimpleSignalWriter
from positronic.drivers.roboarm import RobotStatus, State
from positronic.drivers.roboarm import command as rcmd
from positronic.tests.testing_coutils import ManualCommandReceiver, drive_until, run_scripted_agent


@pytest.fixture
def world():
    with pimm.World(virtual_time=True) as w:
        yield w


class FakeEpisodeWriter(EpisodeWriter[Any]):
    def __init__(self) -> None:
        self.statics: dict[str, Any] = {}
        self.appends: list[tuple[str, Any, int, dict[str, int] | None]] = []
        self.exited = False
        self.discarded: DiscardReason | None = None

    def append(self, signal_name: str, data: Any, ts_ns: int, extra_ts: dict[str, int] | None = None) -> None:
        self.appends.append((signal_name, data, int(ts_ns), extra_ts))

    def set_static(self, name: str, data: Any) -> None:
        self.statics[name] = data

    def __exit__(self, exc_type, exc, tb) -> None:
        self.exited = True

    def discard(self, reason: DiscardReason) -> None:
        self.discarded = reason


class UndiscardableEpisodeWriter(FakeEpisodeWriter):
    """A writer whose discard fails — a video encoder failing to finalize, say."""

    def discard(self, reason: DiscardReason) -> None:
        raise RuntimeError('discard failed')


class FakeDatasetWriter(DatasetWriter):
    def __init__(self, episode_writer: type[FakeEpisodeWriter] = FakeEpisodeWriter) -> None:
        self.created: list[FakeEpisodeWriter] = []
        self._episode_writer = episode_writer

    def new_episode(self) -> FakeEpisodeWriter:
        w = self._episode_writer()
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
    agent = DsWriterAgent(ds_writer, time_mode=time_mode)
    for name, serializer in signals_spec.items():
        agent.add_signal(name, serializer)
    emitters = {name: cast(pimm.SignalEmitter[Any], world.pair(agent.inputs[name])) for name in signals_spec}

    cmd_em = cast(pimm.SignalEmitter[DsWriterCommand], world.pair(agent.command))

    return agent, cmd_em, emitters


def test_start_stop_happy_path(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None, 'b': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE, {'user': 'alice'})), 0.001),
        (partial(emitters['a'].emit, 1), 0.001),
        (partial(emitters['b'].emit, 2), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE, {'done': True})), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.statics.get('user') == 'alice'
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 1), ('b', 2)]
    assert w.exited is True
    assert w.statics.get('done') is True


def test_an_episode_still_open_when_the_run_stops_is_discarded(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 42), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 42)]
    assert w.discarded is DiscardReason.RUN_ENDED
    assert w.exited is True


def test_the_run_ended_discard_is_logged_under_its_own_reason(world, caplog):
    ds = FakeDatasetWriter()
    agent, cmd_em, _ = build_agent_with_pipes({'a': None}, ds, world)

    script = [(partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001)]

    with caplog.at_level(logging.INFO, logger='positronic.dataset.ds_writer_agent'):
        run_scripted_agent(agent, script, world=world)

    assert f'[DISCARD {DiscardReason.RUN_ENDED.value}] Episode 1' in caplog.text
    assert '[ABORT]' not in caplog.text  # the operator aborted nothing


def test_an_episode_that_fails_to_discard_is_not_finalized(world):
    ds = FakeDatasetWriter(episode_writer=UndiscardableEpisodeWriter)
    agent, cmd_em, _ = build_agent_with_pipes({'a': None}, ds, world)

    script = [(partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001)]

    run_scripted_agent(agent, script, world=world)

    assert ds.created[-1].exited is False  # __exit__ commits, which a failed discard must not reach


def test_an_episode_stopped_before_the_run_ends_is_not_discarded(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 42), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert ds.created[-1].discarded is None


def test_a_stop_the_loop_never_read_does_not_commit_the_episode(world):
    ds = FakeDatasetWriter()
    agent = DsWriterAgent(ds)
    commands: ManualCommandReceiver[DsWriterCommand] = ManualCommandReceiver()
    agent.command = cast(pimm.ControlSystemReceiver[DsWriterCommand], commands)

    scheduler = world.interleave(agent.run)
    commands.push(DsWriterCommand(DsWriterCommandType.START_EPISODE))
    drive_until(scheduler, lambda: len(ds.created) == 1)

    # Queued while the loop is between turns: the world stops before the read that would have handled it.
    commands.push(DsWriterCommand(DsWriterCommandType.STOP_EPISODE))
    world.request_stop()
    with pytest.raises(StopIteration):
        next(scheduler)

    assert ds.created[-1].discarded is DiscardReason.RUN_ENDED


def test_ignore_duplicate_commands_and_empty_stop(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'x': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 1
    w = ds.created[-1]
    assert w.exited is True


def test_abort_flow_then_restart(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'s': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['s'].emit, 10), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.ABORT_EPISODE)), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['s'].emit, 11), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    assert len(ds.created) == 2
    w1, w2 = ds.created[0], ds.created[1]
    assert w1.discarded is DiscardReason.ABORTED and w1.exited is True
    assert [(s, v) for (s, v, _, _) in w2.appends] == [('s', 11)]


def test_appends_only_on_updates_and_timestamps_from_clock(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 1), 0.001),
        (None, 0.001),
        (partial(emitters['a'].emit, 2), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert len(w.appends) == 2
    assert w.appends[1][2] > w.appends[0][2]


def test_drains_inputs_latched_before_start(world):
    """The opening turn drains the input channels without recording: a value latched before START — an
    inter-episode home command or a pre-reset frame — is consumed, not appended, so the first recorded
    sample is the next value to arrive (the post-reset scene), never a pre-episode leftover."""
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(emitters['a'].emit, 99), 0.001),  # latched on the channel before the episode opens
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 7), 0.001),  # the first value that arrives in-episode
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 7)]  # 99 was drained on the open turn, not recorded


def test_time_mode_message_uses_signal_timestamp(world):
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

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('a', 1), ('a', 2)]
    assert [ts for (_, _, ts, _) in w.appends] == [ts_first, ts_second]


def test_integration_with_local_dataset_writer(tmp_path, world):
    with LocalDatasetWriter(tmp_path) as writer:
        agent, cmd_em, emitters = build_agent_with_pipes({'a': None, 'b': None}, writer, world)

        script = [
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE, {keys.TASK: 'unit'})), 0.001),
            (partial(emitters['a'].emit, 10), 0.001),
            (partial(emitters['b'].emit, 20), 0.001),
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE, {'ok': True})), 0.001),
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


def test_run_stopping_mid_episode_keeps_the_recording_outside_the_dataset(tmp_path, world):
    root = tmp_path / 'ds'
    with LocalDatasetWriter(root) as writer:
        agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, writer, world)

        script = [
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE, {keys.TASK: 'unit'})), 0.001),
            (partial(emitters['a'].emit, 10), 0.001),
        ]

        run_scripted_agent(agent, script, world=world)

    assert len(LocalDataset(root)) == 0
    discarded = list((tmp_path / 'ds.discarded').iterdir())
    assert len(discarded) == 1
    assert (discarded[0] / 'a.parquet').exists()
    assert json.loads((discarded[0] / DISCARD_MARKER).read_text())[DISCARD_REASON] == DiscardReason.RUN_ENDED.value


def test_a_run_ending_after_a_failed_abort_still_moves_the_episode_out(tmp_path, world, monkeypatch, caplog):
    root = tmp_path / 'ds'
    real_exit = SimpleSignalWriter.__exit__
    failures_left = [1]

    def fail_once(self, exc_type, exc, tb):
        real_exit(self, exc_type, exc, tb)
        if failures_left[0]:
            failures_left[0] -= 1
            raise RuntimeError('encoder failed')

    monkeypatch.setattr(SimpleSignalWriter, '__exit__', fail_once)

    with LocalDatasetWriter(root) as writer:
        agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, writer, world)

        script = [
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE, {keys.TASK: 'unit'})), 0.001),
            (partial(emitters['a'].emit, 10), 0.001),
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.ABORT_EPISODE)), 0.001),
        ]

        with caplog.at_level(logging.INFO, logger='positronic.dataset.ds_writer_agent'):
            with pytest.raises(RuntimeError, match='encoder failed'):
                run_scripted_agent(agent, script, world=world)

    # The abort's discard failed part-way; the run's teardown finished it rather than reporting a move nobody made.
    assert len(LocalDataset(root)) == 0
    assert not list(root.glob('*/*/a.parquet'))
    discarded = list((tmp_path / 'ds.discarded').iterdir())
    assert len(discarded) == 1
    assert (discarded[0] / 'a.parquet').exists()
    assert json.loads((discarded[0] / DISCARD_MARKER).read_text())[DISCARD_REASON] == DiscardReason.ABORTED.value
    assert f'[DISCARD {DiscardReason.RUN_ENDED.value}] Episode 1' in caplog.text


def test_a_committed_episode_is_never_swept_into_the_discarded_dir(tmp_path, world):
    root = tmp_path / 'ds'
    with LocalDatasetWriter(root) as writer:
        agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, writer, world)

        script = [
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE, {keys.TASK: 'unit'})), 0.001),
            (partial(emitters['a'].emit, 10), 0.001),
            (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
        ]

        run_scripted_agent(agent, script, world=world)

    assert len(LocalDataset(root)) == 1
    assert not (tmp_path / 'ds.discarded').exists()


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
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['x'].emit, 3), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
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
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['img'].emit(10), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
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
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['x'].emit(3), 0.001),
        (lambda: emitters['x'].emit(-1), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
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
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['pose'].emit(pose), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    names_vals = [(s, v) for (s, v, _, _) in w.appends]
    assert len(names_vals) == 1 and names_vals[0][0] == 'pose'
    np.testing.assert_allclose(names_vals[0][1][:3], t)
    np.testing.assert_allclose(names_vals[0][1][3:], q.as_quat)


class _FakeState(State):
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


def test_robot_state_serializer_drops_reset_and_emits_components(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'robot_state': Serializers.robot_state}, ds, world)

    q = np.arange(7, dtype=np.float32)
    dq = np.arange(7, dtype=np.float32) + 10
    t = np.array([0.0, 0.1, 0.2])
    pose = geom.Transform3D(translation=t, rotation=geom.Rotation.identity)

    script = [
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['robot_state'].emit(_FakeState(q, dq, pose, RobotStatus.RESETTING)), 0.001),
        (lambda: emitters['robot_state'].emit(_FakeState(q, dq, pose, RobotStatus.AVAILABLE)), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]

    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    items = {name: val for (name, val, _, _) in w.appends}
    # Should not contain any data from RESETTING
    assert set(items.keys()) == {keys.JOINTS, keys.JOINT_VEL, keys.EE_POSE}
    np.testing.assert_allclose(items[keys.JOINTS], q)
    np.testing.assert_allclose(items[keys.JOINT_VEL], dq)
    np.testing.assert_allclose(items[keys.EE_POSE], np.concatenate([t, geom.Rotation.identity.as_quat]))


def test_robot_command_serializer_variants(world):
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'cmd': Serializers.robot_command}, ds, world)

    pose = geom.Transform3D(translation=np.array([0.2, 0.0, -0.1]), rotation=geom.Rotation.identity)
    delta = geom.Transform3D(translation=np.array([0.01, -0.02, 0.03]), rotation=geom.Rotation.identity)
    delta_frame = geom.Transform3D(translation=np.array([0.0, 0.0, 0.05]), rotation=geom.Rotation.identity)
    joints = np.arange(7, dtype=np.float32) * 0.1

    script = [
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.CartesianPosition(pose)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.CartesianDelta(delta, delta_frame)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.JointPosition(joints)), 0.001),
        (lambda: emitters['cmd'].emit(rcmd.Reset()), 0.001),
        (lambda: cmd_em.emit(DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
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
    assert items['cmd.reset'] == 1


def test_multiple_timelines_recorded(world):
    """Test that DsWriterAgent records message, system, and world timelines."""
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'a': None}, ds, world)

    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['a'].emit, 42), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
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
    agent.add_signal('robot_command', TrajectoryOverrideSerializer(Serializers.robot_command))
    agent.add_signal('robot_state', Serializers.robot_state)

    loaded = pickle.loads(pickle.dumps(agent))

    assert set(loaded.inputs) == {'robot_command', 'robot_state'}
    with loaded._telemetry_span():
        pass


def test_trajectory_override_serializer():
    s = TrajectoryOverrideSerializer(None)
    s.reset()

    # First trajectory: nothing is final yet (could be overridden).
    assert s([(1, 'a'), (2, 'b'), (3, 'c')]) == []

    # Next trajectory starts at ts=2 -> only ts<2 ('a') is final; 'b','c' overridden.
    out = s([(2, 'B'), (3, 'C'), (4, 'D')])
    assert [(t.ts, t.value) for t in out] == [(1, 'a')]

    # Episode end drains the still-live buffer.
    assert [(t.ts, t.value) for t in s.flush()] == [(2, 'B'), (3, 'C'), (4, 'D')]


def test_serializer_plain_list_value(world):
    """A serializer returning a plain list (non-`Timestamped`) is appended as one sample.

    The trajectory-stream dispatch must not hijack legitimate list-valued samples.
    """
    ds = FakeDatasetWriter()

    def to_list(_):
        return [1, 2, 3]

    agent, cmd_em, emitters = build_agent_with_pipes({'v': to_list}, ds, world)
    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['v'].emit, 0), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]
    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    assert [(s, v) for (s, v, _, _) in w.appends] == [('v', [1, 2, 3])]


def test_trajectory_override_serializer_empty_cancels_buffer():
    """Empty trajectory is the Harness STOP cancel signal: drop the buffered tail."""
    s = TrajectoryOverrideSerializer(None)
    s.reset()

    # Buffer a trajectory (nothing committed yet).
    assert s([(1, 'a'), (2, 'b'), (3, 'c')]) == []
    # Empty trajectory = cancel: nothing committed AND buffer cleared.
    assert s([]) == []
    # Subsequent flush must not emit the canceled waypoints.
    assert s.flush() == []


def test_trajectory_override_serializer_flush_cutoff():
    """flush(now_ns) commits only points already due; the future tail is dropped."""
    s = TrajectoryOverrideSerializer(None)
    s.reset()

    # Buffer a chunk scheduled at ts 1..4 (nothing committed yet).
    assert s([(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]) == []

    # Episode ends at ts=2: only the due points (ts <= 2) are committed; 'c','d' dropped.
    assert [(t.ts, t.value) for t in s.flush(now_ns=2)] == [(1, 'a'), (2, 'b')]

    # No cutoff keeps the legacy "commit everything" behavior.
    s.reset()
    assert s([(1, 'a'), (2, 'b')]) == []
    assert [(t.ts, t.value) for t in s.flush()] == [(1, 'a'), (2, 'b')]


def test_stop_commits_due_drops_future_trajectory(world):
    """A mid-trajectory STOP commits already-due samples and drops the un-executed tail."""
    ds = FakeDatasetWriter()
    agent, cmd_em, emitters = build_agent_with_pipes({'traj': TrajectoryOverrideSerializer(None)}, ds, world)

    future = 10**18  # far beyond the test clock, so it stays an un-executed tail
    script = [
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.START_EPISODE)), 0.001),
        (partial(emitters['traj'].emit, [(0, 'due'), (future, 'tail')]), 0.001),
        (partial(cmd_em.emit, DsWriterCommand(DsWriterCommandType.STOP_EPISODE)), 0.001),
    ]
    run_scripted_agent(agent, script, world=world)

    w = ds.created[-1]
    # 'due' (ts <= stop time) is committed; the future 'tail' is dropped.
    assert [(s, v) for (s, v, _, _) in w.appends] == [('traj', 'due')]
    assert w.exited is True
