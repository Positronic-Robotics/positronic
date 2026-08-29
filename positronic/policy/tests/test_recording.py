import numpy as np

from positronic import keys
from positronic.drivers.roboarm import command
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Answer, ChunkSession, Done, Policy
from positronic.policy.layers import ChunkPlayer
from positronic.policy.recording import (
    Recorder,
    _build_blueprint,
    _ChunkTapSession,
    _command_field_arrays,
    _CommandTapSession,
    _flat_wire,
    _RecordingTapSession,
    _squeeze_batch,
    _stack_numeric,
)


def _chunk(session, obs, time_ns: int = 0):
    """The chunk a session under a player answers for ``obs``."""
    answer = session(obs, time_ns)
    assert isinstance(answer, Answer), 'the session answers commands, not a chunk'
    return answer.result()


class _TrackingSession(ChunkSession):
    def __init__(self, actions, meta):
        self._actions = actions
        self._meta = meta

    def __call__(self, obs, time_ns):
        return Done(list(self._actions))

    @property
    def meta(self):
        return self._meta


class _TrackingPolicy(Policy):
    """Policy that returns a fixed action chunk and tracks session creation."""

    def __init__(self, actions: list[dict] | None = None):
        self._actions = actions or [{'action': np.array([1.0, 2.0], dtype=np.float32), 'timestamp': 0.0}]
        self.session_count = 0

    def new_session(self, context=None, rt=None):
        self.session_count += 1
        return _TrackingSession(self._actions, {'policy_key': 'policy_value'})

    @property
    def meta(self):
        return {'policy_key': 'policy_value'}


class _CapturingSession(ChunkSession):
    """Innermost session that snapshots the Recorder's carried timeline state when called."""

    def __init__(self, rec, actions):
        self._rec = rec
        self._actions = actions
        self.seen_timeline_values = None
        self.seen_depth = None

    def __call__(self, obs, time_ns):
        self.seen_timeline_values = dict(self._rec._timeline_values)
        self.seen_depth = self._rec._depth
        return Done(list(self._actions))


class _CapturingPolicy(Policy):
    def __init__(self, rec, actions):
        self._rec = rec
        self._actions = actions
        self.last_session = None

    def new_session(self, context=None, rt=None):
        self.last_session = _CapturingSession(self._rec, self._actions)
        return self.last_session


def test_squeeze_batch():
    assert _squeeze_batch(np.zeros((1, 1, 4, 4, 3))).shape == (4, 4, 3)
    assert _squeeze_batch(np.zeros((4, 4, 3))).shape == (4, 4, 3)
    assert _squeeze_batch(np.zeros((2, 4, 4, 3))).shape == (2, 4, 4, 3)


def test_build_blueprint():
    assert _build_blueprint([], []) is None
    assert _build_blueprint(['raw/left', 'raw/right'], []) is not None
    assert _build_blueprint(['raw/left'], ['server/state']) is not None


def test_stack_numeric():
    assert _stack_numeric([0.1, 0.2, 0.3]).shape == (3,)
    assert _stack_numeric([np.zeros(7), np.ones(7)]).shape == (2, 7)
    assert _stack_numeric(['a', 'b']) is None
    # Ragged fields can't form a homogeneous tensor.
    assert _stack_numeric([np.zeros(7), np.ones(3)]) is None


_GAINS = ('kq', 'kqd', 'kx', 'kxd')


def test_flat_wire_unfolds_a_nested_control_mode():
    """The mode is a mapping inside the wire; a recording plots numeric leaves, so it has to come apart."""
    mode = command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    flat = _flat_wire(command.to_wire(command.JointPosition(np.zeros(7), mode=mode)))

    assert set(flat) == {'positions'} | {f'mode.impedance.{k}' for k in _GAINS}
    np.testing.assert_allclose(flat['mode.impedance.kq'], mode.kq)


def test_flat_wire_marks_a_mode_that_names_no_gains():
    """`PositionControl()` names no gains, so it has no numeric leaf to plot; the marker is what shows it."""
    flat = _flat_wire(command.to_wire(command.JointPosition(np.zeros(7), mode=command.PositionControl())))

    assert flat == {'positions': flat['positions'], 'mode.position_control': 1}


def test_a_cartesian_chunk_records_the_mode_beside_its_trajectory(tmp_path):
    """A pose is drawn as a 3D path rather than a field series, so a mode pinned on it has nowhere else to go."""
    mode = command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    pose = Transform3D(translation=np.array([0.1, 0.2, 0.3], dtype=np.float32), rotation=Rotation.identity)
    actions = [{keys.ROBOT_COMMAND: CartesianPosition(pose=pose, mode=mode), 'timestamp': 0.0}]

    rec = Recorder(tmp_path)
    _chunk(rec.tap('t').wrap(_TrackingPolicy(actions)).new_session(), {'x': 1.0, keys.WALL_TIME_NS: 1})

    assert any('mode.impedance.kq' in path for path in rec._series_paths), rec._series_paths
    assert not any(path.endswith('cartesian_pos/pose') for path in rec._series_paths), 'the pose is the trajectory'


def test_a_field_only_some_commands_carry_is_stacked_over_those():
    """A chunk may pin a mode on one command and not the next, and the pinned one is still worth plotting."""
    mode = command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    cmds = [command.JointPosition(np.zeros(7), mode=mode), command.JointPosition(np.ones(7))]

    out = {name: (arr, horizon) for name, arr, horizon in _command_field_arrays('cmd', cmds, np.array([0.0, 0.1]))}

    assert set(out) == {'cmd/joint_pos/positions'} | {f'cmd/joint_pos/mode.impedance.{k}' for k in _GAINS}
    kq, kq_horizon = out['cmd/joint_pos/mode.impedance.kq']
    assert kq.shape == (1, 7), 'the unpinned command has no gains to stack'
    np.testing.assert_allclose(kq_horizon, [0.0])  # stacked against the command that carries it, not both


def test_a_chunk_that_switches_law_records_both():
    """Two commands pinning different laws share no mode field, and dropping what they disagree on loses both."""
    impedance = command.Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)
    cmds = [
        command.JointPosition(np.zeros(7), mode=command.PositionControl()),
        command.JointPosition(np.ones(7), mode=impedance),
    ]

    names = {name for name, _, _ in _command_field_arrays('cmd', cmds, np.array([0.0, 0.1]))}

    assert 'cmd/joint_pos/mode.position_control' in names
    assert {f'cmd/joint_pos/mode.impedance.{k}' for k in _GAINS} <= names


def test_single_tap_file_per_episode(tmp_path):
    rec = Recorder(tmp_path)
    policy = rec.tap('raw').wrap(_TrackingPolicy())
    for _ in range(3):
        session = policy.new_session()
        session.close()
    assert len(list(tmp_path.glob('*.rrd'))) == 3


def test_tap_delegates_inner_call(tmp_path):
    actions = [{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.1}]
    policy = Recorder(tmp_path).tap('t').wrap(_TrackingPolicy(actions))
    session = policy.new_session()
    result = _chunk(session, {'x': 1.0, keys.WALL_TIME_NS: 1_000_000})
    assert result == actions


def test_tap_meta_passthrough(tmp_path):
    """A tap contributes no meta; inner meta passes through."""
    policy = Recorder(tmp_path).tap('t').wrap(_TrackingPolicy())
    assert policy.meta == {'policy_key': 'policy_value'}


def test_obs_log_filtering_uses_pure_tap_names(tmp_path):
    rec = Recorder(tmp_path)
    session = rec.tap('cam').wrap(_TrackingPolicy([{'v': 1.0, 'timestamp': 0.0}])).new_session()
    session(
        {
            keys.WALL_TIME_NS: 1_000_000,
            keys.TASK: 'pick up the cube',
            'camera': np.zeros((4, 4, 3), dtype=np.uint8),
            'joint_pos': np.array([1.0, 2.0], dtype=np.float32),
            'joints_list': [0.1, 0.2, 0.3],
            keys.GRIP: 0.5,
        },
        0,
    )

    assert 'cam/camera' in rec._image_paths
    assert 'cam/joint_pos' in rec._numeric_paths
    assert 'cam/grip' in rec._numeric_paths
    assert 'cam/joints_list' in rec._numeric_paths

    all_paths = rec._image_paths + rec._numeric_paths
    assert not any('time_ns' in p for p in all_paths)
    assert not any('task' in p for p in all_paths)
    # Pure tap-name prefix: no built-in '/image/' segment.
    assert not any('/image/' in p for p in all_paths)


def test_logs_command_chunk_without_mutating(tmp_path):
    pose = Transform3D(translation=np.array([0.1, 0.2, 0.3], dtype=np.float32), rotation=Rotation.identity)
    actions = [
        {keys.ROBOT_COMMAND: CartesianPosition(pose=pose), 'target_grip': 0.5, 'timestamp': 0.0},
        {keys.ROBOT_COMMAND: CartesianPosition(pose=pose), 'target_grip': 0.6, 'timestamp': 0.1},
    ]
    session = Recorder(tmp_path).tap('t').wrap(_TrackingPolicy(actions)).new_session()
    result = _chunk(session, {'x': 1.0, keys.WALL_TIME_NS: 1})
    assert isinstance(result, list)
    assert result[0][keys.ROBOT_COMMAND] is actions[0][keys.ROBOT_COMMAND]  # unchanged on return


def test_handles_an_empty_chunk(tmp_path):
    class _EmptySession(ChunkSession):
        def __call__(self, obs, time_ns):
            return Done([])

    class _EmptyPolicy(Policy):
        def new_session(self, context=None, rt=None):
            return _EmptySession()

    session = Recorder(tmp_path).tap('t').wrap(_EmptyPolicy()).new_session()
    assert isinstance(session, _RecordingTapSession)
    assert _chunk(session, {'x': 1.0}) == []
    assert session._step == 1


def test_a_two_action_chunk_is_not_read_as_a_command_pair(tmp_path):
    """A two-action chunk has the shape of the ``(commands, resume_at_ns)`` pair a session above the player
    answers, and a tap under the player must plot both actions."""
    chunk = [{'v': 1.0, 'timestamp': 0.0}, {'v': 2.0, 'timestamp': 0.5}]
    session = Recorder(tmp_path).tap('t').wrap(_TrackingPolicy(chunk)).new_session()
    assert isinstance(session, _ChunkTapSession)

    assert session({'x': 1.0}, 0).result() == chunk


def test_a_tap_above_the_player_logs_the_command_of_each_round(tmp_path):
    """Above a ``ChunkPlayer`` a session answers commands, so the tap plots a point per round."""
    actions = [{'v': 1.0, 'timestamp': 0.0}, {'v': 2.0, 'timestamp': 0.5}]
    rec = Recorder(tmp_path)
    session = (rec.tap('raw') | ChunkPlayer()).wrap(_TrackingPolicy(actions)).new_session()
    assert isinstance(session, _CommandTapSession)

    assert session({'x': 1.0, keys.WALL_TIME_NS: 1}, 0) == ({'v': 1.0}, int(0.5e9))
    assert session({'x': 1.0, keys.WALL_TIME_NS: 2}, int(0.25e9)) == ({}, int(0.5e9))
    assert 'raw/series/v' in rec._series_paths


def test_two_taps_share_one_file_per_episode(tmp_path):
    rec = Recorder(tmp_path)
    policy = (rec.tap('raw') | rec.tap('server')).wrap(_TrackingPolicy())

    session = policy.new_session()
    assert len(list(tmp_path.glob('*.rrd'))) == 1
    assert rec._live == 2

    session.close()
    assert rec._live == 0

    policy.new_session()
    assert len(list(tmp_path.glob('*.rrd'))) == 2


def test_two_taps_log_both_seams(tmp_path):
    rec = Recorder(tmp_path)
    actions = [{'v': 1.0, 'timestamp': 0.0}]
    policy = (rec.tap('raw') | rec.tap('server')).wrap(_TrackingPolicy(actions))
    session = policy.new_session()
    session({'camera': np.zeros((4, 4, 3), dtype=np.uint8), keys.WALL_TIME_NS: 1}, 0)

    assert 'raw/camera' in rec._image_paths
    assert 'server/camera' in rec._image_paths
    assert len(list(tmp_path.glob('*.rrd'))) == 1


def test_timeline_values_captured_once_and_carried(tmp_path):
    rec = Recorder(tmp_path)
    inner = _CapturingPolicy(rec, [{'v': 1.0, 'timestamp': 0.0}])
    policy = (rec.tap('raw') | rec.tap('server')).wrap(inner)
    session = policy.new_session()

    session({keys.WALL_TIME_NS: 111, keys.OBS_TIME_NS: 222, 'x': 1.0}, 0)

    # Both taps entered before the inner session ran, and both share the values
    # captured once from the raw obs at the outermost tap.
    assert inner.last_session.seen_depth == 2
    assert inner.last_session.seen_timeline_values == {'wall_time': 111, 'obs_time': 222}
    # Per-inference context is cleared once the outermost tap returns.
    assert rec._timeline_values == {}
    assert rec._depth == 0


def test_partial_timelines_only_set_present_keys(tmp_path):
    rec = Recorder(tmp_path)
    inner = _CapturingPolicy(rec, [{'v': 1.0, 'timestamp': 0.0}])
    session = rec.tap('raw').wrap(inner).new_session()

    session({keys.WALL_TIME_NS: 555, 'x': 1.0}, 0)  # no obs_time_ns
    assert inner.last_session.seen_timeline_values == {'wall_time': 555}


def test_concurrent_recorders_write_separate_files(tmp_path):
    """Two overlapping recorders (e.g. one per websocket session) must not share a
    stream or collide on filenames."""
    rec_a = Recorder(tmp_path)
    rec_b = Recorder(tmp_path)
    rec_a.tap('inference').wrap(_TrackingPolicy()).new_session()
    rec_b.tap('inference').wrap(_TrackingPolicy()).new_session()

    assert rec_a._stream is not rec_b._stream
    assert len(list(tmp_path.glob('*.rrd'))) == 2
