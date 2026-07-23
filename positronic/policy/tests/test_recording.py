import numpy as np

from positronic import keys
from positronic.drivers.roboarm.command import CartesianPosition
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Policy, Session
from positronic.policy.recording import Recorder, _build_blueprint, _squeeze_batch, _stack_numeric


class _TrackingSession(Session):
    def __init__(self, actions, meta):
        self._actions = actions
        self._meta = meta

    def __call__(self, obs):
        return list(self._actions)

    @property
    def meta(self):
        return self._meta


class _TrackingPolicy(Policy):
    """Policy that returns a fixed action chunk and tracks session creation."""

    def __init__(self, actions: list[dict] | None = None):
        self._actions = actions or [{'action': np.array([1.0, 2.0], dtype=np.float32), 'timestamp': 0.0}]
        self.session_count = 0

    def new_session(self, context=None):
        self.session_count += 1
        return _TrackingSession(self._actions, {'policy_key': 'policy_value'})

    @property
    def meta(self):
        return {'policy_key': 'policy_value'}


class _CapturingSession(Session):
    """Innermost session that snapshots the Recorder's carried timeline state when called."""

    def __init__(self, rec, actions):
        self._rec = rec
        self._actions = actions
        self.seen_timeline_values = None
        self.seen_depth = None

    def __call__(self, obs):
        self.seen_timeline_values = dict(self._rec._timeline_values)
        self.seen_depth = self._rec._depth
        return list(self._actions)


class _CapturingPolicy(Policy):
    def __init__(self, rec, actions):
        self._rec = rec
        self._actions = actions
        self.last_session = None

    def new_session(self, context=None):
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
    result = session({'x': 1.0, 'wall_time_ns': 1_000_000})
    assert result == actions


def test_tap_meta_passthrough(tmp_path):
    """A tap contributes no meta; inner meta passes through."""
    policy = Recorder(tmp_path).tap('t').wrap(_TrackingPolicy())
    assert policy.meta == {'policy_key': 'policy_value'}


def test_obs_log_filtering_uses_pure_tap_names(tmp_path):
    rec = Recorder(tmp_path)
    session = rec.tap('cam').wrap(_TrackingPolicy([{'v': 1.0, 'timestamp': 0.0}])).new_session()
    session({
        'wall_time_ns': 1_000_000,
        keys.TASK: 'pick up the cube',
        'camera': np.zeros((4, 4, 3), dtype=np.uint8),
        'joint_pos': np.array([1.0, 2.0], dtype=np.float32),
        'joints_list': [0.1, 0.2, 0.3],
        keys.GRIP: 0.5,
    })

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
        {'robot_command': CartesianPosition(pose=pose), 'target_grip': 0.5, 'timestamp': 0.0},
        {'robot_command': CartesianPosition(pose=pose), 'target_grip': 0.6, 'timestamp': 0.1},
    ]
    session = Recorder(tmp_path).tap('t').wrap(_TrackingPolicy(actions)).new_session()
    result = session({'x': 1.0, 'wall_time_ns': 1})
    assert result[0]['robot_command'] is actions[0]['robot_command']  # unchanged on return


def test_handles_none_actions(tmp_path):
    class _NoneSession(Session):
        def __call__(self, obs):
            return None

    class _NonePolicy(Policy):
        def new_session(self, context=None):
            return _NoneSession()

    session = Recorder(tmp_path).tap('t').wrap(_NonePolicy()).new_session()
    assert session({'x': 1.0}) is None
    assert session._step == 1


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
    session({'camera': np.zeros((4, 4, 3), dtype=np.uint8), 'wall_time_ns': 1})

    assert 'raw/camera' in rec._image_paths
    assert 'server/camera' in rec._image_paths
    assert len(list(tmp_path.glob('*.rrd'))) == 1


def test_timeline_values_captured_once_and_carried(tmp_path):
    rec = Recorder(tmp_path)
    inner = _CapturingPolicy(rec, [{'v': 1.0, 'timestamp': 0.0}])
    policy = (rec.tap('raw') | rec.tap('server')).wrap(inner)
    session = policy.new_session()

    session({'wall_time_ns': 111, 'obs_time_ns': 222, 'x': 1.0})

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

    session({'wall_time_ns': 555, 'x': 1.0})  # no obs_time_ns
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
