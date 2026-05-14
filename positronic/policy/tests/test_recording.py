import numpy as np
import rerun as rr

from positronic.drivers.roboarm.command import CartesianPosition
from positronic.geom import Rotation, Transform3D
from positronic.policy.base import Policy, Session
from positronic.policy.codec import RecordingWrapper, _build_blueprint, _RecordingSession, _squeeze_batch


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


def _make_session(tmp_path, inner_session=None):
    rec = rr.RecordingStream(application_id='test')
    rec.save(str(tmp_path / 'test.rrd'))
    inner = inner_session or _TrackingSession([{'v': 1, 'timestamp': 0.0}], {})
    return _RecordingSession(inner, rec)


def test_squeeze_batch():
    assert _squeeze_batch(np.zeros((1, 1, 4, 4, 3))).shape == (4, 4, 3)
    assert _squeeze_batch(np.zeros((4, 4, 3))).shape == (4, 4, 3)
    assert _squeeze_batch(np.zeros((2, 4, 4, 3))).shape == (2, 4, 4, 3)


def test_build_blueprint():
    assert _build_blueprint([], []) is None
    assert _build_blueprint(['input/image/left', 'input/image/right'], []) is not None
    assert _build_blueprint(['input/image/left'], ['encoded/state']) is not None


def test_recording_wrapper_new_session_creates_rrd_files(tmp_path):
    policy = RecordingWrapper(tmp_path).wrap(_TrackingPolicy())

    for _i in range(3):
        policy.new_session()
    assert len(list(tmp_path.glob('*.rrd'))) == 3


def test_recording_wrapper_new_session_calls_inner(tmp_path):
    tracking = _TrackingPolicy()
    policy = RecordingWrapper(tmp_path).wrap(tracking)

    policy.new_session()
    assert tracking.session_count == 1
    policy.new_session()
    assert tracking.session_count == 2


def test_recording_wrapper_session_delegates_inner_call(tmp_path):
    actions = [{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.1}]
    policy = RecordingWrapper(tmp_path).wrap(_TrackingPolicy(actions))
    session = policy.new_session()

    result = session({'x': 1.0, 'wall_time_ns': 1_000_000})
    assert result == actions


def test_recording_wrapper_meta_passthrough(tmp_path):
    """RecordingWrapper doesn't contribute meta; inner meta passes through."""
    policy = RecordingWrapper(tmp_path).wrap(_TrackingPolicy())
    assert policy.meta == {'policy_key': 'policy_value'}


def test_session_call_extracts_wall_time(tmp_path):
    session = _make_session(tmp_path)
    wall_time = 1_000_000_000
    session({'wall_time_ns': wall_time, 'inference_time_ns': 500_000_000, 'x': 1.0})
    assert session._step == 1


def test_session_step_increments(tmp_path):
    session = _make_session(tmp_path)
    assert session._step == 0

    session({'x': 1.0})
    assert session._step == 1

    session({'x': 2.0})
    assert session._step == 2


def test_session_logs_command_as_wire(tmp_path):
    """Command objects are converted to wire format for logging (plain-data)."""
    pose = Transform3D(translation=np.array([0.1, 0.2, 0.3], dtype=np.float32), rotation=Rotation.identity)
    actions = [{'robot_command': CartesianPosition(pose=pose), 'target_grip': 0.5, 'timestamp': 0.0}]
    inner = _TrackingSession(actions, {})
    session = _make_session(tmp_path, inner_session=inner)
    # Should not raise — Command gets to_wire'd internally for logging
    result = session({'x': 1.0})
    assert result[0]['robot_command'] is actions[0]['robot_command']  # unchanged on return


def test_session_log_filtering(tmp_path):
    session = _make_session(tmp_path)
    session({
        'wall_time_ns': 1_000_000,
        'task': 'pick up the cube',
        'camera': np.zeros((4, 4, 3), dtype=np.uint8),
        'joint_pos': np.array([1.0, 2.0], dtype=np.float32),
        'joints_list': [0.1, 0.2, 0.3],
        'grip': 0.5,
    })

    assert any('image' in p for p in session._image_paths)
    assert any('joint_pos' in p or 'grip' in p for p in session._numeric_paths)
    assert any('joints_list' in p for p in session._numeric_paths)

    all_paths = session._image_paths + session._numeric_paths
    assert not any('wall_time_ns' in p for p in all_paths)
    assert not any('task' in p for p in all_paths)


def test_concurrent_sessions_independent_state(tmp_path):
    wrapper = RecordingWrapper(tmp_path)
    policy_a = wrapper.wrap(_TrackingPolicy())
    policy_b = wrapper.wrap(_TrackingPolicy())
    session_a = policy_a.new_session()
    session_b = policy_b.new_session()

    session_a({'x': 1.0})
    session_a({'x': 2.0})
    assert session_a._step == 2
    assert session_b._step == 0

    session_b({'y': 1.0})
    assert session_b._step == 1
    assert session_a._step == 2


def test_recording_wrapper_full_pipeline(tmp_path):
    actions = [{'v': i, 'timestamp': i * 0.1} for i in range(3)]
    policy = RecordingWrapper(tmp_path).wrap(_TrackingPolicy(actions))
    session = policy.new_session()

    result = session({'camera': np.zeros((4, 4, 3), dtype=np.uint8), 'grip': 0.5})
    assert result == actions
    assert len(list(tmp_path.glob('*.rrd'))) == 1


def test_session_handles_none_actions(tmp_path):
    """When inner returns None (no new trajectory), recording should still work."""

    class _NoneSession(Session):
        def __call__(self, obs):
            return None

    session = _make_session(tmp_path, inner_session=_NoneSession())
    result = session({'x': 1.0})
    assert result is None
    assert session._step == 1
