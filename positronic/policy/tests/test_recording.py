import numpy as np
import pytest
import rerun as rr

from positronic.dataset.transforms.episode import Derive
from positronic.policy.base import Policy
from positronic.policy.codec import (
    Codec,
    RecordingCodec,
    _build_blueprint,
    _RecordingPolicy,
    _RecordingSession,
    _squeeze_batch,
)


class _TrackingCodec(Codec):
    """Codec that transforms data for verifying delegation."""

    def __init__(self, *, meta: dict | None = None):
        self._meta = meta or {'action_fps': 10.0}

    def encode(self, data):
        return {**data, 'encoded': True}

    def _decode_single(self, data, context):
        return {**data, 'decoded': True}

    @property
    def meta(self):
        return dict(self._meta)

    def dummy_encoded(self, data=None):
        return {'dummy': True}


class _TrackingPolicy(Policy):
    """Policy that returns a fixed action chunk and tracks reset calls."""

    def __init__(self, actions: list[dict] | None = None):
        self._actions = actions or [{'action': np.array([1.0, 2.0], dtype=np.float32)}]
        self.reset_count = 0

    def select_action(self, obs):
        return list(self._actions)

    def reset(self):
        self.reset_count += 1

    @property
    def meta(self):
        return {'policy_key': 'policy_value'}


def _make_codec(tmp_path, **kw):
    return RecordingCodec(_TrackingCodec(**kw), tmp_path)


def _make_session(tmp_path, *, action_fps=10.0, meta=None):
    inner = _TrackingCodec(meta=meta)
    rec = rr.new_recording(application_id='test')
    rec.save(str(tmp_path / 'test.rrd'))
    return _RecordingSession(inner, rec, action_fps=action_fps)


def test_squeeze_batch():
    assert _squeeze_batch(np.zeros((1, 1, 4, 4, 3))).shape == (4, 4, 3)
    assert _squeeze_batch(np.zeros((4, 4, 3))).shape == (4, 4, 3)
    assert _squeeze_batch(np.zeros((2, 4, 4, 3))).shape == (2, 4, 4, 3)


def test_build_blueprint():
    assert _build_blueprint([], []) is None
    assert _build_blueprint(['input/image/left', 'input/image/right'], []) is not None
    assert _build_blueprint(['input/image/left'], ['encoded/state']) is not None


def test_recording_codec_delegates(tmp_path):
    codec = _make_codec(tmp_path, meta={'action_fps': 30.0, 'custom': 'val'})

    assert codec.encode({'x': 1}) == {'x': 1, 'encoded': True}
    assert codec.decode({'y': 2}) == {'y': 2, 'decoded': True}
    assert codec.decode([{'a': 1}, {'b': 2}]) == [{'a': 1, 'decoded': True}, {'b': 2, 'decoded': True}]
    assert codec.meta == {'action_fps': 30.0, 'custom': 'val'}
    assert codec.dummy_encoded() == {'dummy': True}


def test_recording_policy_select_action_before_reset(tmp_path):
    policy = _make_codec(tmp_path).wrap(_TrackingPolicy())
    with pytest.raises(RuntimeError, match='reset'):
        policy.select_action({'x': 1.0})


def test_recording_policy_reset_creates_rrd_files(tmp_path):
    policy = _make_codec(tmp_path).wrap(_TrackingPolicy())
    assert isinstance(policy, _RecordingPolicy)

    for i in range(1, 4):
        policy.reset()
        assert (tmp_path / f'episode_{i:04d}.rrd').exists()


def test_recording_policy_reset_calls_inner_reset(tmp_path):
    tracking = _TrackingPolicy()
    policy = _make_codec(tmp_path).wrap(tracking)

    policy.reset()
    assert tracking.reset_count == 1
    policy.reset()
    assert tracking.reset_count == 2


def test_recording_policy_select_action_pipeline(tmp_path):
    policy = _make_codec(tmp_path).wrap(_TrackingPolicy([{'v': 1}, {'v': 2}]))
    policy.reset()

    result = policy.select_action({'x': 1.0})
    assert len(result) == 2
    assert all('decoded' in r for r in result)


def test_recording_policy_meta(tmp_path):
    policy = _make_codec(tmp_path, meta={'action_fps': 10.0}).wrap(_TrackingPolicy())
    assert 'action_fps' in policy.meta

    policy.reset()
    meta = policy.meta
    assert meta['policy_key'] == 'policy_value'
    assert meta['action_fps'] == 10.0


def test_session_encode_extracts_wall_time(tmp_path):
    session = _make_session(tmp_path)
    wall_time, inf_time = 1_000_000_000, 500_000_000

    encoded = session.encode({'__wall_time_ns__': wall_time, '__inference_time_ns__': inf_time, 'x': 1.0})

    assert encoded == {'__wall_time_ns__': wall_time, '__inference_time_ns__': inf_time, 'x': 1.0, 'encoded': True}
    assert session._time_ns == wall_time
    assert session._inference_time_ns == inf_time


def test_session_encode_falls_back_to_wall_clock(tmp_path):
    session = _make_session(tmp_path)
    session.encode({'x': 1.0})
    assert session._time_ns > 0
    assert session._inference_time_ns is None


def test_session_step_increments(tmp_path):
    session = _make_session(tmp_path)
    assert session._step == 0

    session.encode({'x': 1.0})
    session.decode([{'v': 1}])
    assert session._step == 1

    session.encode({'x': 2.0})
    session.decode([{'v': 2}])
    assert session._step == 2


def test_session_decode(tmp_path):
    session = _make_session(tmp_path)
    session.encode({'x': 1.0})

    assert session.decode({'v': 42}) == {'v': 42, 'decoded': True}

    session.encode({'x': 2.0})
    chunk = session.decode([{'v': i} for i in range(5)])
    assert len(chunk) == 5
    assert all(r['decoded'] for r in chunk)


def test_session_delegates(tmp_path):
    session = _make_session(tmp_path, meta={'action_fps': 30.0})
    assert session.meta == {'action_fps': 30.0}
    assert session.dummy_encoded() == {'dummy': True}


def test_session_log_filtering(tmp_path):
    session = _make_session(tmp_path)
    session.encode({
        '__wall_time_ns__': 1_000_000,
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
    assert not any('__wall_time_ns__' in p for p in all_paths)
    assert not any('task' in p for p in all_paths)


def test_concurrent_sessions_independent_state(tmp_path):
    codec = _make_codec(tmp_path)
    session_a = codec._new_session()
    session_b = codec._new_session()

    session_a.encode({'x': 1.0})
    session_a.decode([{'v': 1}])
    session_a.encode({'x': 2.0})
    session_a.decode([{'v': 2}])

    assert session_a._step == 2
    assert session_b._step == 0

    session_b.encode({'y': 1.0})
    session_b.decode([{'v': 3}])
    assert session_b._step == 1
    assert session_a._step == 2


def test_full_pipeline(tmp_path):
    policy = _make_codec(tmp_path, meta={'action_fps': 10.0}).wrap(_TrackingPolicy([{'v': i} for i in range(3)]))
    policy.reset()

    result = policy.select_action({'camera': np.zeros((4, 4, 3), dtype=np.uint8), 'grip': 0.5})
    assert len(result) == 3
    assert all('decoded' in r for r in result)
    assert (tmp_path / 'episode_0001.rrd').exists()


def test_recording_codec_none_inner_identity(tmp_path):
    codec = RecordingCodec(None, tmp_path)

    assert codec.encode({'x': 1}) == {'x': 1}
    assert codec.decode({'y': 2}) == {'y': 2}
    assert codec.decode([{'a': 1}, {'b': 2}]) == [{'a': 1}, {'b': 2}]
    assert codec.meta == {}
    assert codec.dummy_encoded() == {}
    assert codec.dummy_encoded({'z': 3}) == {'z': 3}
    assert isinstance(codec.training_encoder, Derive)


def test_recording_codec_none_inner_wrap(tmp_path):
    codec = RecordingCodec(None, tmp_path)
    actions = [{'v': 1}, {'v': 2}]
    policy = codec.wrap(_TrackingPolicy(actions))
    assert isinstance(policy, _RecordingPolicy)

    policy.reset()
    result = policy.select_action({'x': 1.0})
    assert result == actions
    assert (tmp_path / 'episode_0001.rrd').exists()


def test_session_none_inner_identity(tmp_path):
    rec = rr.new_recording(application_id='test')
    rec.save(str(tmp_path / 'test.rrd'))
    session = _RecordingSession(None, rec, action_fps=10.0)

    encoded = session.encode({'x': 1.0, 'img': np.zeros((4, 4, 3), dtype=np.uint8)})
    assert encoded == {'x': 1.0, 'img': encoded['img']}

    single = session.decode({'v': 0})
    assert single == {'v': 0}

    decoded = session.decode([{'v': 1}, {'v': 2}])
    assert decoded == [{'v': 1}, {'v': 2}]

    assert session.meta == {}
    assert session.dummy_encoded() == {}
    assert session.dummy_encoded({'z': 1}) == {'z': 1}
    assert isinstance(session.training_encoder, Derive)


def test_session_none_inner_logs_input_and_model_only(tmp_path):
    rec = rr.new_recording(application_id='test')
    rec.save(str(tmp_path / 'test.rrd'))
    session = _RecordingSession(None, rec, action_fps=10.0)

    session.encode({'grip': 0.5, 'camera': np.zeros((4, 4, 3), dtype=np.uint8)})
    # input paths should be logged
    assert len(session._image_paths) > 0 or len(session._numeric_paths) > 0

    input_paths = session._image_paths + session._numeric_paths
    assert all('input' in p for p in input_paths)
    # no 'encoded' paths since inner is None
    assert not any('encoded' in p for p in input_paths)

    session.decode([{'action': np.array([1.0])}])
    all_paths = session._image_paths + session._numeric_paths
    assert any('model' in p for p in all_paths)
    assert not any('decoded' in p for p in all_paths)
