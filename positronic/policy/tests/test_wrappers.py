"""Unit tests for PolicyWrapper composition, ChunkedSchedule, and ErrorRecovery."""

from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import Recover, to_wire
from positronic.policy.base import Policy, PolicyWrapper, Session
from positronic.policy.codec import ActionTimestamp, Codec
from positronic.policy.harness import ChunkedSchedule, ErrorRecovery


class _ConstSession(Session):
    def __init__(self, actions):
        self._actions = actions
        self.call_count = 0

    def __call__(self, obs):
        self.call_count += 1
        return self._actions


class _ConstPolicy(Policy):
    def __init__(self, actions):
        self._actions = actions
        self._session: _ConstSession | None = None

    def new_session(self, context=None):
        self._session = _ConstSession(self._actions)
        return self._session


def _obs(now_sec=0.0, status=RobotStatus.AVAILABLE):
    return {'inference_time_ns': int(now_sec * 1e9), 'robot_state.status': status}


class TestChunkedSchedule:
    def test_first_call_runs_inference(self):
        policy = ChunkedSchedule().wrap(_ConstPolicy([{'v': 1, 'timestamp': 1.0}, {'v': 2, 'timestamp': 1.5}]))
        session = policy.new_session()
        result = session(_obs(now_sec=1.0))
        assert result is not None
        assert len(result) == 2

    def test_returns_none_while_trajectory_active(self):
        policy = ChunkedSchedule().wrap(_ConstPolicy([{'v': 1, 'timestamp': 1.0}, {'v': 2, 'timestamp': 1.5}]))
        session = policy.new_session()
        session(_obs(now_sec=1.0))
        assert session(_obs(now_sec=1.2)) is None
        assert session(_obs(now_sec=1.4)) is None

    def test_re_infers_after_trajectory_consumed(self):
        inner = _ConstPolicy([{'v': 1, 'timestamp': 1.0}, {'v': 2, 'timestamp': 1.5}])
        session = ChunkedSchedule().wrap(inner).new_session()
        session(_obs(now_sec=1.0))
        assert session(_obs(now_sec=1.3)) is None
        result = session(_obs(now_sec=1.6))
        assert result is not None
        assert inner._session.call_count == 2

    def test_single_action_refires_immediately_after(self):
        """Single action at now → trajectory_end = now → next tick re-infers."""
        policy = ChunkedSchedule().wrap(_ConstPolicy([{'v': 1, 'timestamp': 1.0}]))
        session = policy.new_session()
        session(_obs(now_sec=1.0))
        result = session(_obs(now_sec=1.01))
        assert result is not None


class TestErrorRecovery:
    def test_delegates_when_no_error(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery().wrap(inner).new_session()
        result = session(_obs(status=RobotStatus.AVAILABLE))
        assert result == [{'v': 1}]

    def test_emits_recover_on_first_error(self):
        session = ErrorRecovery().wrap(_ConstPolicy([{'v': 1}])).new_session()
        result = session(_obs(status=RobotStatus.ERROR))
        assert len(result) == 1
        assert result[0]['robot_command'] == to_wire(Recover())
        assert 'target_grip' not in result[0]

    def test_returns_none_on_subsequent_errors(self):
        session = ErrorRecovery().wrap(_ConstPolicy([{'v': 1}])).new_session()
        session(_obs(status=RobotStatus.ERROR))
        assert session(_obs(status=RobotStatus.ERROR)) is None
        assert session(_obs(status=RobotStatus.ERROR)) is None

    def test_resumes_after_recovery(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery().wrap(inner).new_session()
        session(_obs(status=RobotStatus.ERROR))
        session(_obs(status=RobotStatus.ERROR))
        result = session(_obs(status=RobotStatus.AVAILABLE))
        assert result == [{'v': 1}]
        assert inner._session.call_count == 1

    def test_skips_inner_during_error(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery().wrap(inner).new_session()
        session(_obs(status=RobotStatus.AVAILABLE))
        count_before = inner._session.call_count
        session(_obs(status=RobotStatus.ERROR))
        session(_obs(status=RobotStatus.ERROR))
        assert inner._session.call_count == count_before

    def test_delegates_meta(self):
        class _MetaSession(Session):
            def __call__(self, obs):
                return []

            @property
            def meta(self):
                return {'model': 'test'}

        class _MetaPolicy(Policy):
            def new_session(self, context=None):
                return _MetaSession()

            @property
            def meta(self):
                return {'model': 'test'}

        session = ErrorRecovery().wrap(_MetaPolicy()).new_session()
        assert session.meta == {'model': 'test'}


class TestPipelineComposition:
    """Test | operator across PolicyWrapper and Codec types."""

    def test_wrapper_pipe_wrapper(self):
        pipeline = ErrorRecovery() | ChunkedSchedule()
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'v': 1, 'timestamp': 1.0}]))
        session = policy.new_session()
        result = session(_obs(now_sec=1.0, status=RobotStatus.AVAILABLE))
        assert result is not None
        assert result[0]['v'] == 1

    def test_wrapper_pipe_codec(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = ChunkedSchedule() | codec
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'action': 'test'}]))
        session = policy.new_session()
        result = session(_obs(now_sec=1.0))
        assert result is not None
        assert result[0].get('timestamp') is not None

    def test_codec_pipe_wrapper(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = codec | ChunkedSchedule()
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'action': 'test', 'timestamp': 1.0}]))
        session = policy.new_session()
        result = session(_obs(now_sec=1.0))
        assert result is not None

    def test_full_pipeline(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = ErrorRecovery() | ChunkedSchedule() | codec
        assert isinstance(pipeline, PolicyWrapper)
        # 5 raw actions → codec stamps at 1.0, 1.1, 1.2, 1.3, 1.4
        policy = pipeline.wrap(_ConstPolicy([{'action': f'a{i}'} for i in range(5)]))
        session = policy.new_session()
        result = session(_obs(now_sec=1.0))
        assert result is not None
        assert result[0].get('timestamp') is not None
        # Second call within trajectory window returns None (ChunkedSchedule)
        assert session(_obs(now_sec=1.2)) is None

    def test_codec_and_stays_codec_only(self):
        """& only works between codecs, not wrappers."""
        c1 = ActionTimestamp(fps=10.0)
        c2 = ActionTimestamp(fps=5.0)
        composed = c1 & c2
        assert isinstance(composed, Codec)
