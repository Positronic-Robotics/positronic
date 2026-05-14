"""Unit tests for PolicyWrapper composition, ChunkedSchedule, and ErrorRecovery."""

from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import Recover
from positronic.policy.base import Policy, PolicyWrapper, Session
from positronic.policy.codec import ActionTimestamp, Codec
from positronic.policy.harness import ChunkedSchedule, ErrorRecovery


class _FakeClock:
    """Minimal clock stub for unit tests — caller sets ``t`` directly."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t

    def now_ns(self) -> int:
        return int(self.t * 1e9)


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
        # Relative timestamps: trajectory of duration 0.5s
        clock = _FakeClock(t=1.0)
        policy = ChunkedSchedule(clock).wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}]))
        session = policy.new_session()
        result = session(_obs())
        assert result is not None
        assert len(result) == 2
        # Timestamps stamped to absolute by ChunkedSchedule.
        assert result[0]['timestamp'] == 1.0
        assert result[1]['timestamp'] == 1.5

    def test_returns_none_while_trajectory_active(self):
        # Trajectory starts at clock=1.0, ends at 1.0+0.5=1.5.
        clock = _FakeClock(t=1.0)
        policy = ChunkedSchedule(clock).wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}]))
        session = policy.new_session()
        session(_obs())
        clock.t = 1.2
        assert session(_obs()) is None
        clock.t = 1.4
        assert session(_obs()) is None

    def test_re_infers_after_trajectory_consumed(self):
        clock = _FakeClock(t=1.0)
        inner = _ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}])
        session = ChunkedSchedule(clock).wrap(inner).new_session()
        session(_obs())  # trajectory ends at clock=1.5
        clock.t = 1.3
        assert session(_obs()) is None
        clock.t = 1.6
        result = session(_obs())
        assert result is not None
        assert inner._session.call_count == 2

    def test_single_action_refires_immediately_after(self):
        """Single action at ts=0 → trajectory_end = now → next tick re-infers."""
        clock = _FakeClock(t=1.0)
        policy = ChunkedSchedule(clock).wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}]))
        session = policy.new_session()
        session(_obs())
        clock.t = 1.01
        result = session(_obs())
        assert result is not None


class TestErrorRecovery:
    def test_delegates_when_no_error(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery(_FakeClock()).wrap(inner).new_session()
        result = session(_obs(status=RobotStatus.AVAILABLE))
        assert result == [{'v': 1}]

    def test_emits_recover_on_first_error(self):
        clock = _FakeClock(t=2.5)
        session = ErrorRecovery(clock).wrap(_ConstPolicy([{'v': 1}])).new_session()
        result = session(_obs(status=RobotStatus.ERROR))
        assert len(result) == 1
        # Wrappers produce Command objects directly; wire format lives at network boundary.
        assert isinstance(result[0]['robot_command'], Recover)
        assert result[0]['timestamp'] == 2.5
        assert 'target_grip' not in result[0]

    def test_returns_none_on_subsequent_errors(self):
        session = ErrorRecovery(_FakeClock()).wrap(_ConstPolicy([{'v': 1}])).new_session()
        session(_obs(status=RobotStatus.ERROR))
        assert session(_obs(status=RobotStatus.ERROR)) is None
        assert session(_obs(status=RobotStatus.ERROR)) is None

    def test_resumes_after_recovery(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery(_FakeClock()).wrap(inner).new_session()
        session(_obs(status=RobotStatus.ERROR))
        session(_obs(status=RobotStatus.ERROR))
        result = session(_obs(status=RobotStatus.AVAILABLE))
        assert result == [{'v': 1}]
        assert inner._session.call_count == 1

    def test_skips_inner_during_error(self):
        inner = _ConstPolicy([{'v': 1}])
        session = ErrorRecovery(_FakeClock()).wrap(inner).new_session()
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

        session = ErrorRecovery(_FakeClock()).wrap(_MetaPolicy()).new_session()
        assert session.meta == {'model': 'test'}


class TestPipelineComposition:
    """Test | operator across PolicyWrapper and Codec types."""

    def test_wrapper_pipe_wrapper(self):
        clock = _FakeClock(t=1.0)
        pipeline = ErrorRecovery(clock) | ChunkedSchedule(clock)
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}]))
        session = policy.new_session()
        result = session(_obs(status=RobotStatus.AVAILABLE))
        assert result is not None
        assert result[0]['v'] == 1

    def test_wrapper_pipe_codec(self):
        clock = _FakeClock(t=1.0)
        codec = ActionTimestamp(fps=10.0)
        pipeline = ChunkedSchedule(clock) | codec
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'action': 'test'}]))
        session = policy.new_session()
        result = session(_obs())
        assert result is not None
        assert result[0].get('timestamp') is not None

    def test_codec_pipe_wrapper(self):
        clock = _FakeClock(t=1.0)
        codec = ActionTimestamp(fps=10.0)
        pipeline = codec | ChunkedSchedule(clock)
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'action': 'test', 'timestamp': 0.0}]))
        session = policy.new_session()
        result = session(_obs())
        assert result is not None

    def test_full_pipeline(self):
        clock = _FakeClock(t=1.0)
        codec = ActionTimestamp(fps=10.0)
        pipeline = ErrorRecovery(clock) | ChunkedSchedule(clock) | codec
        assert isinstance(pipeline, PolicyWrapper)
        # 5 raw actions → codec stamps relative 0.0, 0.1, 0.2, 0.3, 0.4
        # → ChunkedSchedule shifts to 1.0, 1.1, 1.2, 1.3, 1.4 (clock=1.0).
        policy = pipeline.wrap(_ConstPolicy([{'action': f'a{i}'} for i in range(5)]))
        session = policy.new_session()
        result = session(_obs())
        assert result is not None
        assert result[0]['timestamp'] == 1.0
        # Second call within trajectory window returns None (ChunkedSchedule).
        clock.t = 1.2
        assert session(_obs()) is None

    def test_codec_and_stays_codec_only(self):
        """& only works between codecs, not wrappers."""
        c1 = ActionTimestamp(fps=10.0)
        c2 = ActionTimestamp(fps=5.0)
        composed = c1 & c2
        assert isinstance(composed, Codec)
