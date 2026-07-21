"""Unit tests for PolicyWrapper composition, ChunkedSchedule, TemporalStack, and the definition spec."""

import numpy as np
import pytest

from positronic.policy import spec
from positronic.policy.base import Policy, PolicyWrapper, Session
from positronic.policy.codec import ActionTimestamp, Codec
from positronic.policy.wrappers import ChunkedSchedule, TemporalStack


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

    def new_session(self, context=None, now=None):
        self._session = _ConstSession(self._actions)
        return self._session


def _obs(now_sec=0.0):
    return {'obs_time_ns': int(now_sec * 1e9)}


class TestChunkedSchedule:
    def test_first_call_runs_inference(self):
        # Relative timestamps: trajectory of duration 0.5s
        clock = _FakeClock(t=1.0)
        inner = _ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}])
        policy = ChunkedSchedule().wrap(inner)
        session = policy.new_session(now=clock.now)
        result = session(_obs())
        assert result is not None
        assert len(result) == 2
        # Timestamps stamped to absolute by ChunkedSchedule.
        assert result[0]['timestamp'] == 1.0
        assert result[1]['timestamp'] == 1.5

    def test_returns_none_while_trajectory_active(self):
        # Trajectory starts at clock=1.0, ends at 1.0+0.5=1.5.
        clock = _FakeClock(t=1.0)
        inner = _ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}])
        policy = ChunkedSchedule().wrap(inner)
        session = policy.new_session(now=clock.now)
        session(_obs())
        clock.t = 1.2
        assert session(_obs()) is None
        clock.t = 1.4
        assert session(_obs()) is None

    def test_re_infers_after_trajectory_consumed(self):
        clock = _FakeClock(t=1.0)
        inner = _ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}])
        session = ChunkedSchedule().wrap(inner).new_session(now=clock.now)
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
        policy = ChunkedSchedule().wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}]))
        session = policy.new_session(now=clock.now)
        session(_obs())
        clock.t = 1.01
        result = session(_obs())
        assert result is not None


class TestPipelineComposition:
    """Test | operator across PolicyWrapper and Codec types."""

    def test_wrapper_pipe_wrapper(self):
        clock = _FakeClock(t=1.0)
        pipeline = TemporalStack(keys=('v',), offsets_sec=(0.0,)) | ChunkedSchedule()
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}]))
        session = policy.new_session(now=clock.now)
        result = session({'obs_time_ns': int(1e9), 'v': np.array([5.0])})
        assert result is not None
        assert result[0]['v'] == 1

    def test_codec_pipe_wrapper(self):
        clock = _FakeClock(t=1.0)
        codec = ActionTimestamp(fps=10.0)
        pipeline = codec | ChunkedSchedule()
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'action': 'test', 'timestamp': 0.0}]))
        session = policy.new_session(now=clock.now)
        result = session(_obs())
        assert result is not None

    def test_full_pipeline(self):
        clock = _FakeClock(t=1.0)
        codec = ActionTimestamp(fps=10.0)
        pipeline = ChunkedSchedule() | codec
        assert isinstance(pipeline, PolicyWrapper)
        # 5 raw actions → codec stamps relative 0.0, 0.1, 0.2, 0.3, 0.4
        # → ChunkedSchedule shifts to 1.0, 1.1, 1.2, 1.3, 1.4 (clock=1.0).
        policy = pipeline.wrap(_ConstPolicy([{'action': f'a{i}'} for i in range(5)]))
        session = policy.new_session(now=clock.now)
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


class _CaptureSession(Session):
    def __init__(self):
        self.seen = []

    def __call__(self, obs):
        self.seen.append(obs)
        return []


class _CapturePolicy(Policy):
    def __init__(self):
        self.session = _CaptureSession()

    def new_session(self, context=None, now=None):
        return self.session


def _stack_obs(now_sec, value):
    return {'obs_time_ns': int(now_sec * 1e9), 'v': np.array([value])}


class TestTemporalStack:
    OFFSETS = (-0.2, -0.1, 0.0)

    def test_pad_start_repeats_oldest(self):
        clock = _FakeClock(t=0.0)
        inner = _CapturePolicy()
        session = TemporalStack(keys=('v',), offsets_sec=self.OFFSETS).wrap(inner).new_session(now=clock.now)
        session(_stack_obs(0.0, 1.0))
        stack = inner.session.seen[0]['v']
        assert stack.shape == (3, 1)
        assert (stack == 1.0).all()

    def test_no_pad_start_grows_from_one(self):
        clock = _FakeClock(t=0.0)
        inner = _CapturePolicy()
        wrapper = TemporalStack(keys=('v',), offsets_sec=self.OFFSETS, pad_start=False)
        session = wrapper.wrap(inner).new_session(now=clock.now)

        session(_stack_obs(0.0, 1.0))
        assert inner.session.seen[0]['v'].shape == (1, 1)

        session(_stack_obs(0.1, 2.0))
        assert inner.session.seen[1]['v'].shape == (2, 1)
        assert inner.session.seen[1]['v'][:, 0].tolist() == [1.0, 2.0]

        session(_stack_obs(0.2, 3.0))
        assert inner.session.seen[2]['v'].shape == (3, 1)
        assert inner.session.seen[2]['v'][:, 0].tolist() == [1.0, 2.0, 3.0]

    def test_no_pad_start_full_window_matches_padded(self):
        offsets = self.OFFSETS
        stacks = {}
        for pad_start in (True, False):
            clock = _FakeClock(t=0.0)
            inner = _CapturePolicy()
            wrapper = TemporalStack(keys=('v',), offsets_sec=offsets, pad_start=pad_start)
            session = wrapper.wrap(inner).new_session(now=clock.now)
            for i in range(4):
                session(_stack_obs(0.1 * i, float(i)))
            stacks[pad_start] = inner.session.seen[-1]['v']
        assert stacks[True].shape == stacks[False].shape == (3, 1)
        assert (stacks[True] == stacks[False]).all()


class TestDefinitionSpec:
    """The (local, remote) definition split and the wire spec of the local half."""

    def test_split_on_marker(self):
        stack = TemporalStack(keys=('v',), offsets_sec=(0.0,))
        sched = ChunkedSchedule()
        codec = ActionTimestamp(fps=10.0)
        local, rem = spec.split(stack | sched | spec.remote | codec)
        assert local._pipeline_components() == (stack, sched)
        assert rem is codec

    def test_split_empty_halves(self):
        assert spec.split(spec.remote) == (None, None)
        local, rem = spec.split(ChunkedSchedule() | spec.remote)
        assert rem is None and isinstance(local, ChunkedSchedule)

    def test_split_requires_exactly_one_marker(self):
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(ChunkedSchedule() | ChunkedSchedule())
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(spec.remote | spec.remote)

    def test_split_recomposes_codec_half_as_codec(self):
        rem = spec.split(spec.remote | ActionTimestamp(fps=10.0) | ActionTimestamp(fps=5.0))[1]
        assert isinstance(rem, Codec)

    def test_inline_drops_marker(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = spec.inline(ChunkedSchedule() | spec.remote | codec)
        assert pipeline._pipeline_components()[-1] is codec
        assert spec.inline(spec.remote) is None

    def test_marker_cannot_be_applied(self):
        with pytest.raises(TypeError, match='border'):
            spec.remote.wrap(_ConstPolicy([]))

    def test_spec_round_trip(self):
        stack = TemporalStack(keys=('a', 'b'), offsets_sec=(-0.5, 0.0), pad_start=False) | ChunkedSchedule()
        rebuilt = spec.from_spec(stack.to_spec())
        assert rebuilt.to_spec() == stack.to_spec()

    def test_leaf_without_args_omits_args_key(self):
        assert ChunkedSchedule().to_spec() == {'name': 'chunked_schedule'}

    def test_par_topology_round_trips(self, monkeypatch):
        class _WireCodec(Codec):
            def __init__(self, tag=''):
                self._tag = tag

            def to_spec(self):
                return {'name': 'wire_codec', 'args': {'tag': self._tag}}

        monkeypatch.setitem(spec.WIRE_WRAPPERS, 'wire_codec', _WireCodec)
        composed = _WireCodec('t') | (_WireCodec('a') & _WireCodec('b'))
        rebuilt = spec.from_spec(composed.to_spec())
        assert rebuilt.to_spec() == composed.to_spec()
        assert composed.to_spec() == {
            'seq': [
                {'name': 'wire_codec', 'args': {'tag': 't'}},
                {'par': [{'name': 'wire_codec', 'args': {'tag': 'a'}}, {'name': 'wire_codec', 'args': {'tag': 'b'}}]},
            ]
        }

    def test_par_of_non_codecs_is_rejected(self):
        with pytest.raises(TypeError):
            spec.from_spec({'par': [{'name': 'chunked_schedule'}, {'name': 'chunked_schedule'}]})

    def test_empty_declaration_builds_nothing(self):
        assert spec.from_spec({'seq': []}) is None

    def test_unknown_name_lists_vocabulary(self):
        with pytest.raises(ValueError, match='chunked_schedule'):
            spec.from_spec({'name': 'not_a_wrapper'})

    def test_unknown_arg_fails(self):
        with pytest.raises(TypeError):
            spec.from_spec({'name': 'temporal_stack', 'args': {'keys': ['v'], 'offsets_sec': [0.0], 'bogus': 1}})

    def test_non_deliverable_wrapper_fails_loudly(self):
        with pytest.raises(NotImplementedError, match='not deliverable'):
            ActionTimestamp(fps=10.0).to_spec()

    def test_wire_names_match_table(self):
        instances = {'chunked_schedule': ChunkedSchedule(), 'temporal_stack': TemporalStack(('v',), (0.0,))}
        assert set(instances) == set(spec.WIRE_WRAPPERS)
        for name, instance in instances.items():
            assert instance.to_spec()['name'] == name
            assert type(instance) is spec.WIRE_WRAPPERS[name]
