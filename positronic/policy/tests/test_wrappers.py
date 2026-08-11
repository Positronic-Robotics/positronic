"""Unit tests for PolicyWrapper composition, ChunkedSchedule, TemporalStack, and the policy-pipeline algebra."""

from typing import Any

import numpy as np
import pytest

from positronic import keys
from positronic.geom import Rotation, Transform3D
from positronic.policy import spec
from positronic.policy.action import (
    AbsoluteJointsAction,
    AbsolutePositionAction,
    IKJointsAction,
    JointDeltaAction,
    RelativePositionAction,
)
from positronic.policy.base import Policy, PolicyWrapper, Session
from positronic.policy.codec import (
    ActionHorizon,
    ActionTimestamp,
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    RestrictImageSize,
)
from positronic.policy.observation import ObservationCodec
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
    return {keys.OBS_TIME_NS: int(now_sec * 1e9)}


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
        session(_obs(1.0))  # trajectory ends at clock=1.5
        assert session(_obs(1.3)) is None
        clock.t = 1.6
        result = session(_obs(1.6))
        assert result is not None
        assert inner._session.call_count == 2

    def test_single_action_refires_immediately_after(self):
        """Single action at ts=0 → trajectory_end = now → next tick re-infers."""
        clock = _FakeClock(t=1.0)
        policy = ChunkedSchedule().wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}]))
        session = policy.new_session(now=clock.now)
        session(_obs(1.0))
        clock.t = 1.01
        result = session(_obs(1.01))
        assert result is not None

    def test_expiry_is_judged_at_the_observation_instant(self):
        """Whether the trajectory has run out is a question about the observation, not about ``now``."""
        inner = _ConstPolicy([{'v': 1, 'timestamp': 0.0}, {'v': 2, 'timestamp': 0.5}])
        session = ChunkedSchedule().wrap(inner).new_session(now=_FakeClock(t=2.0).now)
        session(_obs(1.0))  # anchored at now()=2.0, so the trajectory ends at 2.5
        assert session(_obs(2.4)) is None
        assert session(_obs(2.6)) is not None


class TestPipelineComposition:
    """Test | operator across PolicyWrapper and Codec types."""

    def test_wrapper_pipe_wrapper(self):
        clock = _FakeClock(t=1.0)
        pipeline = TemporalStack(keys=('v',), offsets_sec=(0.0,)) | ChunkedSchedule()
        assert isinstance(pipeline, PolicyWrapper)
        policy = pipeline.wrap(_ConstPolicy([{'v': 1, 'timestamp': 0.0}]))
        session = policy.new_session(now=clock.now)
        result = session({keys.OBS_TIME_NS: int(1e9), 'v': np.array([5.0])})
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

    def test_agreeing_declarations_merge(self):
        assert (ActionTimestamp(fps=10.0) | ActionTimestamp(fps=10.0)).meta['action_fps'] == 10.0

    def test_disagreeing_declarations_have_no_merged_answer(self):
        composed = ActionTimestamp(fps=10.0) & ActionTimestamp(fps=5.0)
        with pytest.raises(ValueError, match='action_fps'):
            _ = composed.meta

    def test_two_frame_codecs_refuse_to_advertise_one_frame(self):
        """Poses come out at the product of both transforms, which neither codec's declaration names."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        b = Transform3D(np.array([0.01, 0.0, 0.02]), Rotation.from_euler([0.0, 0.0, -0.4]))
        with pytest.raises(ValueError, match=keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(b)).meta

    def test_the_same_frame_twice_is_still_two_moves(self):
        """The second move starts where the first left off, so the shared value names neither end of the pair."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        with pytest.raises(ValueError, match=keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(a)).meta

    def test_a_relative_decoder_anchors_on_the_pose_the_policy_saw(self):
        """The right half decodes actions the left half re-expressed, so its context is re-expressed too —
        anchoring a policy-frame delta on the raw pose would send the arm somewhere else."""
        quat = Rotation.Representation.QUAT
        t = Transform3D(np.array([0.0, 0.0, 0.1]), Rotation.from_euler([0.0, 0.0, np.pi / 2]))
        raw = Transform3D(np.array([0.3, 0.1, 0.4]), Rotation.from_euler([0.0, 0.0, 0.0]))
        turn, shift = Rotation.from_euler([0.0, 0.0, 0.2]), np.array([0.01, -0.02, 0.03])
        action = {'action': np.concatenate([turn.as_quat, shift, [1.0]]).astype(np.float32)}

        codec = ChangeEEFrame(t) | RelativePositionAction()
        assert isinstance(codec, Codec)
        decoded = codec.decode(action, context={keys.EE_POSE: raw.as_vector(quat)})
        assert isinstance(decoded, dict)

        seen = raw * t
        pose = decoded[keys.ROBOT_COMMAND].pose
        want = Transform3D(seen.translation + shift, seen.rotation * turn) * t.inv
        np.testing.assert_allclose(pose.as_vector(quat), want.as_vector(quat), atol=1e-6)
        anchored_raw = Transform3D(raw.translation + shift, raw.rotation * turn) * t.inv
        assert not np.allclose(pose.translation, anchored_raw.translation)

    def test_parallel_frame_codecs_keep_the_frame_they_share(self):
        """Both halves encode the same input, so one move happens and the shared declaration describes it."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        np.testing.assert_allclose(
            (ChangeEEFrame(a) & ChangeEEFrame(a)).meta[keys.EE_FRAME], a.as_vector(Rotation.Representation.QUAT)
        )


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
    return {keys.OBS_TIME_NS: int(now_sec * 1e9), 'v': np.array([value])}


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


class TestPipelineSpec:
    """The (local, remote) pipeline split and the wire spec of the local half."""

    def test_split_on_marker(self):
        stack = TemporalStack(keys=('v',), offsets_sec=(0.0,))
        sched = ChunkedSchedule()
        codec = ActionTimestamp(fps=10.0)
        local, border, rem = spec.split(stack | sched | spec.remote | codec)
        assert local is not None and local._wrappers() == (stack, sched)
        assert border is spec.remote
        assert rem is codec

    def test_split_empty_halves(self):
        assert spec.split(spec.remote) == (None, spec.remote, None)
        local, _, rem = spec.split(ChunkedSchedule() | spec.remote)
        assert rem is None and isinstance(local, ChunkedSchedule)

    def test_split_requires_exactly_one_marker(self):
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(ChunkedSchedule() | ChunkedSchedule())
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(spec.remote | spec.remote)

    def test_split_recomposes_codec_half_as_codec(self):
        rem = spec.split(spec.remote | ActionTimestamp(fps=10.0) | ActionTimestamp(fps=5.0))[2]
        assert isinstance(rem, Codec)

    def test_border_carries_the_wire_settings(self):
        """``remote`` is the plain border; calling it describes the wire without changing the split."""
        border = spec.split(ChunkedSchedule() | spec.remote(compress_images=True) | ActionTimestamp(fps=10.0))[1]
        assert border.compress_images is True
        assert spec.remote.compress_images is False

    def test_marker_cannot_be_applied(self):
        with pytest.raises(TypeError, match='border'):
            spec.remote.wrap(_ConstPolicy([]))

    def test_spec_round_trip(self):
        stack = TemporalStack(keys=('a', 'b'), offsets_sec=(-0.5, 0.0), pad_start=False) | ChunkedSchedule()
        rebuilt = spec.from_spec(stack.to_spec())
        assert rebuilt is not None and rebuilt.to_spec() == stack.to_spec()

    def test_codec_spec_round_trip(self):
        obs = ObservationCodec(
            state={'observation.state': {'grip': 1}}, images={'left': (keys.WRIST_IMAGE, (224, 224))}
        )
        local = ChunkedSchedule() | ActionTimestamp(fps=10.0) | (obs & AbsolutePositionAction('pose', 'grip'))
        rebuilt = spec.from_spec(local.to_spec())
        assert rebuilt is not None and rebuilt.to_spec() == local.to_spec()

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
        assert rebuilt is not None and rebuilt.to_spec() == composed.to_spec()
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
            IKJointsAction(solver_cls=None).to_spec()

    def test_wire_names_match_table(self):
        instances = {
            'chunked_schedule': ChunkedSchedule(),
            'temporal_stack': TemporalStack(('v',), (0.0,)),
            'action_timestamp': ActionTimestamp(fps=10.0),
            'action_horizon': ActionHorizon(1.0),
            'binarize_grip_training': BinarizeGripTraining(('grip',)),
            'binarize_grip_inference': BinarizeGripInference(),
            'flip_grip': FlipGrip(),
            'restrict_image_size': RestrictImageSize(),
            'observation_codec': ObservationCodec(state={}, images={}),
            'absolute_position_action': AbsolutePositionAction(keys.TARGET_EE_POSE, 'target_grip'),
            'absolute_joints_action': AbsoluteJointsAction(keys.TARGET_JOINTS, 'target_grip'),
            'relative_position_action': RelativePositionAction(),
            'joint_delta_action': JointDeltaAction(),
            'change_ee_frame': ChangeEEFrame(Transform3D.identity),
        }
        assert set(instances) == set(spec.WIRE_WRAPPERS)
        for name, instance in instances.items():
            assert instance.to_spec()['name'] == name
            assert type(instance) is spec.WIRE_WRAPPERS[name]


class _ListSource(spec.ModelSource):
    def __init__(self, models):
        self._models = list(models)

    def get_models(self):
        return list(self._models)

    def load(self, model_id, on_progress=None):
        return _ConstPolicy([{'model': model_id}])


class TestPipe:
    """The source terminal: ``... | source`` closes a wrapper chain into a Pipeline."""

    def test_wrapper_chain_terminates_into_pipe(self):
        stack = TemporalStack(keys=('v',), offsets_sec=(0.0,))
        sched = ChunkedSchedule()
        codec = ActionTimestamp(fps=10.0)
        source = spec.PolicySource(_ConstPolicy([]))
        pipeline = stack | sched | spec.remote | codec | source
        assert isinstance(pipeline, spec.Pipeline)
        assert pipeline.components == (stack, sched, spec.remote, codec)
        assert pipeline.source is source

    def test_lone_codec_terminates_into_pipe(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = codec | spec.PolicySource(_ConstPolicy([]))
        assert isinstance(pipeline, spec.Pipeline)
        assert pipeline.components == (codec,)

    def test_bare_marker_terminates_into_pipe(self):
        pipeline = spec.remote | spec.PolicySource(_ConstPolicy([]))
        assert isinstance(pipeline, spec.Pipeline)
        assert pipeline.components == (spec.remote,)

    def test_split_pipe(self):
        sched = ChunkedSchedule()
        codec = ActionTimestamp(fps=10.0)
        local, border, rem = spec.split(sched | spec.remote | codec | spec.PolicySource(_ConstPolicy([])))
        assert local is sched
        assert border is spec.remote
        assert rem is codec

    def test_split_pipe_requires_exactly_one_marker(self):
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(ChunkedSchedule() | spec.PolicySource(_ConstPolicy([])))

    def test_pipe_refuses_a_frame_declared_on_both_sides_of_the_wire(self):
        """Rig-side and server-side conversion are alternatives; running both puts poses at the product."""
        t = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        chain = ChangeEEFrame(t) | spec.remote | (ActionTimestamp(fps=10.0) | ChangeEEFrame(t))
        with pytest.raises(ValueError, match=keys.EE_FRAME):
            _ = chain | spec.PolicySource(_ConstPolicy([]))

    def test_pipe_composes_no_further(self):
        pipeline: Any = ChunkedSchedule() | spec.remote | spec.PolicySource(_ConstPolicy([]))
        with pytest.raises(TypeError):
            _ = pipeline | ActionTimestamp(fps=10.0)
        with pytest.raises(TypeError):
            _ = ChunkedSchedule() | pipeline
        with pytest.raises(TypeError):
            _ = pipeline | spec.PolicySource(_ConstPolicy([]))

    def test_inline_full_pipe(self):
        clock = _FakeClock(t=1.0)
        inner = _ConstPolicy([{'action': f'a{i}'} for i in range(5)])
        policy = spec.inline(ChunkedSchedule() | spec.remote | ActionTimestamp(fps=10.0) | spec.PolicySource(inner))
        assert isinstance(policy, Policy)
        session = policy.new_session(now=clock.now)
        result = session(_obs())
        assert result is not None
        assert result[0]['timestamp'] == 1.0
        clock.t = 1.2
        assert session(_obs()) is None

    def test_inline_tolerates_marker_less_pipe(self):
        clock = _FakeClock(t=1.0)
        inner = _ConstPolicy([{'v': 1, 'timestamp': 0.0}])
        policy = spec.inline(ChunkedSchedule() | spec.PolicySource(inner))
        session = policy.new_session(now=clock.now)
        result = session(_obs())
        assert result is not None and result[0]['timestamp'] == 1.0

    def test_inline_bare_source_pipe_is_the_loaded_policy(self):
        inner = _ConstPolicy([])
        assert spec.inline(spec.remote | spec.PolicySource(inner)) is inner

    def test_inline_loads_the_latest_model(self):
        policy = spec.inline(spec.remote | _ListSource(['a', 'b']))
        assert isinstance(policy, _ConstPolicy)
        assert policy._actions == [{'model': 'b'}]

    def test_resolve_defaults_to_latest(self):
        source = _ListSource(['a', 'b', 'c'])
        assert source.resolve(None) == 'c'
        assert source.resolve('a') == 'a'
        with pytest.raises(ValueError, match='nope'):
            source.resolve('nope')

    def test_source_equality_is_structural(self):
        policy = _ConstPolicy([])
        assert spec.PolicySource(policy) == spec.PolicySource(policy)
        assert spec.PolicySource(policy, name='x') != spec.PolicySource(policy)
        assert spec.PolicySource(policy) != spec.PolicySource(_ConstPolicy([]))
        assert _ListSource(['a']) == _ListSource(['a'])
        assert _ListSource(['a']) != _ListSource(['b'])
        assert spec.PolicySource(policy) != _ListSource(['a'])

        class _SubSource(spec.PolicySource):
            pass

        assert _SubSource(policy) != spec.PolicySource(policy)

    def test_policy_source(self):
        policy = _ConstPolicy([])
        source = spec.PolicySource(policy, name='const')
        assert source.get_models() == ['const']
        assert source.resolve(None) == 'const'
        progress = []
        assert source.load('const', on_progress=progress.append) is policy
        assert progress == []
        assert source.meta('const') == {}
        assert spec.PolicySource(policy).get_models() == ['default']


def _image(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestRestrictImageSize:
    def test_bounds_every_image(self):
        result = RestrictImageSize(64, 48).encode({
            'cam_a': _image(480, 640),
            'cam_b': _image(240, 320),
            'state': np.array([1.0]),
        })
        assert result['cam_a'].shape == (48, 64, 3)
        assert result['cam_b'].shape == (48, 64, 3)
        np.testing.assert_array_equal(result['state'], np.array([1.0]))

    def test_defaults_to_the_standard_bound(self):
        assert RestrictImageSize().encode({'cam': _image(1080, 1920)})['cam'].shape == (360, 640, 3)

    def test_aspect_is_kept_and_images_only_shrink(self):
        result = RestrictImageSize(160, 160).encode({'wide': _image(480, 640), 'small': _image(24, 32)})
        assert result['wide'].shape == (120, 160, 3)
        assert result['small'].shape == (24, 32, 3)

    def test_image_within_bound_is_the_same_object(self):
        img = _image(48, 64)
        assert RestrictImageSize(64, 48).encode({'cam': img})['cam'] is img

    def test_stacked_frames_are_bounded_per_frame(self):
        stack = np.zeros((3, 480, 640, 3), dtype=np.uint8)
        assert RestrictImageSize(64, 48).encode({'cam': stack})['cam'].shape == (3, 48, 64, 3)

    def test_nested_images_are_reached(self):
        result = RestrictImageSize(64, 48).encode({'video': {'cam': _image(480, 640)}, 'seq': [_image(480, 640)]})
        assert result['video']['cam'].shape == (48, 64, 3)
        assert result['seq'][0].shape == (48, 64, 3)

    def test_non_image_values_pass_through(self):
        obs = {'state': np.array([1.0, 2.0]), 'task': 'pick cube', 'flag': True}
        result = RestrictImageSize(64, 48).encode(obs)
        np.testing.assert_array_equal(result['state'], obs['state'])
        assert result['task'] == 'pick cube'
        assert result['flag'] is True

    def test_actions_pass_through_untouched(self):
        actions = [{'target_grip': 0.5}, {'target_grip': 1.0}]
        assert RestrictImageSize(64, 48).decode(actions) == actions

    def test_training_encoder_refuses(self):
        with pytest.raises(NotImplementedError, match='full-resolution'):
            _ = RestrictImageSize(64, 48).training_encoder

    def test_survives_a_wire_round_trip(self):
        rebuilt = spec.from_spec(RestrictImageSize(64, 48).to_spec())
        assert isinstance(rebuilt, RestrictImageSize)
        assert rebuilt.encode({'cam': _image(480, 640)})['cam'].shape == (48, 64, 3)
