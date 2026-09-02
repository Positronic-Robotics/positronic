"""Unit tests for Layer composition, ChunkedSchedule, TemporalStack, and the policy-pipeline algebra."""

from typing import Any

import numpy as np
import pytest

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.command import Impedance, JointDelta
from positronic.geom import Rotation, Transform3D
from positronic.policy import spec
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, IKJointsAction, JointDeltaAction
from positronic.policy.base import Layer, Policy, Session
from positronic.policy.codec import (
    ActionHorizon,
    ActionTimestamp,
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    RestrictImageSize,
    SetControlMode,
)
from positronic.policy.layers import ChunkedSchedule, StopOnFault, TemporalStack
from positronic.policy.observation import ObservationCodec


class _ConstSession(Session):
    def __init__(self, actions):
        self._actions = actions
        self.call_count = 0

    def __call__(self, obs, time_ns):
        self.call_count += 1
        return self._actions


class _ConstPolicy(Policy):
    def __init__(self, actions):
        self._actions = actions
        self._session: _ConstSession | None = None

    def new_session(self, context=None, rt=None):
        self._session = _ConstSession(self._actions)
        return self._session


def _obs(now_sec=0.0, status=RobotStatus.AVAILABLE):
    return {keys.OBS_TIME_NS: int(now_sec * 1e9), keys.ROBOT_STATUS: status}


class TestStopOnFault:
    @pytest.mark.parametrize('unavailable', [RobotStatus.ERROR, RobotStatus.BUSY])
    def test_an_unavailable_arm_stops_what_is_executing(self, unavailable):
        inner = _ConstSession([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        session = StopOnFault().make_session(inner)

        assert session(_obs(0.0, unavailable), 0) == []
        assert inner.call_count == 0, 'the model was asked about an arm that is not tracking it'

    def test_an_available_arm_reaches_the_model(self):
        inner = _ConstSession([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        session = StopOnFault().make_session(inner)

        assert session(_obs(0.0, RobotStatus.AVAILABLE), 0) is not None
        assert inner.call_count == 1

    def test_an_observation_with_no_arm_status_reaches_the_model(self):
        """A probe replaying a recording has no arm to stop for."""
        inner = _ConstSession([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        session = StopOnFault().make_session(inner)

        assert session({keys.OBS_TIME_NS: 0}, 0) is not None
        assert inner.call_count == 1

    def test_either_arm_of_a_bimanual_rig_stops_the_pair(self):
        """Whichever arm is unavailable stops the pair, and the status counts as its number: a server-side stack
        reads it off a wire with no enum to carry."""
        inner = _ConstSession([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        session = StopOnFault().make_session(inner)
        obs = {
            keys.OBS_TIME_NS: 0,
            f'{keys.ROBOT_STATE}.left.status': int(RobotStatus.AVAILABLE),
            f'{keys.ROBOT_STATE}.right.status': int(RobotStatus.ERROR),
        }

        assert session(obs, 0) == []
        assert inner.call_count == 0

    def test_the_status_a_recording_carries_for_a_taken_arm_stops_the_policy(self):
        """The numbers are the contract between a rig and a server: 1 is an arm its driver has taken."""
        inner = _ConstSession([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        session = StopOnFault().make_session(inner)

        assert (RobotStatus.AVAILABLE, RobotStatus.BUSY, RobotStatus.ERROR) == (0, 1, 3)
        assert session({keys.OBS_TIME_NS: 0, keys.ROBOT_STATUS: 1}, 0) == []
        assert inner.call_count == 0

    def test_the_status_published_for_a_travelling_arm_reaches_the_model(self):
        """The wire protocol publishes 2 for an arm on its way to a setpoint, which is one taking commands."""
        inner = _ConstSession([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        session = StopOnFault().make_session(inner)

        assert session({keys.OBS_TIME_NS: 0, keys.ROBOT_STATUS: 2}, 0) is not None
        assert inner.call_count == 1

    def test_a_status_no_arm_answers_to_raises(self):
        """A number outside ``RobotStatus`` is the rig and the server disagreeing about the protocol, which
        is not something to drive an arm through."""
        session = StopOnFault().make_session(_ConstSession([]))

        with pytest.raises(ValueError):
            session({keys.OBS_TIME_NS: 0, keys.ROBOT_STATUS: 99}, 0)

    def test_recovery_plans_afresh_instead_of_resuming(self):
        """The stop resets the scheduler below it, so the first observation from an available arm infers
        again rather than waiting out the chunk stamped before."""
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {'v': 2, keys.ACTION_TIMESTAMP: 1.0}])
        session = (StopOnFault() | ChunkedSchedule()).wrap(inner).new_session()

        assert session(_obs(0.0), 0) is not None  # a chunk that runs until 1.0
        assert session(_obs(0.2, RobotStatus.ERROR), int(0.2e9)) == []
        assert session(_obs(0.3), int(0.3e9)) is not None


class _ScriptedSession(Session):
    """Answers each of ``script`` in turn — ``None`` where a session has nothing to place yet."""

    def __init__(self, script):
        self._script = list(script)
        self.call_count = 0

    def __call__(self, obs, time_ns):
        self.call_count += 1
        return self._script.pop(0)


class TestChunkedSchedule:
    def test_an_inner_with_no_answer_yet_is_asked_again(self):
        """A session that waits for a served function answers ``None``, which is no trajectory. The layer
        passes the ``None`` on and asks again on the next observation."""
        inner = _ScriptedSession([None, None, [{'v': 1, keys.ACTION_TIMESTAMP: 0.0}]])
        session = ChunkedSchedule().make_session(inner)

        assert session(_obs(0.0), int(1e9)) is None
        assert session(_obs(0.1), int(1e9)) is None
        assert session(_obs(0.2), int(1e9)) == [{'v': 1, keys.ACTION_TIMESTAMP: 1.0}]
        assert inner.call_count == 3

    def test_first_call_runs_inference(self):
        # Relative timestamps: trajectory of duration 0.5s
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {'v': 2, keys.ACTION_TIMESTAMP: 0.5}])
        policy = ChunkedSchedule().wrap(inner)
        session = policy.new_session()
        result = session(_obs(), int(1e9))
        assert result is not None
        assert len(result) == 2
        # Timestamps stamped to absolute by ChunkedSchedule.
        assert result[0][keys.ACTION_TIMESTAMP] == 1.0
        assert result[1][keys.ACTION_TIMESTAMP] == 1.5

    def test_returns_none_while_trajectory_active(self):
        # The trajectory starts at the call time 1.0 and ends at 1.0+0.5=1.5.
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {'v': 2, keys.ACTION_TIMESTAMP: 0.5}])
        policy = ChunkedSchedule().wrap(inner)
        session = policy.new_session()
        session(_obs(), int(1e9))
        assert session(_obs(), int(1.2e9)) is None
        assert session(_obs(), int(1.4e9)) is None

    def test_re_infers_after_trajectory_consumed(self):
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {'v': 2, keys.ACTION_TIMESTAMP: 0.5}])
        session = ChunkedSchedule().wrap(inner).new_session()
        session(_obs(1.0), int(1e9))  # trajectory ends at 1.5
        assert session(_obs(1.3), int(1.3e9)) is None
        result = session(_obs(1.6), int(1.6e9))
        assert result is not None
        assert inner._session.call_count == 2

    def test_single_action_refires_immediately_after(self):
        """Single action at ts=0 → trajectory_end is the call's time → next tick re-infers."""
        policy = ChunkedSchedule().wrap(_ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}]))
        session = policy.new_session()
        session(_obs(1.0), int(1e9))
        result = session(_obs(1.01), int(1.01e9))
        assert result is not None

    def test_expiry_is_judged_at_the_observation_instant(self):
        """Whether the trajectory has run out is a question about the observation, not about the call's time."""
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {'v': 2, keys.ACTION_TIMESTAMP: 0.5}])
        session = ChunkedSchedule().wrap(inner).new_session()
        session(_obs(1.0), int(2e9))  # anchored at the call's time 2.0, so the trajectory ends at 2.5
        assert session(_obs(2.4), int(2e9)) is None
        assert session(_obs(2.6), int(2e9)) is not None


class TestPipelineComposition:
    """Test | operator across Layer and Codec types."""

    def test_layer_pipe_layer(self):
        pipeline = TemporalStack(keys=('v',), offsets_sec=(0.0,)) | ChunkedSchedule()
        assert isinstance(pipeline, Layer)
        policy = pipeline.wrap(_ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}]))
        session = policy.new_session()
        result = session({keys.OBS_TIME_NS: int(1e9), 'v': np.array([5.0])}, int(1e9))
        assert result is not None
        assert result[0]['v'] == 1

    def test_codec_pipe_layer(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = codec | ChunkedSchedule()
        assert isinstance(pipeline, Layer)
        policy = pipeline.wrap(_ConstPolicy([{'action': 'test', keys.ACTION_TIMESTAMP: 0.0}]))
        session = policy.new_session()
        result = session(_obs(), int(1e9))
        assert result is not None

    def test_full_pipeline(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = ChunkedSchedule() | codec
        assert isinstance(pipeline, Layer)
        # 5 raw actions → codec stamps relative 0.0, 0.1, 0.2, 0.3, 0.4
        # → ChunkedSchedule shifts to 1.0, 1.1, 1.2, 1.3, 1.4 (call time 1.0).
        policy = pipeline.wrap(_ConstPolicy([{'action': f'a{i}'} for i in range(5)]))
        session = policy.new_session()
        result = session(_obs(), int(1e9))
        assert result is not None
        assert result[0][keys.ACTION_TIMESTAMP] == 1.0
        # Second call within trajectory window returns None (ChunkedSchedule).
        assert session(_obs(), int(1.2e9)) is None

    def test_codec_and_stays_codec_only(self):
        """& only works between codecs, not layers."""
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
        with pytest.raises(ValueError, match=roboarm_keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(b)).meta

    def test_the_same_frame_twice_is_still_two_moves(self):
        """The second move starts where the first left off, so the shared value names neither end of the pair."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        with pytest.raises(ValueError, match=roboarm_keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(a)).meta

    def test_parallel_frame_codecs_keep_the_frame_they_share(self):
        """Both halves encode the same input, so one move happens and the shared declaration describes it."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        np.testing.assert_allclose(
            (ChangeEEFrame(a) & ChangeEEFrame(a)).meta[roboarm_keys.EE_FRAME], a.as_vector(Rotation.Representation.QUAT)
        )


class _CaptureSession(Session):
    def __init__(self):
        self.seen = []

    def __call__(self, obs, time_ns):
        self.seen.append(obs)
        return []


class _CapturePolicy(Policy):
    def __init__(self):
        self.session = _CaptureSession()

    def new_session(self, context=None, rt=None):
        return self.session


def _stack_obs(now_sec, value):
    return {keys.OBS_TIME_NS: int(now_sec * 1e9), 'v': np.array([value])}


IMPEDANCE = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)


class TestSetControlMode:
    def test_every_command_in_a_chunk_carries_the_mode(self):
        chunk = [
            {keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7)), keys.ACTION_TIMESTAMP: 0.0},
            {keys.ROBOT_COMMAND: JointDelta(velocities=np.ones(7)), keys.ACTION_TIMESTAMP: 0.1},
            {keys.ACTION_TIMESTAMP: 0.2},  # the horizon sentinel carries no command
        ]
        decoded = SetControlMode(IMPEDANCE).decode(chunk)
        assert isinstance(decoded, list)
        for action in decoded[:2]:
            assert isinstance(action, dict)
            assert action[keys.ROBOT_COMMAND].mode == IMPEDANCE
        assert keys.ROBOT_COMMAND not in decoded[2]

    def test_a_single_action_carries_the_mode(self):
        decoded = SetControlMode(IMPEDANCE).decode({keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7))})
        assert isinstance(decoded, dict)
        assert decoded[keys.ROBOT_COMMAND].mode == IMPEDANCE

    def test_every_arm_channel_is_stamped(self):
        """A bimanual action names a channel per arm, and both execute under the mode."""
        action = {
            f'{keys.ROBOT_COMMAND}.left': JointDelta(velocities=np.zeros(7)),
            f'{keys.ROBOT_COMMAND}.right': JointDelta(velocities=np.ones(7)),
            keys.TARGET_JOINTS: np.zeros(7),  # in the command family by name, but a vector
            'target_grip': 0.5,
        }
        decoded = SetControlMode(IMPEDANCE).decode(action)
        assert isinstance(decoded, dict)
        assert decoded[f'{keys.ROBOT_COMMAND}.left'].mode == IMPEDANCE
        assert decoded[f'{keys.ROBOT_COMMAND}.right'].mode == IMPEDANCE
        np.testing.assert_array_equal(decoded[keys.TARGET_JOINTS], np.zeros(7))


class TestTemporalStack:
    OFFSETS = (-0.2, -0.1, 0.0)

    def test_pad_start_repeats_oldest(self):
        inner = _CapturePolicy()
        session = TemporalStack(keys=('v',), offsets_sec=self.OFFSETS).wrap(inner).new_session()
        session(_stack_obs(0.0, 1.0), 0)
        stack = inner.session.seen[0]['v']
        assert stack.shape == (3, 1)
        assert (stack == 1.0).all()

    def test_no_pad_start_grows_from_one(self):
        inner = _CapturePolicy()
        layer = TemporalStack(keys=('v',), offsets_sec=self.OFFSETS, pad_start=False)
        session = layer.wrap(inner).new_session()

        session(_stack_obs(0.0, 1.0), 0)
        assert inner.session.seen[0]['v'].shape == (1, 1)

        session(_stack_obs(0.1, 2.0), int(0.1e9))
        assert inner.session.seen[1]['v'].shape == (2, 1)
        assert inner.session.seen[1]['v'][:, 0].tolist() == [1.0, 2.0]

        session(_stack_obs(0.2, 3.0), int(0.2e9))
        assert inner.session.seen[2]['v'].shape == (3, 1)
        assert inner.session.seen[2]['v'][:, 0].tolist() == [1.0, 2.0, 3.0]

    def test_no_pad_start_full_window_matches_padded(self):
        offsets = self.OFFSETS
        stacks = {}
        for pad_start in (True, False):
            inner = _CapturePolicy()
            layer = TemporalStack(keys=('v',), offsets_sec=offsets, pad_start=pad_start)
            session = layer.wrap(inner).new_session()
            for i in range(4):
                session(_stack_obs(0.1 * i, float(i)), round(0.1 * i * 1e9))
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
        assert local is not None and local._layers() == (stack, sched)
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

        monkeypatch.setitem(spec.WIRE_LAYERS, 'wire_codec', _WireCodec)
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
            spec.from_spec({'name': 'not_a_layer'})

    def test_unknown_arg_fails(self):
        with pytest.raises(TypeError):
            spec.from_spec({'name': 'temporal_stack', 'args': {'keys': ['v'], 'offsets_sec': [0.0], 'bogus': 1}})

    def test_non_deliverable_layer_fails_loudly(self):
        with pytest.raises(NotImplementedError, match='not deliverable'):
            IKJointsAction(solver_cls=None).to_spec()

    def test_the_table_publishes_these_exact_wire_names(self):
        """The strings a deployed server already declares its local stack with. Spelled out here rather than
        read off ``WIRE_NAME``, so renaming an attribute cannot quietly rename the wire."""
        instances = {
            'chunked_schedule': ChunkedSchedule(),
            'stop_on_fault': StopOnFault(),
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
            'joint_delta_action': JointDeltaAction(),
            'change_ee_frame': ChangeEEFrame(Transform3D.identity),
        }
        assert set(instances) == set(spec.WIRE_LAYERS)
        for name, instance in instances.items():
            assert instance.to_spec()['name'] == name
            assert type(instance) is spec.WIRE_LAYERS[name]


class _ListSource(spec.ModelSource):
    def __init__(self, models):
        self._models = list(models)

    def get_models(self):
        return list(self._models)

    def load(self, model_id, on_progress=None):
        return _ConstPolicy([{'model': model_id}])


class TestPipe:
    """The source terminal: ``... | source`` closes a layer chain into a Pipeline."""

    def test_layer_chain_terminates_into_pipe(self):
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
        with pytest.raises(ValueError, match=roboarm_keys.EE_FRAME):
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
        inner = _ConstPolicy([{'action': f'a{i}'} for i in range(5)])
        policy = spec.inline(ChunkedSchedule() | spec.remote | ActionTimestamp(fps=10.0) | spec.PolicySource(inner))
        assert isinstance(policy, Policy)
        session = policy.new_session()
        result = session(_obs(), int(1e9))
        assert result is not None
        assert result[0][keys.ACTION_TIMESTAMP] == 1.0
        assert session(_obs(), int(1.2e9)) is None

    def test_inline_tolerates_marker_less_pipe(self):
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        policy = spec.inline(ChunkedSchedule() | spec.PolicySource(inner))
        session = policy.new_session()
        result = session(_obs(), int(1e9))
        assert result is not None and result[0][keys.ACTION_TIMESTAMP] == 1.0

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
