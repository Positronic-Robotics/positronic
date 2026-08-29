"""Unit tests for Layer composition, ChunkPlayer, TemporalStack, and the policy-pipeline algebra."""

from typing import Any

import numpy as np
import pytest

from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm.command import Impedance, JointDelta
from positronic.geom import Rotation, Transform3D
from positronic.policy import spec
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, IKJointsAction, JointDeltaAction
from positronic.policy.base import Answer, ChunkSession, Done, Layer, Policy, Session
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
from positronic.policy.layers import ChunkPlayer, StopOnFault, TemporalStack
from positronic.policy.observation import ObservationCodec


class _Pending(Answer):
    """A call the test answers by hand, so a player can be watched while it waits."""

    def __init__(self, value=None, failure: BaseException | None = None):
        self._value = value
        self._failure = failure
        self._done = False

    def answer(self):
        self._done = True

    def done(self):
        return self._done

    def result(self):
        if self._failure is not None:
            raise self._failure
        return self._value


class _ConstSession(ChunkSession):
    def __init__(self, actions):
        self._actions = actions
        self.call_count = 0

    def __call__(self, obs, time_ns):
        self.call_count += 1
        return Done(self._actions)


class _ConstPolicy(Policy):
    def __init__(self, actions):
        self._actions = actions
        self._session = _ConstSession(actions)

    def new_session(self, context=None, rt=None) -> _ConstSession:
        self._session = _ConstSession(self._actions)
        return self._session


def _obs(now_sec=0.0, status=RobotStatus.AVAILABLE):
    return {keys.OBS_TIME_NS: int(now_sec * 1e9), keys.ROBOT_STATUS: status}


class _CommandSession(Session):
    """Answers a fixed command mapping, and asks for its next call one period later."""

    POLL_SEC = 0.001

    def __init__(self, commands):
        self._commands = commands
        self.call_count = 0

    def __call__(self, obs, time_ns):
        self.call_count += 1
        return self._commands, time_ns + int(self.POLL_SEC * 1e9)


class TestStopOnFault:
    @pytest.mark.parametrize('unavailable', [RobotStatus.ERROR, RobotStatus.BUSY])
    def test_an_unavailable_arm_stops_what_is_executing(self, unavailable):
        inner = _CommandSession({'v': 1})
        session = StopOnFault().make_session(inner)

        assert session(_obs(0.0, unavailable), 0) == ({}, int(StopOnFault.POLL_SEC * 1e9))
        assert inner.call_count == 0, 'the model was asked about an arm that is not tracking it'

    def test_an_available_arm_reaches_the_model(self):
        inner = _CommandSession({'v': 1})
        session = StopOnFault().make_session(inner)

        assert session(_obs(0.0, RobotStatus.AVAILABLE), 0) == ({'v': 1}, int(_CommandSession.POLL_SEC * 1e9))
        assert inner.call_count == 1

    def test_an_observation_with_no_arm_status_reaches_the_model(self):
        """A probe replaying a recording has no arm to stop for."""
        inner = _CommandSession({'v': 1})
        session = StopOnFault().make_session(inner)

        assert session({keys.OBS_TIME_NS: 0}, 0) == ({'v': 1}, int(_CommandSession.POLL_SEC * 1e9))
        assert inner.call_count == 1

    def test_either_arm_of_a_bimanual_rig_stops_the_pair(self):
        """Whichever arm is unavailable stops the pair, and the status counts as its number: a server-side stack
        reads it off a wire with no enum to carry."""
        inner = _CommandSession({'v': 1})
        session = StopOnFault().make_session(inner)
        obs = {
            keys.OBS_TIME_NS: 0,
            f'{keys.ROBOT_STATE}.left.status': int(RobotStatus.AVAILABLE),
            f'{keys.ROBOT_STATE}.right.status': int(RobotStatus.ERROR),
        }

        assert session(obs, 0) == ({}, int(StopOnFault.POLL_SEC * 1e9))
        assert inner.call_count == 0

    def test_the_status_a_recording_carries_for_a_taken_arm_stops_the_policy(self):
        """The numbers are the contract between a rig and a server: 1 is an arm its driver has taken."""
        inner = _CommandSession({'v': 1})
        session = StopOnFault().make_session(inner)

        assert (RobotStatus.AVAILABLE, RobotStatus.BUSY, RobotStatus.ERROR) == (0, 1, 3)
        assert session({keys.OBS_TIME_NS: 0, keys.ROBOT_STATUS: 1}, 0) == ({}, int(StopOnFault.POLL_SEC * 1e9))
        assert inner.call_count == 0

    def test_the_status_published_for_a_travelling_arm_reaches_the_model(self):
        """The wire protocol publishes 2 for an arm on its way to a setpoint, which is one taking commands."""
        inner = _CommandSession({'v': 1})
        session = StopOnFault().make_session(inner)

        assert session({keys.OBS_TIME_NS: 0, keys.ROBOT_STATUS: 2}, 0) == (
            {'v': 1},
            int(_CommandSession.POLL_SEC * 1e9),
        )
        assert inner.call_count == 1

    def test_a_status_no_arm_answers_to_raises(self):
        """A number outside ``RobotStatus`` is the rig and the server disagreeing about the protocol, which
        is not something to drive an arm through."""
        session = StopOnFault().make_session(_CommandSession({}))

        with pytest.raises(ValueError):
            session({keys.OBS_TIME_NS: 0, keys.ROBOT_STATUS: 99}, 0)

    def test_recovery_plans_afresh_instead_of_resuming(self):
        """The stop drops the chunk the player holds, so the first observation from an available arm plans
        again rather than playing out the chunk stamped before."""
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {'v': 2, keys.ACTION_TIMESTAMP: 1.0}])
        session = (StopOnFault() | ChunkPlayer()).wrap(inner).new_session()

        assert session(_obs(0.0), 0) == ({'v': 1}, int(1e9))  # a chunk that runs until 1.0
        assert session(_obs(0.2, RobotStatus.ERROR), int(0.2e9)) == ({}, int(0.2e9) + int(StopOnFault.POLL_SEC * 1e9))
        assert session(_obs(0.3), int(0.3e9)) == ({'v': 1}, int(1.3e9))
        assert inner._session.call_count == 2


class _ScriptedSession(ChunkSession):
    """Answers each of ``script`` in turn, each entry a handle the player holds."""

    def __init__(self, script):
        self._script = list(script)
        self.call_count = 0

    def __call__(self, obs, time_ns):
        self.call_count += 1
        return self._script.pop(0)


class TestChunkPlayer:
    def test_a_call_that_has_not_answered_leaves_the_player_holding(self):
        """The player asks one time and holds the handle, so a round it waits through costs no second call."""
        pending = _Pending([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        inner = _ScriptedSession([pending])
        session = ChunkPlayer().make_session(inner)

        assert session(_obs(0.0), int(1e9)) == ({}, int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9))
        assert session(_obs(0.1), int(1e9)) == ({}, int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9))
        pending.answer()
        assert session(_obs(0.2), int(1e9)) == ({'v': 1}, int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9))
        assert inner.call_count == 1

    def test_a_cancel_drops_the_chunk_of_the_call_in_flight(self):
        """A cancelled player drops the chunk it waited for, because that chunk describes a world the cancel
        says has gone, and it asks for a new one."""
        pending = _Pending([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        inner = _ScriptedSession([pending, Done([{'v': 2, keys.ACTION_TIMESTAMP: 0.0}])])
        session = ChunkPlayer().make_session(inner)
        poll_ns = int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9)

        assert session(_obs(), int(1e9)) == ({}, poll_ns)
        pending.answer()
        session.cancel()

        assert session(_obs(), int(1e9)) == ({}, poll_ns), 'the cancelled chunk was read and thrown away'
        assert session(_obs(), int(1e9)) == ({'v': 2}, poll_ns)
        assert inner.call_count == 2

    def test_a_cancelled_call_still_raises_what_it_failed_with(self):
        """A dropped chunk drops no failure. The player reads a cancelled call, so a stalled server raises
        to the caller that asked for the episode."""
        pending = _Pending(failure=TimeoutError('server stalled'))
        session = ChunkPlayer().make_session(_ScriptedSession([pending]))

        session(_obs(), int(1e9))
        pending.answer()
        session.cancel()

        with pytest.raises(TimeoutError, match='server stalled'):
            session(_obs(), int(1e9))

    def test_a_cancel_ends_with_the_call_it_was_made_against(self):
        """A cancel ends with the call it was made against, even when that call fails. A caller that catches
        the failure and keeps the session gets the next chunk."""
        pending = _Pending(failure=TimeoutError('server stalled'))
        inner = _ScriptedSession([pending, Done([{'v': 2, keys.ACTION_TIMESTAMP: 0.0}])])
        session = ChunkPlayer().make_session(inner)

        session(_obs(), int(1e9))
        pending.answer()
        session.cancel()
        with pytest.raises(TimeoutError, match='server stalled'):
            session(_obs(), int(1e9))

        assert session(_obs(), int(1e9)) == ({'v': 2}, int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9))

    def test_a_chunk_plays_one_waypoint_at_a_time(self):
        """The player anchors the chunk on the call that receives it and asks for a call at each waypoint."""
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {'v': 2, keys.ACTION_TIMESTAMP: 0.5}])
        session = ChunkPlayer().wrap(inner).new_session()

        assert session(_obs(), int(1e9)) == ({'v': 1}, int(1.5e9))
        assert session(_obs(), int(1.2e9)) == ({}, int(1.5e9))
        assert inner._session.call_count == 1

    def test_waypoints_due_together_keep_every_channel_they_name(self):
        """A round that reaches two waypoints commands both channels, not only the ones the later names."""
        inner = _ConstPolicy([
            {'arm': 1, 'grip': 0.5, keys.ACTION_TIMESTAMP: 0.0},
            {'grip': 0.9, keys.ACTION_TIMESTAMP: 0.01},
            {'arm': 2, keys.ACTION_TIMESTAMP: 0.02},
            {keys.ACTION_TIMESTAMP: 0.03},
        ])
        session = ChunkPlayer().wrap(inner).new_session()

        assert session(_obs(), int(1e9)) == ({'arm': 1, 'grip': 0.5}, int(1.01e9))
        assert session(_obs(), int(1.025e9)) == ({'grip': 0.9, 'arm': 2}, int(1.03e9))

    def test_a_new_chunk_supersedes_the_one_it_replaces(self):
        """The call that drains a chunk loads the next one, and a channel only the drained chunk named
        commands nothing: the driver holds what it last took until the new chunk names that channel."""
        inner = _ScriptedSession([
            Done([{'arm': 1, keys.ACTION_TIMESTAMP: 0.0}, {'grip': 0.9, keys.ACTION_TIMESTAMP: 0.5}]),
            Done([{'arm': 2, keys.ACTION_TIMESTAMP: 0.2}]),
        ])
        session = ChunkPlayer().make_session(inner)

        assert session(_obs(), int(1e9)) == ({'arm': 1}, int(1.5e9))
        assert session(_obs(), int(1.5e9)) == ({}, int(1.7e9))

    def test_a_channel_with_several_waypoints_due_keeps_the_last(self):
        """A round that finds more than one waypoint due commands the latest of them."""
        inner = _ConstPolicy([{'v': i, keys.ACTION_TIMESTAMP: i * 0.1} for i in range(4)])
        session = ChunkPlayer().wrap(inner).new_session()

        assert session(_obs(), int(1e9)) == ({'v': 0}, int(1.1e9))
        assert session(_obs(), int(1.25e9)) == ({'v': 2}, int(1.3e9))

    def test_the_call_that_drains_the_chunk_asks_for_the_next_one(self):
        """Draining and re-querying happen in one call, so the chunk after it starts where this one ended."""
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {keys.ACTION_TIMESTAMP: 0.5}])
        session = ChunkPlayer().wrap(inner).new_session()

        assert session(_obs(), int(1e9)) == ({'v': 1}, int(1.5e9))
        assert session(_obs(), int(1.5e9)) == ({'v': 1}, int(2.0e9))
        assert inner._session.call_count == 2

    def test_a_single_action_is_drained_by_the_call_that_loads_it(self):
        """A chunk of one action at ts=0 is drained by the call that loads it, so the next call re-queries."""
        session = ChunkPlayer().wrap(_ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])).new_session()

        assert session(_obs(1.0), int(1e9)) == ({'v': 1}, int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9))
        assert session(_obs(1.01), int(1.01e9)) == ({'v': 1}, int(1.01e9) + int(ChunkPlayer.POLL_SEC * 1e9))

    def test_a_waypoint_naming_no_channel_commands_nothing(self):
        """The codecs close a chunk with a timestamp-only sentinel; it states where the chunk ends."""
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}, {keys.ACTION_TIMESTAMP: 0.5}])
        session = ChunkPlayer().make_session(inner.new_session())

        assert session(_obs(), int(1e9)) == ({'v': 1}, int(1.5e9))
        assert session(_obs(), int(1.4e9)) == ({}, int(1.5e9))

    def test_a_chunk_timed_against_another_clock_is_refused(self):
        """A chunk that reaches the player already anchored would place its waypoints decades out."""
        session = ChunkPlayer().wrap(_ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 1.77e18 / 1e9}])).new_session()

        with pytest.raises(ValueError, match='clock of their own'):
            session(_obs(), int(1e9))


class TestPipelineComposition:
    """Test | operator across Layer and Codec types."""

    def test_layer_pipe_layer(self):
        pipeline = TemporalStack(keys=('v',), offsets_sec=(0.0,)) | ChunkPlayer()
        assert isinstance(pipeline, Layer)
        policy = pipeline.wrap(_ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}]))
        session = policy.new_session()
        assert session({keys.OBS_TIME_NS: int(1e9), 'v': np.array([5.0])}, int(1e9)) == (
            {'v': 1},
            int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9),
        )

    def test_codec_pipe_layer(self):
        """A codec composes above a player, and refuses the commands it would silently corrupt."""
        pipeline = ActionTimestamp(fps=10.0) | ChunkPlayer()
        assert isinstance(pipeline, Layer)
        session = pipeline.wrap(_ConstPolicy([{'action': 'test'}])).new_session()
        with pytest.raises(AssertionError, match='under the ChunkPlayer'):
            session(_obs(), int(1e9))

    def test_full_pipeline(self):
        codec = ActionTimestamp(fps=10.0)
        pipeline = ChunkPlayer() | codec
        assert isinstance(pipeline, Layer)
        # 5 raw actions → codec stamps relative 0.0, 0.1, 0.2, 0.3, 0.4 and closes the chunk at 0.5
        # → ChunkPlayer plays them from the call's time 1.0.
        policy = pipeline.wrap(_ConstPolicy([{'action': f'a{i}'} for i in range(5)]))
        session = policy.new_session()
        assert session(_obs(), int(1e9)) == ({'action': 'a0'}, int(1.1e9))
        assert session(_obs(), int(1.2e9)) == ({'action': 'a2'}, int(1.3e9))

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
        with pytest.raises(ValueError, match=keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(b)).meta

    def test_the_same_frame_twice_is_still_two_moves(self):
        """The second move starts where the first left off, so the shared value names neither end of the pair."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        with pytest.raises(ValueError, match=keys.EE_FRAME):
            _ = (ChangeEEFrame(a) | ChangeEEFrame(a)).meta

    def test_parallel_frame_codecs_keep_the_frame_they_share(self):
        """Both halves encode the same input, so one move happens and the shared declaration describes it."""
        a = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        np.testing.assert_allclose(
            (ChangeEEFrame(a) & ChangeEEFrame(a)).meta[keys.EE_FRAME], a.as_vector(Rotation.Representation.QUAT)
        )


class _CaptureSession(ChunkSession):
    def __init__(self):
        self.seen = []

    def __call__(self, obs, time_ns):
        self.seen.append(obs)
        return Done([])


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
        sched = ChunkPlayer()
        codec = ActionTimestamp(fps=10.0)
        local, border, rem = spec.split(stack | sched | spec.remote | codec)
        assert local is not None and local._layers() == (stack, sched)
        assert border is spec.remote
        assert rem is codec

    def test_split_empty_halves(self):
        assert spec.split(spec.remote) == (None, spec.remote, None)
        local, _, rem = spec.split(ChunkPlayer() | spec.remote)
        assert rem is None and isinstance(local, ChunkPlayer)

    def test_split_requires_exactly_one_marker(self):
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(ChunkPlayer() | ChunkPlayer())
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(spec.remote | spec.remote)

    def test_split_recomposes_codec_half_as_codec(self):
        rem = spec.split(spec.remote | ActionTimestamp(fps=10.0) | ActionTimestamp(fps=5.0))[2]
        assert isinstance(rem, Codec)

    def test_border_carries_the_wire_settings(self):
        """``remote`` is the plain border; calling it describes the wire without changing the split."""
        border = spec.split(ChunkPlayer() | spec.remote(compress_images=True) | ActionTimestamp(fps=10.0))[1]
        assert border.compress_images is True
        assert spec.remote.compress_images is False

    def test_marker_cannot_be_applied(self):
        with pytest.raises(TypeError, match='border'):
            spec.remote.wrap(_ConstPolicy([]))

    def test_spec_round_trip(self):
        stack = TemporalStack(keys=('a', 'b'), offsets_sec=(-0.5, 0.0), pad_start=False) | ChunkPlayer()
        rebuilt = spec.from_spec(stack.to_spec())
        assert rebuilt is not None and rebuilt.to_spec() == stack.to_spec()

    def test_codec_spec_round_trip(self):
        obs = ObservationCodec(
            state={'observation.state': {'grip': 1}}, images={'left': (keys.WRIST_IMAGE, (224, 224))}
        )
        local = ChunkPlayer() | ActionTimestamp(fps=10.0) | (obs & AbsolutePositionAction('pose', 'grip'))
        rebuilt = spec.from_spec(local.to_spec())
        assert rebuilt is not None and rebuilt.to_spec() == local.to_spec()

    def test_leaf_without_args_omits_args_key(self):
        assert ChunkPlayer().to_spec() == {'name': 'chunk_player'}

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
            spec.from_spec({'par': [{'name': 'chunk_player'}, {'name': 'chunk_player'}]})

    def test_empty_declaration_builds_nothing(self):
        assert spec.from_spec({'seq': []}) is None

    def test_unknown_name_lists_vocabulary(self):
        with pytest.raises(ValueError, match='chunk_player'):
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
            'chunk_player': ChunkPlayer(),
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
        sched = ChunkPlayer()
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
        sched = ChunkPlayer()
        codec = ActionTimestamp(fps=10.0)
        local, border, rem = spec.split(sched | spec.remote | codec | spec.PolicySource(_ConstPolicy([])))
        assert local is sched
        assert border is spec.remote
        assert rem is codec

    def test_split_pipe_requires_exactly_one_marker(self):
        with pytest.raises(ValueError, match='exactly one'):
            spec.split(ChunkPlayer() | spec.PolicySource(_ConstPolicy([])))

    def test_pipe_refuses_a_frame_declared_on_both_sides_of_the_wire(self):
        """Rig-side and server-side conversion are alternatives; running both puts poses at the product."""
        t = Transform3D(np.array([0.0, 0.0, 0.05]), Rotation.from_euler([0.0, 0.0, 0.3]))
        chain = ChangeEEFrame(t) | spec.remote | (ActionTimestamp(fps=10.0) | ChangeEEFrame(t))
        with pytest.raises(ValueError, match=keys.EE_FRAME):
            _ = chain | spec.PolicySource(_ConstPolicy([]))

    def test_pipe_composes_no_further(self):
        pipeline: Any = ChunkPlayer() | spec.remote | spec.PolicySource(_ConstPolicy([]))
        with pytest.raises(TypeError):
            _ = pipeline | ActionTimestamp(fps=10.0)
        with pytest.raises(TypeError):
            _ = ChunkPlayer() | pipeline
        with pytest.raises(TypeError):
            _ = pipeline | spec.PolicySource(_ConstPolicy([]))

    def test_inline_full_pipe(self):
        inner = _ConstPolicy([{'action': f'a{i}'} for i in range(5)])
        policy = spec.inline(ChunkPlayer() | spec.remote | ActionTimestamp(fps=10.0) | spec.PolicySource(inner))
        assert isinstance(policy, Policy)
        session = policy.new_session()
        assert session(_obs(), int(1e9)) == ({'action': 'a0'}, int(1.1e9))
        assert session(_obs(), int(1.2e9)) == ({'action': 'a2'}, int(1.3e9))

    def test_inline_tolerates_marker_less_pipe(self):
        inner = _ConstPolicy([{'v': 1, keys.ACTION_TIMESTAMP: 0.0}])
        policy = spec.inline(ChunkPlayer() | spec.PolicySource(inner))
        session = policy.new_session()
        assert session(_obs(), int(1e9)) == ({'v': 1}, int(1e9) + int(ChunkPlayer.POLL_SEC * 1e9))

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
