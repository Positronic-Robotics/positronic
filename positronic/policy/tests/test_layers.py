"""Scheduling, fault gating, temporal history, and deliverable stack specifications."""

from concurrent.futures import Future
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

import pimm
from pimm.world import VirtualClock
from positronic import keys
from positronic.drivers.roboarm import RobotStatus
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.command import Impedance, JointDelta
from positronic.eval import keys as eval_keys
from positronic.geom import Rotation, Transform3D
from positronic.policy import codec as codec_module
from positronic.policy import spec
from positronic.policy.action import AbsoluteJointsAction, AbsolutePositionAction, IKJointsAction, JointDeltaAction
from positronic.policy.base import Policy, Step
from positronic.policy.codec import (
    BinarizeGripInference,
    BinarizeGripTraining,
    ChangeEEFrame,
    Codec,
    FlipGrip,
    Metadata,
    RestrictImageSize,
    SetControlMode,
)
from positronic.policy.executor import Executor, _UnchargedAnswer
from positronic.policy.layers import ChunkedSchedule, PauseOnUnavailable, TemporalStack
from positronic.policy.observation import ObservationCodec
from positronic.policy.sequential import Sequential

MOTOR = 'motor'
POSITION = 'position'


class Echo(Policy):
    def run(self, runtime):
        obs = yield
        while True:
            obs = yield Step(obs, runtime.time_ns + 100_000_000)


@pytest.mark.parametrize('status', [RobotStatus.ERROR, RobotStatus.BUSY])
@pytest.mark.parametrize('channel', [keys.ROBOT_STATUS, 'robot_state.left.status', 'robot_state.right.status'])
def test_unavailability_withholds_commands_and_resumes_on_recovery(execution, status, channel):
    runtime, _ = execution
    inner = Mock(send=Mock(side_effect=lambda obs: Step(obs, 100_000_000)))
    run = runtime.start(PauseOnUnavailable(), inner)
    try:
        obs = {keys.ROBOT_STATUS: RobotStatus.AVAILABLE, channel: status}
        assert run.send(obs) == Step({}, 1_000_000)
        inner.send.assert_not_called()
        obs[channel] = RobotStatus.AVAILABLE
        assert run.send(obs) == Step(obs, 100_000_000)
        assert run.send({MOTOR: 1}) == Step({MOTOR: 1}, 100_000_000)
    finally:
        run.close()


@pytest.mark.parametrize('pad_start', [True, False])
def test_temporal_history_padding_sampling_and_fresh_runs(execution, pad_start):
    runtime, clock = execution
    definition = Sequential(TemporalStack((POSITION,), (-0.2, -0.1, 0.0), pad_start), Echo())
    run = runtime.start(definition)
    frame = np.array([1])
    first = run.send({POSITION: frame})
    np.testing.assert_array_equal(first.commands[POSITION][:, 0], [1, 1, 1] if pad_start else [1])
    frame[0] = 2
    clock.advance_to_ns(100_000_000)
    second = run.send({POSITION: frame})
    np.testing.assert_array_equal(second.commands[POSITION][:, 0], [1, 1, 2] if pad_start else [1, 2])
    frame[0] = 3
    clock.advance_to_ns(200_000_000)
    third = run.send({POSITION: frame})
    np.testing.assert_array_equal(third.commands[POSITION][:, 0], [1, 2, 3])
    run.close()
    fresh = runtime.start(definition)
    try:
        reset = fresh.send({POSITION: frame})
        np.testing.assert_array_equal(reset.commands[POSITION][:, 0], [3, 3, 3] if pad_start else [3])
    finally:
        fresh.close()


def test_temporal_history_carries_the_last_sample_before_each_offset(execution):
    runtime, clock = execution
    run = runtime.start(Sequential(TemporalStack((POSITION,), (-0.2, -0.1, 0.0)), Echo()))
    try:
        for ns, value in [(0, 1), (150_000_000, 2), (300_000_000, 3)]:
            clock.advance_to_ns(ns)
            result = run.send({POSITION: np.array([value])})
        np.testing.assert_array_equal(result.commands[POSITION][:, 0], [1, 2, 3])
    finally:
        run.close()


@pytest.fixture
def execution(monkeypatch):
    with pimm.World(virtual_time=True) as world:
        runtime = Executor(world.clock.now_ns, simulated=True, charge_inference_time=False)

        def submit(function, *args, **kwargs):
            future = Future()
            future.set_result(function(*args, **kwargs))
            return _UnchargedAnswer(future)

        monkeypatch.setattr(runtime, 'submit', submit)
        try:
            yield runtime, cast(VirtualClock, world.clock)
        finally:
            runtime.close()


def test_ready_answer_emits_without_an_initial_yield(execution, monkeypatch):
    runtime, _ = execution
    future = Future()
    future.set_result([{MOTOR: 1}])
    monkeypatch.setattr(runtime, 'submit', lambda *args: _UnchargedAnswer(future))
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [])
    assert run.send({}) == Step({MOTOR: 1}, 100_000_000)
    run.close()


def test_pending_inference_is_not_resubmitted_on_each_tick(execution, monkeypatch):
    runtime, clock = execution
    future = Future()
    submit = Mock(return_value=_UnchargedAnswer(future))
    monkeypatch.setattr(runtime, 'submit', submit)
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [])
    try:
        for now in (0, 5_000_000, 10_000_000):
            clock.advance_to_ns(now)
            assert run.send({}) == Step({}, now)
        submit.assert_called_once()
        future.set_result([{MOTOR: 1}])
        assert run.send({}) == Step({MOTOR: 1}, 110_000_000)
        submit.assert_called_once()
    finally:
        run.close()


def test_last_action_keeps_its_full_period(execution):
    runtime, clock = execution
    requests = []

    def infer(obs):
        requests.append(runtime.time_ns)
        return [{MOTOR: i} for i in (1, 2, 3)]

    run = runtime.start(ChunkedSchedule(fps=10), infer)
    for index in range(4):
        clock.advance_to_ns(index * 100_000_000)
        step = run.send({})
        assert step == Step({MOTOR: index % 3 + 1}, (index + 1) * 100_000_000)
    assert requests == [0, 300_000_000]
    run.close()


def test_late_wake_merges_due_channels_and_keeps_absolute_deadlines(execution):
    runtime, clock = execution
    chunk = [{MOTOR: 1}, {POSITION: 2}, {MOTOR: 3}, {MOTOR: 4}]
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: chunk)
    run.send({})
    clock.advance_to_ns(225_000_000)
    assert run.send({}) == Step({POSITION: 2, MOTOR: 3}, 300_000_000)
    run.close()


def test_an_overrun_skips_all_but_the_last_due_row_and_counts_the_skip(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [{MOTOR: i} for i in range(5)])
    emitted = []
    for now in (0, 100_000_000, 350_000_000, 400_000_000):
        clock.advance_to_ns(now)
        emitted.append(run.send({}).commands[MOTOR])
    meta = runtime.episode_meta()
    run.close()
    assert emitted == [0, 1, 3, 4]
    prefix = f'{eval_keys.SCHEDULE}.{MOTOR}'
    assert meta == {
        f'{prefix}.{eval_keys.SCHEDULED}': 5,
        f'{prefix}.{eval_keys.EMITTED}': 4,
        f'{prefix}.{eval_keys.DROPPED}': 1,
        f'{prefix}.{eval_keys.LATE_P50_MS}': 0.0,
        f'{prefix}.{eval_keys.LATE_P90_MS}': pytest.approx(35.0),
        f'{prefix}.{eval_keys.LATE_MAX_MS}': 50.0,
        f'{prefix}.{eval_keys.GAP_MAX_MS}': 250.0,
    }


def test_a_new_chunk_counts_the_due_rows_it_replaces_as_dropped(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10), lambda obs: [{MOTOR: i} for i in range(5)])
    emitted = []
    for now in (0, 600_000_000):
        clock.advance_to_ns(now)
        emitted.append(run.send({}).commands[MOTOR])
    meta = runtime.episode_meta()
    run.close()
    assert emitted == [0, 0]
    prefix = f'{eval_keys.SCHEDULE}.{MOTOR}'
    assert meta[f'{prefix}.{eval_keys.SCHEDULED}'] == 10
    assert meta[f'{prefix}.{eval_keys.EMITTED}'] == 2
    assert meta[f'{prefix}.{eval_keys.DROPPED}'] == 4


def test_horizon_cuts_commands_but_preserves_boundary(execution):
    runtime, clock = execution
    run = runtime.start(ChunkedSchedule(fps=10, horizon_sec=0.15), lambda obs: [{MOTOR: i} for i in range(5)])
    assert run.send({}) == Step({MOTOR: 0}, 100_000_000)
    clock.advance_to_ns(100_000_000)
    assert run.send({}) == Step({MOTOR: 1}, 150_000_000)
    clock.advance_to_ns(150_000_000)
    assert run.send({}) == Step({MOTOR: 0}, 250_000_000)
    run.close()


def test_empty_chunks_request_an_immediate_retry(execution):
    runtime, clock = execution
    requests = []

    def infer(obs):
        requests.append(runtime.time_ns)
        return []

    run = runtime.start(ChunkedSchedule(fps=10), infer)
    for now in (0, 0):
        clock.advance_to_ns(now)
        assert run.send({}) == Step({}, now)
    assert requests == [0, 0]
    run.close()


IMPEDANCE = Impedance(kq=(40.0,) * 7, kqd=(4.0,) * 7, kx=(750.0,) * 6, kxd=(37.0,) * 6)


class TestCodecComposition:
    """Codec composition preserves metadata and frame declarations."""

    def test_codec_and_stays_codec_only(self):
        """& only works between codecs, not layers."""
        c1 = Metadata({'action_fps': 10.0})
        c2 = Metadata({'action_fps': 5.0})
        composed = c1 & c2
        assert isinstance(composed, Codec)

    def test_agreeing_declarations_merge(self):
        assert (Metadata({'action_fps': 10.0}) | Metadata({'action_fps': 10.0})).meta['action_fps'] == 10.0

    def test_disagreeing_declarations_have_no_merged_answer(self):
        composed = Metadata({'action_fps': 10.0}) & Metadata({'action_fps': 5.0})
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


class TestSetControlMode:
    def test_every_command_in_a_chunk_carries_the_mode(self):
        chunk = [
            {keys.ROBOT_COMMAND: JointDelta(velocities=np.zeros(7))},
            {keys.ROBOT_COMMAND: JointDelta(velocities=np.ones(7))},
            {keys.TARGET_GRIP: 0.5},
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

    def test_a_threaded_stack_scales_to_the_same_pixels_as_one_thread(self):
        """A stack over the parallel bar scales to the same pixels as the frames taken one at a time."""
        rng = np.random.default_rng(0)
        stack = rng.integers(0, 256, size=(RestrictImageSize._PARALLEL_FROM + 4, 480, 640, 3), dtype=np.uint8)
        codec = RestrictImageSize(64, 48)
        one_at_a_time = np.stack([codec.encode({'cam': frame})['cam'] for frame in stack])
        np.testing.assert_array_equal(codec.encode({'cam': stack})['cam'], one_at_a_time)

    def test_a_single_usable_cpu_stays_serial(self, monkeypatch):
        """A pool wins nothing on a single core, and costs threads to raise."""
        monkeypatch.setattr(codec_module, '_usable_cpus', lambda: 1)
        codec = RestrictImageSize(64, 48)
        assert codec._workers(codec._PARALLEL_FROM + 4) == 1

    def test_the_pool_is_bounded_by_the_cpus_the_process_may_run_on(self, monkeypatch):
        monkeypatch.setattr(codec_module, '_usable_cpus', lambda: 2)
        codec = RestrictImageSize(64, 48)
        assert codec._workers(codec._MAX_WORKERS * 4) == 2

    def test_a_stack_under_the_parallel_bar_still_scales(self):
        stack = np.zeros((RestrictImageSize._PARALLEL_FROM - 1, 480, 640, 3), dtype=np.uint8)
        assert RestrictImageSize(64, 48).encode({'cam': stack})['cam'].shape[1:] == (48, 64, 3)

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


@pytest.mark.parametrize(
    'definition',
    [
        Sequential(TemporalStack(('a', 'b'), (-0.5, 0.0), pad_start=False), ChunkedSchedule(fps=10, horizon_sec=0.5)),
        Sequential(PauseOnUnavailable(), ChunkedSchedule(fps=10), RestrictImageSize(64, 48)),
        ObservationCodec(state={'state': {'grip': 1}}, images={}) & AbsolutePositionAction('pose', 'grip'),
        FlipGrip() | (BinarizeGripInference() & AbsoluteJointsAction('joints', 'grip')),
    ],
)
def test_stack_and_codec_specs_round_trip(definition):
    assert spec.from_spec(definition.to_spec()).to_spec() == definition.to_spec()


@pytest.mark.parametrize(
    'node, error',
    [
        ({'seq': []}, ValueError),
        ({'par': []}, ValueError),
        ({'par': [{'name': 'stop_on_fault'}]}, ValueError),
        ({'name': 'unknown'}, ValueError),
        ({'name': 'temporal_stack', 'args': {'keys': ['v'], 'offsets_sec': [0.0], 'bogus': 1}}, TypeError),
    ],
)
def test_invalid_stack_specs_are_rejected(node, error):
    with pytest.raises(error):
        spec.from_spec(node)


def test_non_deliverable_codec_is_rejected():
    with pytest.raises(NotImplementedError, match='IKJointsAction'):
        IKJointsAction(solver_cls=None).to_spec()


def test_wire_names_match_the_registered_components():
    instances = {
        'chunked_schedule': ChunkedSchedule(fps=10),
        'stop_on_fault': PauseOnUnavailable(),
        'temporal_stack': TemporalStack(('v',), (0.0,)),
        'binarize_grip_training': BinarizeGripTraining(('grip',)),
        'binarize_grip_inference': BinarizeGripInference(),
        'flip_grip': FlipGrip(),
        'metadata': Metadata({'action_fps': 15}),
        'restrict_image_size': RestrictImageSize(),
        'observation_codec': ObservationCodec(state={}, images={}),
        'absolute_position_action': AbsolutePositionAction(keys.TARGET_EE_POSE, keys.TARGET_GRIP),
        'absolute_joints_action': AbsoluteJointsAction(keys.TARGET_JOINTS, keys.TARGET_GRIP),
        'joint_delta_action': JointDeltaAction(),
        'change_ee_frame': ChangeEEFrame(Transform3D.identity),
    }
    registered = spec.COMPONENTS
    assert set(instances) | {'action_timestamp', 'action_horizon'} == set(registered)
    for name, instance in instances.items():
        assert instance.to_spec()['name'] == name
        assert type(instance) is registered[name][instance.WIRE_VERSION].implementation
        assert instance.to_spec()['version'] == instance.WIRE_VERSION
