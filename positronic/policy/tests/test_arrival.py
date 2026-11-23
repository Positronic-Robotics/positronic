import numpy as np
import pytest
from positronic_model_server.spec import component, parallel, sequence

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.transforms import image
from positronic.drivers.roboarm import command
from positronic.policy.arrival import transform_frames_on_arrival
from positronic.policy.base import Processor
from positronic.policy.codecs import TASK_FIELD, TransformImages, TransformOnArrival
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.processors import TemporalStack
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec

CAMERAS = (keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE_2)
FPS = 15.0
STEPS = 30
# A camera frame much larger than the transform's size, as a rig camera is.
FRAME_SHAPE = (72, 128, 3)


IMPEDANCE = command.to_wire(DROID_IMPEDANCE)


def _history(offsets: int) -> dict:
    return component(
        'temporal_stack', version=2, keys=list(CAMERAS), offsets_sec=[-i / FPS for i in reversed(range(offsets))]
    )


def _stacked_transform_spec() -> dict:
    """A history of every camera, a schedule, and a bound on the image size at each call."""
    return sequence(
        component('stop_on_fault'),
        _history(25),
        component('chunked_schedule', fps=FPS),
        component('restrict_image_size', width=32, height=32),
    )


def _three_camera_history_spec() -> dict:
    """Four history frames of three cameras, field selection beside the action codecs, then the image transform."""
    return sequence(
        component('temporal_stack', version=2, keys=[keys.JOINTS, keys.GRIP], offsets_sec=[-0.2, -0.1, 0.0]),
        _history(4),
        component('rtc_schedule', fps=FPS, call_after_sec=0.2, latency_estimate=0.1),
        parallel(
            component(
                'select_observation_fields', mapping={**{name: name for name in CAMERAS}, TASK_FIELD: TASK_FIELD}
            ),
            sequence(
                component('set_control_mode', mode=IMPEDANCE),
                component(
                    'absolute_position_action',
                    tgt_ee_pose_key=keys.TARGET_EE_POSE,
                    tgt_grip_key=keys.TARGET_GRIP,
                    rotation_rep='quat',
                    commands_path=['action_prefix', 'actions'],
                ),
            ),
        ),
        component(
            'transform_images', width=40, height=24, mode='stretch', resample='area', paths=[[name] for name in CAMERAS]
        ),
    )


def _camera_frames(steps: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    return [{name: rng.integers(0, 256, FRAME_SHAPE, dtype=np.uint8) for name in CAMERAS} for _ in range(steps)]


class _Resizes:
    """Count each call of ``transform_frame`` that returns a new frame, by the input frame's shape."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.count = 0
        original = image.transform_frame

        def counted(frame, *args, **kwargs):
            result = original(frame, *args, **kwargs)
            self.count += result is not frame
            return result

        monkeypatch.setattr(image, 'transform_frame', counted)


def _declared(spec: dict) -> Processor:
    stack = from_spec(spec)
    assert isinstance(stack, Processor)
    return stack


def _moved(stack: Processor) -> Sequential:
    moved = transform_frames_on_arrival(stack)
    assert isinstance(moved, Sequential)
    return moved


def _run(stack: Processor, frames: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """The camera fields of each call that the stack makes, over one step per frame set at ``FPS``."""
    calls = []

    def infer(obs):
        calls.append({name: np.array(obs[name]) for name in CAMERAS})
        return None

    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(stack, infer)
    try:
        for step, cameras in enumerate(frames):
            now[0] = round(step * 1e9 / FPS)
            runtime.start_tick(now[0])
            obs = {
                **cameras,
                keys.JOINTS: np.zeros(7),
                keys.GRIP: np.zeros(1),
                keys.EE_POSE: np.zeros(7),
                keys.ROBOT_STATUS: 0,
                TASK_FIELD: 'pick',
            }
            run.send(obs)
            # Each call ends before the next step, so both stacks make the same calls.
            while (waited := runtime.wait(timeout_sec=10).status) is WaitStatus.ANSWERS_READY:
                pass
            assert waited is WaitStatus.CAN_ADVANCE
    finally:
        run.close()
        runtime.close()
    return calls


SPECS = [_stacked_transform_spec, _three_camera_history_spec]


@pytest.mark.parametrize('declared', SPECS)
def test_each_stacked_camera_frame_is_transformed_in_front_of_its_history(declared):
    stack = _moved(_declared(declared()))

    components = stack.components
    history = next(
        i for i, part in enumerate(components) if isinstance(part, TemporalStack) and keys.WRIST_IMAGE in part.keys
    )
    arrival = components[history - 1]
    assert isinstance(arrival, TransformOnArrival)
    assert arrival.fields == CAMERAS


@pytest.mark.parametrize('declared', SPECS)
def test_the_policy_gets_the_same_camera_history_when_frames_are_transformed_on_arrival(declared):
    frames = _camera_frames(STEPS)

    before = _run(_declared(declared()), frames)
    after = _run(transform_frames_on_arrival(_declared(declared())), frames)

    assert len(after) == len(before) > 1
    for old, new in zip(before, after, strict=True):
        for name in CAMERAS:
            assert new[name].shape == old[name].shape
            assert new[name].ndim == 4
            np.testing.assert_array_equal(new[name], old[name])


@pytest.mark.parametrize('declared', SPECS)
def test_each_camera_frame_is_resized_once_and_a_call_resizes_no_history(declared, monkeypatch):
    resizes = _Resizes(monkeypatch)
    calls = _run(transform_frames_on_arrival(_declared(declared())), _camera_frames(STEPS))

    assert len(calls) > 1
    assert resizes.count == STEPS * len(CAMERAS)


@pytest.mark.parametrize('declared', SPECS)
def test_without_the_move_each_call_resizes_the_whole_history(declared, monkeypatch):
    resizes = _Resizes(monkeypatch)
    calls = _run(_declared(declared()), _camera_frames(STEPS))

    assert resizes.count == sum(len(call[name]) for call in calls for name in CAMERAS)


def test_a_transform_in_front_of_the_history_stays_as_declared():
    declared = _declared(
        sequence(
            component('restrict_image_size', width=32, height=32), _history(4), component('chunked_schedule', fps=FPS)
        )
    )

    assert transform_frames_on_arrival(declared) is declared


def test_a_camera_field_renamed_before_the_transform_is_not_moved():
    declared = _declared(
        sequence(
            _history(4),
            component('chunked_schedule', fps=FPS),
            component('select_observation_fields', mapping={'wrist': keys.WRIST_IMAGE}),
            component('transform_images', width=32, height=32, mode='pad', paths=[['wrist']]),
        )
    )

    assert transform_frames_on_arrival(declared) is declared


def test_only_the_camera_fields_that_reach_the_transform_unchanged_move():
    declared = _declared(
        sequence(
            _history(4),
            component('chunked_schedule', fps=FPS),
            component(
                'select_observation_fields',
                # The wrist frame takes a new name, and the second exterior frame goes.
                mapping={'wrist': keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE},
            ),
            component('transform_images', width=32, height=32, mode='pad', paths=[['wrist'], *([n] for n in CAMERAS)]),
        )
    )

    stack = _moved(declared)

    arrival = stack.components[0]
    assert isinstance(arrival, TransformOnArrival)
    assert arrival.fields == (keys.EXTERIOR_IMAGE,)


def test_a_transform_of_other_fields_moves_no_camera_frame():
    declared = _declared(
        sequence(
            _history(4),
            component('chunked_schedule', fps=FPS),
            component('transform_images', width=32, height=32, mode='pad', paths=[['image.other']]),
        )
    )

    assert transform_frames_on_arrival(declared) is declared


def test_two_histories_of_one_camera_transform_each_frame_once(monkeypatch):
    declared = Sequential(
        TemporalStack(CAMERAS, (-0.2, 0.0)),
        TemporalStack(CAMERAS, (-0.1, 0.0)),
        TransformImages(32, 32, mode=image.ResizeMode.PAD, paths=[[name] for name in CAMERAS]),
    )

    stack = _moved(declared)

    assert [type(part) for part in stack.components] == [
        TransformOnArrival,
        TemporalStack,
        TemporalStack,
        TransformImages,
    ]


def test_each_camera_moves_with_the_transform_that_selects_it():
    declared = Sequential(
        TemporalStack(CAMERAS, (-0.2, 0.0)),
        TransformImages(32, 32, mode=image.ResizeMode.PAD, paths=[[keys.WRIST_IMAGE]]),
        TransformImages(48, 48, mode=image.ResizeMode.BOUND, paths=[[name] for name in CAMERAS]),
    )

    stack = _moved(declared)

    wrist, others = stack.components[:2]
    assert isinstance(wrist, TransformOnArrival) and wrist.fields == (keys.WRIST_IMAGE,)
    assert isinstance(others, TransformOnArrival) and others.fields == (keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2)
    assert isinstance(stack.components[2], TemporalStack)
