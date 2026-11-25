import numpy as np
import pytest
from positronic_model_server.spec import component, parallel, sequence

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.transforms import image
from positronic.dataset.transforms.image import Resample, ResizeMode
from positronic.policy.arrival import transform_frames_on_arrival
from positronic.policy.base import Processor
from positronic.policy.codecs import TASK_FIELD, SetControlMode, TransformImages
from positronic.policy.codecs.action import AbsolutePositionAction
from positronic.policy.codecs.observation import SelectObservationFields
from positronic.policy.compatibility import CHUNKED_SCHEDULE, PauseOnUnavailable, RTCScheduleV1, restrict_image_size_v1
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.keys import RTC
from positronic.policy.processors import ChunkedSchedule, TemporalStack
from positronic.policy.sequential import Sequential
from positronic.policy.spec import from_spec

CAMERAS = (keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE_2)
FPS = 15.0
STEPS = 30
# A camera frame much larger than the transform's size, as a rig camera is.
FRAME_SHAPE = (72, 128, 3)
# A field name that a select gives to the wrist frame.
RENAMED_WRIST = 'wrist'
# The time of the simulated clock in ns. A test can move it forward inside a step.
_CLOCK = [0]


def _history(offsets: int) -> dict:
    return TemporalStack(CAMERAS, tuple(-i / FPS for i in reversed(range(offsets)))).to_spec()


def _stacked_transform_spec() -> dict:
    """A history of every camera, a schedule, and a bound on the image size at each call."""
    return sequence(
        PauseOnUnavailable().to_spec(),
        _history(25),
        component(CHUNKED_SCHEDULE, fps=FPS),
        restrict_image_size_v1(width=32, height=32),
    )


def _three_camera_history_spec() -> dict:
    """Four history frames of three cameras, field selection beside the action codecs, then the image transform."""
    return sequence(
        TemporalStack((keys.JOINTS, keys.GRIP), (-0.2, -0.1, 0.0)).to_spec(),
        _history(4),
        RTCScheduleV1(FPS, call_after_sec=0.2, latency_estimate=0.1).to_spec(),
        parallel(
            SelectObservationFields({**{name: name for name in CAMERAS}, TASK_FIELD: TASK_FIELD}).to_spec(),
            sequence(
                SetControlMode(DROID_IMPEDANCE).to_spec(),
                AbsolutePositionAction(
                    keys.TARGET_EE_POSE, keys.TARGET_GRIP, commands_path=(RTC.ACTION_PREFIX, RTC.PREFIX_ACTIONS)
                ).to_spec(),
            ),
        ),
        TransformImages(
            40, 24, mode=ResizeMode.STRETCH, resample=Resample.AREA, paths=[[name] for name in CAMERAS]
        ).to_spec(),
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


def _run(stack: Processor, frames: list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
    """The image fields of each call that the stack makes, over one step per frame set at ``FPS``."""
    calls = []

    def infer(obs):
        calls.append({name: np.array(value) for name, value in obs.items() if isinstance(value, np.ndarray)})
        return None

    runtime = Executor(lambda: _CLOCK[0], simulated=True, charge_inference_time=False)
    run = runtime.start(stack, infer)
    try:
        for step, cameras in enumerate(frames):
            _CLOCK[0] = round(step * 1e9 / FPS)
            runtime.start_tick(_CLOCK[0])
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
def test_the_policy_gets_the_same_camera_history_when_frames_are_transformed_on_arrival(declared):
    frames = _camera_frames(STEPS)

    before = _run(_declared(declared()), frames)
    after = _run(transform_frames_on_arrival(_declared(declared())), frames)

    assert len(after) == len(before) > 1
    for old, new in zip(before, after, strict=True):
        assert new.keys() == old.keys()
        for name in CAMERAS:
            assert new[name].ndim == 4
        for name in new:
            assert new[name].shape == old[name].shape
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


def _compare(declared: Processor, monkeypatch: pytest.MonkeyPatch) -> tuple[int, int]:
    """The resizes of ``declared`` as given and with frames transformed on arrival, over equal calls."""
    resizes = _Resizes(monkeypatch)
    frames = _camera_frames(STEPS)
    before = _run(declared, frames)
    as_given, resizes.count = resizes.count, 0
    after = _run(transform_frames_on_arrival(declared), frames)
    assert len(after) == len(before) > 1
    for old, new in zip(before, after, strict=True):
        assert new.keys() == old.keys()
        for name in new:
            np.testing.assert_array_equal(new[name], old[name])
    return as_given, resizes.count


def test_a_transform_in_front_of_the_history_keeps_its_resizes(monkeypatch):
    declared = _declared(
        sequence(restrict_image_size_v1(width=32, height=32), _history(4), component(CHUNKED_SCHEDULE, fps=FPS))
    )

    as_given, moved = _compare(declared, monkeypatch)

    assert moved == as_given == STEPS * len(CAMERAS)


def test_a_camera_field_renamed_before_the_transform_is_resized_at_the_call(monkeypatch):
    declared = _declared(
        sequence(
            _history(4),
            component(CHUNKED_SCHEDULE, fps=FPS),
            SelectObservationFields({RENAMED_WRIST: keys.WRIST_IMAGE}).to_spec(),
            TransformImages(32, 32, mode=ResizeMode.PAD, paths=[[RENAMED_WRIST]]).to_spec(),
        )
    )

    as_given, moved = _compare(declared, monkeypatch)

    assert moved == as_given > STEPS


def test_only_the_camera_fields_that_reach_the_transform_unchanged_are_resized_on_arrival(monkeypatch):
    declared = _declared(
        sequence(
            _history(4),
            component(CHUNKED_SCHEDULE, fps=FPS),
            # The wrist frame takes a new name, and the second exterior frame goes.
            SelectObservationFields({
                RENAMED_WRIST: keys.WRIST_IMAGE,
                keys.EXTERIOR_IMAGE: keys.EXTERIOR_IMAGE,
            }).to_spec(),
            TransformImages(32, 32, mode=ResizeMode.PAD, paths=[[RENAMED_WRIST], [keys.EXTERIOR_IMAGE]]).to_spec(),
        )
    )

    as_given, moved = _compare(declared, monkeypatch)

    # The exterior frame is resized once on arrival; the renamed wrist history is resized at each call.
    assert STEPS < moved < as_given


def test_a_transform_of_a_camera_outside_the_history_keeps_its_resizes(monkeypatch):
    declared = Sequential(
        TemporalStack((keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE), (-0.2, 0.0)),
        ChunkedSchedule(FPS),
        TransformImages(32, 32, mode=ResizeMode.PAD, paths=[[keys.EXTERIOR_IMAGE_2]]),
    )

    as_given, moved = _compare(declared, monkeypatch)

    assert moved == as_given > 0


def test_a_later_history_of_other_fields_lets_the_camera_frames_move(monkeypatch):
    declared = Sequential(
        TemporalStack(CAMERAS, (-0.2, 0.0)),
        TemporalStack((keys.JOINTS,), (-0.1, 0.0)),
        ChunkedSchedule(FPS),
        TransformImages(32, 32, mode=ResizeMode.PAD, paths=[[name] for name in CAMERAS]),
    )

    as_given, moved = _compare(declared, monkeypatch)

    assert moved == STEPS * len(CAMERAS) < as_given


def test_each_camera_moves_with_the_transform_that_selects_it(monkeypatch):
    declared = Sequential(
        TemporalStack(CAMERAS, (-0.2, 0.0)),
        ChunkedSchedule(FPS),
        TransformImages(32, 32, mode=ResizeMode.PAD, paths=[[keys.WRIST_IMAGE]]),
        TransformImages(48, 48, mode=ResizeMode.BOUND, paths=[[name] for name in CAMERAS]),
    )

    as_given, moved = _compare(declared, monkeypatch)

    assert moved == STEPS * len(CAMERAS) < as_given


def test_a_frame_keeps_its_arrival_time_when_its_resize_takes_time(monkeypatch):
    """A resize that takes time does not change which history frame the policy gets."""
    original = image.transform_frame
    resizes = [0]

    def slow(frame, *args, **kwargs):
        # Every second resize takes 20 ms, so the time that each step spends on resizes varies.
        resizes[0] += 1
        if resizes[0] % 2:
            _CLOCK[0] += 20_000_000
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(image, 'transform_frame', slow)
    frames = _camera_frames(STEPS)

    before = _run(_declared(_stacked_transform_spec()), frames)
    after = _run(transform_frames_on_arrival(_declared(_stacked_transform_spec())), frames)

    assert len(after) == len(before) > 1
    for old, new in zip(before, after, strict=True):
        for name in CAMERAS:
            np.testing.assert_array_equal(new[name], old[name])
