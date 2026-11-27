import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from typing import Any

import numpy as np
import pytest
from positronic_model_server.spec import component, parallel, sequence

from positronic import keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.transforms import image
from positronic.dataset.transforms.image import Resample, ResizeMode
from positronic.policy import keys as policy_keys
from positronic.policy.arrival import transform_frames_on_arrival
from positronic.policy.base import Processor
from positronic.policy.codecs import TASK_FIELD, SetControlMode, TransformImages
from positronic.policy.codecs.action import AbsolutePositionAction
from positronic.policy.codecs.observation import SelectObservationFields
from positronic.policy.compatibility import (
    CHUNKED_SCHEDULE,
    WALL_TIME_NS,
    PauseOnUnavailable,
    RTCScheduleV1,
    StampObservationTimes,
    restrict_image_size_v1,
)
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.keys import RTC
from positronic.policy.processors import (
    ChunkedSchedule,
    EndOnField,
    Periodic,
    TemporalStack,
    TrajectoryPlayer,
    _Arriving,
)
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
# The fields of each observation beside the camera frames.
_STATE = {
    keys.JOINTS: np.zeros(7),
    keys.GRIP: np.zeros(1),
    keys.EE_POSE: np.zeros(7),
    keys.ROBOT_STATUS: 0,
    TASK_FIELD: 'pick',
}


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
    """Count each call of ``transform_frame`` that returns a new frame, on any thread."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        self.count = 0
        lock = threading.Lock()
        original = image.transform_frame

        def counted(frame, *args, **kwargs):
            result = original(frame, *args, **kwargs)
            with lock:
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
            run.send({**cameras, **_STATE})
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
def test_a_frame_that_a_slower_camera_repeats_is_resized_once(declared, monkeypatch):
    # Each frame comes on two steps, as an equal copy, the way a camera at half the loop rate sends it.
    frames = [
        copy
        for cameras in _camera_frames(STEPS // 2)
        for copy in (cameras, {name: np.array(frame) for name, frame in cameras.items()})
    ]
    before = _run(_declared(declared()), frames)
    resizes = _Resizes(monkeypatch)

    after = _run(transform_frames_on_arrival(_declared(declared())), frames)

    assert len(after) == len(before) > 1
    np.testing.assert_equal(after, before)
    assert resizes.count == STEPS // 2 * len(CAMERAS)


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


def _every_clock_reader_spec() -> Processor:
    """A history inside every component that reads the clock in a step, and a transform at the call."""
    return Sequential(
        EndOnField(),
        StampObservationTimes(),
        PauseOnUnavailable(),
        TemporalStack(CAMERAS, (-0.2, -0.1, 0.0)),
        Periodic(FPS / 2),
        TrajectoryPlayer(plan_key=policy_keys.PLAN, replan_on_error=True),
        ChunkedSchedule(FPS),
        TransformImages(32, 32, mode=ResizeMode.PAD, paths=[[name] for name in CAMERAS]),
    )


def _plain(value: Any) -> Any:
    """``value`` with each mapping, sequence and array copied, so a later step cannot change it."""
    if isinstance(value, Mapping):
        return {name: _plain(item) for name, item in value.items() if name != WALL_TIME_NS}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.array(value)
    return value


def _record(stack: Processor, frames: list[dict[str, np.ndarray]], answers: bool) -> tuple[list, list]:
    """Each call input and each step output of ``stack``, with all fields, the wall clock left out.

    With ``answers``, each call answers a chunk of four joint targets; otherwise it answers nothing. A call ends
    after the step that makes it, so no step output depends on how fast a call thread runs.
    """
    calls: list = []
    steps: list = []
    step_done = threading.Event()

    def infer(obs):
        calls.append(_plain(obs))
        assert step_done.wait(timeout=10)
        return [{keys.TARGET_JOINTS: np.full(7, len(calls) / 10)} for _ in range(4)] if answers else None

    runtime = Executor(lambda: _CLOCK[0], simulated=True, charge_inference_time=False)
    run = runtime.start(stack, infer)
    try:
        for step, cameras in enumerate(frames):
            _CLOCK[0] = round(step * 1e9 / FPS)
            runtime.start_tick(_CLOCK[0])
            step_done.clear()
            output = run.send({**cameras, **_STATE})
            step_done.set()
            steps.append(None if output is None else (_plain(output.commands), output.resume_at_ns))
            while (waited := runtime.wait(timeout_sec=10).status) is WaitStatus.ANSWERS_READY:
                pass
            assert waited is WaitStatus.CAN_ADVANCE
    finally:
        run.close()
        runtime.close()
    return calls, steps


def _slow_main_loop_resizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every second resize on the main loop take 20 ms of the clock, so each step spends a different time."""
    original = image.transform_frame
    resizes = [0]

    def slow(frame, *args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            resizes[0] += 1
            if resizes[0] % 2:
                _CLOCK[0] += 20_000_000
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(image, 'transform_frame', slow)


CLOCK_READERS: list[tuple[Callable[[], Processor], bool]] = [
    (lambda: _declared(_stacked_transform_spec()), False),
    (lambda: _declared(_three_camera_history_spec()), False),
    (_every_clock_reader_spec, True),
]


@pytest.mark.parametrize(('declared', 'answers'), CLOCK_READERS)
def test_a_resize_on_arrival_that_takes_time_changes_no_input_and_no_step_of_the_policy(declared, answers, monkeypatch):
    _slow_main_loop_resizes(monkeypatch)
    frames = _camera_frames(STEPS)

    calls_before, steps_before = _record(declared(), frames, answers)
    calls_after, steps_after = _record(transform_frames_on_arrival(declared()), frames, answers)

    assert len(calls_before) > 1
    np.testing.assert_equal(calls_after, calls_before)
    np.testing.assert_equal(steps_after, steps_before)


def _send_until_raised(stack: TemporalStack, frames: Callable[[int], dict[str, np.ndarray]]) -> None:
    """Step ``stack`` with ``frames(step)`` for at most 5 seconds, under an inner stack that reads nothing."""

    def reads_nothing():
        while True:
            yield None

    inner = reads_nothing()
    next(inner)
    runtime = Executor(lambda: _CLOCK[0], simulated=True, charge_inference_time=False)
    run = stack.run(runtime, inner)
    next(run)
    deadline = time.monotonic() + 5
    try:
        step = 0
        while time.monotonic() < deadline:
            _CLOCK[0] = round(step * 1e9 / FPS)
            runtime.start_tick(_CLOCK[0])
            run.send(frames(step))
            step += 1
    finally:
        run.close()
        runtime.close()


def _fail(values):
    raise ValueError('no frame')


def test_a_change_on_arrival_that_fails_stops_the_stack_when_nothing_reads_the_frame():
    frames = _camera_frames(STEPS)

    with pytest.raises(ValueError, match='no frame'):
        _send_until_raised(TemporalStack(CAMERAS, (-0.1, 0.0)).with_arrival(_fail), lambda step: frames[step % STEPS])


def test_a_change_on_arrival_that_fails_stops_the_stack_while_the_camera_repeats_its_frame():
    first = _camera_frames(1)[0]

    with pytest.raises(ValueError, match='no frame'):
        _send_until_raised(
            TemporalStack(CAMERAS, (-0.1, 0.0)).with_arrival(_fail),
            lambda step: {name: np.array(frame) for name, frame in first.items()},
        )


class _HeldChange:
    """A change on arrival that waits for ``release``, and keeps the second and the newest sample of ``inner``.

    The first frame starts on the thread and waits there, so the second frame is the first one that stays queued.
    """

    def __init__(self):
        self.release = threading.Event()
        self.runs = 0
        self.second_sample: Mapping[str, Any] = {}
        self.newest_sample: Mapping[str, Any] = {}

    def __call__(self, values: dict[str, Any]) -> dict[str, Any]:
        assert self.release.wait(timeout=10)
        self.runs += 1
        return {name: frame[::2, ::2] for name, frame in values.items()}

    def inner(self):
        obs = yield None
        obs = yield None
        self.second_sample = obs
        while True:
            self.newest_sample = obs
            obs = yield None


def _drive_held(held: _HeldChange, frames: list[dict[str, np.ndarray]]) -> None:
    """Send ``frames`` while ``held`` blocks the background thread, then release it and read the newest frame."""
    inner = held.inner()
    next(inner)
    runtime = Executor(lambda: _CLOCK[0], simulated=True, charge_inference_time=False)
    run = TemporalStack(CAMERAS, (-0.1, 0.0)).with_arrival(held).run(runtime, inner)
    next(run)
    try:
        for step, cameras in enumerate(frames):
            _CLOCK[0] = round(step * 1e9 / FPS)
            runtime.start_tick(_CLOCK[0])
            run.send(cameras)
        held.release.set()
        # One thread changes the frames in order, so this read waits for every change still queued.
        held.newest_sample[CAMERAS[0]]
    finally:
        run.close()
        runtime.close()


def test_a_slow_change_on_arrival_runs_only_for_the_frames_that_the_history_still_holds():
    held = _HeldChange()

    _drive_held(held, _camera_frames(STEPS))

    # The first frame runs while the thread waits. Of the rest, only the 3 frames that a window of 0.1 s holds
    # at 15 Hz run; the frames that left the history run no change.
    assert held.runs == 1 + 3


def test_a_sample_that_holds_a_dropped_frame_still_reads_the_changed_frame():
    held = _HeldChange()
    frames = _camera_frames(STEPS)

    _drive_held(held, frames)

    for name in CAMERAS:
        expected = np.stack([frames[0][name][::2, ::2], frames[1][name][::2, ::2]])
        np.testing.assert_array_equal(held.second_sample[name], expected)


def _held_frames(periods: dict[str, int]) -> list[dict[str, np.ndarray]]:
    """Camera frames over ``STEPS`` steps, where each camera sends a new frame every ``periods[camera]`` steps."""
    fresh = _camera_frames(STEPS)
    return [
        {name: np.array(fresh[step - step % period][name]) for name, period in periods.items()} for step in range(STEPS)
    ]


def test_a_new_frame_of_one_camera_resizes_only_that_camera(monkeypatch):
    frames = _held_frames({keys.EXTERIOR_IMAGE: 1, keys.WRIST_IMAGE: 3, keys.EXTERIOR_IMAGE_2: 3})
    before = _run(_declared(_three_camera_history_spec()), frames)
    resizes = _Resizes(monkeypatch)

    after = _run(transform_frames_on_arrival(_declared(_three_camera_history_spec())), frames)

    np.testing.assert_equal(after, before)
    assert resizes.count == STEPS + 2 * STEPS // 3


# A stacked value beside the camera frames that changes on each step.
STEP_COUNT = 'step_count'


def test_a_stacked_value_that_changes_each_step_resizes_no_repeated_frame(monkeypatch):
    declared = Sequential(
        TemporalStack((*CAMERAS, STEP_COUNT), (-0.2, -0.1, 0.0)),
        ChunkedSchedule(FPS),
        TransformImages(32, 32, mode=ResizeMode.PAD, paths=[[name] for name in CAMERAS]),
    )
    frames = [
        {**cameras, STEP_COUNT: np.array([step])}
        for step, cameras in enumerate(_held_frames(dict.fromkeys(CAMERAS, 2)))
    ]
    before = _run(declared, frames)
    resizes = _Resizes(monkeypatch)

    after = _run(transform_frames_on_arrival(declared), frames)

    np.testing.assert_equal(after, before)
    assert resizes.count == STEPS // 2 * len(CAMERAS)


def _read_from_threads(arriving: _Arriving, count: int) -> list[Any]:
    """The value or the error that each of ``count`` threads reads from ``arriving``, one thread after another."""
    results: list[Any] = []

    def read():
        try:
            results.append(arriving['frame'])
        except ValueError as error:
            results.append(error)

    for _ in range(count):
        reader = threading.Thread(target=read)
        reader.start()
        reader.join(timeout=10)
    return results


def _cancelled() -> Future:
    result: Future = Future()
    result.cancel()
    return result


def test_concurrent_reads_of_a_cancelled_frame_share_one_resize():
    started, release = threading.Event(), threading.Event()
    runs: list[int] = []

    def change(values):
        runs.append(1)
        started.set()
        assert release.wait(timeout=10)
        return {name: value * 2 for name, value in values.items()}

    arriving = _Arriving({'frame': np.ones(2)}, change, _cancelled())
    results: list[np.ndarray] = []
    readers = [threading.Thread(target=lambda: results.append(arriving['frame'])) for _ in range(2)]

    readers[0].start()
    try:
        assert started.wait(timeout=10)
        # The first reader makes the resize now. The second reader comes while it runs.
        readers[1].start()
    finally:
        release.set()
    for reader in readers:
        reader.join(timeout=10)

    assert len(runs) == 1
    np.testing.assert_equal(results, [np.full(2, 2.0)] * 2)


def test_a_failed_resize_of_a_cancelled_frame_runs_once_and_each_reader_gets_its_error():
    runs: list[int] = []

    def change(values):
        runs.append(1)
        raise ValueError('no frame')

    results = _read_from_threads(_Arriving({'frame': np.ones(2)}, change, _cancelled()), 2)

    assert len(runs) == 1
    assert len(results) == 2 and all(isinstance(result, ValueError) for result in results)
