"""What the RealSense driver puts on its ports, and what it does with a camera that stops answering."""

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.drivers.camera import realsense as realsense_driver
from positronic.tests.testing_coutils import RecordingEmitter

DEPTH_SCALE = 1e-4  # meters per unit of z16, as a D405 reports it


class StopFlag(pimm.SignalReceiver[bool]):
    """``should_stop`` under the test's control."""

    def __init__(self):
        self.stopped = False

    def read(self) -> pimm.Message[bool]:
        return pimm.Message(self.stopped)


class FakeFrame:
    def __init__(self, data: np.ndarray | None):
        self._data = data

    def get_data(self) -> np.ndarray | None:
        return self._data


class FakeFrames:
    """A frameset, or the empty one a poll returns when the camera has nothing ready."""

    def __init__(self, color: np.ndarray | None = None, depth: np.ndarray | None = None):
        self.color = color
        self.depth = depth

    def __bool__(self) -> bool:
        return self.color is not None

    def get_color_frame(self) -> FakeFrame:
        return FakeFrame(self.color)

    def get_depth_frame(self) -> FakeFrame:
        return FakeFrame(self.depth)


class FakeSDK:
    """The ``pyrealsense2`` surface the driver reaches for, with the camera under the test's control.

    ``queued`` is what the next polls hand back; an empty queue polls empty, as a silent camera does.
    ``start_raises`` is what opening the camera raises, and ``streams`` records what each pipeline was
    started on, so a reopen is visible as a second entry.
    """

    stream = realsense_driver.rs.stream
    format = realsense_driver.rs.format

    def __init__(self):
        self.queued: list[FakeFrames] = []
        self.start_raises: Exception | None = None
        self.streams: list[dict] = []
        self.devices: list[str | None] = []
        self.stopped = 0
        self.aligned = 0

    def config(self) -> 'FakeSDK.Config':
        return FakeSDK.Config(self)

    def pipeline(self) -> 'FakeSDK.Pipeline':
        return FakeSDK.Pipeline(self)

    def align(self, to_stream):
        return FakeSDK.Align(self)

    class Config:
        def __init__(self, sdk: 'FakeSDK'):
            self.sdk = sdk
            self.device: str | None = None
            self.enabled: dict = {}

        def enable_device(self, serial: str) -> None:
            self.device = serial

        def enable_stream(self, stream, width, height, fmt, fps) -> None:
            self.enabled[stream] = (width, height, fmt, fps)

    class Pipeline:
        def __init__(self, sdk: 'FakeSDK'):
            self.sdk = sdk

        def start(self, config: 'FakeSDK.Config'):
            if self.sdk.start_raises is not None:
                raise self.sdk.start_raises
            self.sdk.streams.append(config.enabled)
            self.sdk.devices.append(config.device)
            return FakeSDK.Profile()

        def poll_for_frames(self) -> FakeFrames:
            return self.sdk.queued.pop(0) if self.sdk.queued else FakeFrames()

        def stop(self) -> None:
            self.sdk.stopped += 1

    class Align:
        def __init__(self, sdk: 'FakeSDK'):
            self.sdk = sdk

        def process(self, frames: FakeFrames) -> FakeFrames:
            self.sdk.aligned += 1
            return frames

    class Profile:
        def get_device(self) -> 'FakeSDK.Profile':
            return self

        def first_depth_sensor(self) -> 'FakeSDK.Profile':
            return self

        def get_depth_scale(self) -> float:
            return DEPTH_SCALE


@pytest.fixture
def sdk(monkeypatch) -> FakeSDK:
    fake = FakeSDK()
    monkeypatch.setattr(realsense_driver, 'rs', fake)
    return fake


def _driven(sdk: FakeSDK, *, depth: bool = False, serial: str | None = '419122270018', **kwargs):
    """A driver over ``sdk`` with its ports recorded, and its loop ready to pump."""
    camera = realsense_driver.RealSenseCamera(serial_number=serial, **kwargs)
    frames, depths = RecordingEmitter(), RecordingEmitter()
    camera.frame._bind(frames)
    if depth:
        camera.depth._bind(depths)
    clock, stop = MockClock(), StopFlag()
    return camera, frames, depths, clock, stop, camera.run(stop, clock)


def _pump(loop, clock: MockClock, ticks: int = 1) -> None:
    """Run the loop, letting the clock pass whatever each tick asks to sleep for."""
    for _ in range(ticks):
        cmd = next(loop)
        clock.advance(cmd.seconds if isinstance(cmd, pimm.Sleep) else 0.0)


def _image(value: int = 7) -> np.ndarray:
    return np.full((4, 6, 3), value, dtype=np.uint8)


def test_colour_reaches_the_port_stamped_on_arrival(sdk):
    _, frames, _, clock, _, loop = _driven(sdk)
    sdk.queued = [FakeFrames(_image())]

    _pump(loop, clock, ticks=2)

    assert len(frames.emitted) == 1
    ts, adapter = frames.emitted[0]
    np.testing.assert_array_equal(adapter.array, _image())
    assert ts == 0  # the clock stood at zero when the frame was polled


def test_a_poll_with_nothing_ready_emits_nothing(sdk):
    _, frames, _, clock, _, loop = _driven(sdk)

    _pump(loop, clock, ticks=5)

    assert frames.emitted == []


def test_the_camera_is_opened_on_its_serial(sdk):
    _, _, _, clock, _, loop = _driven(sdk, serial='419122270018')
    _pump(loop, clock)
    assert sdk.devices == ['419122270018']


def test_no_serial_takes_whichever_camera_answers(sdk):
    _, _, _, clock, _, loop = _driven(sdk, serial=None)
    _pump(loop, clock)
    assert sdk.devices == [None]


def test_depth_is_not_streamed_until_the_port_is_bound(sdk):
    _, _, depths, clock, _, loop = _driven(sdk, depth=False)
    sdk.queued = [FakeFrames(_image())]

    _pump(loop, clock, ticks=2)

    assert list(sdk.streams[0]) == [sdk.stream.color]
    assert sdk.aligned == 0
    assert depths.emitted == []


def test_a_bound_depth_port_streams_depth_in_meters(sdk):
    _, frames, depths, clock, _, loop = _driven(sdk, depth=True)
    sdk.queued = [FakeFrames(_image(), np.full((4, 6), 3000, dtype=np.uint16))]

    _pump(loop, clock, ticks=2)

    assert sdk.streams[0][sdk.stream.depth] == (640, 480, sdk.format.z16, 30)
    assert sdk.aligned == 1
    ts, adapter = depths.emitted[0]
    np.testing.assert_allclose(adapter.array, np.full((4, 6, 1), 0.3, dtype=np.float32), rtol=1e-6)
    assert ts == frames.emitted[0][0]  # one frameset, one timestamp


def test_silence_past_the_stall_reopens_the_camera(sdk):
    _, frames, _, clock, _, loop = _driven(sdk)
    _pump(loop, clock, ticks=int(realsense_driver._STALL_S / realsense_driver._POLL_S) + 2)

    assert sdk.stopped == 1
    assert len(sdk.streams) == 2

    sdk.queued = [FakeFrames(_image())]
    _pump(loop, clock, ticks=2)
    assert len(frames.emitted) == 1


def test_a_camera_that_stays_down_raises(sdk):
    _, _, _, clock, _, loop = _driven(sdk, max_recovery_time_sec=2.0)
    sdk.start_raises = RuntimeError('No device connected')

    with pytest.raises(RuntimeError, match='No device connected'):
        _pump(loop, clock, ticks=1000)


def test_the_pipeline_is_stopped_when_the_world_comes_down(sdk):
    _, _, _, clock, stop, loop = _driven(sdk)
    _pump(loop, clock)
    stop.stopped = True

    with pytest.raises(StopIteration):
        next(loop)
    assert sdk.stopped == 1
