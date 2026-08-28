"""What the Linux video driver puts on its frame port, from the buffers a device hands it."""

import numpy as np
import pytest

import pimm
from positronic.drivers.camera import linux_video
from positronic.tests.testing_coutils import RecordingEmitter

WIDTH, HEIGHT = 4, 2


class StopFlag(pimm.SignalReceiver[bool]):
    """``should_stop`` under the test's control."""

    def __init__(self):
        self.stopped = False

    def read(self) -> pimm.Message[bool]:
        return pimm.Message(self.stopped)


class FakeFrame:
    def __init__(self, data: bytes, pixel_format):
        self.data = data
        self.pixel_format = pixel_format
        self.width, self.height = WIDTH, HEIGHT


class FakeDevice:
    """A device that hands over the frames a test gives it, and records what it was set to."""

    to_serve: list['FakeFrame'] = []
    opened: 'FakeDevice | None' = None

    def __init__(self, path: str):
        self.path = path
        self.info = type('Info', (), {'buffers': ['capture']})()
        self.frames = list(FakeDevice.to_serve)
        self.format = None
        self.fps = None
        self.closed = False
        FakeDevice.opened = self

    def open(self) -> None:
        pass

    def set_format(self, buffer, width, height, pixel_format) -> None:
        self.format = (buffer, width, height, pixel_format)

    def set_fps(self, buffer, fps) -> None:
        self.fps = (buffer, fps)

    def __iter__(self):
        return iter(self.frames)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def device(monkeypatch):
    monkeypatch.setattr(linux_video, 'Device', FakeDevice)
    return FakeDevice


def _driven(frames, **kwargs):
    """A driver over a device carrying ``frames``, with its port recorded, run to exhaustion."""
    camera = linux_video.LinuxVideo(
        device_path='/dev/null', width=WIDTH, height=HEIGHT, fps=30, pixel_format='YUYV', **kwargs
    )
    emitted = RecordingEmitter()
    camera.frame._bind(emitted)
    FakeDevice.to_serve = frames
    list(camera.run(StopFlag(), pimm.world.SystemClock()))
    opened = FakeDevice.opened
    assert opened is not None, 'the driver opened no device'
    return emitted, opened


def _yuyv(luma: int) -> bytes:
    """One YUYV buffer of a flat grey, which converts to a flat grey RGB image."""
    return bytes([luma, 128] * (WIDTH * HEIGHT))


def test_a_yuyv_buffer_reaches_the_port_as_an_image(device):
    emitted, _ = _driven([FakeFrame(_yuyv(200), linux_video.PixelFormat.YUYV)])

    assert len(emitted.emitted) == 1
    _, adapter = emitted.emitted[0]
    assert adapter.array.shape == (HEIGHT, WIDTH, 3)
    assert adapter.array.dtype == np.uint8
    assert adapter.array.min() > 150  # the grey survives the conversion


def test_every_buffer_is_one_frame(device):
    frames = [FakeFrame(_yuyv(v), linux_video.PixelFormat.YUYV) for v in (50, 120, 200)]

    emitted, _ = _driven(frames)

    assert len(emitted.emitted) == 3


def test_the_device_is_set_to_what_the_driver_was_asked_for(device):
    _, opened = _driven([FakeFrame(_yuyv(100), linux_video.PixelFormat.YUYV)])

    assert opened.format == ('capture', WIDTH, HEIGHT, 'YUYV')
    assert opened.fps == ('capture', 30)


def test_the_device_is_closed_when_the_frames_run_out(device):
    _, opened = _driven([FakeFrame(_yuyv(100), linux_video.PixelFormat.YUYV)])

    assert opened.closed


def test_a_buffer_short_of_a_frame_is_dropped(device, caplog):
    """A busy bus hands over a buffer with its tail missing, and a run outlives it."""
    short = FakeFrame(_yuyv(200)[: WIDTH * HEIGHT], linux_video.PixelFormat.YUYV)
    whole = FakeFrame(_yuyv(200), linux_video.PixelFormat.YUYV)

    emitted, _ = _driven([short, whole, short])

    assert len(emitted.emitted) == 1
    assert 'short of a frame' in caplog.text
    assert 'handed over 2 buffers' in caplog.text


def test_a_buffer_of_three_bytes_a_pixel_is_taken_as_it_is(device):
    raw = bytes(range(WIDTH * HEIGHT * 3))

    emitted, _ = _driven([FakeFrame(raw, linux_video.PixelFormat.RGB24)])

    _, adapter = emitted.emitted[0]
    np.testing.assert_array_equal(adapter.array, np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3))
