"""Driver for the RealSense depth cameras. One USB link carries the colour stream and the depth stream.

The driver starts the depth stream only if the ``depth`` port has a consumer. A depth stream that nobody
reads takes USB bandwidth from the colour stream, and four cameras do not fit on one USB controller. If
the depth stream is on, ``rs.align`` moves each depth pixel to the colour pixel of the same point.

Each frame gets the time of its arrival, and not ``frame.get_timestamp()``. The camera counts
milliseconds from the epoch of the host, and the world clock has a different epoch. A frame is a few
milliseconds old when it arrives. The colour frame and the depth frame of one frameset get the same time.

``run()`` starts the pipeline. ``World.start`` pickles a background control system, a pipeline does not
stay usable after that, and two processes must not open the same camera.
"""

import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

import pimm
from pimm.shared_memory import NumpySMAdapter
from positronic.drivers import vendor_import
from positronic.drivers.camera.device_open_lock import device_open_lock

# pyrealsense2 lives in the `realsense` extra, which the type-check environment does not install.
with vendor_import(
    'pyrealsense2',
    'RealSense camera support',
    hint='Re-run with the realsense extra:\n  uv run --locked --extra realsense ...\n',
    platforms=('linux', 'win32'),
):
    import pyrealsense2 as rs  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# The wait between two polls. It adds latency, but it sets no rate: frames come 33 ms apart at 30 Hz.
_POLL_S = 0.001
# A quiet time this long shows a lost link. The camera sends a frame each 200 ms at its lowest rate, and it
# recovers a dropped frame in one or two periods.
_STALL_S = 1.0
_REOPEN_S = 0.5


@dataclass
class _Stream:
    """The pipeline of one run, and the objects that read a frame from it.

    ``align`` and ``depth_scale`` have a value only if the depth stream is on.
    """

    pipeline: Any
    align: Any | None
    depth_scale: float


class RealSenseCamera(pimm.ControlSystem):
    def __init__(
        self,
        serial_number: str | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_recovery_time_sec: float = 10.0,
    ):
        """RealSense camera driver.

        Args:
            serial_number: (str) Serial number of the camera to open, as ``rs.context`` reports it.
                ``None`` opens the first camera that the SDK finds. Use it only on a host with one camera.
            width: (int) Frame width, for the colour stream and the depth stream.
            height: (int) Frame height, for the colour stream and the depth stream.
            fps: (int) Frames per second. The camera refuses a combination that has no profile.
            max_recovery_time_sec: (float) How long the driver opens a quiet camera again before it
                raises.
        """
        super().__init__()
        # IMPORTANT: This control system may be spawned under multiprocessing "spawn". Keep only plain-Python
        # config on self so the instance is picklable; construct pyrealsense2 objects inside `run()`.
        self._serial_number = serial_number
        self._width = width
        self._height = height
        self._fps = fps
        self._max_recovery_time_sec = max_recovery_time_sec

        self.frame = pimm.ControlSystemEmitter[NumpySMAdapter](self)
        self._frame_adapter = None  # Lazy init

        self.depth = pimm.ControlSystemEmitter[NumpySMAdapter](self)
        self._depth_adapter = None  # Lazy init

    def _open(self, depth_wanted: bool) -> _Stream:
        """Start the streams, and give back the objects that read them."""
        config = rs.config()
        if self._serial_number is not None:
            config.enable_device(self._serial_number)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        if depth_wanted:
            config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)

        pipeline = rs.pipeline()
        with device_open_lock():
            profile = pipeline.start(config)
        if not depth_wanted:
            return _Stream(pipeline, None, 0.0)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        return _Stream(pipeline, rs.align(rs.stream.color), depth_scale)

    def _reopened(
        self, stream: _Stream, depth_wanted: bool, should_stop: pimm.SignalReceiver, clock: pimm.Clock
    ) -> Generator[pimm.Sleep, None, _Stream | None]:
        """Open a quiet camera again, and yield until it sends frames. Drive with ``yield from``.

        Gives back ``None`` if the world stops before the camera sends frames again.
        """
        logger.warning('RealSense %s went silent, reopening', self._serial_number)
        try:
            stream.pipeline.stop()
        except RuntimeError as exc:
            # A disconnected camera has no pipeline to stop. The reopen below ends the run if it also fails.
            logger.error('RealSense %s did not stop cleanly: %s', self._serial_number, exc)

        deadline = clock.now() + self._max_recovery_time_sec
        while not should_stop.value:
            try:
                reopened = self._open(depth_wanted)
            except RuntimeError as exc:
                if clock.now() >= deadline:
                    raise
                logger.warning('RealSense %s did not reopen: %s', self._serial_number, exc)
                yield pimm.Sleep(_REOPEN_S)
                continue
            logger.info('RealSense %s is streaming again', self._serial_number)
            return reopened
        return None

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        depth_wanted = self.depth.num_bound > 0
        fps_counter = pimm.utils.RateCounter(f'RealSense {self._serial_number}')
        stream = self._open(depth_wanted)
        last_frame_at = clock.now()

        while not should_stop.value:
            frames = stream.pipeline.poll_for_frames()
            if not frames:
                if clock.now() - last_frame_at > _STALL_S:
                    stream = yield from self._reopened(stream, depth_wanted, should_stop, clock)
                    if stream is None:
                        return
                    last_frame_at = clock.now()
                yield pimm.Sleep(_POLL_S)
                continue

            last_frame_at = clock.now()
            ts = clock.now_ns()
            if stream.align is not None:
                frames = stream.align.process(frames)
                depth_m = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32)
                depth_m *= np.float32(stream.depth_scale)
                self._depth_adapter = NumpySMAdapter.lazy_init(depth_m[..., np.newaxis], self._depth_adapter)
                self.depth.emit(self._depth_adapter, ts=ts)

            image = np.asanyarray(frames.get_color_frame().get_data())
            self._frame_adapter = NumpySMAdapter.lazy_init(image, self._frame_adapter)
            self.frame.emit(self._frame_adapter, ts=ts)

            fps_counter.tick()
            yield pimm.Sleep(_POLL_S)

        stream.pipeline.stop()


if __name__ == '__main__':
    import argparse
    import time

    from positronic.drivers.camera.video_writer import VideoWriter

    parser = argparse.ArgumentParser(description='RealSense driver smoke: stream a camera into an mp4.')
    parser.add_argument('output', nargs='?', help='where to write the mp4')
    parser.add_argument('--serial', default=None, help='which camera to open; the first one if left out')
    parser.add_argument('--seconds', type=float, default=10.0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--list', action='store_true', help='print the cameras this host can see and exit')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    if args.list:
        for device in rs.context().devices:
            print(
                device.get_info(rs.camera_info.serial_number),
                device.get_info(rs.camera_info.name),
                device.get_info(rs.camera_info.firmware_version),
            )
        raise SystemExit(0)

    if args.output is None:
        parser.error('an output path is required unless --list is given')

    with pimm.World() as world:
        camera = RealSenseCamera(serial_number=args.serial, fps=args.fps)
        writer = VideoWriter(args.output, args.fps)
        world.connect(camera.frame, writer.frame)

        deadline = time.monotonic() + args.seconds
        for cmd in world.start(writer, camera):
            if time.monotonic() > deadline:
                break
            time.sleep(cmd.seconds if isinstance(cmd, pimm.Sleep) else 0)
