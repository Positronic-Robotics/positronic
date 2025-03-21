"""Control systems for V4L2 cameras using linuxpy library"""

import asyncio
from typing import AsyncIterator, Optional
import logging

import cv2
from linuxpy.video.device import Device, Frame, PixelFormat
import numpy as np
import av

import ironic as ir


@ir.ironic_system(output_ports=['frame'])
class LinuxPyCamera(ir.ControlSystem):
    device_path: str
    width: int
    height: int
    fps: int
    pixel_format: str

    _device: Optional[Device] = None
    _aiter: Optional[AsyncIterator[Frame]] = None
    _fps_counter: ir.utils.FPSCounter
    _frame_queue: asyncio.Queue
    _reader_task: Optional[asyncio.Task] = None
    _stopped: asyncio.Event
    _error: Optional[Exception] = None
    _codec_contexts: dict[str, av.CodecContext]
    _logger: logging.Logger
    _codec_mapping = {
        PixelFormat.H264: 'h264',
        PixelFormat.HEVC: 'hevc',
        PixelFormat.VP8: 'vp8',
        PixelFormat.VP9: 'vp9',
        PixelFormat.MPEG4: 'mpeg4',
        PixelFormat.MJPEG: 'mjpeg',
    }

    def __init__(self, device_path: str,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 pixel_format: str = "MJPG"):
        """Video4Linux camera that uses linuxpy library

        Args:
            device_path: Path to video device (e.g. "/dev/video0")
            width: Frame width
            height: Frame height
            fps: Target frames per second
            pixel_format: Pixel format (e.g. "MJPG", "YUYV", "UYVY")
        """
        super().__init__()
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format

        self._fps_counter = ir.utils.FPSCounter(f"Camera {self.device_path}")
        self._frame_queue = asyncio.Queue(maxsize=1)
        self._reader_task = None
        self._stopped = asyncio.Event()
        self._error = None
        self._codec_contexts = {}
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{device_path}")

    async def _frame_generator(self) -> AsyncIterator[dict]:
        """Generate processed frames from the camera"""
        while not self._stopped.is_set():
            try:
                frame = await anext(self._aiter)
                data = np.frombuffer(frame.data, dtype=np.uint8)

                match frame.pixel_format:
                    case PixelFormat.YUYV:
                        data = data.reshape((frame.height, frame.width, 2))
                        rgb_data = cv2.cvtColor(data, cv2.COLOR_YUV2RGB_YUYV)
                        yield {'image': rgb_data}
                    case PixelFormat.UYVY:
                        data = data.reshape((frame.height, frame.width, 2))
                        rgb_data = cv2.cvtColor(data, cv2.COLOR_YUV2RGB_UYVY)
                        yield {'image': rgb_data}
                    case _ if frame.pixel_format in self._codec_mapping:
                        codec_name = self._codec_mapping[frame.pixel_format]
                        codec_ctx = self._get_codec_context(codec_name)
                        packets = codec_ctx.parse(data)
                        for packet in packets:
                            frames = codec_ctx.decode(packet)
                            for frame in frames:
                                yield {'image': frame.to_ndarray(format='rgb24')}
                    case _:
                        # Assume 3 bytes per pixel (RGB/BGR)
                        rgb_data = data.reshape((frame.height, frame.width, 3))
                        yield {'image': rgb_data}
            except StopAsyncIteration:
                break

    async def _frame_reader(self):
        """Background coroutine that reads processed frames and manages the queue"""
        try:
            async for frame in self._frame_generator():
                assert frame is not None
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await self._frame_queue.put(frame)
        except Exception as e:
            self._error = e
            self._logger.error(f"Camera error: {e}")
            raise
        finally:
            self._stopped.set()

    async def setup(self):
        """Set up the camera device and configure format"""
        try:
            self._device = Device(self.device_path)
            self._device.open()

            self._device.set_format(self._device.info.buffers[0],
                                    self.width, self.height, self.pixel_format)

            self._device.set_fps(self._device.info.buffers[0], self.fps)
            self._aiter = aiter(self._device)

            self._stopped.clear()
            self._error = None
            self._reader_task = asyncio.create_task(self._frame_reader())
        except OSError as e:
            self._error = e
            self._stopped.set()
            self._logger.error(f"Failed to setup camera {self.device_path}: {e}")
            raise

    async def cleanup(self):
        """Clean up camera resources"""
        self._stopped.set()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._device:
            if self._aiter:
                await self._aiter.aclose()
                self._aiter = None
            self._device.close()
            self._device = None

    async def step(self):
        if self._stopped.is_set():
            if self._error:
                self._logger.error(f"Camera {self.device_path} failed: {self._error}")
                raise self._error
            return ir.State.FINISHED

        try:
            frame = self._frame_queue.get_nowait()
            self._fps_counter.tick()
            await self.outs.frame.write(ir.Message(data=frame))
        except asyncio.QueueEmpty:
            pass
        return ir.State.ALIVE

    def _get_codec_context(self, codec_name: str) -> av.CodecContext:
        """Lazily initialize and return codec context for given codec"""
        if codec_name not in self._codec_contexts:
            self._codec_contexts[codec_name] = av.CodecContext.create(codec_name, 'r')
        return self._codec_contexts[codec_name]


if __name__ == "__main__":
    import asyncio
    from positronic.tools.video import VideoDumper
    from positronic.tools.rerun_vis import RerunVisualiser
    from positronic.drivers.camera.merge import merge_on_camera

    async def _main():
        camera = LinuxPyCamera(
            '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:2:1.0-video-index0',
            width=1920,
            height=1080,
            pixel_format="MJPG"
        )
        camera2 = LinuxPyCamera(
            '/dev/v4l/by-path/pci-0000:00:14.0-usb-0:6:1.0-video-index0',
            width=1920,
            height=1080,
            pixel_format="MJPG")

        merged = merge_on_camera(("main", camera), {"ext": camera2})

        system = ir.compose(
            merged,
            # VideoDumper("video.mp4", 30, codec='libx264').bind(
            #     image=ir.utils.map_port(lambda x: x['image'], merged.outs.frame)
            # ),
            RerunVisualiser().bind(
                frame=merged.outs.frame
            )
        )

        await ir.utils.run_gracefully(system)
    asyncio.run(_main())
