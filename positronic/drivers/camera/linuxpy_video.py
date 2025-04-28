"""Control systems for V4L2 cameras using linuxpy library"""

import asyncio
from typing import AsyncIterator, Optional
import logging
import multiprocessing as mp
from multiprocessing.synchronize import Event as MPEvent
from queue import Empty as QueueEmpty

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
    _frame_queue: mp.Queue
    _reader_process: Optional[mp.Process] = None
    _stopped: MPEvent
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

    def __init__(
        self,
        device_path: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        pixel_format: str = "MJPG"
    ):
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
        self._frame_queue = mp.Queue(maxsize=1)
        self._reader_process = None
        self._stopped = mp.Event()
        self._error = None
        self._codec_contexts = {}
        self._logger = logging.getLogger(f"{self.__class__.__name__}_{device_path}")

        # Start the processing in a separate process
        self._reader_process = mp.Process(
            target=self._frame_process,
            args=(
                self.device_path,
                self.width,
                self.height,
                self.fps,
                self.pixel_format,
                self._frame_queue,
                self._stopped
            ),
            daemon=True
        )

    @staticmethod
    def _frame_process(device_path: str, width: int, height: int, fps: int,
                       pixel_format: str, frame_queue: mp.Queue, stopped: MPEvent):
        """Process frames from the camera in a separate process"""
        codec_mapping = {
            PixelFormat.H264: 'h264',
            PixelFormat.HEVC: 'hevc',
            PixelFormat.VP8: 'vp8',
            PixelFormat.VP9: 'vp9',
            PixelFormat.MPEG4: 'mpeg4',
            PixelFormat.MJPEG: 'mjpeg',
        }
        codec_contexts = {}

        def get_codec_context(codec_name: str) -> av.CodecContext:
            """Lazily initialize and return codec context for given codec"""
            if codec_name not in codec_contexts:
                codec_contexts[codec_name] = av.CodecContext.create(codec_name, 'r')
            return codec_contexts[codec_name]

        device = Device(device_path)
        device.open()

        device.set_format(device.info.buffers[0], width, height, pixel_format)
        device.set_fps(device.info.buffers[0], fps)

        for frame in device:
            if stopped.is_set():
                break

            data = np.frombuffer(frame.data, dtype=np.uint8)
            result = None

            match frame.pixel_format:
                case PixelFormat.YUYV:
                    data = data.reshape((frame.height, frame.width, 2))
                    result = {'image': cv2.cvtColor(data, cv2.COLOR_YUV2RGB_YUYV)}
                case PixelFormat.UYVY:
                    data = data.reshape((frame.height, frame.width, 2))
                    result = {'image': cv2.cvtColor(data, cv2.COLOR_YUV2RGB_UYVY)}
                case _ if frame.pixel_format in codec_mapping:
                    codec_name = codec_mapping[frame.pixel_format]
                    codec_ctx = get_codec_context(codec_name)
                    packets = codec_ctx.parse(data)
                    for packet in packets:
                        frames = codec_ctx.decode(packet)
                        if len(frames) == 1:
                            result = {'image': frames[0].to_ndarray(format='rgb24')}
                        else:
                            for i, decoded_frame in enumerate(frames):
                                result[f'image_{i}'] = decoded_frame.to_ndarray(format='rgb24')
                case _:
                    # Assume 3 bytes per pixel (RGB/BGR)
                    rgb_data = data.reshape((frame.height, frame.width, 3))
                    result = {'image': rgb_data}

            if result is not None:
                # Clear queue if it's full
                try:
                    frame_queue.get_nowait()
                except QueueEmpty:
                    pass
                frame_queue.put(result)

        device.close()
        stopped.set()

    async def setup(self):
        self.sync_setup()

    async def cleanup(self):
        self.sync_cleanup()

    async def step(self):
        if self._stopped.is_set() or (self._reader_process and not self._reader_process.is_alive()):
            return ir.State.FINISHED

        try:
            frame = self._frame_queue.get_nowait()
            self._fps_counter.tick()
            await self.outs.frame.write(ir.Message(data=frame))
        except QueueEmpty:
            pass
        return ir.State.ALIVE

    def _get_codec_context(self, codec_name: str) -> av.CodecContext:
        """Lazily initialize and return codec context for given codec"""
        if codec_name not in self._codec_contexts:
            self._codec_contexts[codec_name] = av.CodecContext.create(codec_name, 'r')
        return self._codec_contexts[codec_name]

    def get_frame(self):
        """Get the latest frame from the camera"""
        return self._frame_queue.get()

    def sync_setup(self):
        """Set up the camera device and start the frame processing"""
        try:
            self._stopped.clear()

            self._reader_process.start()
        except Exception as e:
            self._stopped.set()
            self._logger.error(f"Failed to setup camera {self.device_path}: {e}")
            raise

    def sync_cleanup(self):
        """Clean up camera resources"""
        self._stopped.set()

        if self._reader_process and self._reader_process.is_alive():
            self._reader_process.join(timeout=1.0)
            if self._reader_process.is_alive():
                self._reader_process.terminate()
            self._reader_process = None


if __name__ == "__main__":
    from positronic.tools.rerun_vis import RerunVisualiser
    from positronic.drivers.camera.merge import merge_on_camera

    async def _main():
        camera = LinuxPyCamera(
            '/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684LEFT-video-index0',
            width=1920,
            height=1080,
            pixel_format="MJPG"
        )
        camera2 = LinuxPyCamera(
            '/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684RIGHT-video-index0',
            width=1920,
            height=1080,
            pixel_format="MJPG")

        merged = merge_on_camera(("main", camera), {"ext": camera2})

        system = ir.compose(
            merged,
            RerunVisualiser().bind(
                frame=merged.outs.frame
            )
        )

        await ir.utils.run_gracefully(system)
    asyncio.run(_main())
