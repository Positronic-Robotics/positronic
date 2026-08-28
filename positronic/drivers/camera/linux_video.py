from collections.abc import Iterator

import av
import cv2
import numpy as np

import pimm
from positronic.drivers import vendor_import

with vendor_import('linuxpy', 'Linux video capture', platforms=('linux',)):
    from linuxpy.video.device import Device, PixelFormat


class LinuxVideo(pimm.ControlSystem):
    def __init__(self, device_path: str, width: int, height: int, fps: int, pixel_format: str):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.fps_counter = pimm.utils.RateCounter(f'LinuxVideo {device_path}')
        self.frame = pimm.ControlSystemEmitter[pimm.shared_memory.NumpySMAdapter](self)
        self._frame_adapter = None  # Lazy init

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
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

        device = Device(self.device_path)
        device.open()

        device.set_format(device.info.buffers[0], self.width, self.height, self.pixel_format)
        device.set_fps(device.info.buffers[0], self.fps)

        for frame in device:
            if should_stop.value:
                break

            data = np.frombuffer(frame.data, dtype=np.uint8)

            match frame.pixel_format:
                case PixelFormat.YUYV:
                    data = data.reshape((frame.height, frame.width, 2))
                    images = [cv2.cvtColor(data, cv2.COLOR_YUV2RGB_YUYV)]
                case PixelFormat.UYVY:
                    data = data.reshape((frame.height, frame.width, 2))
                    images = [cv2.cvtColor(data, cv2.COLOR_YUV2RGB_UYVY)]
                case _ if frame.pixel_format in codec_mapping:
                    codec_ctx = get_codec_context(codec_mapping[frame.pixel_format])
                    # `av` types what it parses as `bytes` and carries `decode` on the subclasses of
                    # `CodecContext`, so the buffer the device hands over and the base class both read wrong.
                    packets = codec_ctx.parse(data)  # pyright: ignore[reportArgumentType]
                    images = [
                        decoded.to_ndarray(format='rgb24')
                        for packet in packets
                        for decoded in codec_ctx.decode(packet)  # pyright: ignore[reportAttributeAccessIssue]
                    ]
                case _:
                    # Assume 3 bytes per pixel (RGB/BGR)
                    images = [data.reshape((frame.height, frame.width, 3))]

            for image in images:
                self._frame_adapter = pimm.shared_memory.NumpySMAdapter.lazy_init(image, self._frame_adapter)
                self.frame.emit(self._frame_adapter)
                self.fps_counter.tick()

            yield pimm.Yield()  # Give control back to the world

        device.close()
