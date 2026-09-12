import logging
from collections.abc import Iterator

import av
import cv2
import numpy as np

import pimm
from positronic.drivers import vendor_import

with vendor_import('linuxpy', 'Linux video capture', platforms=('linux',)):
    from linuxpy.video.device import Device, PixelFormat

logger = logging.getLogger(__name__)


class LinuxVideo(pimm.ControlSystem):
    # The formats a camera may compress in, and what decodes each of them.
    _CODECS = {
        PixelFormat.H264: 'h264',
        PixelFormat.HEVC: 'hevc',
        PixelFormat.VP8: 'vp8',
        PixelFormat.VP9: 'vp9',
        PixelFormat.MPEG4: 'mpeg4',
        PixelFormat.MJPEG: 'mjpeg',
    }

    def __init__(self, device_path: str, width: int, height: int, fps: int, pixel_format: str):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.fps_counter = pimm.utils.RateCounter(f'LinuxVideo {device_path}')
        self.frame = pimm.ControlSystemEmitter[pimm.shared_memory.NumpySMAdapter](self)
        self._frame_adapter = None  # Lazy init

    @staticmethod
    def _framed(data: np.ndarray, frame, bytes_per_pixel: int) -> np.ndarray | None:
        """``data`` shaped as the frame it belongs to, or ``None`` where it is not one frame's worth."""
        if data.size != frame.height * frame.width * bytes_per_pixel:
            return None
        return data.reshape((frame.height, frame.width, bytes_per_pixel))

    def _images(self, frame, codec_context) -> list[np.ndarray] | None:
        """Every image the buffer ``frame`` carries, as RGB, or ``None`` where it is not one frame's worth.

        A compressed buffer whole enough to read may still carry no image: the parser and the decoder both
        hold data back until they have a frame to give, so an empty list is a wait, not a loss.
        """
        data = np.frombuffer(frame.data, dtype=np.uint8)
        match frame.pixel_format:
            case PixelFormat.YUYV:
                raw = self._framed(data, frame, 2)
                return None if raw is None else [cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_YUYV)]
            case PixelFormat.UYVY:
                raw = self._framed(data, frame, 2)
                return None if raw is None else [cv2.cvtColor(raw, cv2.COLOR_YUV2RGB_UYVY)]
            case _ if frame.pixel_format in self._CODECS:
                codec_ctx = codec_context(self._CODECS[frame.pixel_format])
                # `av` types what it parses as `bytes` and carries `decode` on the subclasses of
                # `CodecContext`, so the buffer the device hands over and the base class both read wrong.
                packets = codec_ctx.parse(data)  # pyright: ignore[reportArgumentType]
                return [
                    decoded.to_ndarray(format='rgb24')
                    for packet in packets
                    for decoded in codec_ctx.decode(packet)  # pyright: ignore[reportAttributeAccessIssue]
                ]
            case _:
                raw = self._framed(data, frame, 3)  # assume 3 bytes per pixel (RGB/BGR)
                return None if raw is None else [raw]

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        codec_contexts = {}

        def codec_context(codec_name: str) -> av.CodecContext:
            """Lazily initialize and return codec context for given codec"""
            if codec_name not in codec_contexts:
                codec_contexts[codec_name] = av.CodecContext.create(codec_name, 'r')
            return codec_contexts[codec_name]

        device = Device(self.device_path)
        device.open()

        device.set_format(device.info.buffers[0], self.width, self.height, self.pixel_format)
        device.set_fps(device.info.buffers[0], self.fps)

        misframed = overtaken = 0
        for frame in device:
            if should_stop.value:
                break

            images = self._images(frame, codec_context)
            # rules-allow: swallowed-error — a busy bus hands over a buffer that is not a whole frame as
            # ordinary traffic, roughly one per camera per minute of capture on the station's four D405;
            # the next buffer is a thirtieth of a second away, and the count says how many went
            if images is None:
                misframed += 1
                if misframed == 1:
                    logger.warning('%s handed over a buffer that is not one frame in size', self.device_path)
            elif images:
                # The port holds one image and nothing runs between two emissions of the same tick, so a
                # buffer decoding to several has only its newest to give; the rest are counted, not sent.
                overtaken += len(images) - 1
                self._frame_adapter = pimm.shared_memory.NumpySMAdapter.lazy_init(images[-1], self._frame_adapter)
                self.frame.emit(self._frame_adapter)
                self.fps_counter.tick()

            yield pimm.Yield()  # Give control back to the world

        if misframed:
            logger.warning('%s handed over %d buffers that are not one frame in size', self.device_path, misframed)
        if overtaken:
            logger.warning('%s decoded %d images a newer one of the same buffer overtook', self.device_path, overtaken)
        device.close()
