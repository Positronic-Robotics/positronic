from collections.abc import Iterator

import av
import cv2
import numpy as np
from linuxpy.video.device import Device, PixelFormat

import pimm


class LinuxVideo(pimm.ControlSystem):
    def __init__(self, device_path: str, width: int, height: int, fps: int, pixel_format: str):
        self.device_path = device_path
        self.width = width
        self.height = height
        self.fps = fps
        self.pixel_format = pixel_format
        self.fps_counter = pimm.utils.RateCounter(f'LinuxVideo {device_path}')
        self.frame: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)

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
                self.frame.emit(result)
                self.fps_counter.tick()

            yield pimm.Pass()  # Give control back to the world

        device.close()
