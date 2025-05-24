import functools

import av
import cv2
import numpy as np
from linuxpy.video.device import Device, PixelFormat

import ironic2 as ir


def _linux_video(device_path: str, width: int, height: int, fps: int, pixel_format: str, stopped, frames: ir.Channel):
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
            frames.write(result)

    device.close()
    stopped.set()


def linux_video(device_path: str, width: int = 640, height: int = 480, fps: int = 30, pixel_format: str = "MJPG"):
    return functools.partial(_linux_video, device_path, width, height, fps, pixel_format)
