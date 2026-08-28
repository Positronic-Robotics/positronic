"""A stand-in for the vendor package the Linux video driver imports.

``linuxpy`` ships only in the ``hardware`` extra, so the driver module cannot be imported from a default
sync. The tests drive a device of their own, so a module carrying the names the driver binds at import is
enough. Installed here, before any test module imports the driver, and only where the real package is
absent.
"""

import importlib.util
import sys
import types
from enum import Enum

VENDOR = 'linuxpy'
DEVICE_MODULE = f'{VENDOR}.video.device'

if importlib.util.find_spec(VENDOR) is None:
    # The formats the driver names. The values are the V4L2 four-character codes, as `linuxpy` reports them.
    pixel_format = Enum('PixelFormat', ['YUYV', 'UYVY', 'RGB24', 'H264', 'HEVC', 'VP8', 'VP9', 'MPEG4', 'MJPEG'])

    device = types.ModuleType(DEVICE_MODULE)
    device.__dict__.update(Device=object, PixelFormat=pixel_format)

    video = types.ModuleType(f'{VENDOR}.video')
    video.__dict__.update(device=device)

    package = types.ModuleType(VENDOR)
    package.__dict__.update(video=video)

    sys.modules.update({VENDOR: package, f'{VENDOR}.video': video, DEVICE_MODULE: device})
