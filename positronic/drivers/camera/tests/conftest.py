"""A stand-in for the vendor package the RealSense driver imports.

``pyrealsense2`` ships only in the ``realsense`` extra, so the driver module cannot be imported from a
default sync. The tests drive the SDK's surface through a fake of their own, so a module carrying the
names the driver binds at import is enough. Installed here, before any test module imports the driver,
and only where the real package is absent.
"""

import importlib.util
import sys
import types

VENDOR = 'pyrealsense2'

if importlib.util.find_spec(VENDOR) is None:
    module = types.ModuleType(VENDOR)
    module.__dict__.update(
        config=object,
        pipeline=object,
        align=object,
        context=object,
        stream=types.SimpleNamespace(color='color', depth='depth'),
        format=types.SimpleNamespace(rgb8='rgb8', z16='z16'),
        camera_info=types.SimpleNamespace(serial_number='serial_number', name='name', firmware_version='firmware'),
    )
    sys.modules[VENDOR] = module
