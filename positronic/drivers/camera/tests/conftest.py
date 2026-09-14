"""Stand-in for the ZED vendor package.

``pyzed`` installs with the ZED SDK and nowhere else, so the driver cannot be imported from a default sync.
The tests read the driver's state against their own fake camera and never call into the vendor, so a module
carrying the names the driver binds at import is enough. Installed here, before any test module imports the
driver, and only when the real package is absent.
"""

import importlib.util
import sys
import types
from enum import Enum

PACKAGE = 'pyzed'
SL = f'{PACKAGE}.sl'


class ErrorCode(Enum):
    SUCCESS = 0
    FAILURE = 1


class VideoSettings(Enum):
    EXPOSURE = 0
    GAIN = 1
    WHITEBALANCE_TEMPERATURE = 2
    AEC_AGC = 3
    WHITEBALANCE_AUTO = 4


if importlib.util.find_spec(PACKAGE) is None:
    sl = types.ModuleType(SL)
    sl.__dict__.update(ERROR_CODE=ErrorCode, VIDEO_SETTINGS=VideoSettings)
    package = types.ModuleType(PACKAGE)
    package.__dict__.update(sl=sl)
    sys.modules.update({PACKAGE: package, SL: sl})
