"""Stand-in for the PyAudio vendor package.

``pyaudio`` ships only in the ``hardware`` extra, so the sound driver cannot be imported from a default
sync. The tests drive its loop against their own fake stream and never call into the vendor, so a bare
module carrying the name the driver binds at import is enough. Installed here, before any test module
imports the driver, and only when the real package is absent.
"""

import importlib.util
import sys
import types

PACKAGE = 'pyaudio'

if importlib.util.find_spec(PACKAGE) is None:
    sys.modules[PACKAGE] = types.ModuleType(PACKAGE)
