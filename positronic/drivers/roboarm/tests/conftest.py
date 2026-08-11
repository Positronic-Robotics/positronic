"""Stand-in for the Franka vendor package.

``positronic_franka`` builds against libfranka and ships only in the ``hardware`` extra, so the driver
module cannot be imported from a default sync. Its Python-side logic needs no vendor behaviour, so a stub
carrying the names the module binds at import is enough. Installed here, before any test module imports the
driver, and only when the real package is absent.
"""

import importlib.util
import sys
import types
from enum import Enum


class GoalStatus(Enum):
    REACHED = 'reached'
    IN_FLIGHT = 'in_flight'
    ABORTED = 'aborted'


PACKAGE = 'positronic_franka'
VENDOR = f'{PACKAGE}._franka'
DESK = f'{PACKAGE}.desk'


def _install_vendor_stub() -> None:
    vendor = types.ModuleType(VENDOR)
    vendor.__dict__.update(
        GoalStatus=GoalStatus,
        State=object,
        Robot=object,
        RealtimeConfig=types.SimpleNamespace(Ignore=object()),
        InternalImpedance=lambda stiffness: ('internal_impedance', stiffness),
    )

    desk = types.ModuleType(DESK)
    desk.__dict__.update(Desk=object, SafetyControllerError=type('SafetyControllerError', (Exception,), {}))

    package = types.ModuleType(PACKAGE)
    package.__dict__.update(_franka=vendor, desk=desk)

    sys.modules.update({PACKAGE: package, VENDOR: vendor, DESK: desk})


if importlib.util.find_spec(PACKAGE) is None:
    _install_vendor_stub()
