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


def _install_vendor_stub() -> None:
    vendor = types.ModuleType('positronic_franka._franka')
    vendor.__dict__.update(
        GoalStatus=GoalStatus,
        State=object,
        Robot=object,
        RealtimeConfig=types.SimpleNamespace(Ignore=object()),
        InternalImpedance=lambda stiffness: ('internal_impedance', stiffness),
    )

    desk = types.ModuleType('positronic_franka.desk')
    desk.__dict__.update(Desk=object, SafetyControllerError=type('SafetyControllerError', (Exception,), {}))

    package = types.ModuleType('positronic_franka')
    package.__dict__.update(_franka=vendor, desk=desk)

    sys.modules.update({
        'positronic_franka': package,
        'positronic_franka._franka': vendor,
        'positronic_franka.desk': desk,
    })


if importlib.util.find_spec('positronic_franka') is None:
    _install_vendor_stub()
