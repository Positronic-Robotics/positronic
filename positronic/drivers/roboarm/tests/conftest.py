"""Stand-ins for the vendor packages the arm drivers import.

``positronic_franka`` builds against libfranka, ``scservo_sdk`` talks to a serial servo bus, and ``placo``
solves kinematics; all three ship only in the ``hardware`` extra, so the driver modules cannot be imported
from a default sync. Their Python-side logic needs no vendor behaviour, so a stub carrying the names each
module binds at import is enough — a test that needs a vendor to compute something stands in for the class
that wraps it instead. Installed here, before any test module imports a driver, and only where the real
package is absent.
"""

import importlib.util
import sys
import types
from enum import Enum

import pytest

import pimm


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


# Both are reached for only inside the functions that use them, so an empty module carries the import
_EMPTY_STUBS = ('scservo_sdk', 'placo')

if importlib.util.find_spec(PACKAGE) is None:
    _install_vendor_stub()

for _name in _EMPTY_STUBS:
    if importlib.util.find_spec(_name) is None:
        sys.modules[_name] = types.ModuleType(_name)


@pytest.fixture
def world():
    with pimm.World() as w:
        yield w
