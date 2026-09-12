"""Stand-ins for the vendor packages the arm drivers import.

``positronic_franka`` builds against libfranka, ``scservo_sdk`` talks to a serial servo bus, ``placo``
solves kinematics, and ``trossen_arm`` talks to an arm controller; all four ship only in an extra, so the
driver modules cannot be imported from a default sync. Their Python-side logic needs no vendor behaviour,
so a stub carrying the names each module binds at import is enough — a test that needs a vendor to compute
something stands in for the class that wraps it instead. Installed here, before any test module imports a
driver, and only where the real package is absent.
"""

import importlib.util
import sys
import types
from enum import Enum

import pytest

import pimm

PACKAGE = 'positronic_franka'
VENDOR = f'{PACKAGE}._franka'
DESK = f'{PACKAGE}.desk'
TROSSEN = 'trossen_arm'


def _install_vendor_stub() -> None:
    """Bind the names ``positronic_franka`` gives the Franka driver. Reached as ``franka.pf.*``, never imported."""

    class GoalStatus(Enum):
        REACHED = 'reached'
        IN_FLIGHT = 'in_flight'
        ABORTED = 'aborted'

    class InternalImpedance:
        def __init__(self, k_theta=(3000.0, 3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0)):
            self.k_theta = list(k_theta)

    class SoftwareImpedance:
        def __init__(
            self,
            kq=(40.0, 30.0, 50.0, 25.0, 35.0, 25.0, 10.0),
            kqd=(4.0, 6.0, 5.0, 5.0, 3.0, 2.0, 1.0),
            kx=(750.0, 750.0, 750.0, 15.0, 15.0, 15.0),
            kxd=(37.0, 37.0, 37.0, 2.0, 2.0, 2.0),
        ):
            self.kq, self.kqd, self.kx, self.kxd = list(kq), list(kqd), list(kx), list(kxd)

    vendor = types.ModuleType(VENDOR)
    vendor.__dict__.update(
        GoalStatus=GoalStatus,
        State=object,
        Robot=object,
        RealtimeConfig=types.SimpleNamespace(Ignore=object()),
        InternalImpedance=InternalImpedance,
        SoftwareImpedance=SoftwareImpedance,
    )

    desk = types.ModuleType(DESK)
    desk.__dict__.update(Desk=object, SafetyControllerError=type('SafetyControllerError', (Exception,), {}))

    package = types.ModuleType(PACKAGE)
    package.__dict__.update(_franka=vendor, desk=desk)

    sys.modules.update({PACKAGE: package, VENDOR: vendor, DESK: desk})


def _install_trossen_stub() -> None:
    """Bind the names ``trossen_arm`` gives the Trossen driver, which reaches them as ``trossen_arm.*``."""

    class Mode(Enum):
        idle = 'idle'
        position = 'position'
        external_effort = 'external_effort'

    class Model(Enum):
        wxai_v0 = 'wxai_v0'

    module = types.ModuleType(TROSSEN)
    module.__dict__.update(
        Mode=Mode,
        Model=Model,
        StandardEndEffector=types.SimpleNamespace(wxai_v0_follower=object(), wxai_v0_leader=object()),
        TrossenArmDriver=object,
        RuntimeError=type('RuntimeError', (RuntimeError,), {}),
    )
    sys.modules[TROSSEN] = module


# Both are reached for only inside the functions that use them, so an empty module carries the import
_EMPTY_STUBS = ('scservo_sdk', 'placo')

if importlib.util.find_spec(PACKAGE) is None:
    _install_vendor_stub()

if importlib.util.find_spec(TROSSEN) is None:
    _install_trossen_stub()

for _name in _EMPTY_STUBS:
    if importlib.util.find_spec(_name) is None:
        sys.modules[_name] = types.ModuleType(_name)


@pytest.fixture
def world():
    with pimm.World() as w:
        yield w
