"""Stand-ins for the vendor packages the arm drivers import.

``positronic_franka`` builds against libfranka, ``scservo_sdk`` talks to a serial servo bus, ``placo``
solves kinematics and ``i2rt`` opens a YAM CAN chain; all four ship only in an extra, so the driver modules
cannot be imported from a default sync. Their Python-side logic needs no vendor behaviour, so a stub carrying
the names each module binds at import is enough — a test that needs a vendor to compute something stands in
for the class that wraps it instead. Installed here, before any test module imports a driver, and only where the real
package is absent.
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
I2RT = 'i2rt'


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
        Goal=object,
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


def _install_i2rt_stub() -> None:
    """Bind the names the YAM driver imports. A test hands the driver a chain of its own."""

    def get_yam_robot(*_args, **_kwargs):
        raise RuntimeError('no YAM chain here; a test replaces this factory')

    def motor_interface(*_args, **_kwargs):
        raise RuntimeError('no CAN interface here; a test replaces this class')

    get_robot = types.ModuleType(f'{I2RT}.robots.get_robot')
    get_robot.__dict__.update(get_yam_robot=get_yam_robot)
    utils = types.ModuleType(f'{I2RT}.robots.utils')
    utils.__dict__.update(GripperType=Enum('GripperType', ['LINEAR_4310']))
    robots = types.ModuleType(f'{I2RT}.robots')
    robots.__dict__.update(get_robot=get_robot, utils=utils)
    dm_driver = types.ModuleType(f'{I2RT}.motor_drivers.dm_driver')
    dm_driver.__dict__.update(DMSingleMotorCanInterface=motor_interface)
    motor_drivers = types.ModuleType(f'{I2RT}.motor_drivers')
    motor_drivers.__dict__.update(dm_driver=dm_driver)
    package = types.ModuleType(I2RT)
    package.__dict__.update(robots=robots, motor_drivers=motor_drivers)

    stubs = (package, robots, get_robot, utils, motor_drivers, dm_driver)
    sys.modules.update({stub.__name__: stub for stub in stubs})


# Both are reached for only inside the functions that use them, so an empty module carries the import
_EMPTY_STUBS = ('scservo_sdk', 'placo')

if importlib.util.find_spec(PACKAGE) is None:
    _install_vendor_stub()

if importlib.util.find_spec(I2RT) is None:
    _install_i2rt_stub()

for _name in _EMPTY_STUBS:
    if importlib.util.find_spec(_name) is None:
        sys.modules[_name] = types.ModuleType(_name)


@pytest.fixture
def world():
    with pimm.World() as w:
        yield w
