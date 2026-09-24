import numpy as np

import pimm
from positronic.drivers.roboarm import yam

STATION_GRAVITY_COMP = [1.0, 1.1, 1.4, 1.4, 1.0, 1.0]


def _opened_with(**kwargs) -> dict:
    """Start a ``Robot`` built with ``kwargs`` and return what it asked its vendor factory for."""
    seen = {}

    def connect(channel, sim, gravity_comp_factor):
        seen.update(channel=channel, sim=sim, gravity_comp_factor=gravity_comp_factor)
        return yam._FakeYam()

    robot = yam.Robot('can0', connect=connect, **kwargs)
    with pimm.World() as world:
        loop = world.start([robot])
        next(loop)  # the chain is opened before the driver yields for the first time
    return seen


def test_a_station_hands_its_gravity_compensation_to_the_chain():
    """i2rt holds a joint against a gravity model of its own, and a joint that model reads short settles below
    where it is sent. The factors a station measured are no use to it unless the driver passes them on."""
    passed = _opened_with(gravity_comp_factor=STATION_GRAVITY_COMP)['gravity_comp_factor']
    np.testing.assert_array_equal(passed, STATION_GRAVITY_COMP)


def test_a_station_that_measured_none_leaves_the_vendor_its_own():
    """Every YAM shares i2rt's factors until a station measures better ones; naming none has to mean that,
    rather than a vector of ones that would turn the compensation off."""
    assert _opened_with()['gravity_comp_factor'] is None
