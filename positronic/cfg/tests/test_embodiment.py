import pytest

from positronic import keys
from positronic.cfg import embodiment
from positronic.cfg.hardware import camera, roboarm


def test_each_fake_droid_is_built_by_the_droid_factory_over_the_same_cameras():
    """The factory declares the observation names and serializers, so one that shares it cannot drift."""
    assert embodiment.droid_fake.target is embodiment.droid.target
    assert embodiment.droid_3cam_fake.target is embodiment.droid_3cam.target
    assert list(camera.droid_fake) == list(camera.droid)
    assert list(camera.droid_3cam_fake) == list(camera.droid_3cam)


def test_the_fake_droid_builds_without_the_vendor_packages():
    built = embodiment.droid_fake.instantiate()

    assert set(built.observations) == {keys.ROBOT_STATE, keys.GRIP, *camera.droid}


def test_the_fake_droid_declares_what_the_real_droid_declares():
    for vendor in ('positronic_franka', 'pymodbus', 'pyzed'):
        pytest.importorskip(vendor)
    real = embodiment.droid.override(robot_arm=roboarm.franka_droid.override(manage_desk=False)).instantiate()
    fake = embodiment.droid_fake.instantiate()

    assert {name: obs.serializer for name, obs in fake.observations.items()} == {
        name: obs.serializer for name, obs in real.observations.items()
    }
