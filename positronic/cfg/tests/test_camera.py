import sys
import types

import configuronic as cfn
import pytest

from positronic import keys
from positronic.cfg.hardware import camera


def _serial_settings(cameras: dict) -> dict[str, str]:
    return {name: cfg.kwargs['serial_env'] for name, cfg in cameras.items()}


@pytest.mark.parametrize('value', [None, '', 'left-zed'], ids=['unset', 'blank', 'not-a-number'])
def test_a_camera_whose_serial_setting_is_not_a_number_refuses_and_names_the_setting(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(camera.FRANKA_LEFT_ZED_SERIAL_ENV, raising=False)
    else:
        monkeypatch.setenv(camera.FRANKA_LEFT_ZED_SERIAL_ENV, value)

    with pytest.raises(RuntimeError, match=f'Set {camera.FRANKA_LEFT_ZED_SERIAL_ENV} in the environment'):
        camera.sideview_left.instantiate()


def test_each_franka_camera_opens_the_serial_its_setting_names(monkeypatch):
    monkeypatch.setitem(sys.modules, 'positronic.drivers.camera.zed', types.SimpleNamespace(SLCamera=dict))
    serials = {
        camera.FRANKA_WRIST_ZED_SERIAL_ENV: 11,
        camera.FRANKA_LEFT_ZED_SERIAL_ENV: 22,
        camera.FRANKA_RIGHT_ZED_SERIAL_ENV: 33,
    }
    for env, serial in serials.items():
        monkeypatch.setenv(env, str(serial))

    for cfg in camera.droid_3cam.values():
        assert cfg.instantiate()['serial_number'] == serials[cfg.kwargs['serial_env']]


def test_each_sided_droid_declares_the_three_camera_observation_keys():
    three = {keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2}

    assert set(camera.droid_left) == set(camera.droid_3cam) == three
    assert set(camera.droid_right) == three


def test_the_two_sides_bind_different_sideviews_and_share_the_wrist():
    left, right = _serial_settings(camera.droid_left), _serial_settings(camera.droid_right)

    assert left[keys.EXTERIOR_IMAGE] != right[keys.EXTERIOR_IMAGE]
    assert left[keys.WRIST_IMAGE] == right[keys.WRIST_IMAGE]


def test_each_side_takes_its_own_sideview_first_and_the_other_second():
    """`exterior` holds the named side and `exterior_2` the other, so the two dicts hold them
    swapped."""
    left, right = _serial_settings(camera.droid_left), _serial_settings(camera.droid_right)
    declared_left = camera.sideview_left.kwargs['serial_env']
    declared_right = camera.sideview_right.kwargs['serial_env']

    assert (left[keys.EXTERIOR_IMAGE], left[keys.EXTERIOR_IMAGE_2]) == (declared_left, declared_right)
    assert (right[keys.EXTERIOR_IMAGE], right[keys.EXTERIOR_IMAGE_2]) == (declared_right, declared_left)


def test_an_at_reference_overrides_cameras_and_leaves_a_sibling_override_standing():
    """An `@` reference resolves to a plain dict, and overriding `cameras` alone leaves a sibling
    override on the same config standing."""

    @cfn.config()
    def arm(brake_after_idle_s: float = 0.0):
        return brake_after_idle_s

    @cfn.config(robot_arm=arm, cameras=camera.droid)
    def embodiment(robot_arm, cameras):
        return robot_arm, cameras

    right_sideview_ref = '@positronic.cfg.hardware.camera.droid_right'

    pinned = embodiment.override(robot_arm=arm.override(brake_after_idle_s=600.0))
    bound = cfn.Config(lambda e: e, e=pinned).override(**{'e.cameras': right_sideview_ref}).kwargs['e']

    assert _serial_settings(bound.kwargs['cameras']) == _serial_settings(camera.droid_right)
    assert bound.kwargs['robot_arm'].kwargs == {'brake_after_idle_s': 600.0}
    assert _serial_settings(camera.droid) != _serial_settings(camera.droid_right)  # the module dict is not mutated
