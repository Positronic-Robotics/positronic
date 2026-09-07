import configuronic as cfn

from positronic import keys
from positronic.cfg.hardware import camera


def _serials(cameras: dict) -> dict[str, int]:
    return {name: cfg.kwargs['serial_number'] for name, cfg in cameras.items()}


def test_each_sided_droid_carries_the_same_cameras_as_the_unsided_one():
    assert set(camera.droid_left) == set(camera.droid) == {keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE}
    assert set(camera.droid_right) == set(camera.droid)


def test_the_two_sides_bind_different_sideviews_and_share_the_wrist():
    left, right = _serials(camera.droid_left), _serials(camera.droid_right)

    assert left[keys.EXTERIOR_IMAGE] != right[keys.EXTERIOR_IMAGE]
    assert left[keys.WRIST_IMAGE] == right[keys.WRIST_IMAGE]


def test_each_side_binds_the_sideview_the_station_declares_for_it():
    assert _serials(camera.droid_left)[keys.EXTERIOR_IMAGE] == camera.sideview_left.kwargs['serial_number']
    assert _serials(camera.droid_right)[keys.EXTERIOR_IMAGE] == camera.sideview_right.kwargs['serial_number']


def test_the_launcher_reference_binds_the_side_without_disturbing_the_rest():
    """An `@` reference resolves to a plain dict, and overriding `cameras` alone leaves a sibling
    override on the same config standing."""

    @cfn.config()
    def arm(brake_after_idle_s: float = 0.0):
        return brake_after_idle_s

    @cfn.config(robot_arm=arm, cameras=camera.droid)
    def embodiment(robot_arm, cameras):
        return robot_arm, cameras

    # Written out rather than derived from `camera.__name__`: the dotted path is this module's
    # public name, so a derived reference would follow a rename instead of failing on one.
    right_sideview_ref = '@positronic.cfg.hardware.camera.droid_right'

    pinned = embodiment.override(robot_arm=arm.override(brake_after_idle_s=600.0))
    bound = cfn.Config(lambda e: e, e=pinned).override(**{'e.cameras': right_sideview_ref}).kwargs['e']

    assert _serials(bound.kwargs['cameras']) == _serials(camera.droid_right)
    assert bound.kwargs['robot_arm'].kwargs == {'brake_after_idle_s': 600.0}
    assert _serials(camera.droid) != _serials(camera.droid_right)  # the module dict is not mutated
