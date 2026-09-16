import configuronic as cfn

from positronic import keys
from positronic.cfg.hardware import camera


def _serials(cameras: dict) -> dict[str, int]:
    return {name: cfg.kwargs['serial_number'] for name, cfg in cameras.items()}


def test_each_sided_droid_declares_the_three_camera_observation_keys():
    """A sided set binds three cameras, so one run serves a policy that reads the second exterior
    beside one that does not."""
    three = {keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2}

    assert set(camera.droid_left) == set(camera.droid_3cam) == three
    assert set(camera.droid_right) == three


def test_the_two_sides_bind_different_sideviews_and_share_the_wrist():
    left, right = _serials(camera.droid_left), _serials(camera.droid_right)

    assert left[keys.EXTERIOR_IMAGE] != right[keys.EXTERIOR_IMAGE]
    assert left[keys.WRIST_IMAGE] == right[keys.WRIST_IMAGE]


def test_each_side_takes_its_own_sideview_first_and_the_other_second():
    """The exchange the sided sets make: a side's own sideview is `exterior` and the other one is
    `exterior_2`, so the pair swaps keys between the two sides."""
    left, right = _serials(camera.droid_left), _serials(camera.droid_right)
    declared_left = camera.sideview_left.kwargs['serial_number']
    declared_right = camera.sideview_right.kwargs['serial_number']

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

    # Written out rather than derived from `camera.__name__`: the dotted path is this module's
    # public name, so a derived reference would follow a rename instead of failing on one.
    right_sideview_ref = '@positronic.cfg.hardware.camera.droid_right'

    pinned = embodiment.override(robot_arm=arm.override(brake_after_idle_s=600.0))
    bound = cfn.Config(lambda e: e, e=pinned).override(**{'e.cameras': right_sideview_ref}).kwargs['e']

    assert _serials(bound.kwargs['cameras']) == _serials(camera.droid_right)
    assert bound.kwargs['robot_arm'].kwargs == {'brake_after_idle_s': 600.0}
    assert _serials(camera.droid) != _serials(camera.droid_right)  # the module dict is not mutated
