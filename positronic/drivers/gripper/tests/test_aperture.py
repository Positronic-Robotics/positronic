import pytest

from positronic.drivers.gripper import aperture


@pytest.mark.parametrize(('grip', 'opening'), [(0.0, 1.0), (1.0, 0.0), (0.25, 0.75)])
def test_flip_grip_maps_a_grip_to_its_aperture_and_back(grip, opening):
    assert aperture.flip_grip(grip) == pytest.approx(opening)
    assert aperture.flip_grip(opening) == pytest.approx(grip)


@pytest.mark.parametrize(('value', 'flipped'), [(-0.2, 1.0), (1.3, 0.0)])
def test_flip_grip_clips_a_reading_outside_the_unit_range(value, flipped):
    assert aperture.flip_grip(value) == flipped
