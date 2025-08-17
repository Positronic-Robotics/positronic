import pytest
import numpy as np

from positronic import geom
from positronic.geom.trajectory import AbsoluteTrajectory, RelativeTrajectory


# rotation around z-axis by 90 degrees clockwise
around_z_90 = geom.Rotation.from_rotation_matrix([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

around_z_270 = geom.Rotation.from_rotation_matrix([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
])


@pytest.fixture
def absolute_trajectory() -> AbsoluteTrajectory:
    absolute_positions = [
        geom.Transform3D.identity,
        geom.Transform3D(translation=np.array([1, 0, 0])),
        geom.Transform3D(translation=np.array([1, 0, 0]), rotation=around_z_90),
        geom.Transform3D(translation=np.array([1, 1, 0]), rotation=around_z_90),
        geom.Transform3D(translation=np.array([1, 1, 1]), rotation=around_z_90),
        geom.Transform3D(translation=np.array([0, 1, 1]), rotation=around_z_90),
    ]
    return AbsoluteTrajectory(absolute_positions)


@pytest.fixture
def absolute_trajectory_with_start_position() -> AbsoluteTrajectory:
    absolute_positions = [
        geom.Transform3D(translation=np.array([1, 2, 3]), rotation=around_z_270),
        geom.Transform3D(translation=np.array([1, 1, 3]), rotation=around_z_270),
        geom.Transform3D(translation=np.array([1, 1, 3])),
        geom.Transform3D(translation=np.array([2, 1, 3])),
        geom.Transform3D(translation=np.array([2, 1, 4])),
        geom.Transform3D(translation=np.array([2, 2, 4])),
    ]
    return AbsoluteTrajectory(absolute_positions)


@pytest.fixture
def relative_trajectory() -> RelativeTrajectory:
    relative_positions = [
        geom.Transform3D(translation=np.array([1, 0, 0])),
        geom.Transform3D(rotation=around_z_90),
        geom.Transform3D(translation=np.array([1, 0, 0])),
        geom.Transform3D(translation=np.array([0, 0, 1])),
        geom.Transform3D(translation=np.array([0, 1, 0])),
    ]
    return RelativeTrajectory(relative_positions)


@pytest.fixture
def start_position():
    """Fixture providing a start position for testing."""
    return geom.Transform3D(
        translation=np.array([1.0, 2.0, 3.0]),
        rotation=around_z_270
    )


def test_absolute_to_relative_conversion(
        absolute_trajectory: AbsoluteTrajectory,
        relative_trajectory: RelativeTrajectory,
):
    converted_relative_trajectory = absolute_trajectory.to_relative()

    assert len(converted_relative_trajectory) == len(relative_trajectory)

    for expected, actual in zip(relative_trajectory, converted_relative_trajectory):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)


def test_relative_to_absolute_conversion(
        absolute_trajectory: AbsoluteTrajectory,
        relative_trajectory: RelativeTrajectory,
):
    converted_absolute_trajectory = relative_trajectory.to_absolute(absolute_trajectory[0])

    assert len(converted_absolute_trajectory) == len(absolute_trajectory)

    for expected, actual in zip(absolute_trajectory, converted_absolute_trajectory):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)


def test_cycle_conversion(absolute_trajectory_with_start_position: AbsoluteTrajectory):
    converted_relative = absolute_trajectory_with_start_position.to_relative()
    converted_absolute = converted_relative.to_absolute(absolute_trajectory_with_start_position[0])

    assert len(converted_absolute) == len(absolute_trajectory_with_start_position)

    for expected, actual in zip(absolute_trajectory_with_start_position, converted_absolute):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)


def test_relative_to_absolute_conversion_with_start_position(
        absolute_trajectory_with_start_position: AbsoluteTrajectory,
        relative_trajectory: RelativeTrajectory,
        start_position: geom.Transform3D,
):
    converted_absolute = relative_trajectory.to_absolute(start_position)

    assert len(converted_absolute) == len(absolute_trajectory_with_start_position)

    for expected, actual in zip(absolute_trajectory_with_start_position, converted_absolute):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)


def test_cycle_consistency_twice_produces_original_trajectory(absolute_trajectory: AbsoluteTrajectory):
    restored_1_cycle = absolute_trajectory.to_relative().to_absolute(geom.Transform3D.identity)
    restored_2_cycles = restored_1_cycle.to_relative().to_absolute(geom.Transform3D.identity)

    for expected, actual in zip(absolute_trajectory, restored_2_cycles):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)
