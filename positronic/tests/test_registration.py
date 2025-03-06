import pytest
import numpy as np

import geom
from positronic.tools.registration import AbsoluteTrajectory, RelativeTrajectory, align_with_pca, pca_registration, register_trajectories


T = geom.Transform3D
R = geom.Rotation

# rotation around z-axis by 90 degrees clockwise
around_z_90 = R.from_rotation_matrix([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])

around_z_270 = R.from_rotation_matrix([
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 1]
])


@pytest.fixture
def absolute_trajectory():
    absolute_positions = [
        T(),
        T([1, 0, 0]),
        T([1, 0, 0], around_z_90),
        T([1, 1, 0], around_z_90),
        T([1, 1, 1], around_z_90),
        T([0, 1, 1], around_z_90),
    ]
    return AbsoluteTrajectory(absolute_positions)

@pytest.fixture
def absolute_trajectory_with_start_position():
    absolute_positions = [
        T(translation=np.array([1, 2, 3]), rotation=around_z_270),
        T(translation=np.array([1, 1, 3]), rotation=around_z_270),
        T(translation=np.array([1, 1, 3])),
        T(translation=np.array([2, 1, 3])),
        T(translation=np.array([2, 1, 4])),
        T(translation=np.array([2, 2, 4])),
    ]
    return AbsoluteTrajectory(absolute_positions)

@pytest.fixture
def relative_trajectory():
    relative_positions = [
        T([1, 0, 0]),
        T(rotation=around_z_90),
        T([1, 0, 0]),
        T([0, 0, 1]),
        T([0, 1, 0])
    ]
    return RelativeTrajectory(relative_positions)


@pytest.fixture
def start_position():
    """Fixture providing a start position for testing."""
    return geom.Transform3D(
        translation=np.array([1.0, 2.0, 3.0]),
        rotation=around_z_270
    )


def test_absolute_to_relative_conversion(absolute_trajectory, relative_trajectory):
    converted_relative_trajectory = absolute_trajectory.to_relative()

    assert len(converted_relative_trajectory) == len(relative_trajectory)

    for expected, actual in zip(relative_trajectory, converted_relative_trajectory):
        assert np.allclose(expected.as_matrix, actual.as_matrix)


def test_relative_to_absolute_conversion(absolute_trajectory, relative_trajectory):
    converted_absolute_trajectory = relative_trajectory.to_absolute(absolute_trajectory[0])

    assert len(converted_absolute_trajectory) == len(absolute_trajectory)

    for expected, actual in zip(absolute_trajectory, converted_absolute_trajectory):
        assert np.allclose(expected.as_matrix, actual.as_matrix)

def test_cycle_conversion(absolute_trajectory_with_start_position):
    converted_relative = absolute_trajectory_with_start_position.to_relative()
    converted_absolute = converted_relative.to_absolute(absolute_trajectory_with_start_position[0])

    assert len(converted_absolute) == len(absolute_trajectory_with_start_position)

    for expected, actual in zip(absolute_trajectory_with_start_position, converted_absolute):
        assert np.allclose(expected.as_matrix, actual.as_matrix)

def test_relative_to_absolute_conversion_with_start_position(
        absolute_trajectory_with_start_position,
        relative_trajectory,
        start_position
):
    converted_absolute = relative_trajectory.to_absolute(start_position)

    assert len(converted_absolute) == len(absolute_trajectory_with_start_position)

    for expected, actual in zip(absolute_trajectory_with_start_position, converted_absolute):
        assert np.allclose(expected.as_matrix, actual.as_matrix)


def test_self_distance_equals_zero(absolute_trajectory):
    distance = absolute_trajectory.distance(absolute_trajectory)

    assert np.allclose(distance, 0.0)



def test_registration(absolute_trajectory_with_start_position):
    target = T(
        translation=np.array([3, 1, 2]),
        rotation=R.from_rotvec(np.array([3, 4, 5]))
    )

    start = target * absolute_trajectory_with_start_position[0]
    source = absolute_trajectory_with_start_position.to_relative().to_absolute(start_position=start)
    transform = register_trajectories(source, absolute_trajectory_with_start_position)

    print(transform.as_matrix)
    print(target.as_matrix)

    np.testing.assert_allclose(transform.as_matrix, target.as_matrix, rtol=1e-6)



def test_registration_trajectories_match(absolute_trajectory_with_start_position):
    target = T(
        translation=np.array([31, 21, -9]),
        rotation=R.from_rotvec(np.array([11, 4, 5]))
    )

    start = target * absolute_trajectory_with_start_position[0]
    source = absolute_trajectory_with_start_position.to_relative().to_absolute(start_position=start)
    transform = register_trajectories(source, absolute_trajectory_with_start_position)

    start = transform.inv * source[0]
    restored = source.to_relative().to_absolute(start_position=start)

    for expected, actual in zip(absolute_trajectory_with_start_position, restored):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)

def test_pca_registration(absolute_trajectory_with_start_position):
    target = T(
        translation=np.array([31, 21, -9]),
        rotation=R.from_rotvec(np.array([11, 4, 5]))
    )

    start = target * absolute_trajectory_with_start_position[0]
    source = absolute_trajectory_with_start_position.to_relative().to_absolute(start_position=start)
    transform = pca_registration(source, absolute_trajectory_with_start_position)

    start = transform.inv * source[0]
    restored = source.to_relative().to_absolute(start_position=start)

    for expected, actual in zip(absolute_trajectory_with_start_position, restored):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)


def double_cycle_consistency(absolute_trajectory):
    restored = absolute_trajectory.to_relative().to_absolute().to_relative().to_absolute()

    for expected, actual in zip(absolute_trajectory, restored):
        np.testing.assert_allclose(expected.as_matrix, actual.as_matrix, atol=1e-6)



def test_registration_closest_positions(absolute_trajectory_with_start_position):
    target = T(
        translation=np.array([13, 17, -20]),
        rotation=R.from_rotvec(np.array([-1, 4, 5]))
    )

    start = target * absolute_trajectory_with_start_position[0]
    source = absolute_trajectory_with_start_position.to_relative().to_absolute(start_position=start)
    # duplicate each element in source
    s = []
    for p in source:
        s.append(p)
        s.append(p)
    source = AbsoluteTrajectory(s)

    transform = register_trajectories(source, absolute_trajectory_with_start_position)

    print(transform.as_matrix)
    print(target.as_matrix)

    np.testing.assert_allclose(transform.as_matrix, target.as_matrix, rtol=1e-6)


def test_align_with_pca(absolute_trajectory_with_start_position):
    offset = T(
        translation=np.array([13, 1, -20]),
        rotation=R.from_rotvec(np.array([3, 4, 5]))
    )
    t1 = absolute_trajectory_with_start_position
    t2 = absolute_trajectory_with_start_position.to_relative().to_absolute(start_position=offset * t1[0])

    aligned_1 = align_with_pca(t1)
    aligned_2 = align_with_pca(t2)

    for p1, p2 in zip(aligned_1, aligned_2):
        np.testing.assert_allclose(p1.as_matrix, p2.as_matrix, atol=1e-6)

def test_align_with_pca_sets_mean_to_zero(absolute_trajectory_with_start_position):
    aligned = align_with_pca(absolute_trajectory_with_start_position)

    np.testing.assert_allclose(aligned.mean_translation(), np.zeros(3), atol=1e-6)
