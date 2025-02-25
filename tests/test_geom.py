import unittest
import numpy as np
from geom import (
    Transform3D,
    Quaternion,
    degrees_to_radians,
    radians_to_degrees,
    RotationRepresentation,
)


class TestTransform3D(unittest.TestCase):

    def test_as_matrix(self):
        translation = np.array([1, 2, 3])
        quaternion = Quaternion(
            0.7071, 0.7071, 0, 0
        )  # 90 degrees rotation around x-axis
        transform = Transform3D(translation, quaternion)
        expected_matrix = np.array(
            [[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]]
        )
        np.testing.assert_array_almost_equal(
            transform.as_matrix, expected_matrix, decimal=4
        )

    def test_from_matrix(self):
        matrix = np.array([[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]])
        transform = Transform3D.from_matrix(matrix)
        np.testing.assert_array_almost_equal(transform.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(
            transform.quaternion, [0.7071, 0.7071, 0, 0], decimal=4
        )

    def test_mul(self):
        translation1 = np.array([1, 2, 3])
        quaternion1 = Quaternion(
            0.7071, 0.7071, 0, 0
        )  # 90 degrees rotation around x-axis
        transform1 = Transform3D(translation1, quaternion1)

        translation2 = np.array([4, 5, 6])
        quaternion2 = Quaternion(
            0.7071, 0, 0.7071, 0
        )  # 90 degrees rotation around y-axis
        transform2 = Transform3D(translation2, quaternion2)

        result = transform1 * transform2
        expected_translation = np.array([5, -4, 8])
        expected_quaternion = Quaternion(0.5, 0.5, 0.5, 0.5)  # Combined rotation
        np.testing.assert_array_almost_equal(
            result.translation, expected_translation, decimal=4
        )
        np.testing.assert_array_almost_equal(
            result.quaternion, expected_quaternion, decimal=4
        )

    def test_call(self):
        translation = np.array([1, 2, 3])
        quaternion = Quaternion(
            0.7071, 0.7071, 0, 0
        )  # 90 degrees rotation around x-axis
        transform = Transform3D(translation, quaternion)
        vector = np.array([1, 1, 1])
        result = transform(vector)
        expected_result = np.array([2, 1, 4])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)


class TestQuaternion(unittest.TestCase):

    def test_mul(self):
        q1 = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        q2 = Quaternion(0.7071, 0, 0.7071, 0)  # 90 degrees rotation around y-axis
        result = q1 * q2
        expected_result = Quaternion(0.5, 0.5, 0.5, 0.5)  # Combined rotation
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_call(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        vector = np.array([1, 1, 1])
        result = q(vector)
        expected_result = np.array([1, -1, 1])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_inv(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        result = q.inv
        expected_result = Quaternion(0.7071, -0.7071, 0, 0)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_as_rotation_matrix(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        expected_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_array_almost_equal(
            q.as_rotation_matrix, expected_matrix, decimal=4
        )

    def test_from_rotation_matrix(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        q = Quaternion.from_rotation_matrix(matrix)
        expected_result = Quaternion(0.7071, 0.7071, 0, 0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_euler_to_quat(self):
        euler = np.array([np.pi / 2, 0, 0])  # 90 degrees rotation around x-axis
        q = Quaternion.from_euler(euler)
        expected_result = Quaternion(0.7071, 0.7071, 0, 0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_quat_to_euler(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        euler = q.as_euler
        expected_result = np.array([np.pi / 2, 0, 0])
        np.testing.assert_array_almost_equal(euler, expected_result, decimal=4)

    def test_from_rotvec_zero_rotation_returns_identity_quaternion(self):
        rotvec = np.array([0.0, 0.0, 0.0])
        q = Quaternion.from_rotvec(rotvec)
        expected_result = Quaternion(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(q, expected_result)

    def test_from_rotvec_x_axis_90_degrees_returns_correct_quaternion(self):
        rotvec = np.array([np.pi / 2, 0.0, 0.0])  # 90 degrees around x-axis
        q = Quaternion.from_rotvec(rotvec)
        expected_result = Quaternion(0.7071, 0.7071, 0.0, 0.0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_as_rotvec_zero_quaternion_returns_zero_vector(self):
        q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        rotvec = q.as_rotvec
        expected_result = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rotvec, expected_result)

    def test_as_rotvec_x_axis_90_degrees_returns_correct_vector(self):
        q = Quaternion(0.7071, 0.7071, 0.0, 0.0)  # 90 degrees around x-axis
        rotvec = q.as_rotvec
        expected_result = np.array([np.pi / 2, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rotvec, expected_result, decimal=4)

    def test_rotvec_conversion_cycle_consistency(self):
        original_rotvec = np.array([0.5, -0.3, 0.8])
        q = Quaternion.from_rotvec(original_rotvec)
        recovered_rotvec = q.as_rotvec
        np.testing.assert_array_almost_equal(
            original_rotvec, recovered_rotvec, decimal=4
        )

    def test_to_representation_euler_same_as_as_euler(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        euler = q.as_euler
        np.testing.assert_array_almost_equal(
            euler, q.to(RotationRepresentation.EULER), decimal=4
        )

    def test_to_representation_quat_same_as_as_quat(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(
            q, q.to(RotationRepresentation.QUAT), decimal=4
        )

    def test_to_representation_rotation_matrix_same_as_as_rotation_matrix(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(
            q.as_rotation_matrix,
            q.to(RotationRepresentation.ROTATION_MATRIX),
            decimal=4,
        )

    def test_to_representation_rotvec_same_as_as_rotvec(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(
            q.as_rotvec, q.to(RotationRepresentation.ROTVEC), decimal=4
        )

    def test_create_from_euler_same_as_from_euler(self):
        euler = np.array([np.pi / 2, 0, 0])  # 90 degrees rotation around x-axis
        q = Quaternion.create_from(euler, RotationRepresentation.EULER)
        np.testing.assert_array_almost_equal(q, Quaternion.from_euler(euler), decimal=4)

    def test_create_from_rotation_matrix_same_as_from_rotation_matrix(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        q = Quaternion.create_from(matrix, RotationRepresentation.ROTATION_MATRIX)
        np.testing.assert_array_almost_equal(
            q, Quaternion.from_rotation_matrix(matrix), decimal=4
        )

    def test_create_from_rotvec_same_as_from_rotvec(self):
        rotvec = np.array([0.5, -0.3, 0.8])
        q = Quaternion.create_from(rotvec, RotationRepresentation.ROTVEC)
        np.testing.assert_array_almost_equal(
            q, Quaternion.from_rotvec(rotvec), decimal=4
        )

    def test_create_from_quat_same_as_from_quat(self):
        q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(
            q, Quaternion.create_from(q, RotationRepresentation.QUAT), decimal=4
        )

    def test_representation_size_exists_for_all_representations(self):
        for representation in RotationRepresentation:
            self.assertIsNotNone(representation.size)

    def test_create_from_representation_exists_for_all_representations(self):
        for representation in RotationRepresentation:
            data = np.zeros(representation.shape)
            self.assertIsNotNone(Quaternion.create_from(data, representation))

    def test_from_value_exists_for_all_representations(self):
        for representation in RotationRepresentation:
            data = np.zeros(representation.shape)
            self.assertIsNotNone(representation.from_value(data))

    def test_from_value_same_as_create_from_representation(self):
        for representation in RotationRepresentation:
            data = np.zeros(representation.shape)
            np.testing.assert_array_almost_equal(
                representation.from_value(data),
                Quaternion.create_from(data, representation),
            )

    def test_to_representation_exists_for_all_representations(self):
        for representation in RotationRepresentation:
            q = Quaternion(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
            self.assertIsNotNone(q.to(representation))


class TestUtils(unittest.TestCase):

    def test_degrees_to_radians(self):
        degrees = 180
        radians = degrees_to_radians(degrees)
        self.assertAlmostEqual(radians, np.pi)

    def test_radians_to_degrees(self):
        radians = np.pi
        degrees = radians_to_degrees(radians)
        self.assertAlmostEqual(degrees, 180)


if __name__ == '__main__':
    unittest.main()
