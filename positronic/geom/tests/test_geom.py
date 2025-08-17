import unittest
import numpy as np
from positronic.geom import Transform3D, Rotation, degrees_to_radians, radians_to_degrees


class TestTransform3D(unittest.TestCase):

    def test_as_matrix(self):
        translation = np.array([1, 2, 3])
        rotation = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        transform = Transform3D(translation, rotation)
        expected_matrix = np.array([[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]])
        np.testing.assert_array_almost_equal(transform.as_matrix, expected_matrix, decimal=4)

    def test_from_matrix(self):
        matrix = np.array([[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]])
        transform = Transform3D.from_matrix(matrix)
        np.testing.assert_array_almost_equal(transform.translation, [1, 2, 3])
        np.testing.assert_array_almost_equal(transform.rotation.as_quat, [0.7071, 0.7071, 0, 0], decimal=4)

    def test_mul(self):
        translation1 = np.array([1, 2, 3])
        rotation1 = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        transform1 = Transform3D(translation1, rotation1)

        translation2 = np.array([4, 5, 6])
        rotation2 = Rotation(0.7071, 0, 0.7071, 0)  # 90 degrees rotation around y-axis
        transform2 = Transform3D(translation2, rotation2)

        result = transform1 * transform2
        expected_translation = np.array([5, -4, 8])
        expected_rotation = Rotation(0.5, 0.5, 0.5, 0.5)  # Combined rotation
        np.testing.assert_array_almost_equal(result.translation, expected_translation, decimal=4)
        np.testing.assert_array_almost_equal(result.rotation.as_quat, expected_rotation.as_quat, decimal=4)

    def test_call(self):
        translation = np.array([1, 2, 3])
        rotation = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        transform = Transform3D(translation, rotation)
        vector = np.array([1, 1, 1])
        result = transform(vector)
        expected_result = np.array([2, 1, 4])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)


class TestRotation(unittest.TestCase):

    def test_mul(self):
        q1 = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        q2 = Rotation(0.7071, 0, 0.7071, 0)  # 90 degrees rotation around y-axis
        result = q1 * q2
        expected_result = Rotation(0.5, 0.5, 0.5, 0.5)  # Combined rotation
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_call(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        vector = np.array([1, 1, 1])
        result = q(vector)
        expected_result = np.array([1, -1, 1])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_inv(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        result = q.inv
        expected_result = Rotation(0.7071, -0.7071, 0, 0)
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_as_rotation_matrix(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        expected_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_array_almost_equal(q.as_rotation_matrix, expected_matrix, decimal=4)

    def test_from_rotation_matrix(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        q = Rotation.from_rotation_matrix(matrix)
        expected_result = Rotation(0.7071, 0.7071, 0, 0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_euler_to_quat(self):
        euler = np.array([np.pi / 2, 0, 0])  # 90 degrees rotation around x-axis
        q = Rotation.from_euler(euler)
        expected_result = Rotation(0.7071, 0.7071, 0, 0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_quat_to_euler(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        euler = q.as_euler
        expected_result = np.array([np.pi / 2, 0, 0])
        np.testing.assert_array_almost_equal(euler, expected_result, decimal=4)

    def test_from_rotvec_zero_rotation_returns_identity_rotation(self):
        rotvec = np.array([0.0, 0.0, 0.0])
        q = Rotation.from_rotvec(rotvec)
        expected_result = Rotation(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(q, expected_result)

    def test_from_rotvec_x_axis_90_degrees_returns_correct_rotation(self):
        rotvec = np.array([np.pi / 2, 0.0, 0.0])  # 90 degrees around x-axis
        q = Rotation.from_rotvec(rotvec)
        expected_result = Rotation(0.7071, 0.7071, 0.0, 0.0)
        np.testing.assert_array_almost_equal(q, expected_result, decimal=4)

    def test_as_rotvec_zero_rotation_returns_zero_vector(self):
        q = Rotation(1.0, 0.0, 0.0, 0.0)  # Identity rotation
        rotvec = q.as_rotvec
        expected_result = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rotvec, expected_result)

    def test_as_rotvec_x_axis_90_degrees_returns_correct_vector(self):
        q = Rotation(0.7071, 0.7071, 0.0, 0.0)  # 90 degrees around x-axis
        rotvec = q.as_rotvec
        expected_result = np.array([np.pi / 2, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rotvec, expected_result, decimal=4)

    def test_rotvec_conversion_cycle_consistency(self):
        original_rotvec = np.array([0.5, -0.3, 0.8])
        q = Rotation.from_rotvec(original_rotvec)
        recovered_rotvec = q.as_rotvec
        np.testing.assert_array_almost_equal(original_rotvec, recovered_rotvec, decimal=4)

    def test_to_representation_euler_same_as_as_euler(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        euler = q.as_euler
        np.testing.assert_array_almost_equal(euler, q.to(Rotation.Representation.EULER), decimal=4)

    def test_to_representation_quat_same_as_as_quat(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(q, q.to(Rotation.Representation.QUAT), decimal=4)

    def test_to_representation_rotation_matrix_same_as_as_rotation_matrix(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(q.as_rotation_matrix,
                                             q.to(Rotation.Representation.ROTATION_MATRIX),
                                             decimal=4)

    def test_to_representation_rotvec_same_as_as_rotvec(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(q.as_rotvec, q.to(Rotation.Representation.ROTVEC), decimal=4)

    def test_create_from_euler_same_as_from_euler(self):
        euler = np.array([np.pi / 2, 0, 0])  # 90 degrees rotation around x-axis
        q = Rotation.create_from(euler, Rotation.Representation.EULER)
        np.testing.assert_array_almost_equal(q, Rotation.from_euler(euler), decimal=4)

    def test_create_from_rotation_matrix_same_as_from_rotation_matrix(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        q = Rotation.create_from(matrix, Rotation.Representation.ROTATION_MATRIX)
        np.testing.assert_array_almost_equal(q, Rotation.from_rotation_matrix(matrix), decimal=4)

    def test_create_from_rotvec_same_as_from_rotvec(self):
        rotvec = np.array([0.5, -0.3, 0.8])
        q = Rotation.create_from(rotvec, Rotation.Representation.ROTVEC)
        np.testing.assert_array_almost_equal(q, Rotation.from_rotvec(rotvec), decimal=4)

    def test_create_from_quat_same_as_from_quat(self):
        q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(q, Rotation.create_from(q, Rotation.Representation.QUAT), decimal=4)

    def test_representation_size_exists_for_all_representations(self):
        for representation in Rotation.Representation:
            self.assertIsNotNone(representation.size)

    def test_create_from_representation_exists_for_all_representations(self):
        for representation in Rotation.Representation:
            data = np.zeros(representation.shape)
            self.assertIsNotNone(Rotation.create_from(data, representation))

    def test_from_value_exists_for_all_representations(self):
        for representation in Rotation.Representation:
            data = np.zeros(representation.shape)
            self.assertIsNotNone(representation.from_value(data))

    def test_from_value_same_as_create_from_representation(self):
        for representation in Rotation.Representation:
            data = np.zeros(representation.shape)
            np.testing.assert_array_almost_equal(representation.from_value(data),
                                                 Rotation.create_from(data, representation))

    def test_to_representation_exists_for_all_representations(self):
        for representation in Rotation.Representation:
            q = Rotation(0.7071, 0.7071, 0, 0)  # 90 degrees rotation around x-axis
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
