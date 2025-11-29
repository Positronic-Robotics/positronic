import unittest

import numpy as np

from positronic.geom import Rotation, Transform3D, degrees_to_radians, radians_to_degrees


class TestTransform3D(unittest.TestCase):
    def test_as_matrix(self):
        translation = np.array([1, 2, 3])
        rotation = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
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
        rotation1 = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        transform1 = Transform3D(translation1, rotation1)

        translation2 = np.array([4, 5, 6])
        rotation2 = Rotation.from_quat([0.7071, 0, 0.7071, 0])  # 90 degrees rotation around y-axis
        transform2 = Transform3D(translation2, rotation2)

        result = transform1 * transform2
        expected_translation = np.array([5, -4, 8])
        expected_rotation = Rotation.from_quat([0.5, 0.5, 0.5, 0.5])  # Combined rotation
        np.testing.assert_array_almost_equal(result.translation, expected_translation, decimal=4)
        np.testing.assert_array_almost_equal(result.rotation.as_quat, expected_rotation.as_quat, decimal=4)

    def test_call(self):
        translation = np.array([1, 2, 3])
        rotation = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        transform = Transform3D(translation, rotation)
        vector = np.array([1, 1, 1])
        result = transform(vector)
        expected_result = np.array([2, 1, 4])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_inv_returns_inverse_transform(self):
        translation = np.array([0.5, -1.2, 2.3])
        rotation = Rotation.from_quat([0.7071, 0, 0.7071, 0])  # 90 degrees rotation around y-axis
        transform = Transform3D(translation, rotation)

        composed = transform * transform.inv

        np.testing.assert_allclose(composed.translation, Transform3D.identity.translation, atol=1e-4, rtol=0)
        np.testing.assert_allclose(composed.rotation.as_quat, Transform3D.identity.rotation.as_quat, atol=1e-4, rtol=0)

    def test_identity_is_zero_translation_identity_rotation(self):
        identity = Transform3D.identity

        np.testing.assert_array_almost_equal(identity.translation, np.zeros(3))
        np.testing.assert_array_almost_equal(identity.rotation.as_quat, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_copy_creates_independent_transform(self):
        translation = np.array([1.0, 2.0, 3.0])
        rotation = Rotation.from_quat([0.9238795, 0.3826834, 0.0, 0.0])  # 45 degrees around x-axis
        transform = Transform3D(translation, rotation)
        expected_translation = np.array([1.0, 2.0, 3.0])
        expected_quat = np.array([0.9238795, 0.3826834, 0.0, 0.0])

        copied = transform.copy()

        self.assertIsNot(transform.translation, copied.translation)
        self.assertIsNot(transform.rotation, copied.rotation)

        transform.translation[0] += 10.0
        transform.translation[0] += 10.0
        # Rotation is no longer mutable via array access, but we can check if it's a different object
        # or if we can modify it via internal methods (which we shouldn't).
        # The test originally did: transform.rotation[1] += 0.1
        # Since Rotation is now immutable-ish (or at least not an array), we can't do that easily.
        # We can replace the rotation object.
        new_quat = transform.rotation.as_quat
        new_quat[1] += 0.1
        transform.rotation = Rotation.from_quat(new_quat)

        np.testing.assert_array_almost_equal(copied.translation, expected_translation)
        np.testing.assert_array_almost_equal(copied.rotation.as_quat, expected_quat, decimal=6)

    def test_from_matrix_with_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            Transform3D.from_matrix(np.eye(3))

    def test_from_matrix_with_invalid_last_row_raises(self):
        matrix = np.eye(4)
        matrix[3, 3] = 2

        with self.assertRaises(ValueError):
            Transform3D.from_matrix(matrix)

    def test_as_vector_returns_translation_followed_by_rotation_representation(self):
        translation = np.array([0.4, -0.2, 1.8])
        rotation = Rotation.from_quat([0.9238795, 0.0, 0.3826834, 0.0])  # 45 degrees around y-axis
        transform = Transform3D(translation, rotation)

        vec = transform.as_vector(Rotation.Representation.QUAT)

        np.testing.assert_array_equal(vec[:3], translation)
        np.testing.assert_array_almost_equal(vec[3:], rotation.as_quat)

    def test_as_vector_with_rotation_matrix_representation(self):
        translation = np.array([0.1, 0.2, -0.3])
        rotation = Rotation.from_quat([0.7071068, 0.0, 0.7071068, 0.0])  # 90 degrees around y-axis
        transform = Transform3D(translation, rotation)

        vec = transform.as_vector(Rotation.Representation.ROTATION_MATRIX)

        self.assertEqual(vec.shape, (12,))
        np.testing.assert_array_equal(vec[:3], translation)
        np.testing.assert_array_almost_equal(vec[3:].reshape(3, 3), rotation.as_rotation_matrix)
        reconstructed = Transform3D.from_vector(vec, Rotation.Representation.ROTATION_MATRIX)
        np.testing.assert_array_almost_equal(reconstructed.translation, translation)
        np.testing.assert_array_almost_equal(reconstructed.rotation.as_quat, rotation.as_quat)

    def test_from_vector_recreates_transform_for_rotation_matrix_representation(self):
        translation = np.array([-0.5, 0.8, 0.25])
        rotation = Rotation.from_quat([0.7071068, 0.0, 0.7071068, 0.0])  # 90 degrees around y-axis
        transform = Transform3D(translation, rotation)

        vec = transform.as_vector(Rotation.Representation.ROTATION_MATRIX)
        reconstructed = Transform3D.from_vector(vec, Rotation.Representation.ROTATION_MATRIX)

        np.testing.assert_array_almost_equal(reconstructed.translation, translation)
        np.testing.assert_array_almost_equal(reconstructed.rotation.as_quat, rotation.as_quat)


class TestRotation(unittest.TestCase):
    def test_mul(self):
        q1 = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        q2 = Rotation.from_quat([0.7071, 0, 0.7071, 0])  # 90 degrees rotation around y-axis
        result = q1 * q2
        expected_result = Rotation.from_quat([0.5, 0.5, 0.5, 0.5])  # Combined rotation
        np.testing.assert_array_almost_equal(result.as_quat, expected_result.as_quat, decimal=4)

    def test_call(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        vector = np.array([1, 1, 1])
        result = q(vector)
        expected_result = np.array([1, -1, 1])
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)

    def test_inv(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        result = q.inv
        expected_result = Rotation.from_quat([0.7071, -0.7071, 0, 0])
        np.testing.assert_array_almost_equal(result.as_quat, expected_result.as_quat, decimal=4)

    def test_as_rotation_matrix(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        expected_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_array_almost_equal(q.as_rotation_matrix, expected_matrix, decimal=4)

    def test_from_rotation_matrix(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        q = Rotation.from_rotation_matrix(matrix)
        expected_result = Rotation.from_quat([0.7071, 0.7071, 0, 0])
        np.testing.assert_array_almost_equal(q.as_quat, expected_result.as_quat, decimal=4)

    def test_euler_to_quat(self):
        euler = np.array([np.pi / 2, 0, 0])  # 90 degrees rotation around x-axis
        q = Rotation.from_euler(euler)
        expected_result = Rotation.from_quat([0.7071, 0.7071, 0, 0])
        np.testing.assert_array_almost_equal(q.as_quat, expected_result.as_quat, decimal=4)

    def test_quat_to_euler(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        euler = q.as_euler
        expected_result = np.array([np.pi / 2, 0, 0])
        np.testing.assert_array_almost_equal(euler, expected_result, decimal=4)

    def test_from_rotvec_zero_rotation_returns_identity_rotation(self):
        rotvec = np.array([0.0, 0.0, 0.0])
        q = Rotation.from_rotvec(rotvec)
        expected_result = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(q.as_quat, expected_result.as_quat)

    def test_from_rotvec_x_axis_90_degrees_returns_correct_rotation(self):
        rotvec = np.array([np.pi / 2, 0.0, 0.0])  # 90 degrees around x-axis
        q = Rotation.from_rotvec(rotvec)
        expected_result = Rotation.from_quat([0.7071, 0.7071, 0.0, 0.0])
        np.testing.assert_array_almost_equal(q.as_quat, expected_result.as_quat, decimal=4)

    def test_as_rotvec_zero_rotation_returns_zero_vector(self):
        q = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])  # Identity rotation
        rotvec = q.as_rotvec
        expected_result = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rotvec, expected_result)

    def test_as_rotvec_x_axis_90_degrees_returns_correct_vector(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0.0, 0.0])  # 90 degrees around x-axis
        rotvec = q.as_rotvec
        expected_result = np.array([np.pi / 2, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rotvec, expected_result, decimal=4)

    def test_rotvec_conversion_cycle_consistency(self):
        original_rotvec = np.array([0.5, -0.3, 0.8])
        q = Rotation.from_rotvec(original_rotvec)
        recovered_rotvec = q.as_rotvec
        np.testing.assert_array_almost_equal(original_rotvec, recovered_rotvec, decimal=4)

    def test_to_representation_euler_same_as_as_euler(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        euler = q.as_euler
        np.testing.assert_array_almost_equal(euler, q.to(Rotation.Representation.EULER), decimal=4)

    def test_to_representation_quat_same_as_as_quat(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(q.as_quat, q.to(Rotation.Representation.QUAT), decimal=4)

    def test_to_representation_rotation_matrix_same_as_as_rotation_matrix(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(
            q.as_rotation_matrix, q.to(Rotation.Representation.ROTATION_MATRIX), decimal=4
        )

    def test_to_representation_rotvec_same_as_as_rotvec(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(q.as_rotvec, q.to(Rotation.Representation.ROTVEC), decimal=4)

    def test_create_from_euler_same_as_from_euler(self):
        euler = np.array([np.pi / 2, 0, 0])  # 90 degrees rotation around x-axis
        q = Rotation.create_from(euler, Rotation.Representation.EULER)
        np.testing.assert_array_almost_equal(q.as_quat, Rotation.from_euler(euler).as_quat, decimal=4)

    def test_create_from_rotation_matrix_same_as_from_rotation_matrix(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        q = Rotation.create_from(matrix, Rotation.Representation.ROTATION_MATRIX)
        np.testing.assert_array_almost_equal(q.as_quat, Rotation.from_rotation_matrix(matrix).as_quat, decimal=4)

    def test_create_from_rotvec_same_as_from_rotvec(self):
        rotvec = np.array([0.5, -0.3, 0.8])
        q = Rotation.create_from(rotvec, Rotation.Representation.ROTVEC)
        np.testing.assert_array_almost_equal(q.as_quat, Rotation.from_rotvec(rotvec).as_quat, decimal=4)

    def test_create_from_quat_same_as_from_quat(self):
        q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
        np.testing.assert_array_almost_equal(
            q.as_quat, Rotation.create_from(q.as_quat, Rotation.Representation.QUAT).as_quat, decimal=4
        )

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
            np.testing.assert_array_almost_equal(
                representation.from_value(data).as_quat, Rotation.create_from(data, representation).as_quat
            )

    def test_to_representation_exists_for_all_representations(self):
        for representation in Rotation.Representation:
            q = Rotation.from_quat([0.7071, 0.7071, 0, 0])  # 90 degrees rotation around x-axis
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
