from enum import Enum
from typing import Any

import numpy as np


class RotationRepresentation(Enum):
    QUAT = 'quat'
    EULER = 'euler'
    ROTATION_MATRIX = 'rotation_matrix'
    ROTVEC = 'rotvec'

    def from_value(self, value: Any) -> 'Quaternion':
        if self == RotationRepresentation.QUAT:
            return Quaternion(*value)
        elif self == RotationRepresentation.EULER:
            return Quaternion.from_euler(value)
        elif self == RotationRepresentation.ROTATION_MATRIX:
            return Quaternion.from_rotation_matrix(value)
        elif self == RotationRepresentation.ROTVEC:
            return Quaternion.from_rotvec(value)
        else:
            raise NotImplementedError(f"Rotation representation {self} not implemented")

    @property
    def shape(self) -> int | tuple[int, int]:
        if self == RotationRepresentation.QUAT:
            return 4
        elif self == RotationRepresentation.EULER:
            return 3
        elif self == RotationRepresentation.ROTATION_MATRIX:
            return (3, 3)
        elif self == RotationRepresentation.ROTVEC:
            return 3
        else:
            raise NotImplementedError(f"Rotation representation {self} not implemented")

    @property
    def size(self) -> int:
        shape = self.shape

        if isinstance(shape, int):
            return shape

        return np.prod(shape)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RotationRepresentation):
            return super().__eq__(other)
        elif isinstance(other, str):
            return self.value == other
        else:
            return False


# y = R * x + t
class Transform3D:
    __slots__ = ('translation', 'quaternion')

    def __init__(self, translation=None, quaternion=None):
        if translation is None:
            translation = np.zeros(3)
        if quaternion is None:
            quaternion = Quaternion(1, 0, 0, 0)
        elif not isinstance(quaternion, Quaternion):
            quaternion = Quaternion(*quaternion)
        self.translation = translation
        self.quaternion = quaternion

    @property
    def as_matrix(self):
        """
        Convert the transformation to a 4x4 matrix.
        """
        t = self.translation
        q = self.quaternion

        # Create 4x4 transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = q.as_rotation_matrix
        matrix[:3, 3] = t

        return matrix

    @staticmethod
    def from_matrix(matrix):
        """
        Create a Transform3D object from a 4x4 transformation matrix.
        """
        if matrix.shape != (4, 4):
            raise ValueError("Input matrix must be of shape (4, 4)")

        # Check that the fourth row has the expected form [0, 0, 0, 1]
        if not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("The fourth row of the matrix must be [0, 0, 0, 1]")

        # Extract translation
        translation = matrix[:3, 3]

        # Extract rotation matrix
        R = matrix[:3, :3]

        # Convert rotation matrix to quaternion
        quaternion = Quaternion.from_rotation_matrix(R)

        return Transform3D(translation, quaternion)

    @property
    def inv(self):
        """
        Compute the inverse of the transformation.
        """
        # Inverse of the rotation
        inv_quaternion = self.quaternion.inv

        # Inverse of the translation
        inv_translation = -inv_quaternion(self.translation)

        return Transform3D(inv_translation, inv_quaternion)

    def __mul__(self, other):
        """
        Multiply two Transform3D objects. T1 * T2 means f(x) = T1(T2(x))
        """
        if not isinstance(other, Transform3D):
            raise TypeError("Multiplicand must be an instance of Transform3D")
        return Transform3D(self.translation + self.quaternion(other.translation),
                           self.quaternion * other.quaternion)

    def __call__(self, vector):
        """
        Apply the transformation to a 3D vector.
        """
        if len(vector) != 3:
            raise ValueError("Input vector must be of length 3")
        return self.quaternion(vector) + self.translation

    def __repr__(self):
        translation_str = np.array2string(self.translation, precision=3)
        quaternion_str = np.array2string(self.quaternion, precision=3)
        return f"Transform3D(t={translation_str}, q={quaternion_str})"

    def __str__(self):
        translation_str = np.array2string(self.translation, precision=3)
        quaternion_str = np.array2string(self.quaternion, precision=3)
        return f"Translation: {translation_str}, Quaternion: {quaternion_str}"

    def copy(self):
        return Transform3D(self.translation.copy(), self.quaternion.copy())


class Quaternion(np.ndarray):
    def __new__(cls, w=1.0, x=0.0, y=0.0, z=0.0, dtype=np.float64):
        obj = np.asarray([w, x, y, z], dtype=dtype).view(cls)
        return obj

    def __mul__(self, other):
        """
        Multiply two quaternions.
        """
        if not isinstance(other, Quaternion):
            raise TypeError("Multiplicand must be an instance of Quaternion")

        w1, x1, y1, z1 = self
        w2, x2, y2, z2 = other

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(w, x, y, z)

    def __call__(self, vector):
        """
        Apply rotation to a vector.
        """
        q_vector = Quaternion(0, *vector)
        rotated_vector = self * q_vector * self.inv
        return np.array(rotated_vector[1:])

    @property
    def inv(self):
        """
        Invert the quaternion.
        """
        w, x, y, z = self
        return Quaternion(w, -x, -y, -z)

    @property
    def as_rotation_matrix(self):
        """
        Convert the quaternion to a rotation matrix.
        """
        w, x, y, z = self
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])

    @classmethod
    def from_rotation_matrix(cls, matrix):
        """
        Create a quaternion from a rotation matrix.
        """
        m = np.asarray(matrix, dtype=np.float64)[:3, :3]
        t = np.trace(m)
        if t > 0.0:
            s = np.sqrt(t + 1.0)
            w = 0.5 * s
            s = 0.5 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        else:
            i = np.argmax(np.diag(m))
            if i == 0:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
                x = 0.5 * s
                s = 0.5 / s
                y = (m[1, 0] + m[0, 1]) * s
                z = (m[0, 2] + m[2, 0]) * s
                w = (m[2, 1] - m[1, 2]) * s
            elif i == 1:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
                y = 0.5 * s
                s = 0.5 / s
                x = (m[1, 0] + m[0, 1]) * s
                z = (m[2, 1] + m[1, 2]) * s
                w = (m[0, 2] - m[2, 0]) * s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
                z = 0.5 * s
                s = 0.5 / s
                x = (m[0, 2] + m[2, 0]) * s
                y = (m[2, 1] + m[1, 2]) * s
                w = (m[1, 0] - m[0, 1]) * s
        return cls(w, x, y, z)

    @classmethod
    def from_euler(cls, euler):
        """
        Convert euler angles in radians to a quaternion.
        """
        roll, pitch, yaw = euler
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(w, x, y, z)

    @classmethod
    def from_rotvec(cls, rotvec: np.ndarray) -> 'Quaternion':
        """
        Convert a rotation vector to a quaternion.

        Args:
            rotvec: (numpy.ndarray) 3D rotation vector representing axis-angle rotation

        Returns:
            Quaternion object representing the same rotation
        """
        angle = np.linalg.norm(rotvec)
        if angle < 1e-10:  # Handle small angles to avoid division by zero
            return cls(1.0, 0.0, 0.0, 0.0)

        axis = rotvec / angle
        sin_theta_2 = np.sin(angle/2)
        cos_theta_2 = np.cos(angle/2)

        w = cos_theta_2
        x = axis[0] * sin_theta_2
        y = axis[1] * sin_theta_2
        z = axis[2] * sin_theta_2

        return cls(w, x, y, z)

    @classmethod
    def create_from(cls, value: Any, representation: RotationRepresentation | str) -> 'Quaternion':
        """
        Create a quaternion from any supported rotation representation.

        Args:
            value: (Any) Any supported rotation representation.
            representation: (RotationRepresentation | str) The representation of the input value.

        Returns:
            Quaternion object.
        """
        return RotationRepresentation(representation).from_value(value)

    def to(self, representation: RotationRepresentation | str) -> np.ndarray:
        """
        Convert the quaternion to any supported rotation representation.

        Args:
            representation: (RotationRepresentation | str) The representation to convert to.

        Returns:
            (np.ndarray) The converted rotation representation.
        """
        if representation == RotationRepresentation.QUAT:
            return np.asarray(self)
        elif representation == RotationRepresentation.EULER:
            return self.as_euler
        elif representation == RotationRepresentation.ROTATION_MATRIX:
            return self.as_rotation_matrix
        elif representation == RotationRepresentation.ROTVEC:
            return self.as_rotvec
        else:
            raise ValueError(f"Invalid rotation representation: {representation}")

    @property
    def as_rotvec(self) -> np.ndarray:
        """
        Convert quaternion to rotation vector representation.

        Returns:
            numpy.ndarray: 3D rotation vector representing axis-angle rotation. The direction
                          of the vector indicates the axis of rotation and its magnitude
                          represents the angle in radians.
        """
        angle = 2 * np.arccos(self[0])
        if angle < 1e-10:  # Handle small angles to avoid division by zero
            return np.zeros(3)

        sin_theta_2 = np.sin(angle/2)
        axis = np.array([self[1], self[2], self[3]]) / sin_theta_2
        return axis * angle

    @property
    def as_euler(self) -> np.ndarray:
        """
        Convert a quaternion to euler angles in radians.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self[0] * self[1] + self[2] * self[3])
        cosr_cosp = 1 - 2 * (self[1]**2 + self[2]**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (self[0] * self[2] - self[3] * self[1])
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self[0] * self[3] + self[1] * self[2])
        cosy_cosp = 1 - 2 * (self[2]**2 + self[3]**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @property
    def angle(self):
        """
        Compute the angle of the quaternion in radians.
        """
        return 2 * np.arccos(self[0])


def quat_mul(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """
    Multiply two quaternions, expressed as 4-element numpy arrays.
    Returns a 4-element numpy array. All quaternions are w,
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return Quaternion(w, x, y, z)


def normalise_quat(q: Quaternion) -> Quaternion:
    """
    Normalise a quaternion, expressed as a 4-element numpy array.
    """
    return q / np.linalg.norm(q)


def degrees_to_radians(degrees):
    """
    Convert degrees to radians.
    """
    return degrees * np.pi / 180.0


def radians_to_degrees(radians):
    """
    Convert radians to degrees.
    """
    return radians * 180.0 / np.pi
