from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np


class Transform3DMeta(type):
    @property
    def identity(cls):
        return cls(translation=np.zeros(3), rotation=Rotation.identity)


# y = R * x + t
class Transform3D(metaclass=Transform3DMeta):
    identity: 'Transform3D'
    __slots__ = ('translation', 'rotation')

    def __init__(self, translation=None, rotation=None):
        if translation is None:
            translation = np.zeros(3)
        if rotation is None:
            rotation = Rotation(1, 0, 0, 0)
        elif not isinstance(rotation, Rotation):
            rotation = Rotation(*rotation)
        self.translation = np.asarray(translation)
        self.rotation = rotation

    @property
    def as_matrix(self):
        """
        Convert the transformation to a 4x4 matrix.
        """
        t = self.translation
        q = self.rotation

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
            raise ValueError('Input matrix must be of shape (4, 4)')

        # Check that the fourth row has the expected form [0, 0, 0, 1]
        if not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError('The fourth row of the matrix must be [0, 0, 0, 1]')

        # Extract translation
        translation = matrix[:3, 3]

        # Extract rotation matrix
        R = matrix[:3, :3]

        # Convert rotation matrix to quaternion
        rotation = Rotation.from_rotation_matrix(R)

        return Transform3D(translation, rotation)

    @property
    def inv(self):
        """
        Compute the inverse of the transformation.
        """
        # Inverse of the rotation
        inv_rotation = self.rotation.inv

        # Inverse of the translation
        inv_translation = -inv_rotation(self.translation)

        return Transform3D(inv_translation, inv_rotation)

    def __mul__(self, other):
        """
        Multiply two Transform3D objects. T1 * T2 means f(x) = T1(T2(x))
        """
        if not isinstance(other, Transform3D):
            raise TypeError('Multiplicand must be an instance of Transform3D')
        return Transform3D(self.translation + self.rotation(other.translation), self.rotation * other.rotation)

    def __call__(self, vector):
        """
        Apply the transformation to a 3D vector.
        """
        if len(vector) != 3:
            raise ValueError('Input vector must be of length 3')
        return self.rotation(vector) + self.translation

    def __repr__(self):
        translation_str = np.array2string(self.translation, precision=3, suppress_small=True)
        quaternion_str = np.array2string(self.rotation.as_quat, precision=3, suppress_small=True)
        return f'Transform3D(t={translation_str}, q={quaternion_str})'

    def __str__(self):
        translation_str = np.array2string(self.translation, precision=3, suppress_small=True)
        quaternion_str = np.array2string(self.rotation.as_quat, precision=3, suppress_small=True)
        return f'Translation: {translation_str}, Quaternion: {quaternion_str}'

    def copy(self):
        return Transform3D(self.translation.copy(), self.rotation.copy())


class RotationMeta(type):
    @property
    def identity(cls):
        return cls.from_quat(np.array([1, 0, 0, 0]))


class Rotation(np.ndarray, metaclass=RotationMeta):
    """Class that represents a rotation in 3D space.

    The rotation is stored as a quaternion in the order (w, x, y, z), but users should not rely on this.
    Instead, use the `Rotation.Representation` enum to create a rotation.
    """

    def __new__(cls, w=1.0, x=0.0, y=0.0, z=0.0, *, dtype=np.float64):
        obj = np.asarray([w, x, y, z], dtype=dtype).view(cls)
        return obj

    class Representation(Enum):
        QUAT = 'quat'
        QUAT_XYZW = 'quat_xyzw'
        EULER = 'euler'
        ROTATION_MATRIX = 'rotation_matrix'
        ROTVEC = 'rotvec'

        def from_value(self, value: Any) -> 'Rotation':
            if self == Rotation.Representation.QUAT:
                return Rotation.from_quat(value)
            elif self == Rotation.Representation.QUAT_XYZW:
                return Rotation.from_quat_xyzw(value)
            elif self == Rotation.Representation.EULER:
                return Rotation.from_euler(value)
            elif self == Rotation.Representation.ROTATION_MATRIX:
                return Rotation.from_rotation_matrix(value)
            elif self == Rotation.Representation.ROTVEC:
                return Rotation.from_rotvec(value)
            else:
                raise NotImplementedError(f'Rotation representation {self} not implemented')

        @property
        def shape(self) -> int | tuple[int, int]:
            if self == Rotation.Representation.QUAT:
                return 4
            elif self == Rotation.Representation.QUAT_XYZW:
                return 4
            elif self == Rotation.Representation.EULER:
                return 3
            elif self == Rotation.Representation.ROTATION_MATRIX:
                return (3, 3)
            elif self == Rotation.Representation.ROTVEC:
                return 3
            else:
                raise NotImplementedError(f'Rotation representation {self} not implemented')

        @property
        def size(self) -> int:
            shape = self.shape

            if isinstance(shape, int):
                return shape

            return int(np.prod(shape))

        def __eq__(self, other: Any) -> bool:
            if isinstance(other, Rotation.Representation):
                return super().__eq__(other)
            elif isinstance(other, str):
                return self.value == other
            else:
                return False

    def __mul__(self, other):
        """
        Multiply two rotations.
        """
        if not isinstance(other, Rotation):
            raise TypeError('Multiplicand must be an instance of Rotation')

        w1, x1, y1, z1 = self
        w2, x2, y2, z2 = other

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Rotation(w, x, y, z)

    def __call__(self, vector):
        """
        Apply rotation to a vector.
        """
        q_vector = Rotation(0, *vector)
        rotated_vector = self * q_vector * self.inv
        return np.array(rotated_vector[1:])

    @property
    def inv(self):
        """
        Inverse rotation.
        """
        w, x, y, z = self
        return Rotation(w, -x, -y, -z)

    @classmethod
    def from_quat(cls, quat: Sequence[float]) -> 'Rotation':
        """
        Create a rotation from a 4-element numpy array. The order of the elements is (w, x, y, z).

        Args:
            quat: (Sequence[float]) w, x, y, z elements.

        Returns:
            Rotation object.
        """
        return cls(*quat)

    @classmethod
    def from_quat_xyzw(cls, quat: Sequence[float]) -> 'Rotation':
        """
        Create a rotation from a 4-element numpy array. The order of the elements is (x, y, z, w).

        Args:
            quat: (Sequence[float]) x, y, z, w elements.

        Returns:
            Rotation object.
        """
        return cls(quat[3], quat[0], quat[1], quat[2])

    @classmethod
    def from_rotation_matrix(cls, matrix):
        """
        Create a rotation from a rotation matrix.
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
        Create a rotation from euler angles in radians.
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
    def from_rotvec(cls, rotvec: np.ndarray) -> 'Rotation':
        """
        Create rotation from a rotation vector.

        Args:
            rotvec: (numpy.ndarray) 3D rotation vector representing axis-angle rotation

        Returns:
            Rotation object representing the same rotation
        """
        angle = np.linalg.norm(rotvec)
        if angle < 1e-10:  # Handle small angles to avoid division by zero
            return cls(1.0, 0.0, 0.0, 0.0)

        axis = rotvec / angle
        sin_theta_2 = np.sin(angle / 2)
        cos_theta_2 = np.cos(angle / 2)

        w = cos_theta_2
        x = axis[0] * sin_theta_2
        y = axis[1] * sin_theta_2
        z = axis[2] * sin_theta_2

        return cls(w, x, y, z)

    @classmethod
    def create_from(cls, value: Any, representation: Representation | str) -> 'Rotation':
        """
        Create a rotation from any supported rotation representation.

        Args:
            value: (Any) Any supported rotation representation.
            representation: (Rotation.Representation | str) The representation of the input value.

        Returns:
            Rotation object.
        """
        return Rotation.Representation(representation).from_value(value)

    def to(self, representation: Representation | str) -> np.ndarray:
        """
        Convert the rotation to any supported rotation representation.

        Args:
            representation: (Rotation.Representation | str) The representation to convert to.

        Returns:
            (np.ndarray) The converted rotation representation.
        """
        if representation == Rotation.Representation.QUAT:
            return self.as_quat
        elif representation == Rotation.Representation.QUAT_XYZW:
            return self.as_quat_xyzw
        elif representation == Rotation.Representation.EULER:
            return self.as_euler
        elif representation == Rotation.Representation.ROTATION_MATRIX:
            return self.as_rotation_matrix
        elif representation == Rotation.Representation.ROTVEC:
            return self.as_rotvec
        else:
            raise ValueError(f'Invalid rotation representation: {representation}')

    @property
    def as_rotation_matrix(self):
        """
        Represent the rotation as a rotation matrix.
        """
        w, x, y, z = self
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ])

    @property
    def as_rotvec(self) -> np.ndarray:
        """
        Represent the rotation as a rotation vector.

        Returns:
            numpy.ndarray: 3D rotation vector representing axis-angle rotation. The direction
                          of the vector indicates the axis of rotation and its magnitude
                          represents the angle in radians.
        """
        self[:] = normalise_quat(self)
        angle = 2 * np.arccos(self[0])
        if angle < 1e-10:  # Handle small angles to avoid division by zero
            return np.zeros(3)

        sin_theta_2 = np.sin(angle / 2)
        axis = np.array([self[1], self[2], self[3]]) / sin_theta_2
        return axis * angle

    @property
    def as_euler(self) -> np.ndarray:
        """
        Convert a rotation to euler angles in radians.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self[0] * self[1] + self[2] * self[3])
        cosr_cosp = 1 - 2 * (self[1] ** 2 + self[2] ** 2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (self[0] * self[2] - self[3] * self[1])
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self[0] * self[3] + self[1] * self[2])
        cosy_cosp = 1 - 2 * (self[2] ** 2 + self[3] ** 2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @property
    def as_quat(self) -> np.ndarray:
        """
        Convert the rotation to a quaternion.
        """
        return np.asarray(self)

    @property
    def as_quat_xyzw(self) -> np.ndarray:
        """
        Convert the rotation to a quaternion in the order (x, y, z, w).
        """
        return np.asarray([self[1], self[2], self[3], self[0]])

    @property
    def angle(self):
        """
        Compute the angle of the rotation in radians.
        """
        return 2 * np.arccos(self[0])


def normalise_quat(q: np.ndarray) -> np.ndarray:
    """
    Normalise a rotation, expressed as a 4-element numpy array.
    """
    return q / np.linalg.norm(q)


def degrees_to_radians(degrees: float) -> float:
    """
    Convert degrees to radians.
    """
    return degrees * np.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """
    Convert radians to degrees.
    """
    return radians * 180.0 / np.pi
