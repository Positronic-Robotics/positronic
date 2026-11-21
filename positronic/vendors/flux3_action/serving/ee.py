"""Pose conversions between the client's quaternion poses and DROID's Euler poses, numpy only.

A client pose is ``[x, y, z, qw, qx, qy, qz]``. DROID's ``cartesian_position`` is ``[x, y, z, rx, ry, rz]`` with
``R = Rz(rz) Ry(ry) Rx(rx)``, the convention of the EE-50 statistics and of BFL's ``examples.droid_ee50``.
"""

import numpy as np


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """``[qw, qx, qy, qz]`` with ``qw >= 0``."""
    m = np.asarray(r, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q if q[0] >= 0 else -q


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    sx, cx, sy, cy, sz, cz = np.sin(rx), np.cos(rx), np.sin(ry), np.cos(ry), np.sin(rz), np.cos(rz)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


def matrix_to_euler_xyz(r: np.ndarray) -> np.ndarray:
    return np.array([
        np.arctan2(r[2, 1], r[2, 2]),
        np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])),
        np.arctan2(r[1, 0], r[0, 0]),
    ])


def pose_to_droid_cartesian(pose: np.ndarray) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` -> DROID's ``[x, y, z, rx, ry, rz]``."""
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('ee_pose must be seven finite values [x, y, z, qw, qx, qy, qz]')
    return np.concatenate([pose[:3], matrix_to_euler_xyz(quat_to_matrix(pose[3:]))])


def droid_commands_to_actions(commands: np.ndarray) -> np.ndarray:
    """DROID commands ``(K, 7)`` ``[x, y, z, rx, ry, rz, grip]`` -> ``(K, 8)`` ``[x, y, z, qw, qx, qy, qz, grip]``."""
    commands = np.asarray(commands, dtype=np.float64)
    if commands.ndim != 2 or commands.shape[1] != 7:
        raise ValueError(f'expected (K, 7) DROID commands, got {commands.shape}')
    quats = np.stack([matrix_to_quat(euler_xyz_to_matrix(*c[3:6])) for c in commands])
    return np.concatenate([commands[:, :3], quats, commands[:, 6:7]], axis=1).astype(np.float32)
