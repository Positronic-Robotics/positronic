"""The joint targets of a pose command for one control step of a Panda arm.

Ports ``solve_pose_ik`` of positronic-franka ``src/pose_ik.hpp``, the IK of the FR3 driver: each joint stays within
its limits and within ``MAX_JOINT_STEP`` of the measured joints, the move is damped near a singularity, and the null
space pulls the joints towards ``POSTURE_JOINTS``. Poses are in the ``panda_link0`` frame. Each Gauss-Newton step
backtracks until it lowers the cost: without that, the steps near the stretched-arm singularity alternate between two
poses, and so do the joint targets of successive control steps.

This module must work in an isolated interpreter without Positronic installed.
"""

import numpy as np

# Mirrors of the positronic-franka IK constants, in src/pose_ik.hpp.
DAMPING_SINGULAR_VALUE = 0.1
MAX_JOINT_STEP = 0.5
POSTURE_JOINTS = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
POSTURE_PULL_PER_STEP = 0.02
_POSTURE_WEIGHT = 0.1
_ITERATIONS = 60
_STEP_TOLERANCE = 1e-9
_LEVENBERG = 1e-4
_LINE_SEARCH_STEPS = 20
_ACTIVE_SET_PASSES = 64

# The Panda's modified Denavit-Hartenberg parameters (a, d, alpha) of joints 1 to 7, then of the flange.
_DH = np.array([
    [0.0, 0.333, 0.0],
    [0.0, 0.0, -np.pi / 2],
    [0.0, 0.316, np.pi / 2],
    [0.0825, 0.0, np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0, 0.0, np.pi / 2],
    [0.088, 0.0, np.pi / 2],
    [0.0, 0.107, 0.0],
])


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st, 0.0, a],
        [st * ca, ct * ca, -sa, -d * sa],
        [st * sa, ct * sa, ca, d * ca],
        [0.0, 0.0, 0.0, 1.0],
    ])


def panda_flange(q: np.ndarray) -> np.ndarray:
    """The ``4 x 4`` flange pose at joints ``q``."""
    pose = np.eye(4)
    for (a, d, alpha), theta in zip(_DH, [*q, 0.0], strict=True):
        pose = pose @ _dh_transform(a, d, alpha, theta)
    return pose


def tool_kinematics(q: np.ndarray, flange_to_tool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ``4 x 4`` tool pose and its ``6 x 7`` Jacobian at joints ``q``, linear rows first.

    ``flange_to_tool`` is the tool pose in the flange frame.
    """
    pose = np.eye(4)
    axes, origins = [], []
    for (a, d, alpha), theta in zip(_DH, [*q, 0.0], strict=True):
        pose = pose @ _dh_transform(a, d, alpha, theta)
        axes.append(pose[:3, 2].copy())
        origins.append(pose[:3, 3].copy())
    tool = pose @ flange_to_tool
    axes, origins = np.array(axes[:7]), np.array(origins[:7])
    linear = np.cross(axes, tool[:3, 3] - origins)
    return tool, np.vstack([linear.T, axes.T])


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The angle-axis vector of a rotation matrix."""
    w = np.sqrt(max(0.0, 1.0 + np.trace(rotation))) / 2.0
    v = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]])
    if w > 1e-6:
        v = v / (4.0 * w)
    else:  # a half turn: the axis is the column of rotation + I with the largest norm
        columns = rotation + np.eye(3)
        v = columns[:, np.argmax(np.linalg.norm(columns, axis=0))]
        v = v / np.linalg.norm(v)
        return np.pi * v
    sin_half = np.linalg.norm(v)
    if sin_half < 1e-12:
        return 2.0 * v
    return 2.0 * np.arctan2(sin_half, w) * v / sin_half


def pose_error(tool: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Position error, then rotation error as an angle-axis vector in the base frame. ``J dq`` moves it."""
    rotation = tool[:3, :3]
    return np.concatenate([tool[:3, 3] - target[:3, 3], -rotation @ _rotation_vector(rotation.T @ target[:3, :3])])


def minimize_on_box(h: np.ndarray, g: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """argmin ``0.5 d^T h d + g^T d`` subject to ``lo <= d <= hi``, by a primal active set.

    ``h`` must be positive definite, and ``lo <= 0 <= hi``, so that ``d = 0`` is a feasible start.
    """
    n = len(g)
    hold = np.zeros(n, dtype=np.int8)  # 0 free, -1 at lo, +1 at hi, 2 fixed by a zero-width box
    d = np.zeros(n)
    fixed = lo >= hi
    hold[fixed], d[fixed] = 2, lo[fixed]
    for _ in range(_ACTIVE_SET_PASSES):
        free = hold == 0
        p = d.copy()
        if free.any():
            rhs = -g[free] - h[np.ix_(free, ~free)] @ d[~free]
            p[free] = np.linalg.solve(h[np.ix_(free, free)], rhs)
        edge = np.clip(p, lo, hi)
        crossing = free & (edge != p)
        if crossing.any():
            fraction = np.full(n, np.inf)
            fraction[crossing] = (edge[crossing] - d[crossing]) / (p[crossing] - d[crossing])
            first = int(np.argmin(fraction))
            d += fraction[first] * (p - d)
            at_lower = p[first] < lo[first]
            d[first] = lo[first] if at_lower else hi[first]
            hold[first] = -1 if at_lower else 1
            continue
        d = p
        pull = hold * (h @ d + g)  # a held joint whose move into the box lowers the cost
        pull[hold == 2] = 0.0
        release = int(np.argmax(pull))
        if pull[release] <= 1e-12:
            return d
        hold[release] = 0
    return d


def joint_targets(
    target: np.ndarray, q0: np.ndarray, flange_to_tool: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Joints that put the tool as close as the bounds allow to the ``4 x 4`` pose ``target``.

    ``q0`` is the measured joints, and ``lower`` and ``upper`` are the joint limits.
    """
    jacobian0 = tool_kinematics(q0, flange_to_tool)[1]
    # Eigenvalues of J^T J ascend: the first is the null space, the second is s_min squared.
    eigenvalues, eigenvectors = np.linalg.eigh(jacobian0.T @ jacobian0)
    s_min = np.sqrt(max(0.0, eigenvalues[1]))
    null_direction = eigenvectors[:, 0]
    # Chiaverini's law; mu_max = S0 / sqrt(2) keeps the weak-direction gain s / (s^2 + mu^2) at or below 1 / S0.
    s0 = DAMPING_SINGULAR_VALUE
    mu_sq = 0.5 * s0**2 * (1.0 - (s_min / s0) ** 2) if s_min < s0 else 0.0
    posture_target = -POSTURE_PULL_PER_STEP * null_direction @ (q0 - POSTURE_JOINTS)
    w_sq = _POSTURE_WEIGHT**2

    lo = np.maximum(q0 - MAX_JOINT_STEP, lower)
    hi = np.minimum(q0 + MAX_JOINT_STEP, upper)
    if not np.all(lo <= hi):
        raise ValueError('No joints are within MAX_JOINT_STEP of the measured joints and inside the joint limits')

    def cost(q: np.ndarray) -> float:
        e = pose_error(tool_kinematics(q, flange_to_tool)[0], target)
        return float(e @ e + mu_sq * (q - q0) @ (q - q0) + w_sq * (null_direction @ (q - q0) - posture_target) ** 2)

    q = np.clip(q0, lo, hi)
    q_cost = cost(q)
    eye = np.eye(len(q0))
    for _ in range(_ITERATIONS):
        tool, jacobian = tool_kinematics(q, flange_to_tool)
        e = pose_error(tool, target)
        posture_residual = null_direction @ (q - q0) - posture_target
        h = jacobian.T @ jacobian + (mu_sq + _LEVENBERG) * eye + w_sq * np.outer(null_direction, null_direction)
        g = jacobian.T @ e + mu_sq * (q - q0) + w_sq * posture_residual * null_direction
        d = minimize_on_box(h, g, lo - q, hi - q)
        if np.linalg.norm(d) < _STEP_TOLERANCE:
            break
        for _ in range(_LINE_SEARCH_STEPS):
            trial = np.clip(q + d, lo, hi)
            trial_cost = cost(trial)
            if trial_cost < q_cost:
                break
            d = d / 2
        else:
            break
        q, q_cost = trial, trial_cost
    return q
