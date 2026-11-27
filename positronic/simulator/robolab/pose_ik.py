"""The joint targets of a pose command for one control step of a Panda arm.

Ports ``solve_pose_ik`` of positronic-franka ``src/pose_ik.hpp``, the IK of the FR3 driver: each joint stays within
its limits and within ``MAX_JOINT_STEP`` of the measured joints, the move is damped near a singularity, and the null
space pulls the joints towards ``POSTURE_JOINTS``. The arm links stay apart by the FR3 capsule model of
``src/self_collision.hpp``. At each checked point of the move, each checked pair keeps its floor:
``SELF_COLLISION_CLEARANCE``, or its own clearance at the start of the move when that is less. Between two checked
points a pair's clearance stays within ``SELF_COLLISION_PATH_TOLERANCE`` of its floor. Poses are in the
``panda_link0`` frame. Each Gauss-Newton step backtracks until it lowers the cost: without that, the steps near the
stretched-arm singularity alternate between two poses, and so do the joint targets of successive control steps.

Two parts are not in the driver. A joint that the physics pushed past a limit gets a target at that limit. The solve
keeps j4 at or below ``ELBOW_STRETCHED``.

This module must work in an isolated interpreter without Positronic installed.
"""

import numpy as np

# Mirrors of the positronic-franka IK constants, in src/pose_ik.hpp and src/self_collision.hpp.
DAMPING_SINGULAR_VALUE = 0.1
MAX_JOINT_STEP = 0.5
POSTURE_JOINTS = np.array([0.0, -0.31, 0.0, -1.65, 0.0, 1.522, 0.0])
POSTURE_PULL_PER_STEP = 0.02
# The least clearance between two checked links at each checked point of the move to an answer. The FR3
# self-collision reflex stopped moves at 0.064 m and less.
SELF_COLLISION_CLEARANCE = 0.09
# Below this clearance between two checked links, the solve pushes them apart. It is above SELF_COLLISION_CLEARANCE,
# so the push holds the answer clear of that floor.
SELF_COLLISION_PUSH_CLEARANCE = 0.10
# How far below its floor the clearance may fall between two checked points of a move.
SELF_COLLISION_PATH_TOLERANCE = 0.001
_ACTIVE_SET_PASSES = 64
# The j4 of the longest shoulder-wrist distance: the a = 0.0825 offsets of joints 4 and 5 bend the stretched arm.
ELBOW_STRETCHED = -(np.arctan(0.0825 / 0.316) + np.arctan(0.0825 / 0.384))

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


def _dh_transform(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    """The ``... x 4 x 4`` transforms for the joint angles ``theta`` of shape ``...``."""
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    t = np.zeros((*np.shape(theta), 4, 4))
    t[..., 0, 0], t[..., 0, 1], t[..., 0, 3] = ct, -st, a
    t[..., 1, 0], t[..., 1, 1], t[..., 1, 2], t[..., 1, 3] = st * ca, ct * ca, -sa, -d * sa
    t[..., 2, 0], t[..., 2, 1], t[..., 2, 2], t[..., 2, 3] = st * sa, ct * sa, ca, d * ca
    t[..., 3, 3] = 1.0
    return t


def link_frames(q: np.ndarray) -> np.ndarray:
    """The ``... x 9 x 4 x 4`` poses of link0 to link7 and of the flange at the ``... x 7`` joints ``q``."""
    frames = np.empty((*q.shape[:-1], 9, 4, 4))
    frames[..., 0, :, :] = np.eye(4)
    angles = np.concatenate([q, np.zeros((*q.shape[:-1], 1))], axis=-1)
    for link, (a, d, alpha) in enumerate(_DH):
        frames[..., link + 1, :, :] = frames[..., link, :, :] @ _dh_transform(a, d, alpha, angles[..., link])
    return frames


def panda_flange(q: np.ndarray) -> np.ndarray:
    """The ``4 x 4`` flange pose at joints ``q``."""
    return tool_kinematics(q, np.eye(4))[0]


def tool_kinematics(q: np.ndarray, flange_to_tool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ``4 x 4`` tool pose and its ``6 x 7`` Jacobian at joints ``q``, linear rows first.

    ``flange_to_tool`` is the tool pose in the flange frame.
    """
    frames = link_frames(q)
    tool = frames[8] @ flange_to_tool
    axes, origins = frames[1:8, :3, 2], frames[1:8, :3, 3]
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


# The FR3 self-collision capsules of franka_description (robots/common/franka_arm.xacro, the *_sc links). Each row is
# the link (0 is the base), the centre and the unit axis in link axes, the length between the two sphere centres, and
# the radius, in m. link7's capsules are in axes turned pi/4 about its z axis. The gripper on the flange has no capsule.
_X, _Z, _LINK7_X = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0)
_CAPSULES = (
    (0, (-0.075, 0.0, 0.06), _X, 0.03, 0.06),
    (1, (0.0, 0.0, -0.1915), _Z, 0.283, 0.06),
    (2, (0.0, 0.0, 0.0), _Z, 0.12, 0.06),
    (3, (0.0, 0.0, -0.145), _Z, 0.15, 0.06),
    (4, (0.0, 0.0, 0.0), _Z, 0.12, 0.06),
    (5, (0.0, 0.0, -0.26), _Z, 0.10, 0.06),
    (5, (0.0, 0.08, -0.13), _Z, 0.14, 0.025),
    (6, (0.0, 0.0, -0.03), _Z, 0.08, 0.05),
    (7, (0.0, 0.0, 0.01), _Z, 0.14, 0.04),
    (7, (0.06 * _LINK7_X[0], 0.06 * _LINK7_X[1], 0.082), _LINK7_X, 0.01, 0.03),
)
_CAPSULE_LINK, _CAPSULE_CENTER, _CAPSULE_AXIS, _CAPSULE_LENGTH, _CAPSULE_RADIUS = map(
    np.array, zip(*_CAPSULES, strict=True)
)
_CAPSULE_HALF_AXIS = 0.5 * _CAPSULE_LENGTH[:, None] * _CAPSULE_AXIS  # from the centre to one sphere centre
# The two sphere centres of each capsule, as homogeneous points in link axes: 10 x 4 x 2.
_CAPSULE_ENDS = np.ones((len(_CAPSULES), 4, 2))
_CAPSULE_ENDS[:, :3] = np.stack([_CAPSULE_CENTER - _CAPSULE_HALF_AXIS, _CAPSULE_CENTER + _CAPSULE_HALF_AXIS], axis=-1)

# The arm link pairs that franka_description's SRDF checks.
SELF_COLLISION_PAIRS = ((0, 5), (0, 6), (0, 7), (1, 5), (1, 6), (1, 7), (2, 5), (2, 7))
# Each row is the index of a checked pair, a capsule of its first link, and a capsule of its second link.
_CAPSULE_PAIRS = np.array([
    (pair, a, b)
    for pair, (link_a, link_b) in enumerate(SELF_COLLISION_PAIRS)
    for a in np.flatnonzero(_CAPSULE_LINK == link_a)
    for b in np.flatnonzero(_CAPSULE_LINK == link_b)
])
_ROWS_OF_PAIR = [np.flatnonzero(_CAPSULE_PAIRS[:, 0] == pair) for pair in range(len(SELF_COLLISION_PAIRS))]


def _dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The dot products of ``x`` and ``y`` over their last axis."""
    return np.einsum('...k,...k->...', x, y)


def closest_points_of_segments(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The closest points of the ``... x 3`` segments ``[p1, q1]`` and ``[p2, q2]``, of a length above zero.

    Ericson, Real-Time Collision Detection, 5.1.9.
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f, c, b = _dot(d1, d1), _dot(d2, d2), _dot(d2, r), _dot(d1, r), _dot(d1, d2)
    denom = a * e - b * b
    s = np.clip(np.divide(b * f - c * e, denom, out=np.zeros_like(denom), where=denom > 1e-12), 0.0, 1.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    return p1 + s[..., None] * d1, p2 + t[..., None] * d2


def _capsule_pair_points(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row of ``_CAPSULE_PAIRS``, the closest points of the two capsule axes and the clearance of the capsules.

    ``frames`` is the ``... x 9 x 4 x 4`` output of ``link_frames``.
    """
    # The two sphere centres of each capsule in the base frame, each ``... x 10 x 3``.
    first, second = np.moveaxis(frames[..., _CAPSULE_LINK, :3, :] @ _CAPSULE_ENDS, -1, 0)
    _, a, b = _CAPSULE_PAIRS.T
    near_a, near_b = closest_points_of_segments(
        first[..., a, :], second[..., a, :], first[..., b, :], second[..., b, :]
    )
    gap = near_a - near_b
    clearance = np.sqrt(_dot(gap, gap)) - _CAPSULE_RADIUS[a] - _CAPSULE_RADIUS[b]
    return near_a, near_b, clearance


def _speed_along(frames: np.ndarray, links: np.ndarray, points: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """How fast each point, fixed to its link in ``links``, moves along its direction, per radian of each joint."""
    axes, origins = frames[1:8, :3, 2], frames[1:8, :3, 3]
    speed = np.einsum('pk,pjk->pj', directions, np.cross(axes, points[:, None, :] - origins))
    return np.where(np.arange(7) < links[:, None], speed, 0.0)


def self_collision_clearances(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The clearance of each pair in ``SELF_COLLISION_PAIRS`` at joints ``q``, and its ``8 x 7`` gradient in the joints.

    The clearance of a pair is that of its closest two capsules. It is negative when the capsules overlap.
    """
    frames = link_frames(q)
    near_a, near_b, clearance = _capsule_pair_points(frames)
    closest = [rows[np.argmin(clearance[rows])] for rows in _ROWS_OF_PAIR]
    near_a, near_b = near_a[closest], near_b[closest]
    _, a, b = _CAPSULE_PAIRS[closest].T
    gap = near_a - near_b
    distance = np.linalg.norm(gap, axis=-1, keepdims=True)
    # Crossed axes give no direction to push apart; the solve keeps 0.09 m, far from that depth.
    normal = np.divide(gap, distance, out=np.zeros_like(gap), where=distance > 1e-9)
    speed_a = _speed_along(frames, _CAPSULE_LINK[a], near_a, normal)
    speed_b = _speed_along(frames, _CAPSULE_LINK[b], near_b, normal)
    return clearance[closest], speed_a - speed_b


def pair_clearances(q: np.ndarray) -> np.ndarray:
    """The ``... x 8`` clearance of each pair in ``SELF_COLLISION_PAIRS`` at the ``... x 7`` joints ``q``."""
    clearance = _capsule_pair_points(link_frames(q))[2]
    return np.stack([clearance[..., rows].min(axis=-1) for rows in _ROWS_OF_PAIR], axis=-1)


def _clearance_rate_bounds() -> np.ndarray:
    """For each joint, the most that the clearance of a checked pair changes per radian of that joint, at any joints.

    A joint turns a point at most its distance from the joint origin per radian. The DH chain from the joint origin to
    the link origin, plus the farthest capsule end from the link origin, bounds that distance. A joint before both
    links of a pair turns the two links together and does not change their clearance.
    """
    bound = np.zeros(7)
    for link_a, link_b in SELF_COLLISION_PAIRS:
        for capsule in np.flatnonzero(_CAPSULE_LINK == link_b):
            center, half_axis = _CAPSULE_CENTER[capsule], _CAPSULE_HALF_AXIS[capsule]
            reach = max(np.linalg.norm(center - half_axis), np.linalg.norm(center + half_axis))
            for j in range(link_b - 1, link_a - 1, -1):  # joint j has its origin at link j + 1
                bound[j] = max(bound[j], reach)
                reach += np.hypot(_DH[j, 0], _DH[j, 1])  # from link j + 1 to link j
    return bound


CLEARANCE_RATE_BOUNDS = _clearance_rate_bounds()


_LEAST_PATH_SAMPLES = 16
_PATH_BISECTIONS = 8


def stop_before_self_collision(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """The point of the move from ``start`` to ``end`` farthest along it up to which each checked pair keeps its floor.

    The floor of a pair is ``SELF_COLLISION_CLEARANCE``, or its own clearance at ``start`` when that is less. Between
    two checked points a pair's clearance stays within ``SELF_COLLISION_PATH_TOLERANCE`` of its floor: the points are
    close enough for the rate bound of each joint (``CLEARANCE_RATE_BOUNDS``).
    """
    move = end - start
    # The most a pair's clearance changes over the whole move. Between two points 1 / samples apart that both keep its
    # floor, the clearance falls at most change_over_move / (2 * samples) below that floor.
    change_over_move = CLEARANCE_RATE_BOUNDS @ np.abs(move)
    samples = max(_LEAST_PATH_SAMPLES, int(np.ceil(change_over_move / (2 * SELF_COLLISION_PATH_TOLERANCE))))
    floors = np.minimum(SELF_COLLISION_CLEARANCE, pair_clearances(start))
    parts = np.arange(1, samples + 1) / samples
    clear = np.all(pair_clearances(start + parts[:, None] * move) >= floors, axis=-1)
    if clear.all():
        return end
    first_blocked = int(np.argmin(clear))
    safe, blocked = (parts[first_blocked - 1] if first_blocked else 0.0), parts[first_blocked]
    for _ in range(_PATH_BISECTIONS):
        mid = 0.5 * (safe + blocked)
        if np.all(pair_clearances(start + mid * move) >= floors):
            safe = mid
        else:
            blocked = mid
    return start + safe * move


_POSTURE_WEIGHT = 0.1
# 1 cm inside SELF_COLLISION_PUSH_CLEARANCE costs as much as 20 cm of tool error.
_COLLISION_WEIGHT = 20.0
_ITERATIONS = 60
_STEP_TOLERANCE = 1e-9
_LEVENBERG = 1e-4
_LINE_SEARCH_STEPS = 20


def joint_targets(
    target: np.ndarray, q0: np.ndarray, flange_to_tool: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Joints that put the tool as close as the bounds and the self-collision clearance allow to the ``4 x 4`` pose
    ``target``.

    ``q0`` is the measured joints, and ``lower`` and ``upper`` are the joint limits. The move from ``q0``, or from its
    nearest point within the bounds, keeps the floor of ``stop_before_self_collision``.
    """
    # Past ELBOW_STRETCHED the shoulder-wrist distance falls as j4 rises. A solve there pins j4 at its upper limit for a
    # target that needs a shorter arm, and the folded elbow reaches every distance that the stretched one does.
    # A j4 above it folds by at most MAX_JOINT_STEP a step, and a j4 past its limit goes to that limit first.
    folded_upper = upper.copy()
    folded_upper[3] = min(upper[3], max(ELBOW_STRETCHED, q0[3] - MAX_JOINT_STEP))

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

    # A joint past a limit gets a target at that limit or inside it.
    lo = np.clip(q0 - MAX_JOINT_STEP, lower, folded_upper)
    hi = np.clip(q0 + MAX_JOINT_STEP, lower, folded_upper)

    def collision_push(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """How far each checked pair is inside ``SELF_COLLISION_PUSH_CLEARANCE``, and the gradient of its clearance."""
        clearance, gradient = self_collision_clearances(q)
        return np.minimum(0.0, clearance - SELF_COLLISION_PUSH_CLEARANCE), gradient

    def cost(q: np.ndarray) -> float:
        e = pose_error(tool_kinematics(q, flange_to_tool)[0], target)
        push = collision_push(q)[0]
        return float(
            e @ e
            + mu_sq * (q - q0) @ (q - q0)
            + w_sq * (null_direction @ (q - q0) - posture_target) ** 2
            + _COLLISION_WEIGHT**2 * push @ push
        )

    start = np.clip(q0, lo, hi)
    q = start
    q_cost = cost(q)
    eye = np.eye(len(q0))
    for _ in range(_ITERATIONS):
        tool, jacobian = tool_kinematics(q, flange_to_tool)
        e = pose_error(tool, target)
        posture_residual = null_direction @ (q - q0) - posture_target
        push, push_gradient = collision_push(q)
        pushed = push_gradient[push < 0.0]
        h = (
            jacobian.T @ jacobian
            + (mu_sq + _LEVENBERG) * eye
            + w_sq * np.outer(null_direction, null_direction)
            + _COLLISION_WEIGHT**2 * pushed.T @ pushed
        )
        g = (
            jacobian.T @ e
            + mu_sq * (q - q0)
            + w_sq * posture_residual * null_direction
            + _COLLISION_WEIGHT**2 * push @ push_gradient
        )
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
    return stop_before_self_collision(start, q)
