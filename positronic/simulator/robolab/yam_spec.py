"""The two-arm i2rt YAM robot of the RoboLab env server, and the action that the wire commands map to.

RoboLab's interpreter imports this module as a top-level ``yam_spec``, so it uses only Python and numpy.

Frames:

- The robot root is the centre of the bottom face of the base plate (link ``base_plate``). x points forward from the
  arms, y to the left arm, z up. The root is at the env origin, on the plane of the task table top.
- Each arm base frame is at the bottom of its arm base (the i2rt URDF frame), on the top face of the plate.
- The link ``<side>_gripper`` is at ``GRIPPER_SITE`` of the i2rt model, the point that the rig reports poses at.
"""

import copy
import functools
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

if __package__:
    from positronic.simulator.env_server import arm_action, dls_ik, protocol
    from positronic.simulator.robolab import keys
else:
    import arm_action  # pyright: ignore[reportMissingImports]
    import dls_ik  # pyright: ignore[reportMissingImports]
    import keys  # pyright: ignore[reportMissingImports]
    import protocol  # pyright: ignore[reportMissingImports]

# The arm sides, as the wire channels and ``positronic.keys.BIMANUAL_ARMS`` name them. A test keeps them the same.
LEFT, RIGHT = 'left', 'right'
SIDES = (LEFT, RIGHT)
ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 7))
# joint7 moves tip_left and joint8 moves tip_right. joint8 is a URDF mimic of joint7 (the rack couples them).
FINGER_JOINT = 'joint7'
FINGER_MIMIC_JOINT = 'joint8'
FINGER_LINKS = ('tip_left', 'tip_right')
WRIST_LINK = 'link6'
GRIPPER_SITE = 'gripper'
# Finger travel per finger. At 0 the pads meet on the wrist axis; at 0.0475 the gap is 95 mm.
FINGER_CLOSED = 0.0
FINGER_OPEN = 0.0475

# The bases are 0.61 m apart, in line and parallel, on a 0.022 m plate, as on the rig.
MOUNT_Y = 0.305
PLATE_THICKNESS = 0.022
# The plate is a box under both bases, its front edge at x 0.037 as measured on the rig. A base footprint is x +-0.035,
# y +-0.1 about its mount.
PLATE_SIZE = (0.137, 0.86)
PLATE_CENTER_X = -0.0315
# The robot root in the world frame of the rig recordings: the mounts of ``yam_bimanual_yambox`` minus the plate.
ROOT_IN_RIG_WORLD = (0.30, 0.0, 0.3125 - PLATE_THICKNESS)

# Arm drives: the i2rt gains, the peak motor torques, and assumed velocity limits (the URDF has 1 rad/s placeholders).
# The sim robot has gravity off, as the real arm compensates it.
ARM_STIFFNESS = (80.0, 80.0, 80.0, 10.0, 10.0, 10.0)
ARM_DAMPING = (5.0, 5.0, 5.0, 1.5, 1.5, 1.5)
ARM_EFFORT = (28.0, 28.0, 28.0, 10.0, 10.0, 10.0)
ARM_VELOCITY = (5.0, 5.0, 5.0, 20.0, 20.0, 20.0)
ARM_ARMATURE = (0.032, 0.032, 0.032, 0.0018, 0.0018, 0.0018)
# Finger drive per finger, in N/m, N s/m, N and m/s. The effort limit caps the grasp force of each finger. These values
# are assumed: no measured grasp force of the linear_4310 gripper is available.
FINGER_STIFFNESS = 10000.0
FINGER_DAMPING = 50.0
FINGER_EFFORT = 40.0
FINGER_VELOCITY = 0.2
# The i2rt joint limits.
ARM_LIMITS = (
    (-2.61799, 3.14159),
    (0.0, 3.66519),
    (0.0, 3.14159),
    (-1.69297, 1.5708),
    (-1.5708, 1.5708),
    (-2.0944, 2.0944),
)
# The start pose of the YAM driver: the Menagerie "home" keyframe, folded up and back.
HOME_JOINTS = (0.0, 1.047, 1.047, 0.0, 0.0, 0.0)

# ``GRIPPER_SITE`` in the wrist link, as ``positronic.drivers.roboarm.models`` attaches it to the i2rt model.
GRIPPER_SITE_XYZ = (0.0408, 0.0, -0.1377)
GRIPPER_SITE_RPY = (3.1415926536, 0.0, -1.5707963268)

# Finger collision boxes in the wrist link with the fingers closed, for tip_left. tip_right is the mirror in y.
# The pad face is the plane y = 0 between z -0.146 and -0.078. The carriage is the part below the gripper housing.
FINGER_BOXES = {
    'pad': {'center': (0.0, -0.004, -0.112), 'size': (0.018, 0.008, 0.068)},
    'carriage': {'center': (0.0, -0.0125, -0.078), 'size': (0.068, 0.021, 0.024)},
}
# The point midway between the pads of one hand, in the wrist link.
PAD_MIDDLE_IN_WRIST = (FINGER_BOXES['pad']['center'][0], 0.0, FINGER_BOXES['pad']['center'][2])

# The 14 action values: per arm the six joint targets in rad, then the grip (0 open, 1 closed, linear between).
ACTION_TERMS = tuple(
    term
    for side in SIDES
    for term in (
        (f'{side}_arm', tuple(f'{side}_{j}' for j in ARM_JOINTS)),
        (f'{side}_gripper', (f'{side}_{FINGER_JOINT}',)),
    )
)
ACTION_DIM = sum(len(joints) for _, joints in ACTION_TERMS)

# The proprioception terms of an arm: the joints, the grip (0 open, 1 closed), and the ``<side>_gripper`` link pose in
# the robot root frame. ``proprio_term`` names a term of one arm.
PROPRIO_JOINT_POS = 'arm_joint_pos'
PROPRIO_GRIP = 'gripper_pos'
PROPRIO_EE_POS = 'ee_pos'
PROPRIO_EE_QUAT = 'ee_quat'


def proprio_term(side: str, term: str) -> str:
    return f'{side}_{term}'


def finger_label(side: str, finger: str) -> str:
    return f'{side}_{finger}'


# The RoboLab labels. A contact sensor label must match one prim per env, so each finger has a label of its own. The
# group of a side covers both fingers of that hand, and the group ``CONTACT_GRIPPERS`` covers all four fingers.
CONTACT_GRIPPERS = 'gripper'
CONTACT_GRIPPER = {
    **{
        finger_label(side, finger): f'{{ENV_REGEX_NS}}/robot/{side}_{finger}'
        for side in SIDES
        for finger in FINGER_LINKS
    },
    **{side: [finger_label(side, finger) for finger in FINGER_LINKS] for side in SIDES},
    CONTACT_GRIPPERS: [finger_label(side, finger) for side in SIDES for finger in FINGER_LINKS],
}
EE_RECORDER_BODIES = {f'{side}_ee_pose': f'{side}_{GRIPPER_SITE}' for side in SIDES}
GRIPPER_CLOSURE = {
    side: {'asset': 'robot', 'joint': f'{side}_{FINGER_JOINT}', 'open': FINGER_OPEN, 'closed': FINGER_CLOSED}
    for side in SIDES
}

# The robot stand: a static box under the plate, from the plate down past the floor (robot frame).
STAND_X = (-0.25, 0.037)
STAND_Y = (-0.50, 0.50)
STAND_Z = (-0.75, 0.0)

# The single-arm i2rt model that the two arms copy, as ``positronic.drivers.roboarm.models`` reads it.
I2RT_URDF = Path(__file__).resolve().parents[2] / 'assets' / 'i2rt_yam' / 'yam.urdf'


def mount_xyz(side: str) -> tuple[float, float, float]:
    """The arm base frame of ``side`` in the robot root frame."""
    return (0.0, MOUNT_Y if side == LEFT else -MOUNT_Y, PLATE_THICKNESS)


@dataclass(frozen=True)
class CameraSpec:
    """A pinhole camera at ``pos``, ``rot_wxyz`` in the frame of ``parent`` (a robot link). Axes: x right, y up, look
    along -z."""

    parent: str
    pos: tuple[float, float, float]
    rot_wxyz: tuple[float, float, float, float]
    horizontal_aperture: float
    vertical_aperture: float


# The rig records the cameras at SVGA (960 x 600), rectified, with square pixels.
CAMERA_WIDTH, CAMERA_HEIGHT = 960, 600
CAMERA_FOCAL_LENGTH = 2.8


def _apertures(focal_px: float) -> tuple[float, float]:
    """The horizontal and vertical apertures that give a focal length of ``focal_px`` pixels."""
    return CAMERA_FOCAL_LENGTH * CAMERA_WIDTH / focal_px, CAMERA_FOCAL_LENGTH * CAMERA_HEIGHT / focal_px


# Fitted to recordings of the YAM rig: the top camera to the arm outlines, each wrist camera to its finger outlines.
CAMERAS = {
    keys.TOP_CAMERA: CameraSpec(
        'base_plate', (-0.1274, 0.049, 0.915), (0.658569, 0.25417, -0.22937, -0.670129), *_apertures(364.8)
    ),
    keys.LEFT_WRIST_CAMERA: CameraSpec(
        f'{LEFT}_{WRIST_LINK}', (-0.0624, 0.0035, -0.1163), (0.680659, -0.2334, -0.21422, 0.660559), *_apertures(322.5)
    ),
    keys.RIGHT_WRIST_CAMERA: CameraSpec(
        f'{RIGHT}_{WRIST_LINK}',
        (-0.0636, -0.0044, -0.1153),
        (0.662851, -0.20269, -0.23956, 0.679821),
        *_apertures(323.5),
    ),
}


# ---- the URDF and the stand ----


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def rpy_matrix(rpy: Any) -> np.ndarray:
    """The rotation of a URDF ``rpy``: Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    return _axis_rotation(2, yaw) @ _axis_rotation(1, pitch) @ _axis_rotation(0, roll)


def _fmt(values: Any) -> str:
    return ' '.join(f'{float(v):.10g}' for v in values)


def _origin(parent: ET.Element, xyz: Any, rpy: Any) -> ET.Element:
    return ET.SubElement(parent, 'origin', xyz=_fmt(xyz), rpy=_fmt(rpy))


def _inertial(link: ET.Element, mass: float, inertia: float, xyz: Any = (0.0, 0.0, 0.0)) -> None:
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'mass', value=_fmt([mass]))
    _origin(inertial, xyz, (0.0, 0.0, 0.0))
    value = _fmt([inertia])
    ET.SubElement(inertial, 'inertia', ixx=value, ixy='0', ixz='0', iyy=value, iyz='0', izz=value)


def _box(parent: ET.Element, tag: str, xyz: Any, rpy: Any, size: Any, name: str | None = None) -> ET.Element:
    element = ET.SubElement(parent, tag, {'name': name} if name else {})
    _origin(element, xyz, rpy)
    ET.SubElement(ET.SubElement(element, 'geometry'), 'box', size=_fmt(size))
    return element


def _matrix_rpy(r: np.ndarray) -> tuple[float, float, float]:
    pitch = np.arcsin(-np.clip(r[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-9:
        roll, yaw = np.arctan2(r[2, 1], r[2, 2]), np.arctan2(r[1, 0], r[0, 0])
    else:
        roll, yaw = np.arctan2(-r[1, 2], r[1, 1]), 0.0
    return float(roll), float(pitch), float(yaw)


def _finger_collisions(link: ET.Element, side: str, finger: str, joint_in_wrist: np.ndarray) -> None:
    """The pad and carriage boxes of ``finger`` in its link frame, at finger travel 0."""
    mirror = 1.0 if finger == 'tip_left' else -1.0
    wrist_in_tip = np.linalg.inv(joint_in_wrist)
    for name, box in FINGER_BOXES.items():
        cx, cy, cz = box['center']
        center = wrist_in_tip @ np.array([cx, mirror * cy, cz, 1.0])
        rpy = _matrix_rpy(wrist_in_tip[:3, :3])
        _box(link, 'collision', center[:3], rpy, box['size'], name=f'{side}_{finger}_{name}')


# The links that collide with their visual mesh. The USD import makes each mesh a convex hull.
_MESH_COLLISION_LINKS = ('base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6')
# The mass and inertia of the massless ``gripper`` link, which PhysX cannot take as zero.
_SITE_MASS = 0.001
_SITE_INERTIA = 1e-7


def _prefixed(element: ET.Element, side: str) -> ET.Element:
    """A copy of a link or a joint, with its name prefixed by ``<side>_``."""
    new = copy.deepcopy(element)
    new.set('name', f'{side}_{element.attrib["name"]}')
    return new


def _joint_frame(joint: ET.Element) -> np.ndarray:
    origin = joint.find('origin')
    assert origin is not None  # every joint of the i2rt URDF states its origin
    frame = np.eye(4)
    frame[:3, :3] = rpy_matrix([float(v) for v in origin.attrib['rpy'].split()])
    frame[:3, 3] = [float(v) for v in origin.attrib['xyz'].split()]
    return frame


def _prefixed_arm(source: ET.Element, mesh_dir: Path, side: str) -> list[ET.Element]:
    """The links and joints of one arm, renamed with ``<side>_``, with collisions, limits and the finger mimic."""
    joints = {j.attrib['name']: j for j in source.findall('joint')}
    out = []
    for arm_link in source.findall('link'):
        name = arm_link.attrib['name']
        new = _prefixed(arm_link, side)
        for mesh in new.iter('mesh'):
            mesh.set('filename', str(mesh_dir / mesh.attrib['filename']))
        for material in new.iter('material'):
            material.set('name', f'{side}_{material.attrib["name"]}')
        if name in _MESH_COLLISION_LINKS:
            visual = new.find('visual')
            assert visual is not None  # every arm link of the i2rt URDF has a visual mesh
            collision = ET.SubElement(new, 'collision', name=f'{side}_{name}_hull')
            collision.extend(copy.deepcopy(child) for child in visual if child.tag in ('origin', 'geometry'))
        elif name in FINGER_LINKS:
            joint = FINGER_JOINT if name == 'tip_left' else FINGER_MIMIC_JOINT
            _finger_collisions(new, side, name, _joint_frame(joints[joint]))
        out.append(new)
    for name, joint in joints.items():
        new = _prefixed(joint, side)
        for tag in ('parent', 'child'):
            element = new.find(tag)
            assert element is not None  # every joint of the i2rt URDF names both links
            element.set('link', f'{side}_{element.attrib["link"]}')
        limit = new.find('limit')
        assert limit is not None  # every joint of the i2rt URDF is a moving joint with limits
        if name in ARM_JOINTS:
            i = ARM_JOINTS.index(name)
            lower, upper = ARM_LIMITS[i]
            effort, velocity = ARM_EFFORT[i], ARM_VELOCITY[i]
        else:
            lower, upper, effort, velocity = FINGER_CLOSED, FINGER_OPEN, FINGER_EFFORT, FINGER_VELOCITY
            if name == FINGER_MIMIC_JOINT:
                ET.SubElement(new, 'mimic', joint=f'{side}_{FINGER_JOINT}', multiplier='1', offset='0')
        limit.attrib.update(lower=_fmt([lower]), upper=_fmt([upper]), effort=_fmt([effort]), velocity=_fmt([velocity]))
        out.append(new)
    site = ET.Element('link', name=f'{side}_{GRIPPER_SITE}')
    _inertial(site, _SITE_MASS, _SITE_INERTIA)
    site_joint = ET.Element('joint', name=f'{side}_gripper_site_joint', type='fixed')
    _origin(site_joint, GRIPPER_SITE_XYZ, GRIPPER_SITE_RPY)
    ET.SubElement(site_joint, 'parent', link=f'{side}_{WRIST_LINK}')
    ET.SubElement(site_joint, 'child', link=f'{side}_{GRIPPER_SITE}')
    return [*out, site, site_joint]


def bimanual_urdf(source: Path = I2RT_URDF) -> str:
    """The URDF of the robot: one articulation, whose root link ``base_plate`` carries two copies of the single-arm
    URDF ``source`` on fixed joints at ``mount_xyz``. The meshes stay where ``source`` names them."""
    arm = ET.parse(source).getroot()
    robot = ET.Element('robot', name='yam_bimanual')
    plate = ET.SubElement(robot, 'link', name='base_plate')
    center = (PLATE_CENTER_X, 0.0, PLATE_THICKNESS / 2)
    size = (*PLATE_SIZE, PLATE_THICKNESS)
    _inertial(plate, 2.0, 0.05, xyz=center)
    visual = _box(plate, 'visual', center, (0.0, 0.0, 0.0), size)
    ET.SubElement(ET.SubElement(visual, 'material', name='plate_grey'), 'color', rgba='0.3 0.3 0.32 1')
    _box(plate, 'collision', center, (0.0, 0.0, 0.0), size, name='plate')
    for side in SIDES:
        mount = ET.SubElement(robot, 'joint', name=f'{side}_mount', type='fixed')
        _origin(mount, mount_xyz(side), (0.0, 0.0, 0.0))
        ET.SubElement(mount, 'parent', link='base_plate')
        ET.SubElement(mount, 'child', link=f'{side}_base')
        robot.extend(_prefixed_arm(arm, source.parent, side))
    ET.indent(robot, space='  ')
    return '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding='unicode') + '\n'


def stand_usda() -> str:
    """The robot stand as a USD layer: a static box with a collider, in the robot frame."""
    (x0, x1), (y0, y1), (z0, z1) = STAND_X, STAND_Y, STAND_Z
    center = ', '.join(_fmt([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]).split())
    scale = ', '.join(_fmt([x1 - x0, y1 - y0, z1 - z0]).split())
    return f"""#usda 1.0
(
    defaultPrim = "yam_stand"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "yam_stand"
{{
    def Cube "stand" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        double size = 1
        color3f[] primvars:displayColor = [(0.35, 0.35, 0.38)]
        double3 xformOp:translate = ({center})
        float3 xformOp:scale = ({scale})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
}}
"""


# ---- the kinematics ----


_MOVING_JOINTS = ('revolute', 'prismatic', 'continuous')


class _Joint(NamedTuple):
    """A moving joint of a chain, and the fixed transform from the joint before it, or the root link, to its frame."""

    name: str
    prismatic: bool
    rot: np.ndarray
    pos: np.ndarray
    axis: np.ndarray  # a unit vector in the joint frame
    # The cross-product matrix of ``axis`` and its square: a turn by ``a`` is ``I + sin(a) skew + (1 - cos(a)) skew2``
    skew: np.ndarray
    skew2: np.ndarray


def _values(element: ET.Element | None, name: str, default: str) -> np.ndarray:
    text = default if element is None else element.get(name, default)
    return np.array([float(value) for value in text.split()])


class UrdfChain:
    """The joints of a URDF from the root link to ``link``, and the kinematics of ``link`` in the root link's frame.

    ``joints`` names the moving joints in order, and ``lower`` and ``upper`` hold their limits; a joint with no limit
    has an infinite one.
    """

    def __init__(self, urdf: ET.Element, link: str):
        by_child = {}
        for joint in urdf.findall('joint'):
            child = joint.find('child')
            assert child is not None  # every joint of a URDF names its child link
            by_child[child.get('link')] = joint
        elements = []
        while link in by_child:
            elements.append(by_child[link])
            parent = by_child[link].find('parent')
            assert parent is not None  # every joint of a URDF names its parent link
            link = parent.get('link')
        self._joints: list[_Joint] = []
        limits = []
        rot, pos = np.eye(3), np.zeros(3)  # the fixed transform since the last moving joint
        for element in reversed(elements):
            origin = element.find('origin')
            pos = pos + rot @ _values(origin, 'xyz', '0 0 0')
            rot = rot @ rpy_matrix(_values(origin, 'rpy', '0 0 0'))
            kind = element.get('type')
            if kind not in _MOVING_JOINTS:
                continue
            axis = _values(element.find('axis'), 'xyz', '1 0 0')
            x, y, z = axis = axis / np.linalg.norm(axis)
            skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            self._joints.append(
                _Joint(str(element.get('name')), kind == 'prismatic', rot, pos, axis, skew, skew @ skew)
            )
            limit = element.find('limit')
            bounded = kind != 'continuous' and limit is not None
            limits.append(
                (float(limit.get('lower', 0)), float(limit.get('upper', 0)))
                if bounded and limit is not None
                else (-math.inf, math.inf)
            )
            rot, pos = np.eye(3), np.zeros(3)
        self._tip_rot, self._tip_pos = rot, pos
        self.joints = tuple(joint.name for joint in self._joints)
        self.lower, self.upper = np.array(limits, dtype=np.float64).reshape(-1, 2).T

    def kinematics(self, q: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The position and the rotation matrix of the link at the joints ``q``, and its ``6 x n`` Jacobian."""
        rot, pos = np.eye(3), np.zeros(3)
        points, axes = [], []
        for joint, value in zip(self._joints, np.asarray(q, dtype=np.float64), strict=True):
            pos = pos + rot @ joint.pos
            rot = rot @ joint.rot
            points.append(pos)
            axes.append(rot @ joint.axis)
            if joint.prismatic:
                pos = pos + axes[-1] * value
            else:
                rot = rot @ (np.eye(3) + math.sin(value) * joint.skew + (1 - math.cos(value)) * joint.skew2)
        pos = pos + rot @ self._tip_pos
        rot = rot @ self._tip_rot
        jac = np.zeros((6, len(axes)))
        for column, (joint, point, axis) in enumerate(zip(self._joints, points, axes, strict=True)):
            if joint.prismatic:
                jac[:3, column] = axis
            else:
                lever = pos - point
                jac[:3, column] = [
                    axis[1] * lever[2] - axis[2] * lever[1],
                    axis[2] * lever[0] - axis[0] * lever[2],
                    axis[0] * lever[1] - axis[1] * lever[0],
                ]
                jac[3:, column] = axis
        return pos, rot, jac


@functools.cache
def arm_chains() -> dict[str, UrdfChain]:
    """The chain of each arm of ``bimanual_urdf``, from the root link ``base_plate`` to ``<side>_gripper``."""
    urdf = ET.fromstring(bimanual_urdf())
    return {side: UrdfChain(urdf, f'{side}_{GRIPPER_SITE}') for side in SIDES}


# ---- the action ----


# The root in the rig world frame. The root does not turn in it, so a rotation is the same in the two frames.
_ROOT = np.asarray(ROOT_IN_RIG_WORLD, dtype=np.float64)


def _solve_pose(
    chain: UrdfChain, measured_q: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray
) -> np.ndarray | None:
    """The joints of ``chain`` at a pose of its site in the rig world frame, or a near solution (``dls_ik.solve``);
    None when the solve misses."""
    solution = dls_ik.solve(
        chain.kinematics,
        measured_q=measured_q,
        lower=chain.lower,
        upper=chain.upper,
        target_pos=np.asarray(target_pos, dtype=np.float64) - _ROOT,
        target_rot=target_rot,
    )
    return None if solution.fit is dls_ik.Fit.MISS else solution.q


def _site_pose(chain: UrdfChain, measured_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pose of the site of ``chain`` in the rig world frame, at the measured joints."""
    pos, rot, _ = chain.kinematics(measured_q)
    return pos + _ROOT, rot


class Action(NamedTuple):
    """One clone's action: the 14 values in ``ACTION_TERMS`` order, and the grip of each arm."""

    values: np.ndarray
    grips: np.ndarray


def action(wire: dict[str, Any], measured: np.ndarray, grips: np.ndarray) -> Action:
    """One clone's action, from the clone's wire action.

    ``measured`` holds the measured joints of each arm in ``SIDES`` order, ``(2, 6)``. ``grips`` holds the last grip
    of each arm, which an arm keeps when its wire grip is not finite. A wire grip is in [0, 1], 1 closed, and goes in as
    it is. A pose solves to the joints of its arm at ``<side>_gripper`` (``dls_ik.solve``), in the rig world frame.
    """
    channels = {
        protocol.arm_channel(name, side) for side in SIDES for name in (protocol.ROBOT_COMMAND, protocol.TARGET_GRIP)
    }
    extra = set(wire) - channels
    if extra:
        raise ValueError(f'the RoboLab YAM robot cannot act on channels {sorted(extra)}')
    chains = arm_chains()
    values, new_grips = [], []
    for i, side in enumerate(SIDES):
        grip = float(wire[protocol.arm_channel(protocol.TARGET_GRIP, side)])
        grip = grip if math.isfinite(grip) else float(grips[i])
        joints, chain = np.asarray(measured[i], dtype=np.float64), chains[side]
        targets = arm_action.arm_targets(
            wire[protocol.arm_channel(protocol.ROBOT_COMMAND, side)],
            joints,
            ik=functools.partial(_solve_pose, chain, joints),
            read_eef=functools.partial(_site_pose, chain, joints),
            arm=side,
        ).q
        values += [targets, [grip]]
        new_grips.append(grip)
    return Action(np.concatenate(values).astype(np.float32), np.asarray(new_grips, dtype=np.float32))
