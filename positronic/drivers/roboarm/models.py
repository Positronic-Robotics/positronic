"""Self-contained robot models (URDF + collision meshes + gripper spec) for the 3D
viewer, the sim ``robot_meta``, and offline IK, plus the Robotiq 2F-85 graft shared by the bundled
real arm and the live franka driver."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, NamedTuple

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer, TypeAdapter

from positronic import geom, keys
from positronic.drivers.roboarm import keys as roboarm_keys

FLANGE_LINK = 'link8'
ROBOLAB_EEF_LINK = 'robolab_eef'
# The point a bundled model reported at before ``GRIPPER_SITE``, as a URDF link: the FR3's ``F_T_EE``, the old grasp
# site of the sim panda, the YAM's i2rt grasp site.
EE_LINK = 'end_effector'
# The grip point every bundled model names, on the axes of the arm's last link: z is the approach axis. Every
# embodiment reports its poses here, and every codec transform is measured from here.
GRIPPER_SITE = 'gripper'
# The frame DROID records its poses in: the flange, ``panda_link8``, measured from ``GRIPPER_SITE`` of the
# FR3 with the 2F-85. DROID's ``cartesian_position`` equals FK at ``panda_link8``, and its checkpoints speak it.
DROID_EE_FRAME = geom.Transform3D(np.array([0.0, 0.0, -0.155]))
# ``ROBOLAB_EEF_LINK`` measured from ``GRIPPER_SITE``: RoboLab's ``eef_frame``, at the Robotiq 2F-85 coupler,
# 18.17mm along the flange approach axis and +90deg about it. RoboLab measures and drives its arm there.
ROBOLAB_EE_FRAME = geom.Transform3D(
    np.array([0.0, 0.0, -0.13682597739]), geom.Rotation.from_euler([0.0, 0.0, 1.5707963268])
)
# The FR3's Franka Hand point, ``F_T_EE`` under the Franka Hand setting in Desk, measured from ``GRIPPER_SITE``.
# ``F_T_EE`` is 103.4mm along the flange approach axis and -45deg about it. Real-Franka recordings and checkpoints
# older than ``GRIPPER_SITE`` speak poses there.
FRANKA_HAND_FRAME = geom.Transform3D(
    np.array([0.0, 0.0, -0.0516000006]), geom.Rotation.from_euler([0.0, 0.0, -0.7853981634])
)
# The hand point of the MuJoCo panda, its ``EE_LINK``, measured from ``GRIPPER_SITE``. ``EE_LINK`` is 58.4mm along
# the flange approach axis and -45deg about it. Sim recordings and checkpoints older than ``GRIPPER_SITE`` speak
# poses there.
PANDA_HAND_FRAME = geom.Transform3D(np.array([0.0, 0.0, -0.04525]), geom.Rotation.from_euler([0.0, 0.0, -0.7853981634]))
# The i2rt grasp site of the YAM, ``grasp_site`` in its MJCF, measured from ``GRIPPER_SITE``. ``grasp_site`` is on the
# wrist axis, 134.7mm along it and -90deg about it. YAM recordings and checkpoints older than ``GRIPPER_SITE`` speak
# poses there.
YAM_GRASP_SITE_FRAME = geom.Transform3D(
    np.array([0.0, 0.0408, -0.0033]), geom.Rotation.from_euler([0.0, 0.0, -1.5707963268])
)
_QUAT = geom.Rotation.Representation.QUAT


def _transform_from_vector(value: Any) -> Any:
    if isinstance(value, geom.Transform3D):
        return value
    return geom.Transform3D.from_vector(np.asarray(value, dtype=np.float64), _QUAT)


# A ``[tx,ty,tz,qw,qx,qy,qz]`` vector in a dump.
Offset = Annotated[
    geom.Transform3D,
    BeforeValidator(_transform_from_vector),
    PlainSerializer(lambda transform: transform.as_vector(_QUAT).tolist(), return_type=list[float]),
]


class EEFrame(BaseModel):
    """Where a pose signal sits: ``offset`` from ``site``, a site of the robot model."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra='forbid')

    site: str
    offset: Offset = geom.Transform3D.identity


_EE_FRAMES_SCHEMA = TypeAdapter(dict[str, EEFrame])


def ee_frames_entry(frames: Mapping[str, EEFrame]) -> dict[str, Any]:
    """The ``EE_FRAMES`` entry of a meta or statics dict that states ``frames``."""
    return {roboarm_keys.EE_FRAMES: _EE_FRAMES_SCHEMA.dump_python(dict(frames), mode='json')}


def parse_ee_frames(entry: Any) -> dict[str, EEFrame]:
    """The ``EEFrame`` per pose signal that an ``EE_FRAMES`` entry states."""
    return _EE_FRAMES_SCHEMA.validate_python(entry)


# Seat the gripper on the flange, rotated about the approach axis to match the real 2F-85 coupler
# (a +45deg Z, i.e. 90deg off the franka ``end_effector`` frame).
_2F85_MOUNT_RPY = '0 0 0.7853981634'


class _UrdfRow(NamedTuple):
    """One link, and the joint that attaches it to its parent.

    ``axis`` names a revolute joint and its sign sets the closing direction; without one the joint is
    fixed. ``mesh`` absent makes the link a pure frame, carrying no visual.
    """

    link: str
    parent: str
    joint: str | None
    xyz: str
    rpy: str
    axis: str | None
    mesh: str | None
    visual_xyz: str | None


def _2f85_finger(side: str, sign: int, base_rpy: str) -> list[_UrdfRow]:
    """One 2F-85 finger as URDF rows. ``sign`` mirrors the y-offsets and ``base_rpy`` (180deg Z on the
    left) mirrors the motion, so one positive ``grip`` closes both fingers. The follower rotates about
    an anchor offset from its body, carried as the follower mesh's visual origin."""
    return [
        _UrdfRow(
            f'{side}_driver',
            'gripper_base',
            f'{side}_driver_joint',
            f'0 {sign * 0.0306011} 0.054904',
            base_rpy,
            '1 0 0',
            'driver.stl',
            None,
        ),
        _UrdfRow(
            f'{side}_coupler',
            f'{side}_driver',
            f'{side}_coupler_joint',
            '0 0.0315 -0.0041',
            '0 0 0',
            None,
            'coupler.stl',
            None,
        ),
        _UrdfRow(
            f'{side}_spring_link',
            'gripper_base',
            f'{side}_spring_link_joint',
            f'0 {sign * 0.0132} 0.0609',
            base_rpy,
            '1 0 0',
            'spring_link.stl',
            None,
        ),
        _UrdfRow(
            f'{side}_follower',
            f'{side}_spring_link',
            f'{side}_follower_joint',
            '0 0.037 0.044',
            '0 0 0',
            '-1 0 0',
            'follower.stl',
            '0 0.018 -0.0065',
        ),
        _UrdfRow(f'{side}_pad', f'{side}_follower', None, '0 -0.0009 0.00702', '0 0 0', None, 'pad.stl', None),
        _UrdfRow(f'{side}_silicone_pad', f'{side}_pad', None, '0 0 0', '0 0 0', None, 'silicone_pad.stl', None),
    ]


# The 2F-85's closed pads meet 155 mm along the flange approach axis.
GRIPPER_2F85_XYZ = '0 0 0.155'

# The coupler stays fixed: given an axis, the outer link hangs 19 mm out at full grip.
_ROBOTIQ_2F85 = [
    _UrdfRow('gripper_base_mount', FLANGE_LINK, None, '0 0 0.007', _2F85_MOUNT_RPY, None, 'base_mount.stl', None),
    _UrdfRow('gripper_base', 'gripper_base_mount', None, '0 0 0.0038', '0 0 -1.5707963268', None, 'base.stl', None),
    _UrdfRow(GRIPPER_SITE, FLANGE_LINK, None, GRIPPER_2F85_XYZ, '0 0 0', None, None, None),
    *_2f85_finger('right', 1, '0 0 0'),
    *_2f85_finger('left', -1, '0 0 3.1415926536'),
    # RoboLab's ``eef_frame`` (``Robotiq_2F_85/base_link`` ∘ ``EEF_OFFSET_ROT``), measured off its DROID USD
    # ``franka_robotiq_2f_85_flattened.usd`` as 18.17mm along the flange Z and +90deg about it.
    _UrdfRow(ROBOLAB_EEF_LINK, FLANGE_LINK, None, '0 0 0.01817402261', '0 0 1.5707963268', None, None, None),
]


def _build_2f85_elements() -> list[ET.Element]:
    elements = []
    for row in _ROBOTIQ_2F85:
        link_el = ET.Element('link', name=row.link)
        # A nominal inertial keeps each link a valid MuJoCo moving body: ik._prepare_spec strips the
        # visuals and compiles the URDF, which rejects zero-inertia bodies. IK is kinematic and the
        # viewer ignores inertials, so the magnitude is arbitrary.
        inertial = ET.SubElement(link_el, 'inertial')
        ET.SubElement(inertial, 'mass', value='0.01')
        ET.SubElement(inertial, 'inertia', ixx='1e-5', iyy='1e-5', izz='1e-5', ixy='0', ixz='0', iyz='0')
        if row.mesh is not None:
            visual = ET.SubElement(link_el, 'visual')
            if row.visual_xyz is not None:
                ET.SubElement(visual, 'origin', xyz=row.visual_xyz, rpy='0 0 0')
            ET.SubElement(ET.SubElement(visual, 'geometry'), 'mesh', filename=row.mesh)
        joint_name = row.joint or f'{row.link}_fixed'
        joint_el = ET.Element('joint', name=joint_name, type='revolute' if row.axis else 'fixed')
        ET.SubElement(joint_el, 'origin', xyz=row.xyz, rpy=row.rpy)
        ET.SubElement(joint_el, 'parent', link=row.parent)
        ET.SubElement(joint_el, 'child', link=row.link)
        if row.axis is not None:
            ET.SubElement(joint_el, 'axis', xyz=row.axis)
            ET.SubElement(joint_el, 'limit', effort='10', velocity='2', lower='-1.6', upper='0.9')
        elements += [link_el, joint_el]
    return elements


# A fixed joint keeps its rest pose, so ``grip`` does not drive it.
_2F85_GRIP_ACTUATED_JOINTS = [row.joint for row in _ROBOTIQ_2F85 if row.joint and row.axis]


@lru_cache(maxsize=1)
def _bundled_robotiq_2f85() -> dict:
    """The Robotiq 2F-85 gripper for the 3D viewer, converted from MuJoCo Menagerie ``robotiq_2f85``
    (meshes scaled to metres): a URDF subtree that mounts on the franka flange (``link8``), its visual
    meshes, and the ``grip``-driven joint spec. The recorded ``grip`` is 1 at closed, 0 at open."""
    mesh_dir = Path(__file__).resolve().parents[2] / 'assets' / 'robotiq_2f85'
    subtree = ''.join(ET.tostring(el, encoding='unicode') for el in _build_2f85_elements())
    return {
        'subtree': subtree,
        roboarm_keys.MESHES: {f.name: f.read_bytes() for f in sorted(mesh_dir.glob('*.stl'))},
        roboarm_keys.GRIPPER: {'signal': keys.GRIP, 'joints': _2F85_GRIP_ACTUATED_JOINTS, 'travel': 0.8},
    }


def attach_frame(arm_root: ET.Element, name: str, parent: str, xyz: str = '0 0 0', rpy: str = '0 0 0') -> None:
    """Add a link ``name`` to a URDF in place, fixed at ``xyz`` and ``rpy`` in the ``parent`` link."""
    ET.SubElement(arm_root, 'link', name=name)
    joint = ET.SubElement(arm_root, 'joint', name=f'{parent}_to_{name}', type='fixed')
    ET.SubElement(joint, 'origin', xyz=xyz, rpy=rpy)
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=name)


def attach_robotiq_2f85(arm_root: ET.Element, meshes: dict[str, bytes]) -> dict:
    """Mount the 2F-85 on a franka arm URDF in place: append its links/joints under the flange and
    merge its meshes into ``meshes``. Returns the ``grip``-driven gripper spec for the viewer."""
    gripper = _bundled_robotiq_2f85()
    arm_root.extend(ET.fromstring(f'<robot>{gripper["subtree"]}</robot>'))
    meshes.update(gripper[roboarm_keys.MESHES])
    return gripper[roboarm_keys.GRIPPER]


@lru_cache(maxsize=1)
def bundled_franka_model() -> dict:
    """FR3 and Robotiq 2F-85 model metadata."""
    here = Path(__file__).resolve()
    arm_root = ET.fromstring((here.parent / 'fr3.urdf').read_text())
    mesh_dir = here.parents[2] / 'assets' / 'fr3_collision'
    meshes = {f.name: f.read_bytes() for f in sorted(mesh_dir.glob('*.stl'))}
    gripper = attach_robotiq_2f85(arm_root, meshes)
    return {
        roboarm_keys.URDF: ET.tostring(arm_root, encoding='unicode'),
        roboarm_keys.MESHES: meshes,
        roboarm_keys.JOINT_NAMES: [f'joint{i}' for i in range(1, 8)],
        roboarm_keys.GRIPPER: gripper,
    }


YAM_JOINT_NAMES = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
YAM_WRIST_LINK = 'link_6'
# The centroid of the face where the closed YAM fingertips meet, in the i2rt CAD mesh of ``YAM_WRIST_LINK``.
GRIPPER_YAM_XYZ = '0 -0.0408 0.138'
# The YAM MJCF's ``grasp_site``: 134.7mm along the wrist axis, turned -90deg about it.
_YAM_GRASP_SITE_XYZ, _YAM_GRASP_SITE_RPY = '0 0 0.1347', '0 0 -1.5707963268'


@lru_cache(maxsize=1)
def bundled_yam_model() -> dict:
    """i2rt YAM model metadata: its URDF and visual meshes, and its joint names."""
    urdf_path = Path(__file__).resolve().parents[2] / 'assets' / 'i2rt_yam' / 'yam.urdf'
    root = ET.fromstring(urdf_path.read_text())
    mesh_files = {mesh.attrib['filename'] for mesh in root.iter('mesh')}
    attach_frame(root, GRIPPER_SITE, YAM_WRIST_LINK, GRIPPER_YAM_XYZ)
    attach_frame(root, EE_LINK, YAM_WRIST_LINK, _YAM_GRASP_SITE_XYZ, _YAM_GRASP_SITE_RPY)
    return {
        'robot': 'i2rt_yam',
        roboarm_keys.URDF: ET.tostring(root, encoding='unicode'),
        roboarm_keys.MESHES: {name: (urdf_path.parent / name).read_bytes() for name in sorted(mesh_files)},
        roboarm_keys.JOINT_NAMES: list(YAM_JOINT_NAMES),
    }


@lru_cache(maxsize=1)
def bundled_panda_model() -> dict:
    """The bundled simulated Franka panda (arm + hand) for the 3D viewer and offline IK: the panda
    URDF, its collision meshes, the joint names, and the ``gripper`` spec that slides the fingers from the recorded
    ``grip`` signal. Supplied to sim datasets by ``SIM_ROBOT_TRANSFORM``, mirroring how
    ``bundled_franka_model`` backfills the real arm.
    """
    urdf_path = Path(__file__).resolve().parents[2] / 'assets' / 'mujoco' / 'panda.urdf'
    root = ET.fromstring(urdf_path.read_text())
    mesh_dir = urdf_path.parent / 'assets'
    mesh_files = {mesh.attrib['filename'] for mesh in root.iter('mesh')}
    return {
        roboarm_keys.URDF: ET.tostring(root, encoding='unicode'),
        roboarm_keys.MESHES: {name: (mesh_dir / name).read_bytes() for name in sorted(mesh_files)},
        roboarm_keys.JOINT_NAMES: [f'joint{i}' for i in range(1, 8)],
        # ``grip`` is recorded in [0, 1] (open→closed); each finger slides 0..0.04 m along its axis.
        roboarm_keys.GRIPPER: {'signal': keys.GRIP, 'joints': ['finger_joint1', 'finger_joint2'], 'travel': 0.04},
    }
