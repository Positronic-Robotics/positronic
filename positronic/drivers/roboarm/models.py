"""Self-contained robot models (URDF + collision meshes + control frame + gripper spec) for the 3D
viewer, the sim ``robot_meta``, and offline IK, plus the Robotiq 2F-85 graft shared by the bundled
real arm and the live franka driver."""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

from positronic import geom, keys
from positronic.drivers.roboarm import keys as roboarm_keys

FLANGE_LINK = 'link8'
DROID_EEF_LINK = 'droid_eef'
# The franka's own tool frame, ``F_T_EE`` off the flange. TODO(#550): move ``DEFAULT_FRAME`` to the flange.
EE_LINK = 'end_effector'
# The frame every embodiment publishes its poses in and every codec transform is measured from. Recommended at
# the flange whenever the gripper can be swapped: recorded data outlives grippers, and an episode anchored to a
# gripper frame becomes unplaceable once that gripper comes off.
DEFAULT_FRAME = 'default'
# ``DROID_EEF_LINK`` measured from ``DEFAULT_FRAME``: the Robotiq 2F-85 coupler, 18.17mm along the flange
# approach axis and +90deg about it. DROID's checkpoints express poses here, and stating it as a transform is
# what lets one declare its frame without any rig consulting a robot model to serve it.
# TODO(#550): the awkward numbers are the FR3's ``F_T_EE`` offset riding along, because ``DEFAULT_FRAME`` is not
# the flange yet. Moving it there turns this into the authored gripper geometry and makes the value portable —
# as it stands it is right on a franka rig and 45mm off on the sim, whose ``DEFAULT_FRAME`` sits elsewhere again.
DROID_EE_FRAME = geom.Transform3D(np.array([0.0, 0.0, -0.085225977]), geom.Rotation.from_euler([0.0, 0.0, 2.35619449]))
# Seat the gripper on the flange, rotated about the approach axis to match the real 2F-85 coupler
# (a +45deg Z, i.e. 90deg off the franka ``end_effector`` frame).
_2F85_MOUNT_RPY = '0 0 0.7853981634'


def _2f85_finger(side: str, sign: int, base_rpy: str) -> list[tuple]:
    """One 2F-85 finger as URDF rows. ``sign`` mirrors the y-offsets and ``base_rpy`` (180deg Z on the
    left) mirrors the motion, so one positive ``grip`` closes both fingers. The follower rotates about
    an anchor offset from its body, carried as the follower mesh's visual origin."""
    return [
        (
            f'{side}_driver',
            'gripper_base',
            f'{side}_driver_joint',
            f'0 {sign * 0.0306011} 0.054904',
            base_rpy,
            '1 0 0',
            'driver.stl',
            None,
        ),
        (
            f'{side}_coupler',
            f'{side}_driver',
            f'{side}_coupler_joint',
            '0 0.0315 -0.0041',
            '0 0 0',
            '-1 0 0',
            'coupler.stl',
            None,
        ),
        (
            f'{side}_spring_link',
            'gripper_base',
            f'{side}_spring_link_joint',
            f'0 {sign * 0.0132} 0.0609',
            base_rpy,
            '1 0 0',
            'spring_link.stl',
            None,
        ),
        (
            f'{side}_follower',
            f'{side}_spring_link',
            f'{side}_follower_joint',
            '0 0.037 0.044',
            '0 0 0',
            '-1 0 0',
            'follower.stl',
            '0 0.018 -0.0065',
        ),
        (f'{side}_pad', f'{side}_follower', None, '0 -0.0009 0.00702', '0 0 0', None, 'pad.stl', None),
        (f'{side}_silicone_pad', f'{side}_pad', None, '0 0 0', '0 0 0', None, 'silicone_pad.stl', None),
    ]


# Rows: (link, parent, joint | None, origin xyz, origin rpy, axis | None, mesh | None, visual xyz | None).
# A row with an axis is a revolute joint whose axis sign sets its closing direction, so one positive
# ``grip`` drives the whole 4-bar: driver/spring_link swing the finger in (+X), coupler/follower
# counter-rotate (-X) to keep the pad parallel. Rows without an axis are fixed; rows without a mesh are pure
# frames, carrying no visual.
_ROBOTIQ_2F85 = [
    ('gripper_base_mount', FLANGE_LINK, None, '0 0 0.007', _2F85_MOUNT_RPY, None, 'base_mount.stl', None),
    ('gripper_base', 'gripper_base_mount', None, '0 0 0.0038', '0 0 -1.5707963268', None, 'base.stl', None),
    *_2f85_finger('right', 1, '0 0 0'),
    *_2f85_finger('left', -1, '0 0 3.1415926536'),
    # RoboLab's ``eef_frame`` (``Robotiq_2F_85/base_link`` ∘ ``EEF_OFFSET_ROT``), measured off its DROID USD
    # ``franka_robotiq_2f_85_flattened.usd`` as 18.17mm along the flange Z and +90deg about it.
    (DROID_EEF_LINK, FLANGE_LINK, None, '0 0 0.01817402261', '0 0 1.5707963268', None, None, None),
]
_ROBOTIQ_2F85_JOINTS = [row[2] for row in _ROBOTIQ_2F85 if row[2]]


def _build_2f85_elements() -> list[ET.Element]:
    elements = []
    for link, parent, joint, xyz, rpy, axis, mesh, visual_xyz in _ROBOTIQ_2F85:
        link_el = ET.Element('link', name=link)
        # A nominal inertial keeps each link a valid MuJoCo moving body: ik._prepare_spec strips the
        # visuals and compiles the URDF, which rejects zero-inertia bodies. IK is kinematic and the
        # viewer ignores inertials, so the magnitude is arbitrary.
        inertial = ET.SubElement(link_el, 'inertial')
        ET.SubElement(inertial, 'mass', value='0.01')
        ET.SubElement(inertial, 'inertia', ixx='1e-5', iyy='1e-5', izz='1e-5', ixy='0', ixz='0', iyz='0')
        if mesh is not None:
            visual = ET.SubElement(link_el, 'visual')
            if visual_xyz is not None:
                ET.SubElement(visual, 'origin', xyz=visual_xyz, rpy='0 0 0')
            ET.SubElement(ET.SubElement(visual, 'geometry'), 'mesh', filename=mesh)
        joint_el = ET.Element('joint', name=joint or f'{link}_fixed', type='revolute' if axis else 'fixed')
        ET.SubElement(joint_el, 'origin', xyz=xyz, rpy=rpy)
        ET.SubElement(joint_el, 'parent', link=parent)
        ET.SubElement(joint_el, 'child', link=link)
        if axis is not None:
            ET.SubElement(joint_el, 'axis', xyz=axis)
            ET.SubElement(joint_el, 'limit', effort='10', velocity='2', lower='-1.6', upper='0.9')
        elements += [link_el, joint_el]
    return elements


@lru_cache(maxsize=1)
def _bundled_robotiq_2f85() -> dict:
    """The Robotiq 2F-85 gripper for the 3D viewer, converted from MuJoCo Menagerie ``robotiq_2f85``
    (meshes scaled to metres): a URDF subtree that mounts on the franka flange (``link8``), its visual
    meshes, and the ``grip``-driven joint spec. The recorded ``grip`` is 1 at closed, 0 at open."""
    mesh_dir = Path(__file__).resolve().parents[2] / 'assets' / 'robotiq_2f85'
    subtree = ''.join(ET.tostring(el, encoding='unicode') for el in _build_2f85_elements())
    return {
        'subtree': subtree,
        'meshes': {f.name: f.read_bytes() for f in sorted(mesh_dir.glob('*.stl'))},
        'gripper': {'signal': keys.GRIP, 'joints': _ROBOTIQ_2F85_JOINTS, 'travel': 0.8},
    }


def add_default_frame(arm_root: ET.Element, parent: str) -> None:
    """Declare ``DEFAULT_FRAME`` on a URDF in place, coincident with the ``parent`` link."""
    ET.SubElement(arm_root, 'link', name=DEFAULT_FRAME)
    joint = ET.SubElement(arm_root, 'joint', name=f'{parent}_to_{DEFAULT_FRAME}', type='fixed')
    ET.SubElement(joint, 'origin', xyz='0 0 0', rpy='0 0 0')
    ET.SubElement(joint, 'parent', link=parent)
    ET.SubElement(joint, 'child', link=DEFAULT_FRAME)


def attach_robotiq_2f85(arm_root: ET.Element, meshes: dict[str, bytes]) -> dict:
    """Mount the 2F-85 on a franka arm URDF in place: append its links/joints under the flange and
    merge its meshes into ``meshes``. Returns the ``grip``-driven gripper spec for the viewer."""
    gripper = _bundled_robotiq_2f85()
    arm_root.extend(ET.fromstring(f'<robot>{gripper["subtree"]}</robot>'))
    meshes.update(gripper['meshes'])
    return gripper['gripper']


@lru_cache(maxsize=1)
def bundled_franka_model() -> dict:
    """The bundled real franka arm + Robotiq 2F-85 for the 3D viewer: the FR3 URDF and its collision
    meshes with the 2F-85 grafted onto the flange, plus the canonical joint names and control frame.

    Backfills real-robot datasets recorded before they stored their own model.
    """
    here = Path(__file__).resolve()
    arm_root = ET.fromstring((here.parent / 'fr3.urdf').read_text())
    mesh_dir = here.parents[2] / 'assets' / 'fr3_collision'
    meshes = {f.name: f.read_bytes() for f in sorted(mesh_dir.glob('*.stl'))}
    gripper = attach_robotiq_2f85(arm_root, meshes)
    add_default_frame(arm_root, EE_LINK)
    return {
        roboarm_keys.URDF: ET.tostring(arm_root, encoding='unicode'),
        'meshes': meshes,
        roboarm_keys.JOINT_NAMES: [f'joint{i}' for i in range(1, 8)],
        roboarm_keys.CONTROL_FRAME: DEFAULT_FRAME,
        'gripper': gripper,
    }


@lru_cache(maxsize=1)
def bundled_panda_model() -> dict:
    """The bundled simulated Franka panda (arm + hand) for the 3D viewer and offline IK: the panda
    URDF, its collision meshes, the joint names, the ``DEFAULT_FRAME`` the sim measures
    ``robot_state.ee_pose`` at, and the ``gripper`` spec that slides the fingers from the recorded
    ``grip`` signal. Supplied to sim datasets by ``SIM_ROBOT_TRANSFORM``, mirroring how
    ``bundled_franka_model`` backfills the real arm.
    """
    urdf_path = Path(__file__).resolve().parents[2] / 'assets' / 'mujoco' / 'panda.urdf'
    root = ET.fromstring(urdf_path.read_text())
    mesh_dir = urdf_path.parent / 'assets'
    mesh_files = {mesh.get('filename') for mesh in root.iter('mesh')}
    add_default_frame(root, EE_LINK)
    return {
        roboarm_keys.URDF: ET.tostring(root, encoding='unicode'),
        'meshes': {name: (mesh_dir / name).read_bytes() for name in sorted(mesh_files)},
        roboarm_keys.JOINT_NAMES: [f'joint{i}' for i in range(1, 8)],
        roboarm_keys.CONTROL_FRAME: DEFAULT_FRAME,
        # ``grip`` is recorded in [0, 1] (open→closed); each finger slides 0..0.04 m along its axis.
        'gripper': {'signal': keys.GRIP, 'joints': ['finger_joint1', 'finger_joint2'], 'travel': 0.04},
    }
