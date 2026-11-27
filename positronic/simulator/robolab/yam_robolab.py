"""The Isaac Lab and RoboLab configs of the two-arm YAM robot, made from ``yam_spec``.

Import it only after the Isaac app starts. It runs in RoboLab's interpreter and imports nothing from positronic.
"""

import copy
from pathlib import Path
from typing import Any

# Isaac Lab, its USD libraries and robolab resolve from RoboLab's own uv project at run time, so the checker cannot see
# these modules.
import isaaclab.envs.mdp as mdp  # pyright: ignore[reportMissingImports]
import isaaclab.sim as sim_utils  # pyright: ignore[reportMissingImports]
import yam_spec as spec
from isaaclab.actuators import ImplicitActuatorCfg  # pyright: ignore[reportMissingImports]
from isaaclab.assets import ArticulationCfg  # pyright: ignore[reportMissingImports]
from isaaclab.envs import ManagerBasedRLEnv  # pyright: ignore[reportMissingImports]
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # pyright: ignore[reportMissingImports]
from isaaclab.managers import ObservationTermCfg as ObsTerm  # pyright: ignore[reportMissingImports]
from isaaclab.sensors import TiledCameraCfg  # pyright: ignore[reportMissingImports]
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # pyright: ignore[reportMissingImports]
from isaaclab.utils import configclass  # pyright: ignore[reportMissingImports]
from isaaclab.utils.math import subtract_frame_transforms  # pyright: ignore[reportMissingImports]
from pxr import Sdf, Usd, UsdPhysics  # pyright: ignore[reportMissingImports]

from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR  # pyright: ignore[reportMissingImports]
from robolab.core.environments.factory import auto_discover_and_create_cfgs  # pyright: ignore[reportMissingImports]
from robolab.core.environments.scene_fixture import TableFixtureCfg  # pyright: ignore[reportMissingImports]
from robolab.core.observations.observation_utils import (  # pyright: ignore[reportMissingImports]
    generate_image_obs_from_cameras,
    generate_obs_cfg,
)
from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz  # pyright: ignore[reportMissingImports]
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg  # pyright: ignore[reportMissingImports]
from robolab.variations.camera import EgocentricMirroredWideAngleHighCameraCfg  # pyright: ignore[reportMissingImports]
from robolab.variations.lighting import SphereLightCfg  # pyright: ignore[reportMissingImports]

_MIMIC_SCHEMA = 'PhysxMimicJointAPI:'


def _make_mimic_rigid(stage: Any) -> None:
    """Couple each finger mimic joint rigidly, inside the finger limits.

    The importer writes each mimic joint with ``physxMimicJoint:<axis>:naturalFrequency`` 25 and widens its limits to
    [-0.0095, 0.057]. With these values joint8 does not follow joint7 in a grasp.
    """
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        schemas = [str(s) for s in prim.GetAppliedSchemas() if str(s).startswith(_MIMIC_SCHEMA)]
        for axis in (s[len(_MIMIC_SCHEMA) :] for s in schemas):
            prim.CreateAttribute(f'physxMimicJoint:{axis}:naturalFrequency', Sdf.ValueTypeNames.Float).Set(0.0)
            joint = UsdPhysics.PrismaticJoint(prim)
            joint.GetLowerLimitAttr().Set(spec.FINGER_CLOSED)
            joint.GetUpperLimitAttr().Set(spec.FINGER_OPEN)
    stage.GetRootLayer().Save()


def convert(out_dir: Path) -> tuple[Path, Path]:
    """The robot USD and the stand USD, written into ``out_dir``.

    The fixed joints stay, so the ``gripper`` links keep prims of their own. The mesh colliders become convex hulls.
    """
    urdf = out_dir / 'yam_bimanual.urdf'
    urdf.write_text(spec.bimanual_urdf())
    stand = out_dir / 'yam_stand.usda'
    stand.write_text(spec.stand_usda())
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(urdf),
            usd_dir=str(out_dir),
            usd_file_name='yam_bimanual.usd',
            fix_base=True,
            merge_fixed_joints=False,
            # Isaac Lab 2.2 passes this flag to the importer's ``set_parse_mimic``, so True keeps the mimic joints.
            convert_mimic_joints_to_normal_joints=True,
            collider_type='convex_hull',
            self_collision=False,
            make_instanceable=False,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0)
            ),
        )
    )
    _make_mimic_rigid(Usd.Stage.Open(converter.usd_path))
    return Path(converter.usd_path), stand


def _per_joint(values: Any) -> dict[str, float]:
    return {f'.*_{joint}': float(v) for joint, v in zip(spec.ARM_JOINTS, values, strict=True)}


def _camera(name: str) -> TiledCameraCfg:
    cam = spec.CAMERAS[name]
    return TiledCameraCfg(
        prim_path=f'{{ENV_REGEX_NS}}/robot/{cam.parent}/{name}',
        height=spec.CAMERA_HEIGHT,
        width=spec.CAMERA_WIDTH,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=spec.CAMERA_FOCAL_LENGTH,
            focus_distance=28.0,
            horizontal_aperture=cam.horizontal_aperture,
            vertical_aperture=cam.vertical_aperture,
            clipping_range=(0.01, 20.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=cam.pos, rot=cam.rot_wxyz, convention='opengl'),
    )


_CAMERAS = {name: _camera(name) for name in spec.CAMERAS}


def _robot_cfg(robot_usd: Path, stand_usd: Path) -> type:
    """The robot cfg class: the two arms on one base plate as one fixed-base articulation, and the three cameras.

    Gravity is off on the robot, as the real arms compensate it.
    """

    class YamCfg:
        robot = ArticulationCfg(
            prim_path='{ENV_REGEX_NS}/robot',
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(robot_usd),
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=64,
                    solver_velocity_iteration_count=4,
                    fix_root_link=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    **{f'.*_{joint}': q for joint, q in zip(spec.ARM_JOINTS, spec.HOME_JOINTS, strict=True)},
                    f'.*_{spec.FINGER_JOINT}': spec.FINGER_OPEN,
                    f'.*_{spec.FINGER_MIMIC_JOINT}': spec.FINGER_OPEN,
                },
                joint_vel={'.*': 0.0},
            ),
            soft_joint_pos_limit_factor=1.0,
            actuators={
                'arms': ImplicitActuatorCfg(
                    joint_names_expr=[f'.*_{joint}' for joint in spec.ARM_JOINTS],
                    stiffness=_per_joint(spec.ARM_STIFFNESS),
                    damping=_per_joint(spec.ARM_DAMPING),
                    effort_limit_sim=_per_joint(spec.ARM_EFFORT),
                    velocity_limit_sim=_per_joint(spec.ARM_VELOCITY),
                    armature=_per_joint(spec.ARM_ARMATURE),
                ),
                'fingers': ImplicitActuatorCfg(
                    joint_names_expr=[f'.*_{spec.FINGER_JOINT}'],
                    stiffness=spec.FINGER_STIFFNESS,
                    damping=spec.FINGER_DAMPING,
                    effort_limit_sim=spec.FINGER_EFFORT,
                    velocity_limit_sim=spec.FINGER_VELOCITY,
                ),
                # The mimic joint follows joint7, so it gets no drive of its own.
                'finger_mimic': ImplicitActuatorCfg(
                    joint_names_expr=[f'.*_{spec.FINGER_MIMIC_JOINT}'], stiffness=0.0, damping=0.0
                ),
            },
        )

    # The cameras go on before configclass, as fields named for their image obs keys. The labels go on after it, so
    # that configclass does not make them fields.
    for name, camera in _CAMERAS.items():
        setattr(YamCfg, name, camera)
    YamCfg = configclass(YamCfg)
    YamCfg.table_fixture = TableFixtureCfg(usd_path=str(stand_usd), frame='robot')  # pyright: ignore[reportAttributeAccessIssue]
    YamCfg.ee_recorder_bodies = dict(spec.EE_RECORDER_BODIES)  # pyright: ignore[reportAttributeAccessIssue]
    return YamCfg


# Names the three robot cameras for ``generate_image_obs_from_cameras``. The robot cfg spawns them.
_YamCameraCfg = configclass(type('_YamCameraCfg', (), dict(_CAMERAS)))


def _joint_ids(robot: Any, names: list[str]) -> list[int]:
    return [robot.data.joint_names.index(name) for name in names]


def arm_joint_pos(env: ManagerBasedRLEnv, side: str) -> Any:
    robot = env.scene['robot']
    return as_torch(robot.data.joint_pos)[:, _joint_ids(robot, [f'{side}_{j}' for j in spec.ARM_JOINTS])]


def gripper_pos(env: ManagerBasedRLEnv, side: str) -> Any:
    """The grip of ``side``: 0 open, 1 closed."""
    robot = env.scene['robot']
    q = as_torch(robot.data.joint_pos)[:, _joint_ids(robot, [f'{side}_{spec.FINGER_JOINT}'])]
    return (spec.FINGER_OPEN - q) / (spec.FINGER_OPEN - spec.FINGER_CLOSED)


def _ee_in_root(env: ManagerBasedRLEnv, side: str) -> Any:
    robot = env.scene['robot']
    body = robot.data.body_names.index(f'{side}_{spec.GRIPPER_SITE}')
    return subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        as_torch(robot.data.body_pos_w)[:, body, :],
        as_torch(robot.data.body_quat_w)[:, body, :],
    )


def ee_pos(env: ManagerBasedRLEnv, side: str) -> Any:
    """The ``gripper`` link position of ``side`` in the robot root frame."""
    return _ee_in_root(env, side)[0]


def ee_quat(env: ManagerBasedRLEnv, side: str) -> Any:
    """The ``gripper`` link orientation of ``side`` in the robot root frame, wxyz."""
    return quat_isaaclab_to_wxyz(_ee_in_root(env, side)[1])


def _proprio_post_init(self: Any) -> None:
    self.enable_corruption = False
    self.concatenate_terms = False


# The terms of each arm, named by ``yam_spec.proprio_term``.
_PROPRIO_TERMS = {
    spec.PROPRIO_JOINT_POS: (arm_joint_pos, None),
    spec.PROPRIO_GRIP: (gripper_pos, (0.0, 1.0)),
    spec.PROPRIO_EE_POS: (ee_pos, None),
    spec.PROPRIO_EE_QUAT: (ee_quat, None),
}
YamProprioceptionObservationCfg = configclass(
    type(
        'YamProprioceptionObservationCfg',
        (ObsGroup,),
        {
            **{
                spec.proprio_term(side, term): ObsTerm(func=func, params={'side': side}, clip=clip)
                for side in spec.SIDES
                for term, (func, clip) in _PROPRIO_TERMS.items()
            },
            '__post_init__': _proprio_post_init,
        },
    )
)


def _action_term(joints: tuple[str, ...]) -> Any:
    """The joint targets of ``joints``. A grip in [0, 1] maps linearly from the open finger to the closed finger."""
    grip = joints[0].endswith(spec.FINGER_JOINT)
    return mdp.JointPositionActionCfg(
        asset_name='robot',
        joint_names=list(joints),
        preserve_order=True,
        use_default_offset=False,
        scale=spec.FINGER_CLOSED - spec.FINGER_OPEN if grip else 1.0,
        offset=spec.FINGER_OPEN if grip else 0.0,
    )


# The 14 action values, one term per ``yam_spec.ACTION_TERMS`` entry, in its order.
YamJointPositionActionCfg = configclass(
    type('YamJointPositionActionCfg', (), {name: _action_term(joints) for name, joints in spec.ACTION_TERMS})
)


def register_envs(task: list[str], robot_usd: Path, stand_usd: Path) -> Any:
    """Register the RoboLab task envs of ``task`` with the YAM robot, its three cameras and joint position actions.

    The envs step physics at 120 Hz and the policy at 15 Hz.
    """
    viewport_cam_cfg = generate_image_obs_from_cameras([EgocentricMirroredWideAngleHighCameraCfg])
    image_obs_cfg = generate_image_obs_from_cameras([_YamCameraCfg])
    observation_cfg = generate_obs_cfg({
        'image_obs': image_obs_cfg(),
        'proprio_obs': YamProprioceptionObservationCfg(),
        'viewport_cam': viewport_cam_cfg(),
    })
    return auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=DEFAULT_TASK_SUBFOLDERS,
        tasks=task,
        pattern='*.py',
        env_postfix='YamJointPosition',
        observations_cfg=observation_cfg(),
        actions_cfg=YamJointPositionActionCfg(),
        robot_cfg=_robot_cfg(robot_usd, stand_usd),
        # The robot cfg spawns the three cameras. Listing ``_YamCameraCfg`` here spawns them before the robot.
        camera_cfg=[EgocentricMirroredWideAngleHighCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=copy.deepcopy(spec.CONTACT_GRIPPER),
        gripper_closure_cfg=copy.deepcopy(spec.GRIPPER_CLOSURE),
        dt=1 / 120,
        render_interval=8,
        decimation=8,
        seed=1,
    )
