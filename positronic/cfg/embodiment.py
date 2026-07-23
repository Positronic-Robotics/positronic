import configuronic as cfn
import numpy as np

import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
from positronic import geom
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.drivers.roboarm import yam as yam_driver
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    cameras={
        'image.wrist': positronic.cfg.hardware.camera.zed_m.override(
            view='left', resolution='hd720', fps=30, image_enhancement=True
        ),
        'image.exterior': positronic.cfg.hardware.camera.zed_2i.override(
            view='left', resolution='hd720', fps=30, image_enhancement=True
        ),
    },
)
def droid(robot_arm, gripper, cameras):
    """Real single-arm Franka (DROID) + Robotiq gripper + ZED cameras."""
    observations = {
        'robot_state': Observation(robot_arm.state, Serializers.robot_state),
        'grip': Observation(gripper.grip, None),
        **{name: Observation(cam.frame, Serializers.camera_images) for name, cam in cameras.items()},
    }
    commands = {
        'robot_command': Command(robot_arm.commands, roboarm_command.Reset(), Serializers.robot_command),
        'target_grip': Command(gripper.target_grip, 0.0, None),
    }
    return Embodiment(
        descriptor='',
        observations=observations,
        commands=commands,
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=robot_arm.robot_meta,
        control_systems=(*cameras.values(), robot_arm, gripper),
        simulated=False,
    )


@cfn.config(robot_arm=positronic.cfg.hardware.roboarm.yam, cameras={})
def yam(robot_arm, cameras):
    """Real single-arm i2rt YAM: the arm driver carries the gripper (they share one CAN chain)."""
    observations = {
        'robot_state': Observation(robot_arm.state, Serializers.robot_state),
        'grip': Observation(robot_arm.grip, None),
        **{name: Observation(cam.frame, Serializers.camera_images) for name, cam in cameras.items()},
    }
    commands = {
        'robot_command': Command(robot_arm.commands, roboarm_command.Reset(), Serializers.robot_command),
        'target_grip': Command(robot_arm.target_grip, 0.0, None),
    }
    return Embodiment(
        descriptor='yam',
        observations=observations,
        commands=commands,
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=robot_arm.robot_meta,
        control_systems=(*cameras.values(), robot_arm),
        simulated=False,
    )


@cfn.config(
    left_channel='can0',
    right_channel='can1',
    cameras={
        'image.exterior': positronic.cfg.hardware.camera.zed_x_top.override(resolution='svga', fps=30),
        'image.wrist_left': positronic.cfg.hardware.camera.zed_x_one_left.override(resolution='svga', fps=30),
        'image.wrist_right': positronic.cfg.hardware.camera.zed_x_one_right.override(resolution='svga', fps=30),
    },
)
def yam_bimanual(left_channel: str, right_channel: str, cameras):
    """Real bimanual i2rt YAM — channel names and static_meta match ``mujoco_yam_bimanual`` by convention.

    Per-arm channels are the flat names the whole stack shares: ``robot_state.{side}`` expands into
    ``robot_state.{side}.q/.dq/.ee_pose`` on record, commands are ``robot_command.{side}`` +
    ``target_grip.{side}``, and static_meta lists ``arms=[left, right]``. Per-side ``base_pose`` mounts
    each arm at the sim world-frame mount, so real ``ee_pose`` lands in the training world frame.
    """
    mount_z = yam_driver.TABLE_Z + yam_driver.YAM_MOUNT_LIFT
    arms = {
        side: yam_driver.Robot(channel, base_pose=geom.Transform3D([*yam_driver.YAM_MOUNTS[side], mount_z]))
        for side, channel in (('left', left_channel), ('right', right_channel))
    }
    observations = {
        **{f'robot_state.{s}': Observation(arm.state, Serializers.robot_state) for s, arm in arms.items()},
        **{f'grip.{s}': Observation(arm.grip, None) for s, arm in arms.items()},
        **{name: Observation(cam.frame, Serializers.camera_images) for name, cam in cameras.items()},
    }
    commands = {
        **{
            f'robot_command.{s}': Command(arm.commands, roboarm_command.Reset(), Serializers.robot_command)
            for s, arm in arms.items()
        },
        **{f'target_grip.{s}': Command(arm.target_grip, 0.0, None) for s, arm in arms.items()},
    }
    static_meta = {
        'joint_signals': [f'robot_state.{s}.q' for s in arms],
        'pose_signals': [f'robot_state.{s}.ee_pose' for s in arms],
        'command_pose_signals': [f'robot_command.{s}.pose' for s in arms],
        'arms': list(arms),
    }
    return Embodiment(
        descriptor='yam_bimanual',
        observations=observations,
        commands=commands,
        static_meta=static_meta,
        # Both drivers emit the identical per-arm meta; record one copy.
        meta_source=arms['left'].robot_meta,
        control_systems=(*cameras.values(), *arms.values()),
        simulated=False,
    )


def mujoco_franka(sim, camera_dict):
    """Mujoco single-arm Franka + gripper over a given sim — pure robot, no scene.

    The scene (loaders) and privileged ground-truth are the eval's concern; this maps
    the sim's arm, gripper, and camera ports into an embodiment. 3 cameras because
    Mujoco does not render the second image when using only 2 cameras.
    """
    observations = {
        'robot_state': Observation(sim.state, Serializers.robot_state),
        'grip': Observation(sim.grip, None),
        **{name: Observation(sim.cameras[orig], Serializers.camera_images) for name, orig in camera_dict.items()},
    }
    # Home to the scene's initial pose, not `Reset()`: in MujocoSim `Reset()` rebuilds the whole scene, wiping the
    # trial's end state right when the operator reviews it.
    home = roboarm_command.JointPosition(np.array(sim.initial_ctrl[:7]))
    commands = {
        'robot_command': Command(sim.commands, home, Serializers.robot_command),
        'target_grip': Command(sim.target_grip, 0.0, None),
    }
    return Embodiment(
        descriptor='mujoco.franka',
        observations=observations,
        commands=commands,
        static_meta={**ROBOT_STATIC_META, 'simulation.mujoco_model_path': sim.mujoco_model_path},
        meta_source=sim.robot_meta,
        control_systems=(sim,),
        simulated=True,
    )
