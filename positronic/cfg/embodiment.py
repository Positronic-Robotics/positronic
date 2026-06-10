import configuronic as cfn

import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation
from positronic.simulator.mujoco.sim import MujocoCameras, MujocoFranka, MujocoGripper


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


def mujoco_franka(sim, camera_fps, camera_dict):
    """Mujoco single-arm Franka + gripper over a given sim — pure robot, no scene.

    The scene (loaders) and privileged ground-truth are the eval's concern; this builds
    only the robot from the sim the eval holds. 3 cameras because Mujoco does not render
    the second image when using only 2 cameras.
    """
    robot_arm = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=camera_fps)
    observations = {
        'robot_state': Observation(robot_arm.state, Serializers.robot_state),
        'grip': Observation(gripper.grip, None),
        **{name: Observation(cameras.cameras[orig], Serializers.camera_images) for name, orig in camera_dict.items()},
    }
    commands = {
        'robot_command': Command(robot_arm.commands, roboarm_command.Reset(), Serializers.robot_command),
        'target_grip': Command(gripper.target_grip, 0.0, None),
    }
    return Embodiment(
        descriptor='mujoco.franka',
        observations=observations,
        commands=commands,
        static_meta={**ROBOT_STATIC_META, 'simulation.mujoco_model_path': sim.mujoco_model_path},
        meta_source=robot_arm.robot_meta,
        control_systems=(cameras, sim, robot_arm, gripper),
        simulated=True,
    )
