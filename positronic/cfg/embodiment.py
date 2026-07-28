import configuronic as cfn
import numpy as np

import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
from positronic import keys
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.franka_droid,
    gripper=positronic.cfg.hardware.gripper.robotiq,
    cameras={
        keys.WRIST_IMAGE: positronic.cfg.hardware.camera.zed_m.override(
            view='left', resolution='hd720', fps=30, image_enhancement=True
        ),
        keys.EXTERIOR_IMAGE: positronic.cfg.hardware.camera.zed_2i.override(
            view='left', resolution='hd720', fps=30, image_enhancement=True
        ),
    },
)
def droid(robot_arm, gripper, cameras):
    """Real single-arm Franka (DROID) + Robotiq gripper + ZED cameras."""
    observations = {
        'robot_state': Observation(robot_arm.state, Serializers.robot_state),
        keys.GRIP: Observation(gripper.grip, None),
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


def mujoco_franka(sim, camera_dict):
    """Mujoco single-arm Franka + gripper over a given sim — pure robot, no scene.

    The scene (loaders) and privileged ground-truth are the eval's concern; this maps
    the sim's arm, gripper, and camera ports into an embodiment. 3 cameras because
    Mujoco does not render the second image when using only 2 cameras.
    """
    observations = {
        'robot_state': Observation(sim.state, Serializers.robot_state),
        keys.GRIP: Observation(sim.grip, None),
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
