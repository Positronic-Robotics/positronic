import configuronic as cfn

import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.simulator
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.embodiment import ROBOT_STATIC_META, Command, Embodiment, Observation, Privileged
from positronic.simulator.mujoco.sim import FullSimState, MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.utils import package_assets_path


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
        'robot_command': Command(
            robot_arm.commands, roboarm_command.Reset(), 'robot_commands', Serializers.robot_command
        ),
        'target_grip': Command(gripper.target_grip, 0.0, 'target_grip', None),
    }
    return Embodiment(
        descriptor='',
        observations=observations,
        commands=commands,
        privileged={},
        static_meta=dict(ROBOT_STATIC_META),
        meta_source=robot_arm.robot_meta,
        control_systems=(*cameras.values(), robot_arm, gripper),
        simulated=False,
    )


@cfn.config(
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    camera_fps=15,
    # 3 cameras because Mujoco does not render the second image when using only 2 cameras
    camera_dict={'image.wrist': 'handcam_left_ph', 'image.exterior': 'back_view_ph', 'image.agent_view': 'agentview'},
    # Full sim state is the privileged ground truth; scoring is computed downstream.
    observers={'sim_state': FullSimState()},
)
def mujoco_franka(mujoco_model_path, loaders, camera_fps, camera_dict, observers):
    """Mujoco single-arm Franka + gripper, with privileged sim-state ground truth."""
    sim = MujocoSim(mujoco_model_path, loaders, observers=observers)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    cameras = MujocoCameras(sim.model, sim.data, resolution=(320, 240), fps=camera_fps)
    observations = {
        'robot_state': Observation(robot_arm.state, Serializers.robot_state),
        'grip': Observation(gripper.grip, None),
        **{name: Observation(cameras.cameras[orig], Serializers.camera_images) for name, orig in camera_dict.items()},
    }
    commands = {
        'robot_command': Command(
            robot_arm.commands, roboarm_command.Reset(), 'robot_commands', Serializers.robot_command
        ),
        'target_grip': Command(gripper.target_grip, 0.0, 'target_grip', None),
    }
    privileged = {name: Privileged(sim.observations[name], None) for name in observers}
    return Embodiment(
        descriptor='mujoco.franka',
        observations=observations,
        commands=commands,
        privileged=privileged,
        static_meta={**ROBOT_STATIC_META, 'simulation.mujoco_model_path': mujoco_model_path},
        meta_source=robot_arm.robot_meta,
        control_systems=(cameras, sim, robot_arm, gripper),
        simulated=True,
    )


mujoco_franka_pnp = mujoco_franka.override(loaders=positronic.cfg.simulator.multi_tote_loaders, observers={})
