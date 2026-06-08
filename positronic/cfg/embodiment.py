from collections.abc import Sequence
from typing import Any

import configuronic as cfn

import pimm
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.simulator
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.embodiment import ROBOT_STATIC_META, Command, Embodiment, Observation, Privileged
from positronic.simulator.mujoco.sim import FullSimState, MujocoCameras, MujocoFranka, MujocoGripper, MujocoSim
from positronic.utils import package_assets_path


def franka(
    robot_arm: pimm.ControlSystem,
    gripper: pimm.ControlSystem,
    *,
    descriptor: str,
    cameras: dict[str, pimm.SignalEmitter] | None = None,
    privileged: dict[str, pimm.SignalEmitter] | None = None,
    static_meta: dict[str, Any] | None = None,
    control_systems: Sequence[pimm.ControlSystem] = (),
    simulated: bool = False,
) -> Embodiment:
    """Build a single-arm Franka + gripper embodiment from separate device CSs.

    Generic over the backing devices: ``MujocoFranka``/``MujocoGripper`` in sim, the
    DROID drivers on hardware. ``control_systems`` are those devices for the runner to
    schedule; ``simulated`` selects the sim runtime.
    """
    observations = {
        'robot_state': Observation(robot_arm.state, Serializers.robot_state),
        'grip': Observation(gripper.grip, None),
    }
    for name, emitter in (cameras or {}).items():
        observations[name] = Observation(emitter, Serializers.camera_images)

    commands = {
        'robot_command': Command(
            robot_arm.commands, roboarm_command.Reset(), 'robot_commands', Serializers.robot_command
        ),
        'target_grip': Command(gripper.target_grip, 0.0, 'target_grip', None),
    }

    privileged_specs = {name: Privileged(source, None) for name, source in (privileged or {}).items()}

    return Embodiment(
        descriptor=descriptor,
        observations=observations,
        commands=commands,
        privileged=privileged_specs,
        static_meta={**ROBOT_STATIC_META, **(static_meta or {})},
        meta_source=robot_arm.robot_meta,
        control_systems=tuple(control_systems),
        simulated=simulated,
    )


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
    return franka(
        robot_arm,
        gripper,
        descriptor='',
        cameras={name: cam.frame for name, cam in cameras.items()},
        control_systems=[*cameras.values(), robot_arm, gripper],
    )


@cfn.config(
    mujoco_model_path=package_assets_path('assets/mujoco/franka_table.xml'),
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
    camera_fps=15,
    # We use 3 cameras not because we need it, but because Mujoco does not render
    # the second image when using only 2 cameras
    camera_dict={'image.wrist': 'handcam_left_ph', 'image.exterior': 'back_view_ph', 'image.agent_view': 'agentview'},
    # Full sim state is the privileged ground truth; scoring is computed downstream.
    observers={'sim_state': FullSimState()},
)
def sim(mujoco_model_path, loaders, camera_fps, camera_dict, observers):
    """Mujoco single-arm Franka + gripper, with privileged sim-state ground truth."""
    mj_sim = MujocoSim(mujoco_model_path, loaders, observers=observers)
    robot_arm = MujocoFranka(mj_sim, suffix='_ph')
    gripper = MujocoGripper(mj_sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    mujoco_cameras = MujocoCameras(mj_sim.model, mj_sim.data, resolution=(320, 240), fps=camera_fps)
    cameras = {name: mujoco_cameras.cameras[orig_name] for name, orig_name in camera_dict.items()}
    return franka(
        robot_arm,
        gripper,
        descriptor='mujoco.franka',
        cameras=cameras,
        privileged={name: mj_sim.observations[name] for name in observers},
        static_meta={'simulation.mujoco_model_path': mujoco_model_path},
        control_systems=[mujoco_cameras, mj_sim, robot_arm, gripper],
        simulated=True,
    )
