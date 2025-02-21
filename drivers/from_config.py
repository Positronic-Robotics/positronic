from typing import Dict

import hydra
from omegaconf import DictConfig

import ironic as ir


def add_image_mapping(mapping: Dict[str, str], camera: ir.ControlSystem):

    def map_images(frame):
        return {new_k: frame[k] for k, new_k in mapping.items()}

    map_system = ir.utils.MapPortCS(map_images)
    map_system.bind(input=camera.outs.frame)
    return ir.compose(camera, map_system, outputs={'frame': map_system.outs.output})


def sl_camera(cfg: DictConfig):
    from drivers.camera.sl import SLCamera
    import pyzed.sl as sl

    view = getattr(sl.VIEW, cfg.view)
    resolution = getattr(sl.RESOLUTION, cfg.resolution)
    kwargs = {}
    if 'depth_mode' in cfg:
        kwargs['depth_mode'] = getattr(sl.DEPTH_MODE, cfg.depth_mode)
    if 'coordinate_units' in cfg:
        kwargs['coordinate_units'] = getattr(sl.UNIT, cfg.coordinate_units)
    if 'max_depth' in cfg:
        kwargs['max_depth'] = cfg.max_depth
    if 'depth_mask' in cfg:
        kwargs['depth_mask'] = cfg.depth_mask

    camera = SLCamera(cfg.fps, view, resolution, **kwargs)

    if 'image_mapping' in cfg:
        return add_image_mapping(cfg.image_mapping, camera)
    else:
        return camera


def realsense_camera(cfg: DictConfig):
    from drivers.camera.realsense import RealsenseCamera, RealsenseCameraCS

    camera = RealsenseCameraCS(
        RealsenseCamera(
            resolution=cfg.resolution,
            fps=cfg.fps,
            enable_color=cfg.enable_color,
            enable_depth=cfg.enable_depth,
            enable_infrared=cfg.enable_infrared,
        ))

    if 'image_mapping' in cfg:
        return add_image_mapping(cfg.image_mapping, camera)
    else:
        return camera


def luxonis_camera(cfg: DictConfig):
    from drivers.camera.luxonis import LuxonisCamera, LuxonisCameraCS

    camera = LuxonisCameraCS(LuxonisCamera())

    return camera


def opencv_camera(cfg: DictConfig):
    from drivers.camera.opencv import OpenCVCamera, OpenCVCameraCS

    camera = OpenCVCameraCS(OpenCVCamera(
        camera_id=cfg.camera_id,
        resolution=cfg.resolution,
        fps=cfg.fps,
    ))

    return camera


def linuxpy_video_camera(cfg: DictConfig):
    from drivers.camera.linuxpy_video import LinuxPyCamera
    return LinuxPyCamera(cfg.device_path, cfg.width, cfg.height, cfg.fps, cfg.pixel_format)


def get_camera(cfg: DictConfig):
    match cfg.type:
        case 'sl':
            return sl_camera(cfg)
        case 'realsense':
            return realsense_camera(cfg)
        case 'luxonis':
            return luxonis_camera(cfg)
        case 'opencv':
            return opencv_camera(cfg)
        case 'linuxpy_video':
            return linuxpy_video_camera(cfg)
        case _:
            raise ValueError(f"Invalid camera type: {cfg.type}")


def robot_setup(cfg: DictConfig):  # noqa: C901  Function is too complex
    """Setup and returns control system with robot and camera(s).
    inputs:
        target_position: Target position for the robot
        target_grip: Target grip for the gripper
    outputs:
        robot_position: Current position of the robot
        grip: Current gripper state
        frame: Current camera frame

    Returns:
        control system, metadata (as a dict)
    """
    components, inputs, outputs = [], {}, {}
    if cfg.type == 'physical':
        from drivers.roboarm import franka
        kwargs = {}
        if 'collision_behavior' in cfg.franka:
            kwargs['collision_behavior'] = cfg.franka.collision_behavior
        if 'home_joints_config' in cfg.franka:
            kwargs['home_joints_config'] = cfg.franka.home_joints_config
        if 'cartesian_mode' in cfg.franka:
            kwargs['cartesian_mode'] = getattr(franka.CartesianMode, cfg.franka.cartesian_mode)

        franka = franka.Franka(cfg.franka.ip, cfg.franka.relative_dynamics_factor, cfg.franka.gripper_force, **kwargs)
        components.append(franka)
        inputs['target_position'] = (franka, 'target_position')
        inputs['reset'] = (franka, 'reset')

        outputs['robot_position'] = franka.outs.position
        outputs['joint_positions'] = franka.outs.joint_positions
        outputs['ext_force_base'] = franka.outs.ext_force_base
        outputs['ext_force_ee'] = franka.outs.ext_force_ee
        outputs['robot_status'] = franka.outs.status

        if 'dh_gripper' in cfg:
            from drivers.gripper.dh import DHGripper
            gripper = DHGripper(cfg.dh_gripper)
            components.append(gripper)
            outputs['grip'] = gripper.outs.grip
            inputs['target_grip'] = (gripper, 'target_grip')
        else:
            outputs['grip'] = franka.outs.grip
            inputs['target_grip'] = (franka, 'target_grip')

        if 'camera' in cfg:
            camera = get_camera(cfg.camera)
            components.append(camera)
            outputs['frame'] = camera.outs.frame
        else:
            outputs['frame'] = None
        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    elif cfg.type == 'mujoco':
        from simulator.mujoco.sim import MujocoSimulator, MujocoRenderer, InverseKinematics
        from simulator.mujoco.environment import MujocoSimulatorCS

        simulator_factory = lambda: create_from_config(cfg)

        # Create MujocoSimulatorCS
        simulator_cs = MujocoSimulatorCS(
            simulator_factory=simulator_factory,
            simulation_rate=1 / cfg.mujoco.simulation_hz,
            render_rate=1 / cfg.mujoco.observation_hz,
        )
        components.append(simulator_cs)

        # Map the interface
        inputs['target_position'] = (simulator_cs, 'robot_target_position')
        inputs['target_grip'] = (simulator_cs, 'gripper_target_grasp')
        inputs['reset'] = (simulator_cs, 'reset')

        outputs['robot_position'] = simulator_cs.outs.robot_position
        outputs['actuator_values'] = simulator_cs.outs.actuator_values
        outputs['joint_positions'] = simulator_cs.outs.joints
        outputs['ext_force_base'] = simulator_cs.outs.ext_force_base
        outputs['ext_force_ee'] = simulator_cs.outs.ext_force_ee
        outputs['grip'] = simulator_cs.outs.grip
        outputs['frame'] = simulator_cs.outs.images
        outputs['robot_status'] = simulator_cs.outs.robot_status
        outputs['episode_metadata'] = simulator_cs.outs.episode_metadata

        metadata = {'mujoco_model_path': cfg.mujoco.model_path, 'simulation_hz': cfg.mujoco.simulation_hz}
        return ir.compose(*components, inputs=inputs, outputs=outputs), metadata
    elif cfg.type == 'umi':
        from drivers.umi import UmiCS

        umi = UmiCS()
        components.append(umi)

        if 'camera' in cfg:
            camera = get_camera(cfg.camera)
            components.append(camera)
            outputs['frame'] = camera.outs.frame
        else:
            outputs['frame'] = None

        inputs['target_position'] = (umi, 'tracker_position')
        inputs['target_grip'] = (umi, 'target_grip')
        inputs['reset'] = None

        outputs['robot_position'] = umi.outs.ee_position
        outputs['grip'] = umi.outs.grip
        outputs['robot_status'] = None

        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    else:
        raise ValueError(f"Invalid robot type: {cfg.type}")
