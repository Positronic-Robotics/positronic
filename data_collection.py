import asyncio
import logging
from typing import List

import hydra
import numpy as np
from omegaconf import DictConfig

import geom
import ironic as ir
from drivers import from_config
from ironic.compose import extend
from tools.dataset_dumper import DatasetDumper

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def setup_interface(cfg: DictConfig):
    if cfg.type == 'teleop':
        from webxr import WebXR
        from teleop import TeleopSystem, front_position, back_position

        components, inputs, outputs = [], {}, {}
        webxr = WebXR(port=cfg.webxr.port)
        components.append(webxr)

        if cfg.operator_position == 'front':
            pos_parser = front_position.instantiate()
        elif cfg.operator_position == 'back':
            pos_parser = back_position.instantiate()
        else:
            raise ValueError(f"Invalid operator position: {cfg.operator_position}")

        teleop = TeleopSystem(pos_parser=pos_parser, operator_position=cfg.operator_position)
        components.append(teleop)

        teleop.bind(
            teleop_transform=webxr.outs.transform,
            teleop_buttons=webxr.outs.buttons,
        )

        if cfg.get('stream_to_webxr'):
            get_frame_for_webxr = ir.utils.MapPortCS(lambda frame: frame[cfg.stream_to_webxr])
            components.append(get_frame_for_webxr)
            inputs['images'] = (get_frame_for_webxr, 'input')
            webxr.bind(frame=get_frame_for_webxr.outs.output, )
        else:
            inputs['images'] = None

        inputs['robot_position'] = (teleop, 'robot_position')
        inputs['robot_grip'] = None
        inputs['robot_status'] = None

        outputs['robot_target_position'] = teleop.outs.robot_target_position
        outputs['gripper_target_grasp'] = teleop.outs.gripper_target_grasp
        outputs['start_recording'] = teleop.outs.start_recording
        outputs['stop_recording'] = teleop.outs.stop_recording
        outputs['reset'] = teleop.outs.reset
        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    elif cfg.type == 'gui':
        from simulator.mujoco.mujoco_gui import DearpyguiUi

        return DearpyguiUi(cfg.mujoco.camera_names), {}
    elif cfg.type == 'teleop_gui':
        # TODO: refactor this in a way which allows to use multiple control interfaces at once
        # i. e. teleop_gui = inputs from teleop + outputs to gui
        from simulator.mujoco.mujoco_gui import DearpyguiUi
        from teleop import TeleopSystem
        from webxr import WebXR

        components, inputs, outputs = [], {}, {}

        webxr = WebXR(port=cfg.webxr.port)
        components.append(webxr)
        teleop = TeleopSystem(operator_position=cfg.operator_position)
        components.append(teleop)
        gui = DearpyguiUi(cfg.mujoco.camera_names)
        components.append(gui)

        teleop.bind(
            teleop_transform=webxr.outs.transform,
            teleop_buttons=webxr.outs.buttons,
        )

        def adjust_rotations(transform: geom.Transform3D) -> geom.Transform3D:
            """
            Adjust the rotations of the transform by swapping roll and yaw angles.
            """
            euler = transform.quaternion.as_euler

            # empirically found transformation that works
            new_euler = [-euler[2], np.pi + euler[1], euler[0]]

            new_quat = geom.Rotation.from_euler(new_euler)

            return geom.Transform3D(translation=transform.translation, quaternion=new_quat)

        inputs['robot_position'] = [(teleop, 'robot_position'), (gui, 'robot_position')]
        inputs['robot_grip'] = (gui, 'robot_grip')
        inputs['images'] = (gui, 'images')
        inputs['robot_status'] = (gui, 'robot_status')

        outputs['robot_target_position'] = ir.utils.map_port(adjust_rotations, teleop.outs.robot_target_position)
        outputs['gripper_target_grasp'] = teleop.outs.gripper_target_grasp
        outputs['start_recording'] = teleop.outs.start_recording
        outputs['stop_recording'] = teleop.outs.stop_recording
        outputs['reset'] = teleop.outs.reset

        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    elif cfg.type == 'spacemouse':
        from drivers.spacemouse import SpacemouseCS

        components, inputs, outputs = [], {}, {}

        spacemouse = SpacemouseCS(**cfg.spacemouse)
        components.append(spacemouse)
        # TODO: figure out the way to remap inputs/outputs / add empty ones
        inputs['robot_position'] = (spacemouse, 'robot_position')
        inputs['robot_grip'] = None
        inputs['images'] = None
        inputs['robot_status'] = None

        outputs['robot_target_position'] = spacemouse.outs.robot_target_position
        outputs['gripper_target_grasp'] = spacemouse.outs.gripper_target_grasp
        outputs['start_recording'] = spacemouse.outs.start_recording
        outputs['stop_recording'] = spacemouse.outs.stop_recording
        outputs['reset'] = spacemouse.outs.reset

        return ir.compose(*components, inputs=inputs, outputs=outputs), {}
    else:
        raise ValueError(f"Invalid control type: {cfg.type}")


async def main_async(cfg: DictConfig):  # noqa: C901  Function is too complex
    metadata = {}
    control, md = setup_interface(cfg.control_ui)
    metadata.update(md)

    hardware, md = from_config.robot_setup(cfg.hardware)
    metadata.update(md)

    hardware = extend(
        hardware, {
            'robot_position_translation':
            ir.utils.map_property(lambda t: t.translation.copy(), hardware.outs.robot_position),
            'robot_position_quaternion':
            ir.utils.map_property(lambda t: t.quaternion.copy(), hardware.outs.robot_position),
        })

    control.bind(
        robot_grip=hardware.outs.grip,
        robot_position=hardware.outs.robot_position,
        images=hardware.outs.frame,
        robot_status=hardware.outs.robot_status,
    )
    hardware.bind(
        target_position=control.outs.robot_target_position,
        target_grip=control.outs.gripper_target_grasp,
        reset=control.outs.reset,
    )

    components: List[ir.ControlSystem] = []
    components.append(control)
    components.append(hardware)

    if cfg.get('rerun'):
        from tools.rerun_vis import RerunVisualiser

        visualizer = RerunVisualiser()
        visualizer.bind(
            frame=hardware.outs.frame,
            new_recording=control.outs.start_recording,
            ext_force_ee=hardware.outs.ext_force_ee,
            ext_force_base=hardware.outs.ext_force_base,
            robot_position=hardware.outs.robot_position,
        )
        components.append(visualizer)  # Add visualizer to components

    if cfg.get('sound'):
        from drivers.sound import SoundSystem

        force_feedback_volume = cfg.sound.get('force_feedback_volume', 0)
        sound_system = SoundSystem(master_volume=force_feedback_volume)

        if force_feedback_volume > 0:

            def force_to_level(force: np.ndarray) -> float:
                # TODO: figure out if L2 norm is better
                return np.abs(force).max()

            sound_system.bind(level=ir.utils.map_property(force_to_level, hardware.outs.ext_force_ee), )

        if cfg.sound.get('record_notifications'):
            sound_system.bind(wav_path=ir.utils.map_port(lambda _: 'assets/sounds/recording-has-started.wav',
                                                         control.outs.start_recording))
            sound_system.bind(wav_path=ir.utils.map_port(lambda _: 'assets/sounds/recording-has-stopped.wav',
                                                         control.outs.stop_recording))

        components.append(sound_system)

    # Setup data collection if enabled
    if cfg.data_output_dir is not None:
        robot_properties = {}

        for stored_name, property_name in cfg.state_mappings.items():
            robot_properties[stored_name] = getattr(hardware.outs, property_name)

        properties_to_dump = ir.utils.properties_dict(**robot_properties)

        data_dumper = DatasetDumper(cfg.data_output_dir, additional_metadata=metadata, video_fps=cfg.get('video_fps'))

        if hasattr(hardware.outs, 'episode_metadata'):

            async def send_episode_metadata(_: ir.Message):
                episode_metadata = (await hardware.outs.episode_metadata()).data

                return episode_metadata

            start_recording = ir.utils.map_port(send_episode_metadata, control.outs.start_recording)
        else:
            start_recording = control.outs.start_recording

        components.append(
            data_dumper.bind(
                # TODO: Let user disable images, like in mujoco_gui
                image=hardware.outs.frame,
                target_grip=control.outs.gripper_target_grasp,
                target_robot_position=control.outs.robot_target_position,
                start_episode=start_recording,
                end_episode=control.outs.stop_recording,
                robot_data=properties_to_dump,
            ))
    # Run the system
    system = ir.compose(*components)
    await ir.utils.run_gracefully(system)


@hydra.main(version_base=None, config_path="configs", config_name="data_collection")
def main(cfg: DictConfig):
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
