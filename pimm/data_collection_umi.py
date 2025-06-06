import time
from typing import Dict

import fire

import ironic2 as ir
from ironic.utils import FPSCounter
from pimm.drivers.camera.linux_video import LinuxVideo
from pimm.drivers.gripper.dh import DHGripper
from pimm.drivers.webxr import WebXR
from positronic.tools.buttons import ButtonHandler
from positronic.tools.dataset_dumper import SerialDumper


def _parse_buttons(buttons: ir.Message | ir.NoValueType, button_handler: ButtonHandler):
    for side in ['left', 'right']:
        if buttons[side] is None:
            continue

        mapping = {
            f'{side}_A': buttons[side][4],
            f'{side}_B': buttons[side][5],
            f'{side}_trigger': buttons[side][0],
            f'{side}_thumb': buttons[side][1],
            f'{side}_stick': buttons[side][3]
        }
        button_handler.update_buttons(mapping)


def main(gripper_port: str = "/dev/ttyUSB0",
         webxr_port: int = 8000,
         camera_device: Dict[str, str] = {
             "left": "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684LEFT-video-index0",
             "right": "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_UC684_UC684RIGHT-video-index0",
         },
         output_dir: str = "data_collection_umi",
         fps: int = 30,
         stream_video_to_webxr: str | None = None,
         ):
    gripper = DHGripper(gripper_port)
    webxr = WebXR(webxr_port)

    world = ir.mp.MPWorld()

    frame_readers = {}
    cameras = {}
    for camera_name, camera_device in camera_device.items():
        camera = LinuxVideo(camera_device, 1280, 720, fps, "h264")
        camera.frame, frame_reader = world.pipe()
        frame_readers[camera_name] = frame_reader
        cameras[camera_name] = camera

    webxr.controller_positions, controller_positions_reader = world.pipe()
    webxr.buttons, buttons_reader = world.pipe()
    if stream_video_to_webxr is not None:
        raise NotImplementedError("TODO: fix video streaming to webxr, since it's currently lagging")
        webxr.frame = ir.map(frame_readers[stream_video_to_webxr], lambda x: x['image'])
    target_grip_emitter, gripper.target_grip = world.pipe()

    world.run(
        webxr.run,
        gripper.run,
        *[camera.run for camera in cameras.values()],
    )

    codec: str = 'libx264'
    tracked = False
    dumper = SerialDumper(output_dir, video_fps=fps, codec=codec)
    button_handler = ButtonHandler()

    frame_readers = {name: ir.ValueUpdated(reader) for name, reader in frame_readers.items()}

    meta = {}

    fps_counter = FPSCounter("Data Collection")

    while not world.should_stop:
        try:
            buttons = ir.signal_value(buttons_reader)
            _parse_buttons(buttons, button_handler)
            if button_handler.just_pressed('right_B'):
                tracked = not tracked
                if tracked:
                    meta['episode_start'] = ir.system_clock()
                    dumper.start_episode()
                    print(f"Episode {dumper.episode_count} started")
                else:
                    dumper.end_episode(meta)
                    meta = {}
                    print(f"Episode {dumper.episode_count} ended")
            # TODO: Support aborting current episode.

            frame_messages = {name: reader.value() for name, reader in frame_readers.items()}
            any_frame_updated = any(
                is_updated and frame is not ir.NoValue
                for frame, is_updated in frame_messages.values()
            )

            if not tracked or not any_frame_updated:
                time.sleep(0.01)
                continue

            frame_messages = {name: frame for name, (frame, _) in frame_messages.items()}

            controller_position = ir.signal_value(controller_positions_reader)['right']
            target_grip = button_handler.get_value('right_trigger')
            target_grip_emitter.emit(ir.Message(target_grip, ir.system_clock()))

            ep_dict = {
                'target_grip': target_grip,
                'target_robot_position_translation': controller_position.translation.copy(),
                'target_robot_position_quaternion': controller_position.rotation.as_quat.copy(),
                **{f'{name}_timestamp': frame.ts for name, frame in frame_messages.items()},
            }

            dumper.write(
                data=ep_dict,
                video_frames={name: frame.data['image'] for name, frame in frame_messages.items()}
            )
            fps_counter.tick()

        except ir.NoValueException:
            time.sleep(0.01)
            continue

    world.stop()


if __name__ == "__main__":
    fire.Fire(main)
