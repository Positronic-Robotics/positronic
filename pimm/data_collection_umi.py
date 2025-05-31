import time
from functools import partial

import fire

import ironic2 as ir
from pimm.drivers.camera.linux_video import LinuxVideo
from pimm.drivers.gripper.dh import DHGripper
from pimm.drivers.webxr import WebXR
from positronic.tools.buttons import ButtonHandler
from positronic.tools.dataset_dumper import SerialDumper


def _parse_buttons(buttons: ir.Message | ir.NoValueType, button_handler: ButtonHandler):
    if buttons is ir.NoValue:
        return

    mapping = {
        'A': buttons.data[4],
        'B': buttons.data[5],
        'trigger': buttons.data[0],
        'thumb': buttons.data[1],
        'stick': buttons.data[3]
    }
    button_handler.update_buttons(mapping)


def main_loop(output_dir: str, fps: int, buttons_reader: ir.SignalReader, controller_positions_reader: ir.SignalReader,
              frame_reader: ir.SignalReader, target_grip_emitter: ir.SignalEmitter, should_stop: ir.SignalReader):
    codec: str = 'libx264'
    tracked = False
    dumper = SerialDumper(output_dir, video_fps=fps, codec=codec)
    button_handler = ButtonHandler()

    last_ts = None
    meta = {}

    while not ir.is_true(should_stop):
        try:
            buttons = ir.signal_value(buttons_reader)
            _parse_buttons(buttons, button_handler)
            if button_handler.just_pressed('B'):
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

            frame_msg = frame_reader.value()
            if not tracked or frame_msg is ir.NoValue or last_ts == frame_msg.ts:
                time.sleep(0.01)
                continue

            last_ts = frame_msg.ts

            controller_position = ir.signal_value(controller_positions_reader)['right']
            target_grip = button_handler.get_value('trigger')
            target_grip_emitter.emit(ir.Message(target_grip, ir.system_clock()))

            ep_dict = {
                'target_grip': target_grip,
                'target_robot_position_translation': controller_position.translation.copy(),
                'target_robot_position_quaternion': controller_position.rotation.as_quat.copy(),
                'image_timestamp': frame_msg.ts,
            }

            dumper.write(data=ep_dict, video_frames=frame_msg.data)

        except ir.NoValueException:
            time.sleep(0.01)
            continue


def main(gripper_port: str = "/dev/ttyUSB0",
         webxr_port: int = 8000,
         camera_device: str = "/dev/video0",
         output_dir: str = "data_collection_umi",
         fps: int = 30):
    gripper = DHGripper(gripper_port)
    webxr = WebXR(webxr_port)
    camera = LinuxVideo(camera_device, 640, 480, fps, "h264")

    world = ir.mp.MPWorld()

    camera.frame, frame_reader = world.pipe()
    webxr.controller_positions, controller_positions_reader = world.pipe()
    webxr.buttons, buttons_reader = world.pipe()
    target_grip_emitter, gripper.target_grip = world.pipe()

    world.run(
        partial(main_loop, output_dir, fps, buttons_reader, controller_positions_reader, frame_reader,
                target_grip_emitter))


if __name__ == "__main__":
    fire.Fire(main)
