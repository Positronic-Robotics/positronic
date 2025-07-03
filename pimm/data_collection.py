import time
from typing import Any, Dict

import geom
import ironic as ir1
import ironic2 as ir
from ironic.utils import FPSCounter
from pimm.drivers import roboarm
from pimm.drivers.sound import SoundSystem
from pimm.drivers.camera.linux_video import LinuxVideo
from pimm.drivers.gripper.dh import DHGripper
from pimm.drivers.webxr import WebXR
from positronic.tools.buttons import ButtonHandler
from positronic.tools.dataset_dumper import SerialDumper

import pimm.cfg.hardware.gripper
import pimm.cfg.webxr
import pimm.cfg.hardware.camera
import pimm.cfg.sound


def _parse_buttons(buttons: ir.Message | None, button_handler: ButtonHandler):
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


class _Tracker:
    on = False
    _offset = geom.Transform3D()
    _teleop_t = geom.Transform3D()

    def __init__(self, operator_position: geom.Transform3D | None):
        self._operator_position = operator_position
        self.on = self.umi_mode

    @property
    def umi_mode(self):
        return self._operator_position is None

    def turn_on(self, robot_pos: geom.Transform3D):
        if self.umi_mode:
            print("Ignoring tracking on/off in UMI mode")
            return

        self.on = True
        print("Starting tracking")
        self._offset = geom.Transform3D(
            -self._teleop_t.translation + robot_pos.translation,
            self._teleop_t.rotation.inv * robot_pos.rotation,
        )

    def turn_off(self):
        if self.umi_mode:
            print("Ignoring tracking on/off in UMI mode")
            return
        self.on = False
        print("Stopped tracking")

    def update(self, tracker_pos: geom.Transform3D):
        if self.umi_mode:
            return tracker_pos

        self._teleop_t = self._operator_position * tracker_pos * self._operator_position.inv
        return geom.Transform3D(
            self._teleop_t.translation + self._offset.translation,
            self._teleop_t.rotation * self._offset.rotation
        )


# TODO: Support aborting current episode.
class _Recorder:
    def __init__(self, dumper: SerialDumper | None, sound_emitter: ir.SignalEmitter):
        self.on = False
        self._dumper = dumper
        self._sound_emitter = sound_emitter
        self._start_wav_path = "positronic/assets/sounds/recording-has-started.wav"
        self._end_wav_path = "positronic/assets/sounds/recording-has-stopped.wav"
        self._meta = {}

    def turn_on(self):
        if self._dumper is None:
            print("No dumper, ignoring 'start recording' command")
            return

        if not self.on:
            self._meta['episode_start'] = ir.system_clock()
            if self._dumper is not None:
                self._dumper.start_episode()
                print(f"Episode {self._dumper.episode_count} started")
                self._sound_emitter.emit(self._start_wav_path)
            self.on = True
        else:
            print("Already recording, ignoring 'start recording' command")

    def turn_off(self):
        if self._dumper is None:
            print("No dumper, ignoring 'stop recording' command")
            return

        if self.on:
            self._dumper.end_episode(self._meta)
            print(f"Episode {self._dumper.episode_count} ended")
            self._sound_emitter.emit(self._end_wav_path)
            self.on = False
        else:
            print("Not recording, ignoring turn_off")

    def update(self, data: dict, video_frames: dict):
        if not self.on:
            return

        if self._dumper is not None:
            self._dumper.write(data=data, video_frames=video_frames)


# map xyz -> zxy
FRANKA_FRONT_TRANSFORM = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
# map xyz -> zxy + flip x and y
FRANKA_BACK_TRANSFORM = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))


def main(robot_arm: Any | None,  # noqa: C901  Function is too complex
         gripper: DHGripper | None,
         webxr: WebXR,
         sound: SoundSystem | None,
         cameras: Dict[str, LinuxVideo] | None,
         output_dir: str | None = None,
         fps: int = 30,
         stream_video_to_webxr: str | None = None,
         operator_position: geom.Transform3D = FRANKA_FRONT_TRANSFORM,
         ):

    with ir.World() as world:
        frame_readers = {}
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, frame_reader = world.pipe()
            frame_readers[camera_name] = ir.ValueUpdated(ir.DefaultReader(frame_reader, None))

        webxr.controller_positions, controller_positions_reader = world.pipe()
        controller_positions_reader = ir.ValueUpdated(controller_positions_reader)
        webxr.buttons, buttons_reader = world.pipe()

        if stream_video_to_webxr is not None:
            raise NotImplementedError("TODO: fix video streaming to webxr, since it's currently lagging")
            webxr.frame = ir.map(frame_readers[stream_video_to_webxr], lambda x: x['image'])

        world.start(webxr.run, *[camera.run for camera in cameras.values()])

        robot_state, robot_commands = ir.NoOpReader(), ir.NoOpEmitter()
        if robot_arm is not None:
            robot_arm.state, robot_state = world.pipe(1)  # TODO: Shared variable
            robot_commands, robot_arm.commands = world.pipe(1)
            world.start(robot_arm.run)

        target_grip_emitter = ir.NoOpEmitter()
        if gripper is not None:
            target_grip_emitter, gripper.target_grip = world.pipe(1)
            world.start(gripper.run)

        sound_emitter = ir.NoOpEmitter()
        if sound is not None:
            sound_emitter, sound.wav_path = world.pipe()
            world.start(sound.run)

        tracker = _Tracker(operator_position if robot_arm is not None else None)
        recorder = _Recorder(
            SerialDumper(output_dir, video_fps=fps) if output_dir is not None else None, sound_emitter)
        button_handler = ButtonHandler()

        fps_counter = FPSCounter("Data Collection")
        while not world.should_stop:
            try:
                _parse_buttons(buttons_reader.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    recorder.turn_off() if recorder.on else recorder.turn_on()
                elif button_handler.just_pressed('right_A'):
                    tracker.turn_off() if tracker.on else tracker.turn_on(robot_state.value.position)
                elif button_handler.just_pressed('right_stick') and not tracker.umi_mode:
                    print("Resetting robot")
                    recorder.turn_off()
                    tracker.turn_off()
                    robot_commands.emit(roboarm.command.Reset())

                target_grip = button_handler.get_value('right_trigger')
                target_grip_emitter.emit(target_grip)

                controller_positions, controller_positions_updated = controller_positions_reader.value
                target_robot_pos = tracker.update(controller_positions['right'])
                if tracker.on and controller_positions_updated:  # Don't spam the robot with commands.
                    robot_commands.emit(roboarm.command.CartesianMove(target_robot_pos))

                frame_messages = {name: reader.read() for name, reader in frame_readers.items()}
                any_frame_updated = any(msg.data[1] and msg.data[0] is not None for msg in frame_messages.values())

                fps_counter.tick()
                if not recorder.on or not any_frame_updated:
                    time.sleep(0.001)
                    continue

                frame_messages = {name: ir.Message(msg.data[0], msg.ts) for name, msg in frame_messages.items()}

                ep_dict = {
                    'target_grip': target_grip,
                    'target_robot_position_translation': target_robot_pos.translation.copy(),
                    'target_robot_position_quaternion': target_robot_pos.rotation.as_quat.copy(),
                    **{f'{name}_timestamp': frame.ts for name, frame in frame_messages.items()},
                }
                if controller_positions['right'] is not None:
                    ep_dict['right_controller_translation'] = controller_positions['right'].translation.copy()
                    ep_dict['right_controller_quaternion'] = controller_positions['right'].rotation.as_quat.copy()
                if controller_positions['left'] is not None:
                    ep_dict['left_controller_translation'] = controller_positions['left'].translation.copy()
                    ep_dict['left_controller_quaternion'] = controller_positions['left'].rotation.as_quat.copy()

                recorder.update(data=ep_dict,
                                video_frames={name: frame.data['image'] for name, frame in frame_messages.items()})

            except ir.NoValueException:
                time.sleep(0.001)
                continue


main = ir1.Config(
    main,
    robot_arm=None,
    gripper=pimm.cfg.hardware.gripper.dh_gripper,
    webxr=pimm.cfg.webxr.webxr,
    sound=pimm.cfg.sound.sound,
    cameras=ir1.Config(
        dict,
        left=pimm.cfg.hardware.camera.arducam_left,
        right=pimm.cfg.hardware.camera.arducam_right,
    ),
    operator_position=FRANKA_FRONT_TRANSFORM,
)

if __name__ == "__main__":
    ir1.cli(main)
