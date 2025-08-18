import time
from typing import Any, Callable, Dict, Iterator

from mujoco import Sequence

from positronic import geom
import configuronic as cfn
import pimm
from positronic.drivers import roboarm
from positronic.drivers.webxr import WebXR
from positronic.gui.dpg import DearpyguiUi
from positronic.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils.buttons import ButtonHandler
from positronic.utils.dataset_dumper import SerialDumper

import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.webxr
import positronic.cfg.hardware.camera
import positronic.cfg.sound
import positronic.cfg.simulator


def _parse_buttons(buttons: Dict, button_handler: ButtonHandler):
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
class Recorder:
    def __init__(self, dumper: SerialDumper | None, sound_emitter: pimm.SignalEmitter[str], clock: pimm.Clock):
        self.on = False
        self._dumper = dumper
        self._sound_emitter = sound_emitter
        self._start_wav_path = "positronic/assets/sounds/recording-has-started.wav"
        self._end_wav_path = "positronic/assets/sounds/recording-has-stopped.wav"
        self._meta = {}
        self._clock = clock
        self._fps_counter = pimm.utils.RateCounter("Recorder")

    def turn_on(self):
        if self._dumper is None:
            print("No dumper, ignoring 'start recording' command")
            return

        if not self.on:
            self._meta['episode_start'] = self._clock.now_ns()
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
            self._meta = {}
        else:
            print("Not recording, ignoring turn_off")

    def add_metadata(self, metadata: dict):
        self._meta.update(metadata)

    def update(self, data: dict, video_frames: dict):
        if not self.on:
            return

        if self._dumper is not None:
            self._dumper.write(data=data, video_frames=video_frames)
            self._fps_counter.tick()


# map xyz -> zxy
FRANKA_FRONT_TRANSFORM = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
# map xyz -> zxy + flip x and y
FRANKA_BACK_TRANSFORM = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))


class DataCollection:
    frame_readers : Dict[str, pimm.SignalReader] = {}
    controller_positions_reader : pimm.SignalReader[Dict[str, geom.Transform3D]] = pimm.NoOpReader()
    buttons_reader : pimm.SignalReader[Dict] = pimm.NoOpReader()
    robot_state : pimm.SignalReader[roboarm.State] = pimm.NoOpReader()
    gripper_state : pimm.SignalReader[float] = pimm.NoOpReader()

    robot_commands : pimm.SignalEmitter[roboarm.command.CommandType] = pimm.NoOpEmitter()
    target_grip_emitter : pimm.SignalEmitter[float] = pimm.NoOpEmitter()
    sound_emitter : pimm.SignalEmitter[str] = pimm.NoOpEmitter()

    def __init__(
        self,
        operator_position: geom.Transform3D | None,
        output_dir: str | None,
        fps: int,
        metadata_getter: Callable[[], Dict] | None = None,
    ):
        self.operator_position = operator_position
        self.output_dir = output_dir
        self.fps = fps
        self.metadata_getter = metadata_getter or (lambda: {})

    def run(self, should_stop: pimm.SignalReader, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        frame_readers = {
            camera_name: pimm.DefaultReader(pimm.ValueUpdated(frame_reader), ({}, False))
            for camera_name, frame_reader in self.frame_readers.items()
        }
        controller_positions_reader = pimm.ValueUpdated(self.controller_positions_reader)

        tracker = _Tracker(self.operator_position)
        recorder = Recorder(
            SerialDumper(self.output_dir, video_fps=self.fps) if self.output_dir is not None else None,
            self.sound_emitter,
            clock,
        )
        button_handler = ButtonHandler()

        fps_counter = pimm.utils.RateCounter("Data Collection")
        while not should_stop.value:
            try:
                _parse_buttons(self.buttons_reader.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    if recorder.on:
                        recorder.turn_off()
                    else:
                        recorder.turn_on()
                        recorder.add_metadata(self.metadata_getter())

                elif button_handler.just_pressed('right_A'):
                    if tracker.on:
                        tracker.turn_off()
                    else:
                        robot_state = self.robot_state.value
                        tracker.turn_on(robot_state.ee_pose)
                elif button_handler.just_pressed('right_stick') and not tracker.umi_mode:
                    print("Resetting robot")
                    recorder.turn_off()
                    tracker.turn_off()
                    self.robot_commands.emit(roboarm.command.Reset())

                target_grip = button_handler.get_value('right_trigger')
                self.target_grip_emitter.emit(target_grip)

                controller_positions, controller_positions_updated = controller_positions_reader.value
                target_robot_pos = tracker.update(controller_positions['right'])
                # TODO: that's not clear how to deal with the fact that target_ts from webxr use real clock
                # and `clock` could be simulator/real clock.
                last_target_ts = clock.now_ns()

                if tracker.on and controller_positions_updated:  # Don't spam the robot with commands.
                    self.robot_commands.emit(roboarm.command.CartesianMove(target_robot_pos))

                # TODO: fix frame synchronization. Two 30 FPS cameras is updated at 60 FPS
                frame_messages, any_frame_updated = pimm.utils.is_any_updated(frame_readers)
                fps_counter.tick()
                if not recorder.on or not any_frame_updated:
                    yield pimm.Sleep(0.001)
                    continue

                ep_dict = {
                    'target_grip': target_grip,
                    'target_robot_position_translation': target_robot_pos.translation.copy(),
                    'target_robot_position_quaternion': target_robot_pos.rotation.as_quat.copy(),
                    'target_timestamp': last_target_ts,
                    **{f'{name}_timestamp': frame.ts for name, frame in frame_messages.items()},
                }

                value = self.robot_state.read()

                if value is not None:
                    value = value.data
                    ep_dict = {
                        **ep_dict,
                        'robot_position_translation': value.ee_pose.translation.copy(),
                        'robot_position_rotation': value.ee_pose.rotation.as_quat.copy(),
                        'robot_joints': value.q.copy(),
                    }

                value = self.gripper_state.read()
                if value is not None:
                    ep_dict['grip'] = value.data

                if controller_positions['right'] is not None:
                    ep_dict['right_controller_translation'] = controller_positions['right'].translation.copy()
                    ep_dict['right_controller_quaternion'] = controller_positions['right'].rotation.as_quat.copy()
                if controller_positions['left'] is not None:
                    ep_dict['left_controller_translation'] = controller_positions['left'].translation.copy()
                    ep_dict['left_controller_quaternion'] = controller_positions['left'].rotation.as_quat.copy()

                recorder.update(data=ep_dict,
                                video_frames={name: frame.data['image'] for name, frame in frame_messages.items()})
                yield pimm.Sleep(0.001)

            except pimm.NoValueException:
                yield pimm.Sleep(0.001)
                continue


def main(robot_arm: Any | None,
         gripper: Any | None,
         webxr: WebXR,
         sound: Any | None,
         cameras: Dict[str, Any] | None,
         output_dir: str | None = None,
         fps: int = 30,
         stream_video_to_webxr: str | None = None,
         operator_position: geom.Transform3D = FRANKA_FRONT_TRANSFORM,
         ):

    with pimm.World() as world:
        data_collection = DataCollection(operator_position, output_dir, fps)
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, data_collection.frame_readers[camera_name] = world.mp_pipe()

        webxr.controller_positions, data_collection.controller_positions_reader = world.mp_pipe()
        webxr.buttons, data_collection.buttons_reader = world.mp_pipe()

        if stream_video_to_webxr is not None:
            emitter, reader = world.mp_pipe()
            cameras[stream_video_to_webxr].frame = pimm.BroadcastEmitter(
                [emitter, cameras[stream_video_to_webxr].frame]
            )

            webxr.frame = pimm.map(reader, lambda x: x['image'])

        world.start_in_subprocess(webxr.run, *[camera.run for camera in cameras.values()])

        if robot_arm is not None:
            robot_arm.state, data_collection.robot_state = world.shared_memory()
            data_collection.robot_commands, robot_arm.commands = world.mp_pipe(1)
            if gripper is not robot_arm:
                world.start_in_subprocess(robot_arm.run)

        if gripper is not None:
            data_collection.target_grip_emitter, gripper.target_grip = world.mp_pipe()
            gripper.grip, data_collection.gripper_state = world.mp_pipe()
            world.start_in_subprocess(gripper.run)

        if sound is not None:
            data_collection.sound_emitter, sound.wav_path = world.mp_pipe()
            world.start_in_subprocess(sound.run)

        dc_steps = iter(world.interleave(data_collection.run))

        while not world.should_stop:
            try:
                time.sleep(next(dc_steps).seconds)
            except StopIteration:
                break


def main_sim(
        mujoco_model_path: str,
        webxr: WebXR,
        sound: Any | None = None,
        loaders: Sequence[MujocoSceneTransform] = (),
        output_dir: str | None = None,
        fps: int = 30,
        operator_position: geom.Transform3D = FRANKA_FRONT_TRANSFORM,
):

    sim = MujocoSim(mujoco_model_path, loaders)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    cameras = {
        'handcam_left': MujocoCamera(sim.model, sim.data, 'handcam_left_ph', (320, 240), fps=fps),
        'handcam_right': MujocoCamera(sim.model, sim.data, 'handcam_right_ph', (320, 240), fps=fps),
    }
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    gui = DearpyguiUi()

    with pimm.World(clock=sim) as world:
        data_collection = DataCollection(operator_position, output_dir, fps, metadata_getter=sim.save_state)
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, (data_collection.frame_readers[camera_name],
                           gui.cameras[camera_name]) = world.mp_one_to_many_pipe(2)

        webxr.controller_positions, data_collection.controller_positions_reader = world.mp_pipe()
        webxr.buttons, data_collection.buttons_reader = world.mp_pipe()

        world.start_in_subprocess(webxr.run, gui.run)

        robot_arm.state, data_collection.robot_state = world.local_pipe()
        data_collection.robot_commands, robot_arm.commands = world.local_pipe()

        data_collection.target_grip_emitter, gripper.target_grip = world.local_pipe()
        gripper.grip, data_collection.gripper_state = world.local_pipe()

        if sound is not None:
            data_collection.sound_emitter, sound.wav_path = world.mp_pipe()
            world.start_in_subprocess(sound.run)

        sim_iter = world.interleave(
            sim.run,
            *[camera.run for camera in cameras.values()],
            robot_arm.run,
            gripper.run,
            data_collection.run,
        )

        sim_iter = iter(sim_iter)

        start_time = pimm.world.SystemClock().now_ns()
        sim_start_time = sim.now_ns()

        while not world.should_stop:
            try:
                time_since_start = pimm.world.SystemClock().now_ns() - start_time
                if sim.now_ns() < sim_start_time + time_since_start:
                    next(sim_iter)
                else:
                    time.sleep(0.001)
            except StopIteration:
                break


main_cfg = cfn.Config(
    main,
    robot_arm=None,
    gripper=positronic.cfg.hardware.gripper.dh_gripper,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    cameras={
        'left': positronic.cfg.hardware.camera.arducam_left,
        'right': positronic.cfg.hardware.camera.arducam_right,
    },
    operator_position=FRANKA_FRONT_TRANSFORM,
)

main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path="positronic/assets/mujoco/franka_table.xml",
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=FRANKA_BACK_TRANSFORM,
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.so101,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=FRANKA_FRONT_TRANSFORM,
    cameras={'right': positronic.cfg.hardware.camera.arducam_right}
)
def so101cfg(robot_arm, **kwargs):
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


if __name__ == "__main__":
    # TODO: add ability to specify multiple targets in CLI
    cfn.cli(main_sim_cfg)
    # cfn.cli(so101cfg)
