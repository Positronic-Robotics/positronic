from enum import Enum
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterator

import configuronic as cfn
from mujoco import Sequence

from positronic import geom
import configuronic as cfn
import pimm
import positronic.cfg.hardware.camera
import positronic.cfg.hardware.gripper
import positronic.cfg.hardware.roboarm
import positronic.cfg.simulator
import positronic.cfg.sound
import positronic.cfg.webxr
from positronic.dataset.core import DatasetWriter
from positronic.drivers import roboarm
from positronic.drivers.webxr import WebXR
from positronic.gui.dpg import DearpyguiUi
from positronic.simulator.mujoco.sim import (
    MujocoCamera,
    MujocoFranka,
    MujocoGripper,
    MujocoSim,
)
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils.buttons import ButtonHandler

from positronic.dataset.local_dataset import LocalDatasetWriter


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
        return geom.Transform3D(self._teleop_t.translation + self._offset.translation,
                                self._teleop_t.rotation * self._offset.rotation)


# TODO: Support aborting current episode.
class Recorder:

    def __init__(self, ds_writer: DatasetWriter | None, sound_emitter: pimm.SignalEmitter[str], clock: pimm.Clock):
        self.on = False

        self._episode_count = 0
        self._ds_writer = ds_writer
        self._ep_writer = None

        self._sound_emitter = sound_emitter
        self._start_wav_path = "positronic/assets/sounds/recording-has-started.wav"
        self._end_wav_path = "positronic/assets/sounds/recording-has-stopped.wav"
        self._meta = {}
        self._clock = clock
        self._fps_counter = pimm.utils.RateCounter("Recorder")

    def turn_on(self, **static_data: dict):
        if self._ds_writer is None:
            print("No dumper, ignoring 'start recording' command")
            return

        if not self.on:
            if self._ds_writer is not None and self._ep_writer is None:
                self._ep_writer = self._ds_writer.new_episode()
                for k, v in static_data.items():
                    self._ep_writer.set_static(k, v)
                self._episode_count += 1
                print(f"Episode {self._episode_count} started")
                self._sound_emitter.emit(self._start_wav_path)
            self.on = True
        else:
            print("Already recording, ignoring 'start recording' command")

    def turn_off(self, **static_data: dict):
        if self._ds_writer is None:
            print("No dumper, ignoring 'stop recording' command")
            return

        if self.on:
            if self._ds_writer is not None:
                for k, v in static_data.items():
                    self._ep_writer.set_static(k, v)
                self._ep_writer.__exit__(None, None, None)
                self._ep_writer = None

            print(f"Episode {self._episode_count} ended")
            self._sound_emitter.emit(self._end_wav_path)
            self.on = False
        else:
            print("Not recording, ignoring turn_off")

    def write(self, signal_name: str, data: Any, ts_ns: int | None = None):
        if not self.on:
            return

        if self._ds_writer is not None:
            self._ep_writer.append(signal_name, data, ts_ns or self._clock.now_ns())


class OperatorPosition(Enum):
    # map xyz -> zxy
    FRONT = geom.Transform3D(rotation=geom.Rotation.from_quat([0.5, 0.5, 0.5, 0.5]))
    # map xyz -> zxy + flip x and y
    BACK = geom.Transform3D(rotation=geom.Rotation.from_quat([-0.5, -0.5, 0.5, 0.5]))


class DataCollection:
    frame_readers: Dict[str, pimm.SignalReader] = {}
    controller_positions_reader: pimm.SignalReader[Dict[str, geom.Transform3D]] = pimm.NoOpReader()
    buttons_reader: pimm.SignalReader[Dict] = pimm.NoOpReader()
    robot_state: pimm.SignalReader[roboarm.State] = pimm.NoOpReader()
    gripper_state: pimm.SignalReader[float] = pimm.NoOpReader()

    robot_commands: pimm.SignalEmitter[roboarm.command.CommandType] = pimm.NoOpEmitter()
    target_grip_emitter: pimm.SignalEmitter[float] = pimm.NoOpEmitter()
    sound_emitter: pimm.SignalEmitter[str] = pimm.NoOpEmitter()

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
        gripper_state_reader = pimm.DefaultReader(pimm.ValueUpdated(self.gripper_state), (None, False))

        tracker = _Tracker(self.operator_position)
        writer = LocalDatasetWriter(Path(self.output_dir)) if self.output_dir is not None else None
        recorder = Recorder(writer, self.sound_emitter, clock)
        button_handler = ButtonHandler()

        while not should_stop.value:
            try:
                _parse_buttons(self.buttons_reader.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    if recorder.on:
                        recorder.turn_off()
                    else:
                        recorder.turn_on(**self.metadata_getter())
                elif button_handler.just_pressed('right_A'):
                    if tracker.on:
                        tracker.turn_off()
                    else:
                        tracker.turn_on(self.robot_state.value.ee_pose)
                elif button_handler.just_pressed('right_stick') and not tracker.umi_mode:
                    print("Resetting robot")
                    recorder.turn_off()
                    tracker.turn_off()
                    self.robot_commands.emit(roboarm.command.Reset())

                target_grip = button_handler.get_value('right_trigger')
                recorder.write('target_grip', target_grip)
                self.target_grip_emitter.emit(target_grip)

                controller_pos, controller_pos_updated = controller_positions_reader.value
                target_ts = clock.now_ns()
                if controller_pos_updated:
                    target_robot_pos = tracker.update(controller_pos['right'])
                    if tracker.on:  # Don't spam the robot with commands.
                        recorder.write('target_robot_position_translation', target_robot_pos.translation, target_ts)
                        recorder.write('target_robot_position_quaternion', target_robot_pos.rotation.as_quat, target_ts)
                        self.robot_commands.emit(roboarm.command.CartesianMove(target_robot_pos))

                    if controller_pos['right'] is not None:
                        recorder.write('right_controller_translation', controller_pos['right'].translation, target_ts)
                        recorder.write('right_controller_quaternion', controller_pos['right'].rotation.as_quat,
                                       target_ts)
                    if controller_pos['left'] is not None:
                        recorder.write('left_controller_translation', controller_pos['left'].translation, target_ts)
                        recorder.write('left_controller_quaternion', controller_pos['left'].rotation.as_quat, target_ts)

                for name, reader in frame_readers.items():
                    frame, updated = reader.value
                    image_ts = clock.now_ns()
                    if updated:
                        for k, image in frame.items():
                            out_name = 'image.' + (name if k == 'image' else f'{name}.{k}')
                            recorder.write(out_name, image, image_ts)

                value = self.robot_state.read()

                if value is not None:
                    robot_ts = clock.now_ns()
                    value = value.data
                    recorder.write('robot_position_translation', value.ee_pose.translation, robot_ts)
                    recorder.write('robot_position_rotation', value.ee_pose.rotation.as_quat, robot_ts)
                    recorder.write('robot_joints', value.q, robot_ts)
                    recorder.write('robot_joints_velocity', value.dq, robot_ts)

                gripper_state, gripper_state_updated = gripper_state_reader.value
                if gripper_state_updated:
                    recorder.write('grip', gripper_state, target_ts)

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
         operator_position: OperatorPosition = OperatorPosition.FRONT,
         ):
    """Runs data collection in real hardware."""
    with pimm.World() as world:
        data_collection = DataCollection(operator_position.value, output_dir, fps)
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, data_collection.frame_readers[camera_name] = world.mp_pipe()

        webxr.controller_positions, data_collection.controller_positions_reader = world.mp_pipe()
        webxr.buttons, data_collection.buttons_reader = world.mp_pipe()

        if stream_video_to_webxr is not None:
            emitter, reader = world.mp_pipe()
            cameras[stream_video_to_webxr].frame = pimm.BroadcastEmitter(
                [emitter, cameras[stream_video_to_webxr].frame])

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
        operator_position: OperatorPosition = OperatorPosition.FRONT,
):
    """Runs data collection in simulator."""

    sim = MujocoSim(mujoco_model_path, loaders)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    cameras = {
        'handcam_left': MujocoCamera(sim.model, sim.data, 'handcam_left_ph', (320, 240), fps=fps),
        'handcam_right': MujocoCamera(sim.model, sim.data, 'handcam_right_ph', (320, 240), fps=fps),
    }
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')
    gui = DearpyguiUi()

    with pimm.World(clock=sim) as world:
        def metadata_getter():
            return {k: v.tolist() for k, v in sim.save_state().items()}
        data_collection = DataCollection(operator_position.value, output_dir, fps, metadata_getter=metadata_getter)
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
    operator_position=OperatorPosition.FRONT,
)

main_sim_cfg = cfn.Config(
    main_sim,
    mujoco_model_path="positronic/assets/mujoco/franka_table.xml",
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)


@cfn.config(
    robot_arm=positronic.cfg.hardware.roboarm.so101,
    webxr=positronic.cfg.webxr.oculus,
    sound=positronic.cfg.sound.sound,
    operator_position=OperatorPosition.BACK,
    cameras={'right': positronic.cfg.hardware.camera.arducam_right}
)
def so101cfg(robot_arm, **kwargs):
    """Runs data collection on SO101 robot"""
    main(robot_arm=robot_arm, gripper=robot_arm, **kwargs)


if __name__ == "__main__":
    cfn.cli({
        "real": main_cfg,
        "so101": so101cfg,
        "sim": main_sim_cfg,
    })
