import asyncio
from typing import Dict

import mujoco

import ironic as ir1
import ironic2 as ir
from ironic.utils import FPSCounter
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


class DataCollection:
    buttons_reader: ir.SignalReader = ir.NoOpReader()
    controller_positions_reader: ir.SignalReader = ir.NoOpReader()
    target_grip_emitter: ir.SignalEmitter = ir.NoOpEmitter()
    wav_path_emitter: ir.SignalEmitter = ir.NoOpEmitter()
    frame_readers: Dict[str, ir.SignalReader] = {}

    def __init__(self, output_dir: str, fps: int):
        self.output_dir = output_dir
        self.fps = fps

    async def run(self, should_stop: ir.SignalReader, clock: ir.Clock):
        tracked = False
        dumper = SerialDumper(self.output_dir, video_fps=self.fps)
        button_handler = ButtonHandler()

        meta = {}
        start_wav_path = "positronic/assets/sounds/recording-has-started.wav"
        end_wav_path = "positronic/assets/sounds/recording-has-stopped.wav"

        fps_counter = FPSCounter("Data Collection")
        while not should_stop.value:
            try:
                _parse_buttons(self.buttons_reader.value, button_handler)
                if button_handler.just_pressed('right_B'):
                    tracked = not tracked
                    if tracked:
                        meta['episode_start'] = ir.system_clock()
                        dumper.start_episode()
                        print(f"Episode {dumper.episode_count} started")
                        self.wav_path_emitter.emit(start_wav_path)
                    else:
                        dumper.end_episode(meta)
                        meta = {}
                        print(f"Episode {dumper.episode_count} ended")
                        self.wav_path_emitter.emit(end_wav_path)
                # TODO: Support aborting current episode.

                frame_messages = {name: reader.read() for name, reader in self.frame_readers.items()}
                any_frame_updated = any(msg.data[1] and msg.data[0] is not None for msg in frame_messages.values())

                target_grip = button_handler.get_value('right_trigger')
                self.target_grip_emitter.emit(target_grip)

                if not tracked or not any_frame_updated:
                    await clock.sleep(0.001)
                    continue

                frame_messages = {name: ir.Message(msg.data[0], msg.ts) for name, msg in frame_messages.items()}
                controller_positions = self.controller_positions_reader.value
                right_controller_position = controller_positions['right']
                left_controller_position = controller_positions['left']

                ep_dict = {
                    'target_grip': target_grip,
                    'target_robot_position_translation': right_controller_position.translation.copy(),
                    'target_robot_position_quaternion': right_controller_position.rotation.as_quat.copy(),
                    'umi_right_translation': right_controller_position.translation.copy(),
                    'umi_right_quaternion': right_controller_position.rotation.as_quat.copy(),
                    'umi_left_translation': left_controller_position.translation.copy(),
                    'umi_left_quaternion': left_controller_position.rotation.as_quat.copy(),
                    **{
                        f'{name}_timestamp': frame.ts
                        for name, frame in frame_messages.items()
                    },
                }

                dumper.write(data=ep_dict,
                             video_frames={
                                 name: frame.data['image']
                                 for name, frame in frame_messages.items()
                             })
                fps_counter.tick()

            except ir.NoValueException:
                await clock.sleep(0.001)
                continue


def main(
    gripper: DHGripper | None,  # noqa: C901  Function is too complex
    webxr: WebXR,
    sound: SoundSystem | None,
    cameras: Dict[str, LinuxVideo],
    output_dir: str = "data_collection_umi",
    fps: int = 30,
    stream_video_to_webxr: str | None = None,
):

    # TODO: this function modifies outer objects with pipes

    with ir.World(clock=ir.RealClock()) as world:
        data_collection = DataCollection(output_dir, fps)

        for camera_name, camera in cameras.items():
            camera.frame, frame_reader = world.pipe()
            data_collection.frame_readers[camera_name] = ir.ValueUpdated(ir.DefaultReader(frame_reader, None))

        webxr.controller_positions, controller_positions_reader = world.pipe()
        webxr.buttons, data_collection.buttons_reader = world.pipe()
        if stream_video_to_webxr is not None:
            raise NotImplementedError("TODO: fix video streaming to webxr, since it's currently lagging")
            webxr.frame = ir.map(data_collection.frame_readers[stream_video_to_webxr], lambda x: x['image'])

        world.start(webxr.run, *[camera.run for camera in cameras.values()])

        if gripper is not None:
            target_grip_emitter, gripper.target_grip = world.pipe()
            data_collection.target_grip_emitter = target_grip_emitter
            world.start(gripper.run)

        if sound is not None:
            wav_path_emitter, sound.wav_path = world.pipe()
            data_collection.wav_path_emitter = wav_path_emitter
            world.start(sound.run)

        asyncio.run(data_collection.run(world.should_stop, world.clock))


class MuJoCoClock(ir.Clock):

    def __init__(self, mujoco_model: mujoco.MjModel, mujoco_data: mujoco.MjData, realtime=False):
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.realtime = realtime

    def now(self) -> float:
        return self.mujoco_data.time

    async def sleep(self, seconds: float):
        # TODO: Correct
        await asyncio.sleep(seconds)

    @property
    def is_realtime(self) -> bool:
        return self.realtime


class SimCamera:
    frame: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, mujoco_model: mujoco.MjModel, mujoco_data: mujoco.MjData, camera_name: str, width: int,
                 height: int, fps: int):
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps

    async def run(self, should_stop: ir.SignalReader, clock: ir.Clock):
        while not should_stop.value:
            # TODO: Real render
            self.frame.emit({'image': self.mujoco_data.camera_image(self.camera_name)}, ts=clock.now())
            await clock.sleep(1 / self.fps)  # TODO: Correct


class SimRobot:
    robot_state: ir.SignalEmitter = ir.NoOpEmitter()
    commands: ir.SignalReader = ir.NoOpReader()

    def __init__(self, mujoco_model: mujoco.MjModel, mujoco_data: mujoco.MjData, hz: int = 100):
        self.mujoco_model = mujoco_model
        self.mujoco_data = mujoco_data
        self.hz = hz

    async def run(self, should_stop: ir.SignalReader, clock: ir.Clock):
        while not should_stop.value:
            self.mujoco_data.ctrl = self.commands.value
            self.robot_state.emit(self.mujoco_data.qpos.copy(), ts=clock.now())

            await clock.sleep(1 / self.hz)


def main_sim(mujoco_model: mujoco.MjModel, mujoco_data: mujoco.MjData, output_dir: str, fps: int):
    clock = MuJoCoClock(mujoco_model, mujoco_data)
    with ir.World(clock=clock) as world:
        data_collection = DataCollection(output_dir, fps)
        cameras = {name: SimCamera(mujoco_model, mujoco_data, name, 1280, 720, fps) for name in ['left', 'right']}
        for camera_name, camera in cameras.items():
            camera.frame, frame_reader = world.pipe()
            data_collection.frame_readers[camera_name] = ir.ValueUpdated(ir.DefaultReader(frame_reader, None))

        robot = SimRobot(mujoco_model, mujoco_data)
        data_collection.commands, robot.commands = world.pipe()  # Must be usual deque
        robot.state, data_collection.robot_state = world.pipe()  # Must be usual deque

        # sim_camera1, data_collection.camera = world.deque()
        # sim_camera2, ui.camera = world.pipe()
        # camera.frame = CombinedEmitter(sim_camera1, sim_camera2)

        asyncio.run(
            asyncio.gather(*[camera.run(world.should_stop, world.clock) for camera in cameras.values()],
                           robot.run(world.should_stop, world.clock),
                           data_collection.run(world.should_stop, world.clock)))


main = ir1.Config(
    main,
    gripper=pimm.cfg.hardware.gripper.dh_gripper,
    webxr=pimm.cfg.webxr.webxr,
    sound=pimm.cfg.sound.sound,
    cameras=ir1.Config(
        dict,
        left=pimm.cfg.hardware.camera.arducam_left,
        right=pimm.cfg.hardware.camera.arducam_right,
    ),
)

if __name__ == "__main__":
    ir1.cli(main)
