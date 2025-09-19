from typing import Dict, Iterator

from mujoco import Sequence
import torch
import tqdm

from positronic import geom
import configuronic as cfn
import pimm
from positronic.drivers import roboarm
from positronic.simulator.mujoco.sim import MujocoCamera, MujocoFranka, MujocoGripper, MujocoSim
from positronic.simulator.mujoco.transforms import MujocoSceneTransform
from positronic.utils.dataset_dumper import SerialDumper
from positronic.data_collection import Recorder

import positronic.cfg.simulator


# TODO: This is not working anymore, after we moved to the new dataset format and dumping code
# This must be fixed or deleted
class DataDumper:
    def __init__(self, output_dir: str | None, fps: int) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self.frame_readers: Dict[str, pimm.SignalReceiver] = {}
        self.robot_state: pimm.SignalReceiver = pimm.NoOpReceiver()
        self.robot_commands: pimm.SignalReceiver[roboarm.command.CommandType] = pimm.NoOpReceiver()
        self.target_grip: pimm.SignalReceiver[float] = pimm.NoOpReceiver()

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:  # noqa: C901
        frame_readers = {
            camera_name: pimm.DefaultReceiver(pimm.ValueUpdated(frame_receiver), ({}, False))
            for camera_name, frame_receiver in self.frame_readers.items()
        }
        target_grip_receiver = pimm.DefaultReceiver(self.target_grip, None)
        target_robot_pos_receiver = pimm.DefaultReceiver(self.robot_commands, None)

        recorder = Recorder(
            SerialDumper(self.output_dir, video_fps=self.fps) if self.output_dir is not None else None,
            pimm.NoOpEmitter(),
            clock)
        recorder.turn_on()
        fps_counter = pimm.utils.RateCounter("Data Collection")

        while not should_stop.value:
            # TODO: fix frame synchronization. Two 30 FPS cameras is updated at 60 FPS
            frame_messages, any_frame_updated = pimm.utils.is_any_updated(frame_readers)

            fps_counter.tick()

            if not any_frame_updated:
                yield pimm.Sleep(0.001)
                continue

            target_grip = target_grip_receiver.value
            target_robot_pos = target_robot_pos_receiver.value

            if target_robot_pos is None or target_grip is None:
                yield pimm.Sleep(0.001)
                continue

            target_robot_pos = target_robot_pos.pose

            ep_dict = {
                'target_grip': target_grip,
                'target_robot_position_translation': target_robot_pos.translation.copy(),
                'target_robot_position_quaternion': target_robot_pos.rotation.as_quat.copy(),
                'target_timestamp': clock.now_ns(),
                **{f'{name}_timestamp': frame.ts for name, frame in frame_messages.items()},
            }
            recorder.update(data=ep_dict,
                            video_frames={name: frame.data['image'] for name, frame in frame_messages.items()})
            yield pimm.Sleep(0.001)

        recorder.turn_off()


class RecordReplay:
    def __init__(self, record_path: str):
        self.record = torch.load(record_path)
        self.robot_commands: pimm.SignalEmitter = pimm.NoOpEmitter()
        self.target_grip: pimm.SignalEmitter[float] = pimm.NoOpEmitter()

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        timestamps = self.record['target_timestamp'] - self.record['target_timestamp'][0]

        self._emit_commands(0)
        for i, ts in enumerate(timestamps[1:], start=1):
            yield pimm.Sleep(max(0, ts - clock.now_ns()) / 1e9)
            self._emit_commands(i)

    def _emit_commands(self, idx):
        target_grip = self.record['target_grip'][idx].item()
        target_robot_pos = self.record['target_robot_position_translation'][idx]
        target_robot_pos_rotation = geom.Rotation.from_quat(self.record['target_robot_position_quaternion'][idx])

        self.target_grip.emit(target_grip)
        self.robot_commands.emit(roboarm.command.CartesianMove(
            pose=geom.Transform3D(translation=target_robot_pos, rotation=target_robot_pos_rotation)),
        )


def main(
        record_path: str,
        mujoco_model_path: str,
        loaders: Sequence[MujocoSceneTransform] = (),
        output_dir: str | None = None,
        fps: int = 30,
):
    record_replay = RecordReplay(record_path)
    sim = MujocoSim(mujoco_model_path, loaders)
    sim.load_state(record_replay.record)
    robot_arm = MujocoFranka(sim, suffix='_ph')
    cameras = {
        'handcam_left': MujocoCamera(sim.model, sim.data, 'handcam_left_ph', (320, 240), fps=fps),
        'handcam_right': MujocoCamera(sim.model, sim.data, 'handcam_right_ph', (320, 240), fps=fps),
    }
    gripper = MujocoGripper(sim, actuator_name='actuator8_ph', joint_name='finger_joint1_ph')

    with pimm.World(clock=sim) as world:
        data_collection = DataDumper(output_dir, fps)
        cameras = cameras or {}
        for camera_name, camera in cameras.items():
            camera.frame, data_collection.frame_readers[camera_name] = world.local_pipe()

        robot_arm.state, data_collection.robot_state = world.local_pipe()

        record_replay.robot_commands, (robot_arm.commands,
                                       data_collection.robot_commands) = world.local_one_to_many_pipe(2)

        record_replay.target_grip, (gripper.target_grip,
                                    data_collection.target_grip) = world.local_one_to_many_pipe(2)

        sim_iter = world.interleave(
            *[camera.run for camera in cameras.values()],
            robot_arm.run,
            gripper.run,
            record_replay.run,
            data_collection.run,
            sim.run,
        )

        for _ in tqdm.tqdm(sim_iter):
            pass


main_cfg = cfn.Config(
    main,
    mujoco_model_path="positronic/assets/mujoco/franka_table.xml",
    loaders=positronic.cfg.simulator.stack_cubes_loaders,
)

if __name__ == "__main__":
    cfn.cli(main_cfg)
